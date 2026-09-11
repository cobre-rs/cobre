//! Policy loading and compatibility validation.
//!
//! [`validate_policy_load`] is the single entry point for compatibility
//! validation; every load path (full-FCF warm-start/resume/simulation-only and
//! boundary-cut injection) routes through it and returns a [`PolicyLoadProof`]
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
use cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL;
use cobre_io::EntitySlot;
use cobre_io::GraphManifest;
use cobre_io::OwnedPolicyBasisRecord;
use cobre_io::OwnedPolicyCutRecord;
use cobre_io::PolicyCheckpoint;
use cobre_io::STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL;
use cobre_io::STAGE_CUTS_NODE_ID_SENTINEL;
use cobre_io::StageCutsReadResult;
use cobre_io::read_policy_checkpoint;
use cobre_solver::{Basis, BasisStatus};

use crate::SddpError;
use crate::cut::pool::CutPool;
use crate::policy::reconcile::{
    BoundaryReconciliationReport, RebindOp, build_boundary_fold, build_rebind,
    build_reconciliation_report, decode_month_anchor, dropped_source_positions, overlap_hours,
    rebind_cut,
};
use crate::setup::{BoundaryStateRequirements, NodeId, NodePos, StudySetup, TypedVec};
use crate::workspace::CapturedBasis;
use cobre_io::StateFamily;

use std::marker::PhantomData;
use std::ops::Deref;
use std::path::Path;

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

/// `rescale_cut_records_for_load` applied to every stage of a full policy
/// checkpoint — the [`FullFcf`] load path (training warm-start/resume and
/// simulation-only runs both route through this before the records reach
/// [`crate::FutureCostFunction::from_deserialized`] /
/// [`crate::FutureCostFunction::new_with_warm_start`]).
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
    /// `false` ([`BoundaryInjection`]) defers to the per-slot reconciliation in
    /// [`load_boundary_cuts`], which tolerates a differing source/current
    /// dimension.
    const CHECK_STATE_DIMENSION: bool;
    /// Whether `num_stages` equality is hard-rejected for this kind.
    const CHECK_NUM_STAGES: bool;
    /// Whether the pool count and graph-manifest identity are hard-rejected for
    /// this kind. The pool-count analogue of [`Self::CHECK_NUM_STAGES`].
    const CHECK_N_POOLS: bool;
    /// Whether per-slot identity is checked here as an EXACT positional match
    /// ([`compare_manifest_slot_identity`]). `false` means the kind reconciles
    /// slot identity by its own mechanism instead, confined to its own load
    /// path (currently only [`BoundaryInjection`], via
    /// [`crate::policy::reconcile::build_rebind`] in [`load_boundary_cuts`]).
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

/// Validate that `source`'s state layout is compatible with `current`'s, per
/// `K`'s check matrix: `state_dimension` equality is hard-rejected only when
/// `K::CHECK_STATE_DIMENSION` ([`FullFcf`]); [`BoundaryInjection`] skips it and
/// defers to the per-slot reconciliation in [`load_boundary_cuts`]. `num_stages`
/// equality is hard-rejected only when `K::CHECK_NUM_STAGES` ([`FullFcf`]).
/// Per-slot identity is an EXACT positional match (delegated to
/// [`compare_manifest_slot_identity`]) only when `K::CHECK_SLOT_IDENTITY_EXACT`
/// ([`FullFcf`]); [`BoundaryInjection`] skips it here — its slot identity is
/// RECONCILED separately, by [`crate::policy::reconcile::build_rebind`] in
/// [`load_boundary_cuts`], never by this lower-level manifest check.
/// `col_scale`/scaling is never a compatibility dimension. This is the single
/// entry point for policy-load validation — its success is the only way to
/// construct a [`PolicyLoadProof<K>`], so every load path routes through it.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] under [`FullFcf`] on a
/// `state_dimension` mismatch, a `num_stages` mismatch, or a per-slot
/// identity mismatch (see [`compare_manifest_slot_identity`]).
pub fn validate_policy_load<K: PolicyLoadKind>(
    source: &PolicyStageManifest<'_>,
    current: &PolicyStageManifest<'_>,
) -> Result<PolicyLoadProof<K>, SddpError> {
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

/// Build a basis cache from deserialized checkpoint basis records.
///
/// Returns a `Vec<Option<CapturedBasis>>` of length `n_nodes` (`== node_ids.len()`),
/// one entry per canonical node position; nodes without a matching record get
/// `None`. Each basis record is keyed by its own node ordinal (its `stage_id`),
/// so leaves sharing a pool land in distinct node slots — no `>= num_stages`
/// drop, no cross-node collision, both of which the pre-branching per-stage
/// sizing produced once `n_nodes > n_stages`. `u8` status codes decode via
/// `from_discriminant_code`, the mirror of `convert_basis_cache`'s export-side
/// `to_discriminant_code`; a pre-existing checkpoint (bytes `0..=4`) decodes
/// identically, since that range means the same in the canonical and `HiGHS`
/// code spaces.
///
/// # Cut-slot reconstruction
///
/// `row_status` is `[template rows…, cut rows…]`, the trailing `num_cut_rows` in
/// capture-time [`CutPool::active_cuts`](crate::cut::pool::CutPool::active_cuts)
/// order (active slots, increasing). A node's cut records live in its OWN pool's
/// [`StageCutsReadResult`] (`sc.stage_id == node_pools[node]`), never the record
/// whose pool id happens to equal the node ordinal — the two diverge once
/// `n_pools != n_nodes` on a branching graph. Slot identity is recovered from
/// that pool's active records' `slot_index` in increasing order, so
/// `reconstruct_basis` preserves stored cut-row statuses across cut-set churn.
///
/// # Graceful fallback
///
/// When the derived active-slot count ≠ `num_cut_rows` (cut selection deactivated
/// cuts between capture and export) or no cut record matches, fall back to safe
/// all-template behavior (empty `cut_row_slots`; every cut row reconstructs
/// BASIC). This changes only the warm-start solve path, never the optimum.
///
/// `node_ids` / `node_pools` are the CURRENT study's `NodeGraph::node_ids` and
/// `NodeGraph::node_pool_ids` (both length `n_nodes`): a resume/warm-start
/// continues the SAME node topology, so each reconstructed basis's `node_id` is
/// `node_ids[node]` and its owning pool is `node_pools[node]` — never a value
/// recovered from the checkpoint itself (the checkpoint wire carries no node id).
#[must_use]
pub fn build_basis_cache_from_checkpoint(
    stage_bases: &[OwnedPolicyBasisRecord],
    stage_cuts: &[StageCutsReadResult],
    node_ids: &TypedVec<NodePos, NodeId>,
    node_pools: &TypedVec<NodePos, usize>,
) -> Vec<Option<CapturedBasis>> {
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
        let col_status: Vec<BasisStatus> = record
            .column_status
            .iter()
            .map(|&c| BasisStatus::from_discriminant_code(c))
            .collect();
        let row_status: Vec<BasisStatus> = record
            .row_status
            .iter()
            .map(|&r| BasisStatus::from_discriminant_code(r))
            .collect();

        let num_cut = record.num_cut_rows as usize;
        let pool = node_pools[node];
        let active_slots: Option<Vec<u32>> = stage_cuts
            .iter()
            .find(|sc| sc.stage_id as usize == pool)
            .map(|sc| {
                sc.cuts
                    .iter()
                    .filter(|c| c.is_active)
                    .map(|c| c.slot_index)
                    .collect()
            });

        let (base_row_count, cut_row_slots) = match active_slots {
            Some(slots) if slots.len() == num_cut && num_cut <= row_status.len() => {
                (row_status.len() - num_cut, slots)
            }
            _ => (row_status.len(), Vec::new()),
        };
        debug_assert_eq!(
            cut_row_slots.len(),
            row_status.len() - base_row_count,
            "build_basis_cache_from_checkpoint: cut_row_slots length must equal the trailing \
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
    cache
}

/// Positional identity of one state-vector slot; `was_active` is excluded —
/// adding it would reject a cut whose entity merely changed activity across
/// studies.
fn slot_identity(slot: &EntitySlot) -> (u8, i32, u32) {
    (slot.entity_type, slot.entity_id, slot.subindex)
}

/// Whether `source`/`current` can be identity-verified at all: an empty
/// manifest on either side (a pre-manifest checkpoint) cannot be verified —
/// warn once and answer `false`, leaving the caller's `state_dimension` check
/// as the sole compatibility guard. Shared by [`compare_manifest_slot_identity`]
/// (the [`FullFcf`] exact-match path) and [`load_boundary_cuts`]'s reconcile
/// path, so the two report the identical warning on an absent manifest.
fn manifest_identity_verifiable(
    source: &[EntitySlot],
    current: &[EntitySlot],
    on_warning: &mut dyn FnMut(&str),
) -> bool {
    if source.is_empty() || current.is_empty() {
        on_warning(&format!(
            "entity manifest absent (source slots: {}, current slots: {}); slot identity \
             could not be verified, relying on state_dimension alone",
            source.len(),
            current.len(),
        ));
        return false;
    }
    true
}

/// Compare two entity manifests slot-for-slot by `slot_identity`.
///
/// `source` (a loaded policy's manifest) and `current` (the current study's
/// terminal manifest) describe the same-length state vector of two studies. A
/// per-slot `(entity_type, entity_id, subindex)` mismatch means a cut coefficient
/// would attach to the wrong state variable and is REJECTED. `was_active` is
/// excluded from `slot_identity`, so a `source`-dormant slot now active only warns.
/// An empty manifest on either side (a pre-manifest checkpoint) cannot be
/// verified: warn and return `Ok`, leaving the caller's `state_dimension` check
/// standing.
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
    if !manifest_identity_verifiable(source, current, on_warning) {
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
/// max over every pool (not a chosen `source_stage`): the inflow-lag block is a
/// property of the policy's one state space, so the deepest slot is pool-invariant
/// for a coherent policy and the max never under-reserves. `0` when the policy
/// carries no inflow-lag slot.
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

/// Load boundary cuts from the `source_stage` of a source Cobre policy checkpoint.
///
/// Pool resolution and the source cost scale are read entirely from the
/// resolved pool's own `cuts/<pool>.bin` — its `graph_stage_id` and
/// `cost_scale_factor` self-describing facts — never from `metadata.json`'s
/// `graph_manifest`/`producer.cost_scale_factor`. A resolved pool whose
/// `cost_scale_factor` reads `None` predates those facts and REJECTS: there is
/// no silent default and no `metadata.json` fallback (see the resolution
/// section of `# Errors` below). A resolved pool shared by more than one
/// node (`node_id == STAGE_CUTS_NODE_ID_SENTINEL`) REJECTS too — a boundary
/// source must be a single-node terminal pool.
///
/// Compares the source stage's manifest against the current TERMINAL-stage
/// manifest (`current_manifest`, built via
/// [`StudySetup::build_terminal_entity_manifest`](crate::StudySetup::build_terminal_entity_manifest));
/// `num_stages` may differ. `state_dimension`/`num_stages` compatibility
/// routes through [`validate_policy_load`] typed to [`BoundaryInjection`];
/// per-slot identity is then RECONCILED, never exact-matched, via
/// [`crate::policy::reconcile::build_rebind`]/`rebind_cut` — storage and
/// inflow-lag reject a target slot with no source counterpart, naming the
/// offending hydro or lag depth (the entity is never relaxed); a live, dated
/// anticipated target slot fans out against the source's own priced
/// anticipated months by calendar overlap
/// (`target_delivery_intervals`, aligned 1:1 with `current_manifest` — built
/// via
/// [`StudySetup::build_terminal_anticipated_delivery_intervals`](crate::StudySetup::build_terminal_anticipated_delivery_intervals)).
/// An empty manifest on either side skips reconciliation and warns, relying
/// on `state_dimension` alone; a `was_active == false` boundary slot whose
/// current counterpart is active warns and loads. Wraps the result in a
/// [`ValidatedBoundaryCuts`] — the sole constructor [`inject_boundary_cuts`]
/// accepts — carrying a [`crate::policy::reconcile::build_reconciliation_report`]
/// tally on the reconcile path, or the `reconciled: false` default on the
/// skipped path; read via [`ValidatedBoundaryCuts::report`].
///
/// `effective_inflow_lag_depth` is the resolved lag depth the current state
/// layout reserves — inferred from this boundary policy via
/// [`resolve_boundary_state_requirements`]. A cut
/// referencing inflow-lag state deeper than it is a coupling regression (the
/// layout should have been sized to hold the loaded cuts), rejected before the
/// manifest checks so the lag-depth-specific message wins over the generic
/// `state_dimension` reject.
///
/// `fixed_windows` are the current study's fixed post-horizon anticipated
/// commitments (built via
/// [`StudySetup::build_terminal_fixed_post_horizon_windows`](crate::StudySetup::build_terminal_fixed_post_horizon_windows)).
/// When the manifest is verifiable,
/// [`crate::policy::reconcile::build_boundary_fold`] folds their displacement
/// value into each boundary cut's intercept on the RAW records, BEFORE
/// `rescale_cut_records_for_load`, so the folded term rides both rescale
/// transforms with the rest of the intercept; an empty fold leaves every
/// intercept bit-identical, and an unverifiable manifest skips the fold (it
/// shares the rebind's gate).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read
/// - `source_stage` matches no pool's `graph_stage_id` (or, on a checkpoint
///   where no pool carries a resolvable `graph_stage_id`, no pool's own id),
///   or more than one pool's `graph_stage_id`
/// - The resolved pool's `.bin` predates self-describing cuts
///   (`cost_scale_factor` reads `None`); re-export with a current Cobre
/// - The resolved pool is shared by more than one node (`node_id` reads the
///   sentinel); a boundary source must be a single-node terminal pool
/// - A cut references inflow-lag state deeper than `effective_inflow_lag_depth`
///   (a layout-sizing coupling regression)
/// - The source stage's state dimension does not match `current_state_dimension`
/// - A target storage or inflow-lag slot has no source counterpart under
///   identity reconciliation
/// - A live anticipated source slot's `delivery_date` fails to decode to a real
///   calendar month (propagated from the intercept fold, before the rebind)
///
/// A dated anticipated target slot with no resolved `target_delivery_intervals`
/// entry is NOT an error: it is an in-study ring slot (a within-horizon
/// delivery) the boundary does not price, resolved to `Zero` (see
/// `reconcile::resolve_anticipated`).
pub fn load_boundary_cuts(
    boundary_path: &Path,
    source_stage: u32,
    current_state_dimension: u32,
    current_manifest: &[EntitySlot],
    target_delivery_intervals: &[Option<(NaiveDate, NaiveDate)>],
    fixed_windows: &[AnticipatedCommitmentHistory],
    effective_inflow_lag_depth: Option<u32>,
    loading_cost_scale_factor: f64,
    on_warning: &mut dyn FnMut(&str),
) -> Result<ValidatedBoundaryCuts, SddpError> {
    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
        SddpError::Validation(format!(
            "failed to read boundary policy checkpoint at {}: {e}",
            boundary_path.display()
        ))
    })?;

    // Resolve `source_stage` to its pool through each pool's own
    // `graph_stage_id` (never `metadata.graph_manifest`), rejecting a
    // multi-node stage: boundary injection requires a single-node source and
    // the frozen `policy.boundary` config offers no node selector. A pool
    // with no resolvable `graph_stage_id` at all is the .bin-level mirror of
    // the old empty-graph-manifest fallback — but the fallback triggers only
    // when EVERY pool in the checkpoint shares that property, so a
    // genuinely branching checkpoint missing just one stage's pool never
    // mis-resolves through an unrelated pool whose raw id happens to match
    // `source_stage` numerically.
    let source_stage_i32 = i32::try_from(source_stage).map_err(|_| {
        SddpError::Validation(format!(
            "boundary policy: source_stage {source_stage} overflows the stage id space"
        ))
    })?;
    let graph_less = checkpoint
        .stage_cuts
        .iter()
        .all(|sr| sr.graph_stage_id == STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL);

    let stage_result = if graph_less {
        checkpoint
            .stage_cuts
            .iter()
            .find(|sr| sr.stage_id == source_stage)
    } else {
        let mut at_stage = checkpoint
            .stage_cuts
            .iter()
            .filter(|sr| sr.graph_stage_id == source_stage_i32);
        let first = at_stage.next();
        if first.is_some() && at_stage.next().is_some() {
            return Err(SddpError::Validation(format!(
                "boundary policy: source_stage {source_stage} names a multi-node stage; boundary \
                 injection requires a single-node source"
            )));
        }
        first
    }
    .ok_or_else(|| {
        SddpError::Validation(format!(
            "boundary policy: source_stage {} not found in checkpoint \
             (available pools: {:?})",
            source_stage,
            checkpoint
                .stage_cuts
                .iter()
                .map(|sr| sr.stage_id)
                .collect::<Vec<_>>()
        ))
    })?;

    let Some(source_cost_scale_factor) = stage_result.cost_scale_factor else {
        return Err(boundary_predates_self_describing_cuts(boundary_path));
    };

    // A shared pool (more than one owning node, `node_id ==
    // STAGE_CUTS_NODE_ID_SENTINEL`) can never be a boundary source — there is
    // no single terminal leaf its state belongs to.
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
                "internal: boundary policy stage {source_stage} references inflow-lag state to \
                 depth {depth}, but the resolved state layout reserves only {reserved} inflow-lag \
                 slots — resolve_boundary_state_requirements must infer the depth from the boundary \
                 policy before the layout is built; reaching here means that coupling broke and cut \
                 coefficients would truncate"
            )));
        }
    }

    // `BoundaryInjection` checks neither `num_stages`, `n_pools`, nor the
    // graph manifest, so these unchecked fields carry placeholders.
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
    let proof = validate_policy_load::<BoundaryInjection>(&source, &current)?;
    for warning in &proof.warnings {
        on_warning(warning);
    }

    let mut records = stage_result.cuts.clone();

    let verifiable =
        manifest_identity_verifiable(&stage_result.entity_manifest, current_manifest, on_warning);

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

    if verifiable {
        // The fold reads SOURCE coefficients by source_pos and MUST run before rebind_cut
        // replaces record.coefficients with the target-aligned vector; placed before rescale
        // so the folded future-cost term rides both rescale transforms with the rest of the
        // intercept.
        let fold = build_boundary_fold(&stage_result.entity_manifest, fixed_windows)?;
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
    }

    rescale_cut_records_for_load(
        &mut records,
        Some(source_cost_scale_factor),
        loading_cost_scale_factor,
    );

    let report = if verifiable {
        let rebind = build_rebind(
            &stage_result.entity_manifest,
            current_manifest,
            target_delivery_intervals,
        )?;
        for record in &mut records {
            record.coefficients = rebind_cut(record, &rebind);
        }
        warn_dropped_source_couplings(&stage_result.entity_manifest, &rebind, on_warning);
        build_reconciliation_report(
            &stage_result.entity_manifest,
            current_manifest,
            target_delivery_intervals,
            &rebind,
        )
    } else {
        BoundaryReconciliationReport::default()
    };

    Ok(ValidatedBoundaryCuts { records, report })
}

/// Human-readable label for a source slot's state family, for the dropped-coupling
/// warning below.
fn family_label(family: Option<StateFamily>) -> &'static str {
    match family {
        Some(StateFamily::HydroStorage) => "hydro-storage",
        Some(StateFamily::HydroInflowLag) => "inflow-lag",
        Some(StateFamily::AnticipatedThermalState) => "anticipated-thermal",
        Some(StateFamily::HydroTransitBucket) => "transit-bucket",
        None => "other",
    }
}

/// Warn — never reject — when a boundary SOURCE cut couples a state slot the current
/// study does not model, so its coefficient is dropped during reconciliation. A
/// superset source (more plants/arcs than this study declares) drops legitimately,
/// so this is informational and the load proceeds; it exists to make the drop
/// visible rather than silent (the transit-bucket family is the motivating case, but
/// the drop is family-blind). Rejecting would break a legitimate superset boundary,
/// and no widening path can fabricate a slot the study's topology never declared.
fn warn_dropped_source_couplings(
    source: &[EntitySlot],
    rebind: &[RebindOp],
    on_warning: &mut dyn FnMut(&str),
) {
    let dropped = dropped_source_positions(source.len(), rebind);
    if dropped.is_empty() {
        return;
    }
    let mut by_family: std::collections::BTreeMap<&'static str, (usize, Vec<String>)> =
        std::collections::BTreeMap::new();
    for pos in dropped {
        if let Some(slot) = source.get(pos) {
            let entry = by_family.entry(family_label(slot.family())).or_default();
            entry.0 += 1;
            if entry.1.len() < 5 {
                entry.1.push(format!(
                    "(id {}, subindex {})",
                    slot.entity_id, slot.subindex
                ));
            }
        }
    }
    for (label, (count, examples)) in by_family {
        on_warning(&format!(
            "boundary policy prices {count} {label} state slot(s) this study does not model \
             (e.g. {}); their coupling is dropped",
            examples.join(", ")
        ));
    }
}

/// Auto-resolve an absent `policy.boundary.source_stage`: decode each source
/// pool's live anticipated `delivery_date` months
/// ([`decode_month_anchor`]) and pick the pool whose months overlap
/// `current_terminal_delivery_intervals` — the same `target_delivery_intervals`
/// axis [`load_boundary_cuts`] fans coefficients onto (built by
/// [`StudySetup::build_terminal_anticipated_delivery_intervals`](crate::StudySetup::build_terminal_anticipated_delivery_intervals)).
/// The winning candidate is a POOL id (`checkpoint.stage_cuts`'s own
/// `stage_id` field names a pool, not a graph stage — see
/// [`cobre_io::StageCutsPayload`]'s doc); [`load_boundary_cuts`]'s
/// `source_stage` parameter speaks graph-stage-id, so this resolver maps the
/// winning pool back to a stage id via that SAME pool's own `graph_stage_id`
/// self-describing fact — never `metadata.graph_manifest`. A winning pool
/// whose `graph_stage_id` is the [`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL`]
/// (a pre-change checkpoint, or an internally unresolved stage key) REJECTS
/// rather than falling back to the raw pool id: unlike `load_boundary_cuts`'s
/// own graph-less fallback (licensed only when the checkpoint has no
/// resolvable `graph_stage_id` anywhere), silently returning an unmapped pool
/// id HERE is the reproduced defect this guards — threaded back into
/// `load_boundary_cuts` as `source_stage` on a branching graph, it can match
/// an unrelated pool whose OWN id happens to equal the winning pool's numeric
/// value, silently resolving to a different pool.
///
/// This only PICKS a candidate; it is not a new trust boundary — a
/// calendar-matched pool from an incompatible source still rejects once
/// [`load_boundary_cuts`] reconciles it by storage/lag identity, and a
/// calendar-matched pool shared by more than one node still rejects there too
/// (a boundary source must be a single-node terminal pool).
///
/// # Errors
///
/// Returns [`SddpError::Validation`] if:
/// - The checkpoint cannot be read.
/// - More than one source pool's decoded months overlap the current terminal
///   window (ambiguous; names the count and advises an explicit
///   `source_stage`).
/// - No source pool carries any decodable live anticipated `delivery_date`
///   (a pre-dated-slot checkpoint, or a pure storage/lag boundary; advises
///   re-exporting or an explicit `source_stage`).
/// - Every source pool that does carry decodable months has none overlapping
///   the current terminal window (advises an explicit `source_stage`).
/// - A live anticipated source slot's `delivery_date` fails to decode to a
///   real calendar month.
/// - The winning pool's own `graph_stage_id` is the sentinel (advises an
///   explicit `source_stage`), or the pool has no entry in
///   `checkpoint.stage_cuts` at all (an internal inconsistency: the pool id
///   was drawn from `checkpoint.stage_cuts` itself).
pub fn resolve_boundary_source_stage(
    boundary_path: &Path,
    current_terminal_delivery_intervals: &[Option<(NaiveDate, NaiveDate)>],
) -> Result<u32, SddpError> {
    let checkpoint = read_policy_checkpoint(boundary_path).map_err(|e| {
        SddpError::Validation(format!(
            "failed to read boundary policy checkpoint at {}: {e}",
            boundary_path.display()
        ))
    })?;

    let target_intervals: Vec<(NaiveDate, NaiveDate)> = current_terminal_delivery_intervals
        .iter()
        .copied()
        .flatten()
        .collect();

    let mut any_decodable = false;
    let mut candidates: Vec<u32> = Vec::new();
    for stage in &checkpoint.stage_cuts {
        let months = decode_pool_anticipated_months(&stage.entity_manifest)?;
        if months.is_empty() {
            continue;
        }
        any_decodable = true;
        let overlaps = months.iter().any(|&month| {
            target_intervals
                .iter()
                .any(|&iv| overlap_hours(month, iv) > 0.0)
        });
        if overlaps {
            candidates.push(stage.stage_id);
        }
    }

    if candidates.len() > 1 {
        return Err(SddpError::Validation(format!(
            "ambiguous: {} source pools match the terminal date; set \
             policy.boundary.source_stage explicitly",
            candidates.len()
        )));
    }
    if let Some(&pool) = candidates.first() {
        let stage_result = checkpoint
            .stage_cuts
            .iter()
            .find(|sr| sr.stage_id == pool)
            .ok_or_else(|| {
                SddpError::Validation(format!(
                    "boundary policy: pool {pool} has no entry in checkpoint.stage_cuts \
                     (internal inconsistency)"
                ))
            })?;
        if stage_result.graph_stage_id == STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL {
            return Err(SddpError::Validation(format!(
                "boundary policy: pool {pool} carries no resolvable graph_stage_id (a \
                 pre-change checkpoint, or an internally unresolved stage key); set \
                 policy.boundary.source_stage explicitly"
            )));
        }
        return u32::try_from(stage_result.graph_stage_id).map_err(|_| {
            SddpError::Validation(format!(
                "boundary policy: pool {pool}'s graph_stage_id {} is negative",
                stage_result.graph_stage_id
            ))
        });
    }
    if any_decodable {
        return Err(SddpError::Validation(
            "no source pool's anticipated delivery calendar aligns with the current study's \
             terminal delivery window; set policy.boundary.source_stage explicitly"
                .to_string(),
        ));
    }
    Err(SddpError::Validation(
        "this boundary predates dated slots; re-export it, or set policy.boundary.source_stage \
         explicitly"
            .to_string(),
    ))
}

/// Decode `manifest`'s live (non-sentinel) `AnticipatedThermalState` slots'
/// `delivery_date` anchors into `[month_start, month_end)` spans, via
/// [`decode_month_anchor`] — the per-pool candidate set
/// [`resolve_boundary_source_stage`] matches against the current study's
/// terminal delivery window.
fn decode_pool_anticipated_months(
    manifest: &[EntitySlot],
) -> Result<Vec<(NaiveDate, NaiveDate)>, SddpError> {
    manifest
        .iter()
        .filter(|slot| {
            slot.entity_type == StateFamily::AnticipatedThermalState.code()
                && slot.delivery_date != ENTITY_SLOT_DELIVERY_DATE_SENTINEL
        })
        .map(|slot| decode_month_anchor(slot.delivery_date).map(|(start, end, _)| (start, end)))
        .collect()
}

/// Boundary cut records that passed [`validate_policy_load`]'s
/// [`BoundaryInjection`] check matrix. The private fields mean the only way to
/// obtain one is [`load_boundary_cuts`], so [`inject_boundary_cuts`] cannot
/// compile against a bare, unvalidated `Vec<OwnedPolicyCutRecord>`. Derefs to
/// `[OwnedPolicyCutRecord]` for read access; carries the load's
/// [`BoundaryReconciliationReport`], read via [`Self::report`].
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
/// stage as boundary-loaded (`terminal_has_boundary_cuts`) and skip theta zeroing.
///
/// `boundary_cuts` must come from [`load_boundary_cuts`] — its private field
/// means a bare `Vec<OwnedPolicyCutRecord>`/slice cannot substitute, so an
/// unvalidated boundary load cannot compile.
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
pub fn inject_boundary_cuts(setup: &mut StudySetup, boundary_cuts: &ValidatedBoundaryCuts) {
    let fcf = &mut setup.fcf;
    let terminal_idx = fcf.pools.len() - 1;
    let state_dimension = fcf.state_dimension;
    let forward_passes = fcf.forward_passes;
    // The terminal pool never receives a new cut (a leaf generates none), so
    // `max_iterations = 0` reserves no growable training capacity.
    fcf.pools[terminal_idx] =
        CutPool::new_with_warm_start(state_dimension, forward_passes, 0, boundary_cuts);
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{AnticipatedCommitmentHistory, EntityId};
    use cobre_io::{
        EntitySlot, GraphManifest, ManifestEdge, ManifestNode, ProducerBlock, StageCutsPayload,
    };

    use super::{
        BoundaryInjection, BoundaryReconciliationReport, BoundaryStateRequirements, CutPool,
        FullFcf, NodeId, NodePos, PolicyStageManifest, TypedVec, ValidatedBoundaryCuts,
        boundary_policy_required_lag_depth, compare_manifest_slot_identity, decode_month_anchor,
        inject_boundary_cuts, load_boundary_cuts, overlap_hours, resolve_boundary_source_stage,
        validate_policy_load,
    };
    use crate::SddpError;
    use crate::test_support;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// All-`None` delivery intervals aligned to `len` — for
    /// [`load_boundary_cuts`] tests whose manifests carry no live-dated
    /// anticipated slot.
    fn no_intervals(len: usize) -> Vec<Option<(NaiveDate, NaiveDate)>> {
        vec![None; len]
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
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 50,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        }
    }

    /// A 1:1 chain graph manifest over `n_stages` nodes (node id == stage id ==
    /// pool id) — the shape a chain-degenerate study writes.
    fn chain_manifest(n_stages: u32) -> GraphManifest {
        let nodes = (0..n_stages)
            .map(|t| ManifestNode {
                id: i32::try_from(t).unwrap(),
                stage_id: i32::try_from(t).unwrap(),
                pool_id: t,
            })
            .collect();
        let edges = (0..n_stages.saturating_sub(1))
            .map(|t| ManifestEdge {
                source_id: i32::try_from(t).unwrap(),
                target_id: i32::try_from(t + 1).unwrap(),
                probability: 1.0,
            })
            .collect();
        GraphManifest {
            n_pools: n_stages,
            nodes,
            edges,
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
            })
            .collect();

        let metadata =
            test_support::checkpoint_metadata(n_stages, chain_manifest(n_stages), producer_block());

        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
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
        };
        let metadata = test_support::checkpoint_metadata(
            stage_id + 1,
            chain_manifest(stage_id + 1),
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

            let cuts = load_boundary_cuts(
                tmp.path(),
                0,
                2,
                &[],
                &[],
                &[],
                None,
                loading_factor,
                &mut ignore_warnings(),
            )
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

        let cuts = load_boundary_cuts(
            tmp.path(),
            2,
            10,
            &[],
            &[],
            &[],
            None,
            NEUTRAL_LOADING_FACTOR,
            &mut ignore_warnings(),
        )
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

    /// Given a checkpoint without stage 99, when `load_boundary_cuts` is called
    /// for stage 99, then it returns `Err(SddpError::Validation)` with a message
    /// containing `"source_stage"` and `"99"`.
    #[test]
    fn load_boundary_cuts_missing_stage_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(
            tmp.path(),
            99,
            10,
            &[],
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(result.is_err(), "should fail for missing stage");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("source_stage"),
            "error should mention 'source_stage': {msg}"
        );
        assert!(
            msg.contains("99"),
            "error should include the missing stage index: {msg}"
        );
    }

    /// Given a checkpoint with `state_dimension=10`, when `load_boundary_cuts` is
    /// called with `current_state_dimension=5`, then it returns
    /// `Err(SddpError::Validation)` with a message containing `"state_dimension"`.
    #[test]
    fn load_boundary_cuts_state_dimension_mismatch_returns_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_checkpoint(tmp.path(), 5, 10, &[1.0]);

        let result = load_boundary_cuts(
            tmp.path(),
            0,
            5,
            &[],
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
        let result = load_boundary_cuts(
            std::path::Path::new("/nonexistent/path/to/policy"),
            0,
            10,
            &[],
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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

    /// A single active `HydroStorage` slot (`entity_type 0`, `subindex 0`).
    fn storage_slot(id: i32) -> EntitySlot {
        EntitySlot {
            entity_type: 0,
            entity_id: id,
            subindex: 0,
            was_active: true,
            delivery_date: cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    /// A single active `HydroTransitBucket` slot (`entity_type 3`): `id` is the
    /// downstream hydro, `lag` the maturity subindex.
    fn transit_bucket_slot(id: i32, lag: u32) -> EntitySlot {
        EntitySlot {
            entity_type: 3,
            entity_id: id,
            subindex: lag,
            was_active: true,
            delivery_date: cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
    }

    /// A single active `HydroInflowLag` slot (`entity_type 1`): `id` is the
    /// hydro, `lag_depth` the 1-based lag (as `build_stage_entity_manifest` emits).
    fn inflow_lag_slot(id: i32, lag_depth: u32) -> EntitySlot {
        EntitySlot {
            entity_type: 1,
            entity_id: id,
            subindex: lag_depth,
            was_active: true,
            delivery_date: cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
        }
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
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            Some(6),
            1_000_000.0,
            &mut ignore_warnings(),
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
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            Some(12),
            1_000_000.0,
            &mut ignore_warnings(),
        )
        .unwrap();

        assert_eq!(
            cuts.len(),
            2,
            "a boundary within the declared depth must load"
        );
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
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching manifest must load all cuts");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot match must emit no warning: {warnings:?}"
        );
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
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
        let result = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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

    /// Given a boundary checkpoint with an empty manifest and a matching
    /// `current_state_dimension`, `load_boundary_cuts` returns `Ok` (no hard fail on
    /// absence) and surfaces an "identity could not be verified" warning.
    #[test]
    fn load_boundary_cuts_absent_manifest_loads_with_warning() {
        let tmp = tempfile::tempdir().unwrap();
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &[]);

        let current = storage_manifest(1, 2);
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
        .unwrap();

        assert_eq!(cuts.len(), 2, "absent manifest must still load cuts");
        assert_eq!(warnings.len(), 1, "absence must surface one warning");
        assert!(
            warnings[0].contains("manifest absent"),
            "warning must flag the absent manifest: {}",
            warnings[0]
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
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
        .unwrap();

        assert_eq!(cuts.len(), 2, "was_active divergence must still load cuts");
        assert!(
            warnings.is_empty(),
            "a dormant-to-active transition is expected, not warned: {warnings:?}"
        );
    }

    /// A manifest carrying a `HydroTransitBucket` slot (`entity_type 3`, the
    /// downstream hydro id, the maturity lag as `subindex`) round-trips: written to a
    /// checkpoint and reloaded against a slot-for-slot matching current manifest,
    /// `load_boundary_cuts` returns `Ok` with no warning — the bucket slot passes
    /// `slot_identity`.
    #[test]
    fn load_boundary_cuts_matching_transit_bucket_manifest_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &manifest);

        let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            1_000_000.0,
            &mut |m| {
                warnings.push(m.to_string());
            },
        )
        .unwrap();

        assert_eq!(cuts.len(), 2, "matching bucket manifest must load all cuts");
        assert!(
            warnings.is_empty(),
            "a slot-for-slot bucket match must emit no warning: {warnings:?}"
        );
    }

    /// A policy exported WITHOUT travel-time buckets loaded by bucket-aware
    /// code whose terminal manifest has a `HydroTransitBucket` (type 3) slot
    /// at the same `state_dimension` succeeds: a target transit bucket with no
    /// source match defaults to `0.0` (distinct from storage/lag's
    /// reject-on-miss; a matching source transit slot is instead copied). The
    /// storage slot still loads its identity-matched coefficient.
    #[test]
    fn load_boundary_cuts_missing_transit_bucket_slot_identity_defaults_to_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let boundary = storage_manifest(1, 2);
        write_checkpoint_with_manifest(tmp.path(), 5, 2, &[10.0, 20.0], &boundary);

        let current = vec![storage_slot(1), transit_bucket_slot(2, 1)];
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            NEUTRAL_LOADING_FACTOR,
            &mut ignore_warnings(),
        )
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
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            3,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            NEUTRAL_LOADING_FACTOR,
            &mut ignore_warnings(),
        )
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

    // ── load_boundary_cuts graph_stage_id resolution (pool id != stage id) ────

    /// Write a checkpoint with one pool per `(pool_id, graph_stage_id,
    /// intercepts)` entry, each pool's own `graph_stage_id` self-describing
    /// fact set directly (never through `metadata.graph_manifest`) — so
    /// `load_boundary_cuts` must resolve `source_stage` to a pool through that
    /// fact alone, not by a stage == pool coincidence.
    fn write_pools_with_graph_stage_ids(dir: &std::path::Path, pools: &[(u32, i32, &[f64])]) {
        let coefficients = [1.0_f64, 1.0];
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
                |(((pool_id, graph_stage_id, intercepts), cuts), active)| StageCutsPayload {
                    stage_id: *pool_id,
                    state_dimension: 2,
                    capacity: intercepts.len() as u32,
                    warm_start_count: 0,
                    cuts,
                    active_cut_indices: active,
                    populated_count: intercepts.len() as u32,
                    entity_manifest: &[],
                    cost_scale_factor: 1_000_000.0,
                    node_id: i32::try_from(*pool_id).unwrap_or(-1),
                    graph_stage_id: *graph_stage_id,
                },
            )
            .collect();
        let metadata =
            test_support::checkpoint_metadata(6, GraphManifest::default(), producer_block());
        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// A `source_stage` matching more than one pool's own `graph_stage_id` is
    /// rejected: the frozen `policy.boundary` config offers no pool selector,
    /// so an ambiguous source is a named `SddpError::Validation` (no remedy
    /// field).
    #[test]
    fn load_boundary_cuts_multi_pool_source_stage_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        // Two distinct pools (2 and 5) both self-describe graph_stage_id 3.
        write_pools_with_graph_stage_ids(tmp.path(), &[(2, 3, &[10.0, 20.0]), (5, 3, &[99.0])]);

        let result = load_boundary_cuts(
            tmp.path(),
            3,
            2,
            &[],
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(result.is_err(), "a multi-pool source_stage must reject");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("multi-node"),
            "rejection must name the stage as multi-node: {msg}"
        );
        assert!(
            msg.contains("source_stage 3"),
            "rejection must name the stage: {msg}"
        );
    }

    /// A `source_stage` resolves through a pool's own `graph_stage_id`, even
    /// when the pool id differs from the stage id (pool 2 self-describes
    /// stage 5): the cuts loaded are pool 2's.
    #[test]
    fn load_boundary_cuts_single_pool_resolves_through_graph_stage_id() {
        let tmp = tempfile::tempdir().unwrap();
        // Pool 2 self-describes graph_stage_id 5 (pool id != stage id).
        write_pools_with_graph_stage_ids(tmp.path(), &[(2, 5, &[10.0, 20.0, 30.0])]);

        let cuts = load_boundary_cuts(
            tmp.path(),
            5,
            2,
            &[],
            &[],
            &[],
            None,
            NEUTRAL_LOADING_FACTOR,
            &mut ignore_warnings(),
        )
        .expect("a source_stage must resolve through the pool's own graph_stage_id");

        assert_eq!(
            cuts.len(),
            3,
            "must load all of pool 2's cuts, resolved from stage 5 via graph_stage_id"
        );
        let intercepts: Vec<f64> = cuts.iter().map(|c| c.intercept).collect();
        assert_eq!(intercepts, vec![10.0, 20.0, 30.0]);
    }

    // ── resolve_boundary_source_stage tests ───────────────────────────────────

    fn ymd(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("valid calendar date")
    }

    /// A single `AnticipatedThermalState` slot (`entity_type 2`), dated at
    /// `delivery_date` (`YYYYMM01`).
    fn anticipated_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
        EntitySlot {
            entity_type: 2,
            entity_id: thermal_id,
            subindex: ring_slot,
            was_active: true,
            delivery_date,
        }
    }

    /// Write a checkpoint with one pool per `manifests` entry (pool id ==
    /// index == `graph_stage_id`, a chain-degenerate graph), each pool holding
    /// one cut sized to its own manifest — for [`resolve_boundary_source_stage`]
    /// tests exercising distinct per-pool anticipated calendars.
    fn write_checkpoint_with_pool_manifests(dir: &std::path::Path, manifests: &[Vec<EntitySlot>]) {
        let n_pools = manifests.len() as u32;
        let coefficients_per_pool: Vec<Vec<f64>> =
            manifests.iter().map(|m| vec![1.0_f64; m.len()]).collect();
        let cuts_per_pool: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = coefficients_per_pool
            .iter()
            .map(|coeffs| {
                vec![cobre_io::PolicyCutRecord {
                    cut_id: 0,
                    slot_index: 0,
                    iteration: 0,
                    forward_pass_index: 0,
                    intercept: 0.0,
                    coefficients: coeffs,
                    is_active: true,
                }]
            })
            .collect();
        let active_indices = [0_u32];
        let payloads: Vec<StageCutsPayload<'_>> = manifests
            .iter()
            .enumerate()
            .map(|(pool, manifest)| StageCutsPayload {
                stage_id: pool as u32,
                state_dimension: manifest.len() as u32,
                capacity: 1,
                warm_start_count: 0,
                cuts: &cuts_per_pool[pool],
                active_cut_indices: &active_indices,
                populated_count: 1,
                entity_manifest: manifest,
                cost_scale_factor: 1_000_000.0,
                node_id: pool as i32,
                graph_stage_id: pool as i32,
            })
            .collect();
        let metadata =
            test_support::checkpoint_metadata(n_pools, chain_manifest(n_pools), producer_block());
        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// A checkpoint whose pool 1 alone carries an anticipated slot dated
    /// inside the current terminal window resolves to pool 1.
    #[test]
    fn resolve_boundary_source_stage_unique_match_returns_pool_index() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![
            vec![storage_slot(1)],
            vec![anticipated_slot(9, 0, 20_260_301)],
        ];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![None, Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let resolved = resolve_boundary_source_stage(tmp.path(), &target)
            .expect("a unique calendar match must resolve");

        assert_eq!(resolved, 1, "pool 1 carries the matching anticipated month");
    }

    /// Two pools both carrying an anticipated slot dated inside the current
    /// terminal window are ambiguous: rejects naming the candidate count and
    /// advising an explicit `source_stage`.
    #[test]
    fn resolve_boundary_source_stage_ambiguous_multiple_pools_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![
            vec![anticipated_slot(9, 0, 20_260_301)],
            vec![anticipated_slot(9, 0, 20_260_301)],
        ];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let err = resolve_boundary_source_stage(tmp.path(), &target).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("ambiguous"), "must name the ambiguity: {msg}");
        assert!(msg.contains('2'), "must name the candidate count 2: {msg}");
        assert!(
            msg.contains("policy.boundary.source_stage"),
            "must advise the explicit override: {msg}"
        );
    }

    /// A source with no anticipated slot at all, on any pool, has no
    /// decodable delivery date to match against: rejects with the
    /// re-export hint.
    #[test]
    fn resolve_boundary_source_stage_fully_sentinel_dated_source_rejects_with_reexport_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![vec![storage_slot(1)], vec![storage_slot(1)]];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let err = resolve_boundary_source_stage(tmp.path(), &target).unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("re-export"),
            "must advise re-exporting the boundary: {msg}"
        );
        assert!(
            msg.contains("policy.boundary.source_stage"),
            "must advise the explicit override: {msg}"
        );
    }

    /// A source pool DOES carry a decodable anticipated month, but it falls
    /// outside the current terminal window: rejects advising an explicit
    /// `source_stage`, distinct from both the ambiguous and the
    /// sentinel-dated fallback.
    #[test]
    fn resolve_boundary_source_stage_no_overlap_among_decodable_pools_rejects() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![vec![anticipated_slot(9, 0, 20_260_301)]];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target = vec![Some((ymd(2026, 6, 1), ymd(2026, 7, 1)))];
        let err = resolve_boundary_source_stage(tmp.path(), &target).unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("policy.boundary.source_stage"),
            "must advise the explicit override: {msg}"
        );
        assert!(
            !msg.contains("ambiguous") && !msg.contains("re-export"),
            "a decodable-but-non-overlapping source is neither ambiguous nor sentinel-dated: {msg}"
        );
    }

    /// A non-existent boundary path surfaces the same IO-failure message
    /// [`load_boundary_cuts`] reports.
    #[test]
    fn resolve_boundary_source_stage_nonexistent_path_returns_error() {
        let err =
            resolve_boundary_source_stage(std::path::Path::new("/nonexistent/path/to/policy"), &[])
                .unwrap_err();

        let msg = err.to_string();
        assert!(
            msg.contains("failed to read boundary policy checkpoint"),
            "error should describe the IO failure: {msg}"
        );
    }

    /// The resolver only PICKS a candidate by calendar: a pool that matches
    /// the terminal date but belongs to a storage-identity-incompatible
    /// source still rejects once `load_boundary_cuts` reconciles it — the
    /// resolver is not a new trust boundary.
    #[test]
    fn resolve_boundary_source_stage_wrong_system_still_rejected_by_load_boundary_cuts_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let manifests = vec![vec![storage_slot(7), anticipated_slot(9, 0, 20_260_301)]];
        write_checkpoint_with_pool_manifests(tmp.path(), &manifests);

        let target_intervals = vec![None, Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let resolved = resolve_boundary_source_stage(tmp.path(), &target_intervals)
            .expect("a unique calendar match must resolve, even from an incompatible system");
        assert_eq!(resolved, 0);

        let current = vec![storage_slot(42), anticipated_slot(9, 100, 20_260_301)];
        let result = load_boundary_cuts(
            tmp.path(),
            resolved,
            current.len() as u32,
            &current,
            &target_intervals,
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

        assert!(
            result.is_err(),
            "a calendar-matched but identity-incompatible source must still reject"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("42"),
            "the storage/lag identity reject must name the unpriced hydro 42: {msg}"
        );
    }

    /// Write a checkpoint from explicit `(pool_id, graph_stage_id,
    /// entity_manifest, intercept)` pools — each pool's own `graph_stage_id`
    /// self-describing fact set directly, for tests where a pool id must
    /// diverge from its own `graph_stage_id`.
    fn write_checkpoint_with_explicit_graph_stage_ids(
        dir: &std::path::Path,
        pools: &[(u32, i32, Vec<EntitySlot>, f64)],
    ) {
        let coefficients_per_pool: Vec<Vec<f64>> = pools
            .iter()
            .map(|(_, _, manifest, _)| vec![1.0_f64; manifest.len()])
            .collect();
        let cuts_per_pool: Vec<Vec<cobre_io::PolicyCutRecord<'_>>> = pools
            .iter()
            .zip(&coefficients_per_pool)
            .map(|((_, _, _, intercept), coeffs)| {
                vec![cobre_io::PolicyCutRecord {
                    cut_id: 0,
                    slot_index: 0,
                    iteration: 0,
                    forward_pass_index: 0,
                    intercept: *intercept,
                    coefficients: coeffs,
                    is_active: true,
                }]
            })
            .collect();
        let active_indices = [0_u32];
        let payloads: Vec<StageCutsPayload<'_>> = pools
            .iter()
            .zip(&cuts_per_pool)
            .map(
                |((pool_id, graph_stage_id, manifest, _), cuts)| StageCutsPayload {
                    stage_id: *pool_id,
                    state_dimension: manifest.len() as u32,
                    capacity: 1,
                    warm_start_count: 0,
                    cuts,
                    active_cut_indices: &active_indices,
                    populated_count: 1,
                    entity_manifest: manifest,
                    cost_scale_factor: 1_000_000.0,
                    node_id: i32::try_from(*pool_id).unwrap_or(-1),
                    graph_stage_id: *graph_stage_id,
                },
            )
            .collect();
        let metadata =
            test_support::checkpoint_metadata(8, GraphManifest::default(), producer_block());
        cobre_io::write_policy_checkpoint(dir, &payloads, &[], &metadata, &[]).unwrap();
    }

    /// The reproduced defect: the resolver's winning POOL id (2) numerically
    /// coincides with an unrelated DECOY pool's own raw id (7), and the decoy
    /// pool's `graph_stage_id` (99) carries different cuts. Returning the raw
    /// pool id (the pre-fix behavior) would make `load_boundary_cuts` resolve
    /// `source_stage` through the decoy pool's numeric id instead, silently
    /// loading the decoy's cuts (999.0) rather than the calendar-winning
    /// pool's (777.0). The fix maps the winning pool to its OWN
    /// `graph_stage_id` (7 — distinct from every pool id and from the decoy's
    /// `graph_stage_id`), so `load_boundary_cuts` follows the correct pool.
    #[test]
    fn resolve_boundary_source_stage_maps_winning_pool_to_its_owning_node_stage_id() {
        let tmp = tempfile::tempdir().unwrap();
        let winning_pool: u32 = 2;
        let winning_graph_stage_id: i32 = 7;
        let decoy_pool: u32 = 7; // numerically equals winning_graph_stage_id
        let decoy_graph_stage_id: i32 = 99;

        let winning_manifest = vec![anticipated_slot(9, 0, 20_260_301)];
        let decoy_manifest = vec![storage_slot(1)];

        write_checkpoint_with_explicit_graph_stage_ids(
            tmp.path(),
            &[
                (
                    winning_pool,
                    winning_graph_stage_id,
                    winning_manifest,
                    777.0,
                ),
                (decoy_pool, decoy_graph_stage_id, decoy_manifest, 999.0),
            ],
        );

        let target = vec![Some((ymd(2026, 3, 1), ymd(2026, 4, 1)))];
        let resolved = resolve_boundary_source_stage(tmp.path(), &target)
            .expect("a unique calendar match must resolve");

        assert_eq!(
            resolved,
            u32::try_from(winning_graph_stage_id).unwrap(),
            "must return the winning pool's own graph_stage_id (7), not the raw pool id (2)"
        );

        let cuts = load_boundary_cuts(
            tmp.path(),
            resolved,
            1,
            &[],
            &no_intervals(1),
            &[],
            None,
            NEUTRAL_LOADING_FACTOR,
            &mut ignore_warnings(),
        )
        .expect("the resolved stage id must thread correctly through load_boundary_cuts");

        assert_eq!(
            cuts.len(),
            1,
            "must load the winning pool's one cut, not the decoy's"
        );
        assert_eq!(
            cuts[0].intercept, 777.0,
            "must load the CORRECT pool's cut (777.0), never the decoy's (999.0)"
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

    /// Identical `state_dimension`, `num_stages`, and slot-for-slot matching
    /// manifests pass `FullFcf` with no warnings.
    #[test]
    fn validate_policy_load_full_fcf_identical_oks_without_warning() {
        let slots = storage_manifest(1, 2);
        let source = psm(2, 12, &slots);
        let current = psm(2, 12, &slots);

        let report = validate_policy_load::<FullFcf>(&source, &current).unwrap();

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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let result = validate_policy_load::<BoundaryInjection>(&source, &current);

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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let full_fcf_result = validate_policy_load::<FullFcf>(&source, &current);
        assert!(
            full_fcf_result.is_err(),
            "num_stages mismatch must reject FullFcf"
        );
        let msg = full_fcf_result.unwrap_err().to_string();
        assert!(msg.contains("num_stages"), "{msg}");
        assert!(msg.contains("12"), "should include source value: {msg}");
        assert!(msg.contains("24"), "should include current value: {msg}");

        let boundary_result = validate_policy_load::<BoundaryInjection>(&source, &current);
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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let result = validate_policy_load::<FullFcf>(&source, &current);

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

        let result = validate_policy_load::<BoundaryInjection>(&source, &current);

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
    /// up) reconciles by identity (storage) and by date (the anticipated slots
    /// fan onto the current ring by calendar overlap) and loads, so boundary
    /// injection across the version boundary is unaffected.
    #[test]
    fn lane_era_wider_checkpoint_rejects_full_fcf_but_boundary_injection_reconciles() {
        let march = 20_260_301;
        let april = 20_260_401;
        let lane_era = vec![
            storage_slot(1),
            anticipated_slot(9, 0, march),
            anticipated_slot(9, 1, april),
        ];

        let current_narrow = vec![storage_slot(1), anticipated_slot(9, 0, march)];
        let full_fcf = validate_policy_load::<FullFcf>(
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
            anticipated_slot(9, 0, april),
            storage_slot(1),
            anticipated_slot(9, 1, march),
        ];
        let target_intervals = vec![
            Some((ymd(2026, 4, 1), ymd(2026, 5, 1))),
            None,
            Some((ymd(2026, 3, 1), ymd(2026, 4, 1))),
        ];

        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            current_reordered.len() as u32,
            &current_reordered,
            &target_intervals,
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        )
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

        let report = validate_policy_load::<FullFcf>(&source, &current).unwrap();

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

        let result = load_boundary_cuts(
            tmp.path(),
            0,
            3,
            &[],
            &[],
            &[],
            None,
            1_000_000.0,
            &mut ignore_warnings(),
        );

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
    /// decode via `build_basis_cache_from_checkpoint`.
    #[test]
    fn basis_status_round_trips_through_export_and_load_for_every_variant() {
        use cobre_io::{PolicyBasisRecord, deserialize_stage_basis, serialize_stage_basis};
        use cobre_solver::{Basis, BasisStatus};

        use super::build_basis_cache_from_checkpoint;
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

        let cache = build_basis_cache_from_checkpoint(
            std::slice::from_ref(&owned),
            &[],
            &vec![NodeId(0)].into(),
            &vec![0].into(),
        );
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

        use super::build_basis_cache_from_checkpoint;

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

        let cache = build_basis_cache_from_checkpoint(
            std::slice::from_ref(&owned),
            &[],
            &vec![NodeId(0)].into(),
            &vec![0].into(),
        );
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
        use super::build_basis_cache_from_checkpoint;

        let node_ids: TypedVec<NodePos, NodeId> = vec![10, 11, 12, 13, 14, 15, 16]
            .into_iter()
            .map(NodeId)
            .collect();
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2, 3, 3, 3, 3].into();
        let stage_cuts = vec![
            pool_cuts(0, &[0]),
            pool_cuts(1, &[1]),
            pool_cuts(2, &[2]),
            pool_cuts(3, &[5, 7]),
        ];
        let stage_bases = vec![
            node_basis(0, 1),
            node_basis(1, 1),
            node_basis(2, 1),
            node_basis(3, 2),
            node_basis(4, 2),
            node_basis(5, 2),
            node_basis(6, 2),
        ];

        let cache =
            build_basis_cache_from_checkpoint(&stage_bases, &stage_cuts, &node_ids, &node_pools);

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

        // Interior node 2 (pool 2) recovers pool 2's single active slot.
        assert_eq!(cache[2].as_ref().unwrap().cut_row_slots, vec![2_u32]);
        // Every leaf (nodes 3..=6) recovers the SHARED pool 3's active slots,
        // keyed by node_pools[node] == 3 — a node-ordinal key would drop 4/5/6.
        for (node, slot) in cache.iter().enumerate().skip(3) {
            assert_eq!(
                slot.as_ref().unwrap().cut_row_slots,
                vec![5_u32, 7_u32],
                "leaf node {node} must recover shared pool 3's active slots"
            );
        }
    }

    /// Chain degeneracy: `node_pools` is the identity (`node_pools[t] == t`), so
    /// node-keyed sizing and pool-keyed cut matching reduce exactly to the
    /// pre-branching per-stage behavior — one basis per stage, keyed by ordinal.
    #[test]
    fn build_basis_cache_from_checkpoint_chain_is_identity_keyed() {
        use super::build_basis_cache_from_checkpoint;

        let node_ids: TypedVec<NodePos, NodeId> = vec![0, 1, 2].into_iter().map(NodeId).collect();
        let node_pools: TypedVec<NodePos, usize> = vec![0, 1, 2].into();
        let stage_cuts = vec![pool_cuts(0, &[0]), pool_cuts(1, &[3]), pool_cuts(2, &[9])];
        let stage_bases = vec![node_basis(0, 1), node_basis(1, 1), node_basis(2, 1)];

        let cache =
            build_basis_cache_from_checkpoint(&stage_bases, &stage_cuts, &node_ids, &node_pools);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache[0].as_ref().unwrap().cut_row_slots, vec![0_u32]);
        assert_eq!(cache[1].as_ref().unwrap().cut_row_slots, vec![3_u32]);
        assert_eq!(cache[2].as_ref().unwrap().cut_row_slots, vec![9_u32]);
    }

    // ── inject_boundary_cuts tests ──────────────────────────────────────────────

    #[test]
    fn inject_boundary_cuts_produces_fixed_capacity_terminal_pool() {
        let mut setup = test_support::oracle_chain_setup(10);
        let terminal_idx = setup.fcf.pools.len() - 1;
        let state_dimension = setup.fcf.state_dimension;
        let records = vec![
            owned_cut(5.0, vec![1.0; state_dimension]),
            owned_cut(6.0, vec![2.0; state_dimension]),
        ];
        let boundary_cuts = ValidatedBoundaryCuts {
            records: records.clone(),
            report: BoundaryReconciliationReport::default(),
        };

        inject_boundary_cuts(&mut setup, &boundary_cuts);

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
        let terminal_idx = setup.fcf.pools.len() - 1;
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

        inject_boundary_cuts(&mut setup, &boundary_cuts);

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
        inject_boundary_cuts(&mut setup_direct, &direct);
        inject_boundary_cuts(&mut setup_bcast, &broadcast);

        let terminal_idx = setup_direct.fcf.pools.len() - 1;
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
        };
        let metadata = test_support::checkpoint_metadata(
            1,
            chain_manifest(1),
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
        let manifest = vec![storage_slot(1), anticipated_slot(9, 0, 20_260_301)];
        let s = 1_000_000.0;
        write_marked_checkpoint_with_manifest(
            tmp.path(),
            raw_intercept,
            &raw_coefficients,
            &manifest,
            s,
        );

        let current = vec![storage_slot(1), anticipated_slot(9, 0, 20_260_301)];
        let value_mw = 50.0;
        let windows = vec![fixed_window(9, ymd(2026, 3, 1), ymd(2026, 4, 1), value_mw)];

        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &windows,
            None,
            s,
            &mut ignore_warnings(),
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
        let manifest = vec![storage_slot(1), anticipated_slot(9, 0, 20_260_301)];
        let s = 1_000_000.0;
        write_marked_checkpoint_with_manifest(
            tmp.path(),
            raw_intercept,
            &raw_coefficients,
            &manifest,
            s,
        );

        let current = vec![storage_slot(1), anticipated_slot(9, 0, 20_260_301)];
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[],
            None,
            s,
            &mut ignore_warnings(),
        )
        .unwrap();

        assert_eq!(
            cuts[0].intercept.to_bits(),
            (raw_intercept / s).to_bits(),
            "an empty fold must leave the intercept bit-identical to the marked-rescale-only load"
        );
    }

    /// An empty (unverifiable) source manifest with a non-empty `fixed_windows`
    /// applies no fold — the fold shares the rebind's `verifiable` gate — and the
    /// hoisted `manifest_identity_verifiable` call fires the absent-manifest
    /// warning exactly once (a second call would double-fire it). The loaded
    /// intercept is the plain marked rescale (`raw / loading_factor`) with no
    /// fold term added, isolating the fold-skip from the mandatory rescale.
    #[test]
    fn load_boundary_cuts_unverifiable_manifest_skips_fold_and_warns_once() {
        let tmp = tempfile::tempdir().unwrap();
        let raw_intercept = 42.0;
        write_checkpoint_with_manifest(tmp.path(), 1, 2, &[raw_intercept], &[]);

        let current = storage_manifest(1, 2);
        let windows = vec![fixed_window(9, ymd(2026, 3, 1), ymd(2026, 4, 1), 50.0)];
        let loading_cost_scale_factor = 1_000_000.0;
        let mut warnings: Vec<String> = Vec::new();
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &windows,
            None,
            loading_cost_scale_factor,
            &mut |m| warnings.push(m.to_string()),
        )
        .unwrap();

        assert_eq!(
            cuts[0].intercept.to_bits(),
            (raw_intercept / loading_cost_scale_factor).to_bits(),
            "an unverifiable manifest must skip the fold, leaving the intercept equal to the \
             plain marked rescale with no fold term added"
        );
        assert_eq!(
            warnings.len(),
            1,
            "the absent-manifest warning must fire exactly once (single hoisted gate): {warnings:?}"
        );
    }

    // ── intercept-fold analytics (hand-computed deltas, both frames) ──────────

    /// The `overlap(window, M) / H_M` weight a window contributes at a source
    /// month anchor, via the SAME accessors `build_boundary_fold` uses — never a
    /// decimal literal, so a weight that drifted together with production would
    /// not be masked.
    fn fold_weight(window: (NaiveDate, NaiveDate), source_anchor: i32) -> f64 {
        let (month_start, month_end, h_m) = decode_month_anchor(source_anchor).unwrap();
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
        let manifest = vec![storage_slot(1), anticipated_slot(9, 0, anchor)];
        let c = 3.0;
        let raw_intercept = 200.0;
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(tmp.path(), raw_intercept, &[1.0, c], &manifest, s);

        let (ws, we) = (ymd(2026, 4, 1), ymd(2026, 5, 1)); // whole month → weight 1.0
        let v = 50.0;
        let current = manifest.clone();
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            2,
            &current,
            &no_intervals(current.len()),
            &[fixed_window(9, ws, we, v)],
            None,
            s,
            &mut ignore_warnings(),
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
        let manifest = vec![storage_slot(1), anticipated_slot(9, 0, anchor)];
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(tmp.path(), 1000.0, &[1.0, 3.0], &manifest, s);

        let current = manifest.clone();
        let load = |windows: &[AnticipatedCommitmentHistory]| {
            load_boundary_cuts(
                tmp.path(),
                0,
                2,
                &current,
                &no_intervals(current.len()),
                windows,
                None,
                s,
                &mut ignore_warnings(),
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
        let manifest = vec![storage_slot(1), anticipated_slot(9, 0, anchor)];
        let s = 2_000_000.0;
        write_marked_checkpoint_with_manifest(tmp.path(), 1000.0, &[1.0, 3.0], &manifest, s);

        let current = manifest.clone();
        let load = |windows: &[AnticipatedCommitmentHistory]| {
            load_boundary_cuts(
                tmp.path(),
                0,
                2,
                &current,
                &no_intervals(current.len()),
                windows,
                None,
                s,
                &mut ignore_warnings(),
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
            anticipated_slot(9, 0, march),
            anticipated_slot(9, 1, april),
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
            tmp.path(),
            0,
            3,
            &current,
            &no_intervals(current.len()),
            &[fixed_window(9, ws, we, v)],
            None,
            s,
            &mut ignore_warnings(),
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
