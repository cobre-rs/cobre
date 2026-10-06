//! Input and output record types for value-function artifact serialization.
//!
//! Input types (`PolicyCutRecord`, `PolicyBasisRecord`, `StageStatesPayload`,
//! `StageCutsPayload`) borrow from caller-owned buffers; owned output types
//! (`Owned*`, `*ReadResult`, `PolicyCheckpoint`) own their vectors. All use
//! generic names to maintain infrastructure crate genericity; conversion from
//! algorithm-specific types is the calling crate's responsibility. Field names
//! correspond to the tables in `schemas/policy.fbs`.

use chrono::{Datelike, NaiveDate};

use crate::output::SoftwareIdentity;

/// Current on-disk value-function artifact format version.
///
/// [`CheckpointManifest::format_version`] must equal this;
/// [`crate::read_policy_checkpoint`] rejects any other value — and absence —
/// with a named error before parsing any payload, so a pre-marker artifact is
/// cleanly rejected, never read positionally. Version 3 records
/// `StageBasis.num_cut_rows` as the basis's own trailing affine-piece row count.
pub const FORMAT_VERSION: u32 = 3;

/// Sentinel [`EntitySlot`] date-field value — [`EntitySlot::reference_date`],
/// [`EntitySlot::interval_start`], and [`EntitySlot::interval_end`] all
/// default to it — for a slot whose family does not populate that field; also
/// the value a reader yields when a field is absent from a buffer older than
/// that field's own id (see `schemas/policy.fbs` for each field's introducing
/// id).
pub const ENTITY_SLOT_DATE_SENTINEL: i32 = i32::MIN;

/// One per-slot entity-identity record for a state-vector dimension.
///
/// `entity_type` is the raw discriminant byte of the `EntityType` enum in
/// `schemas/policy.fbs`; [`EntitySlot::family`] reads it as the typed
/// [`StateFamily`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntitySlot {
    /// Raw [`StateFamily`] discriminant byte; see [`EntitySlot::family`].
    pub entity_type: u8,
    /// Owning entity's id; `int32` because a sentinel id can be `-1`.
    pub entity_id: i32,
    /// Secondary index within the owning entity (per-type meaning is the caller's).
    pub subindex: u32,
    /// Whether the owning entity was operationally active at this slot's stage.
    pub was_active: bool,
    /// `HydroInflowLag`'s reference past stage's `start_date`, `YYYYMMDD`
    /// encoded (`year * 10000 + month * 100 + day`);
    /// [`ENTITY_SLOT_DATE_SENTINEL`] for every other family.
    pub reference_date: i32,
    /// Half-open delivery/arrival interval's inclusive start, `YYYYMMDD`
    /// encoded: `HydroTransitBucket`'s `arrival_start` or
    /// `AnticipatedThermalState`'s `delivery_start`;
    /// [`ENTITY_SLOT_DATE_SENTINEL`] for storage and inflow-lag.
    pub interval_start: i32,
    /// Half-open delivery/arrival interval's exclusive end, paired with
    /// [`interval_start`](Self::interval_start);
    /// [`ENTITY_SLOT_DATE_SENTINEL`] for storage and inflow-lag.
    pub interval_end: i32,
}

impl EntitySlot {
    /// Builds a slot for `family` with every date field at
    /// [`ENTITY_SLOT_DATE_SENTINEL`] — the shared body of the four
    /// per-family constructors below.
    fn at_sentinel_dates(
        family: StateFamily,
        entity_id: i32,
        subindex: u32,
        was_active: bool,
    ) -> Self {
        Self {
            entity_type: family.code(),
            entity_id,
            subindex,
            was_active,
            reference_date: ENTITY_SLOT_DATE_SENTINEL,
            interval_start: ENTITY_SLOT_DATE_SENTINEL,
            interval_end: ENTITY_SLOT_DATE_SENTINEL,
        }
    }

    /// Builds a [`StateFamily::HydroStorage`] slot; `subindex` is always `0`.
    #[must_use]
    pub fn storage(entity_id: i32, was_active: bool) -> Self {
        Self::at_sentinel_dates(StateFamily::HydroStorage, entity_id, 0, was_active)
    }

    /// Builds a [`StateFamily::HydroInflowLag`] slot; `subindex` is the
    /// 1-based AR lag order.
    #[must_use]
    pub fn inflow_lag(entity_id: i32, lag_order: u32, was_active: bool) -> Self {
        Self::at_sentinel_dates(
            StateFamily::HydroInflowLag,
            entity_id,
            lag_order,
            was_active,
        )
    }

    /// Builds a [`StateFamily::HydroTransitBucket`] slot; `entity_id` is the
    /// downstream hydro, `subindex` the maturity lag.
    #[must_use]
    pub fn transit_bucket(downstream_entity_id: i32, maturity_lag: u32, was_active: bool) -> Self {
        Self::at_sentinel_dates(
            StateFamily::HydroTransitBucket,
            downstream_entity_id,
            maturity_lag,
            was_active,
        )
    }

    /// Builds a [`StateFamily::AnticipatedThermalState`] slot; `subindex` is
    /// the ring-buffer slot.
    #[must_use]
    pub fn anticipated(entity_id: i32, ring_slot: u32, was_active: bool) -> Self {
        Self::at_sentinel_dates(
            StateFamily::AnticipatedThermalState,
            entity_id,
            ring_slot,
            was_active,
        )
    }

    /// Returns `self` with `reference_date` replaced; every other field is
    /// unchanged.
    #[must_use]
    pub fn with_reference_date(self, reference_date: i32) -> Self {
        Self {
            reference_date,
            ..self
        }
    }

    /// Returns `self` with `interval_start` and `interval_end` replaced; every
    /// other field is unchanged.
    #[must_use]
    pub fn with_interval(self, start: i32, end: i32) -> Self {
        Self {
            interval_start: start,
            interval_end: end,
            ..self
        }
    }
}

/// State-vector dimension class of an [`EntitySlot`] — the typed Rust view of
/// the `EntityType` enum in `schemas/policy.fbs`. The wire representation stays
/// the raw [`EntitySlot::entity_type`] byte; this enum is the checked reading of
/// it, so the discriminants MUST match the schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StateFamily {
    /// Reservoir storage volume; see [`EntitySlot::storage`].
    HydroStorage = 0,
    /// Hydro inflow AR lag; see [`EntitySlot::inflow_lag`].
    HydroInflowLag = 1,
    /// Anticipated thermal commitment; see [`EntitySlot::anticipated`].
    AnticipatedThermalState = 2,
    /// Water in-transit bucket; see [`EntitySlot::transit_bucket`].
    HydroTransitBucket = 3,
}

impl StateFamily {
    /// The raw `EntityType` discriminant byte for this family.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The family for a raw `EntityType` byte, or `None` for a discriminant no
    /// `schemas/policy.fbs` `EntityType` variant defines.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::HydroStorage),
            1 => Some(Self::HydroInflowLag),
            2 => Some(Self::AnticipatedThermalState),
            3 => Some(Self::HydroTransitBucket),
            _ => None,
        }
    }
}

impl EntitySlot {
    /// The typed [`StateFamily`] of this slot, or `None` when [`Self::entity_type`]
    /// is a byte no `EntityType` variant defines.
    #[must_use]
    pub fn family(&self) -> Option<StateFamily> {
        StateFamily::from_code(self.entity_type)
    }
}

/// One affine-piece record for value-function artifact serialization.
///
/// `'a` borrows the coefficient slice without copying (vectors can be large).
#[derive(Debug, Clone)]
pub struct PolicyCutRecord<'a> {
    /// Unique identifier for this piece across all iterations.
    pub cut_id: u64,
    /// LP row position (required for artifact reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this piece.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed affine intercept.
    pub intercept: f64,
    /// Gradient coefficients, length must equal `state_dimension`.
    ///
    /// Positional only: index `i` is the i-th state-vector dimension, whose
    /// identity is carried by slot `i` of the co-located [`EntitySlot`] manifest
    /// (`entity_manifest`); no labels are stored inline.
    pub coefficients: &'a [f64],
    /// Whether this piece is currently active in the LP.
    pub is_active: bool,
}

/// One stage's solver basis for value-function artifact serialization.
#[derive(Debug, Clone)]
pub struct PolicyBasisRecord<'a> {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Training iteration that produced this basis.
    pub iteration: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: &'a [u8],
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: &'a [u8],
    /// Number of trailing rows in `row_status` that correspond to affine-piece rows.
    pub num_cut_rows: u32,
}

/// Sentinel [`StageStatesPayload::node_id`]/[`StageStatesReadResult::node_id`]
/// value for a policy-graph node identity absent from the write path (a
/// caller that never resolved one) or from a pre-`id:5` buffer
/// (forward-compatible default).
pub const STAGE_STATES_NODE_ID_SENTINEL: i32 = -1;

/// Sentinel [`StageCutsPayload::node_id`]/[`StageCutsReadResult::node_id`] value
/// for a pool with no single owning node (a shared pool, never a boundary
/// source) or a pre-`id:8` buffer (forward-compatible default).
pub const STAGE_CUTS_NODE_ID_SENTINEL: i32 = -1;

/// Sentinel [`StageCutsPayload::graph_stage_id`]/[`StageCutsReadResult::graph_stage_id`]
/// value for an unresolved owning-stage key or a pre-`id:8` buffer
/// (forward-compatible default).
pub const STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL: i32 = -1;

/// Sentinel [`StageCutsPayload::priced_state_date`]/[`StageCutsReadResult::priced_state_date`]
/// value for a not-yet-recorded priced instant or a pre-`id:11` buffer
/// (forward-compatible default). `i32::MIN` rather than `-1`, which is a
/// decodable value in the `YYYYMMDD` space this field encodes.
pub const STAGE_CUTS_PRICED_STATE_DATE_SENTINEL: i32 = i32::MIN;

/// Encodes `date` as `YYYYMMDD` (`year * 10000 + month * 100 + day`) — the wire
/// encoding [`EntitySlot`]'s and [`StageCutsPayload`]'s date fields document.
/// Total; [`decode_slot_date`] is its exact inverse.
#[must_use]
pub fn encode_slot_date(date: NaiveDate) -> i32 {
    date.year() * 10_000
        + i32::try_from(date.month()).unwrap_or(1) * 100
        + i32::try_from(date.day()).unwrap_or(1)
}

/// Decodes `value` from the `YYYYMMDD` encoding [`encode_slot_date`] produces —
/// its exact inverse. `None` for [`ENTITY_SLOT_DATE_SENTINEL`]/
/// [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] (`i32::MIN`) and for any other
/// value that is not a real calendar date; `-1` is deliberately rejected here
/// too, since it is a distinct non-date sentinel elsewhere in this module
/// (e.g. [`STAGE_CUTS_NODE_ID_SENTINEL`]) that must never be mistaken for one.
#[must_use]
pub fn decode_slot_date(value: i32) -> Option<NaiveDate> {
    let year = value / 10_000;
    let month = (value / 100) % 100;
    let day = value % 100;
    let (Ok(month), Ok(day)) = (u32::try_from(month), u32::try_from(day)) else {
        return None;
    };
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Payload for writing per-stage visited states to a value-function artifact.
///
/// The `data` slice contains the flat state vectors (row-major, each of length
/// `state_dimension`). The total number of stored states is `count`.
#[derive(Debug, Clone)]
pub struct StageStatesPayload<'a> {
    /// Study stage index (0-based).
    pub stage_id: u32,
    /// Policy-graph node identity (the declared node id on a branching graph;
    /// [`STAGE_STATES_NODE_ID_SENTINEL`] when absent). Distinct from
    /// `stage_id` the moment a graph carries more than one node per stage.
    pub node_id: i32,
    /// Length of each state vector.
    pub state_dimension: u32,
    /// Number of states stored.
    pub count: u32,
    /// Flat data buffer: `count * state_dimension` f64 elements.
    pub data: &'a [f64],
    /// Per-slot entity identity; length equals `state_dimension` when populated.
    /// An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
}

/// Per-pool affine-piece data payload for [`crate::write_policy_checkpoint`],
/// grouping the arguments of [`crate::serialize_stage_cuts`].
#[derive(Debug)]
pub struct StageCutsPayload<'a> {
    /// Pool id (0-based) — the storage-unit key naming this payload's file
    /// `cuts/<pool>.bin`. Equals the stage index on a chain.
    pub stage_id: u32,
    /// Number of state variables; determines coefficient vector length per piece.
    pub state_dimension: u32,
    /// Total preallocated affine-piece slots in the pool.
    pub capacity: u32,
    /// Number of slots `[0..warm_start_count)` loaded from a prior artifact.
    pub warm_start_count: u32,
    /// Slice of affine-piece records to serialize; length equals `populated_count`.
    pub cuts: &'a [PolicyCutRecord<'a>],
    /// Indices of pieces currently active in the LP.
    pub active_cut_indices: &'a [u32],
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Per-slot entity identity; length equals `state_dimension` when populated.
    /// An empty slice means no manifest is written.
    pub entity_manifest: &'a [EntitySlot],
    /// Objective cost-scale factor the writing study resolved; the provenance
    /// marker making each affine piece scale-independent at rest.
    pub cost_scale_factor: f64,
    /// Owning node's policy-graph id, or [`STAGE_CUTS_NODE_ID_SENTINEL`] for a
    /// shared pool.
    pub node_id: i32,
    /// Graph-stage id of the node(s) owning this pool — the boundary-resolution
    /// key; [`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL`] when unresolved.
    pub graph_stage_id: i32,
    /// Owning stage's `end_date`, encoded `year * 10000 + month * 100 + day`
    /// — the instant this pool's pieces price. [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`]
    /// when not recorded.
    pub priced_state_date: i32,
}

/// One node of the value-function artifact's graph manifest: its declared id,
/// the stage it sits at, and the pool whose payload carries its affine pieces.
///
/// `pool_id` **is** the node → pool map: a node references one pool, and a
/// reader resolves node `id`'s pieces as `pool_id`'s payload (leaf nodes sharing
/// a pool all name the same `pool_id`).
#[derive(Debug, Clone)]
pub struct ManifestNode {
    /// Declared node id.
    pub id: i32,
    /// Stage id this node sits at.
    pub stage_id: i32,
    /// Pool whose payload holds this node's affine pieces.
    pub pool_id: u32,
}

/// One directed edge of the graph manifest, with its transition probability.
#[derive(Debug, Clone)]
pub struct ManifestEdge {
    /// Source node id.
    pub source_id: i32,
    /// Target node id.
    pub target_id: i32,
    /// Transition probability `P(source -> target)`.
    pub probability: f64,
}

/// The graph manifest: the node list (each node carrying its own node → pool
/// assignment), the edge list, and the pool-set size.
///
/// This is the identity source the positional format never had — a reader
/// resolves a node's payload through it (node `n`'s pieces are `pool(n)`'s
/// payload), rather than trusting a filename.
#[derive(Debug, Clone, Default)]
pub struct GraphManifest {
    /// Number of distinct pools (the pool-set size).
    pub n_pools: u32,
    /// Every node, in canonical order, each with its stage and pool.
    pub nodes: Vec<ManifestNode>,
    /// Every directed edge with its transition probability.
    pub edges: Vec<ManifestEdge>,
}

/// [`SeasonManifest::cycle_code`] discriminant: a monthly season cycle.
pub const SEASON_CYCLE_CODE_MONTHLY: u8 = 0;
/// [`SeasonManifest::cycle_code`] discriminant: a weekly season cycle.
pub const SEASON_CYCLE_CODE_WEEKLY: u8 = 1;
/// [`SeasonManifest::cycle_code`] discriminant: a custom (neither monthly nor
/// weekly) season cycle.
pub const SEASON_CYCLE_CODE_CUSTOM: u8 = 2;
/// [`SeasonManifest::cycle_code`] discriminant: no season map declared — the
/// value [`SeasonManifest::default`] carries and a reader yields for a
/// pre-`id:19` buffer.
pub const SEASON_CYCLE_CODE_ABSENT: u8 = 255;

/// One hydro's per-season autoregressive order vector. Element type of
/// [`SeasonManifest::hydro_orders`].
#[derive(Debug, Clone)]
pub struct HydroSeasonOrders {
    /// Owning hydro's id.
    pub hydro_id: i32,
    /// Consecutive-lag AR order per dense season ordinal — a count, never a
    /// lag set (PAR order selection returns a scalar and the coefficient
    /// vector is dense). Length equals [`SeasonManifest::n_seasons`].
    pub orders: Vec<u32>,
}

/// Study-global season cycle and per-hydro PAR-order descriptor carried on
/// [`CheckpointManifest::season_manifest`]. The date-driven boundary gate
/// compares two studies' descriptors to reject a source whose season
/// definitions or per-season AR orders differ from the loading study's.
#[derive(Debug, Clone)]
pub struct SeasonManifest {
    /// Season cycle discriminant; one of the `SEASON_CYCLE_CODE_*`
    /// constants. A raw `u8`, never a `cobre-core` season type — the
    /// crate-genericity rule forbids the algorithm-specific dependency here.
    pub cycle_code: u8,
    /// Number of distinct seasons in the cycle; the length of every
    /// [`HydroSeasonOrders::orders`] vector.
    pub n_seasons: u32,
    /// Per-hydro AR order vectors, in canonical ascending `hydro_id` order.
    pub hydro_orders: Vec<HydroSeasonOrders>,
}

impl CheckpointManifest {
    /// The software this checkpoint records as its writer.
    #[must_use]
    pub fn written_by(&self) -> SoftwareIdentity<'_> {
        SoftwareIdentity {
            name: self.software.as_deref(),
            version: &self.software_version,
        }
    }
}

impl Default for SeasonManifest {
    /// The absent descriptor: [`SEASON_CYCLE_CODE_ABSENT`], zero seasons, no
    /// hydros — what a pre-`id:19` buffer decodes to.
    fn default() -> Self {
        Self {
            cycle_code: SEASON_CYCLE_CODE_ABSENT,
            n_seasons: 0,
            hydro_orders: Vec::new(),
        }
    }
}

/// Producer-namespaced metadata: everything specific to how the artifact was
/// produced (the training algorithm's own recorded state). Segregated from the
/// neutral core carried on [`CheckpointManifest`], whose doc states the
/// segregation rationale.
#[derive(Debug, Clone)]
pub struct ProducerBlock {
    /// Number of training iterations completed at write time.
    pub completed_iterations: u32,
    /// Lower bound value after the final completed iteration.
    pub final_lower_bound: f64,
    /// Last iteration's upper bound, if available (the final value, not a
    /// min-tracked best).
    pub best_upper_bound: Option<f64>,
    /// Maximum number of iterations configured for the run.
    pub max_iterations: u32,
    /// Number of forward passes per iteration.
    pub forward_passes: u32,
    /// Number of pieces loaded from a previous artifact at run start.
    pub warm_start_cuts: u32,
    /// Per-pool warm-start piece counts, in pool-id order.
    ///
    /// When non-empty, supersedes [`warm_start_cuts`] for per-pool accuracy.
    ///
    /// [`warm_start_cuts`]: Self::warm_start_cuts
    pub warm_start_counts: Vec<u32>,
    /// RNG seed used by the scenario sampler.
    ///
    /// Per-draw seeds are derived from `(rng_seed, iteration, scenario, stage)`,
    /// so resume needs only the seed — no accumulated RNG state is persisted.
    pub rng_seed: u64,
    /// Total visited states across all nodes.
    pub total_visited_states: u64,
    /// Block mode the artifact was trained under: the shared lowercase mode when
    /// every study stage agrees, else `"mixed"`.
    pub training_block_mode: String,
    /// Per-study-stage training block modes, in study-stage order.
    ///
    /// Populated only for mixed-mode studies.
    pub training_block_mode_per_stage: Vec<String>,
    /// Objective cost-scale factor the writing study resolved
    /// (`modeling.cost_scale_factor`) — the provenance marker that makes piece
    /// `coefficients`/`intercept` scale-independent at rest (canonical
    /// currency units, not the writer's internal scaled cost space).
    ///
    /// Absent when unmarked; a missing marker is interpreted as
    /// scaled-at-`1_000_000.0`, the constant every unmarked artifact was
    /// unconditionally written under.
    pub cost_scale_factor: Option<f64>,
    /// Lower bound after each recorded iteration, oldest first; the last entry
    /// is iteration [`completed_iterations`]. A writer records every completed
    /// iteration it has; an empty vector means none was recorded, and is what a
    /// buffer without the field reads as.
    ///
    /// [`completed_iterations`]: Self::completed_iterations
    pub lower_bound_history: Vec<f64>,
}

/// Study-global checkpoint metadata carried on the `FlatBuffers`
/// `CheckpointManifest` root at `manifest.bin`: the neutral core
/// (`format_version`, `software`, `software_version`, `created_at`,
/// `num_stages`, the [`GraphManifest`] descriptors) plus the namespaced
/// [`ProducerBlock`]. Read
/// first by [`crate::read_policy_checkpoint`], whose version gate rejects a
/// stale `format_version` before any payload is parsed.
///
/// The neutral core describes the artifact itself; the algorithm's own recorded
/// state lives under the namespaced [`producer`] block, so a reader that does
/// not know the producer reads the core from the core's own vocabulary.
///
/// [`producer`]: Self::producer
#[derive(Debug, Clone)]
pub struct CheckpointManifest {
    /// On-disk format version; must equal [`FORMAT_VERSION`] on read.
    pub format_version: u32,
    /// Name of the software that wrote this checkpoint; `None` for a buffer
    /// older than the field.
    pub software: Option<String>,
    /// Version of the software that wrote this checkpoint.
    pub software_version: String,
    /// ISO 8601 timestamp when the checkpoint was written.
    pub created_at: String,
    /// Number of stages the graph manifest spans.
    pub num_stages: u32,
    /// Graph manifest: node list, edge list, node → pool map, and pool-set size.
    pub graph_manifest: GraphManifest,
    /// Producer-namespaced metadata (the training algorithm's own state).
    pub producer: ProducerBlock,
    /// Study-global season cycle and per-hydro PAR-order descriptor; absent
    /// ([`SEASON_CYCLE_CODE_ABSENT`]) for a pre-`id:19` buffer.
    pub season_manifest: SeasonManifest,
}

// ── Owned output types for deserialization ───────────────────────────────────

/// Owned version of [`PolicyCutRecord`] returned by [`crate::deserialize_stage_cuts`].
///
/// Unlike [`PolicyCutRecord<'a>`], this type owns its `coefficients` vector so it
/// can be returned from a deserialization function that does not borrow from the
/// input buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyCutRecord {
    /// Unique identifier for this piece across all iterations.
    pub cut_id: u64,
    /// LP row position (required for artifact reproducibility).
    pub slot_index: u32,
    /// Training iteration that generated this piece.
    pub iteration: u32,
    /// Forward pass index within the generating iteration.
    pub forward_pass_index: u32,
    /// Pre-computed affine intercept.
    pub intercept: f64,
    /// Gradient coefficients; positional per the [`PolicyCutRecord::coefficients`] contract.
    pub coefficients: Vec<f64>,
    /// Whether this piece is currently active in the LP.
    pub is_active: bool,
}

/// Owned version of [`PolicyBasisRecord`] returned by [`crate::deserialize_stage_basis`].
///
/// Unlike [`PolicyBasisRecord<'a>`], this type owns its status byte vectors so it
/// can be returned from a deserialization function.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwnedPolicyBasisRecord {
    /// Stage index (0-based).
    pub stage_id: u32,
    /// Training iteration that produced this basis.
    pub iteration: u32,
    /// One status code per LP column (variable). Encoding is solver-specific.
    pub column_status: Vec<u8>,
    /// One status code per LP row (constraint). Encoding is solver-specific.
    pub row_status: Vec<u8>,
    /// Number of trailing rows in `row_status` that correspond to affine-piece rows.
    pub num_cut_rows: u32,
}

/// Stage-level metadata and affine-piece records returned by [`crate::deserialize_stage_cuts`].
///
/// Contains the stage-level fields stored in the `StageCuts` root table plus the
/// vector of deserialized affine-piece records.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageCutsReadResult {
    /// Pool id (0-based), as written by the pool-keyed payload.
    pub stage_id: u32,
    /// Number of state variables; equals the length of each piece's `coefficients` vector.
    pub state_dimension: u32,
    /// Total preallocated affine-piece slots in the pool.
    pub capacity: u32,
    /// Number of slots loaded from a prior artifact.
    pub warm_start_count: u32,
    /// Number of filled slots in the pool.
    pub populated_count: u32,
    /// Deserialized affine-piece records.
    pub cuts: Vec<OwnedPolicyCutRecord>,
    /// Per-slot entity identity; empty when the field is absent from the buffer.
    pub entity_manifest: Vec<EntitySlot>,
    /// Cost-scale provenance factor; `None` when absent from a pre-`id:8` buffer.
    pub cost_scale_factor: Option<f64>,
    /// Owning node's policy-graph id; [`STAGE_CUTS_NODE_ID_SENTINEL`] for a shared
    /// pool or a pre-`id:8` buffer.
    pub node_id: i32,
    /// Graph-stage id key; [`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL`] when unresolved
    /// or absent from a pre-`id:8` buffer.
    pub graph_stage_id: i32,
    /// Owning stage's `end_date`, encoded `year * 10000 + month * 100 + day`;
    /// [`STAGE_CUTS_PRICED_STATE_DATE_SENTINEL`] when not recorded or absent
    /// from a pre-`id:11` buffer.
    pub priced_state_date: i32,
}

/// Owned version of [`StageStatesPayload`] returned by [`crate::deserialize_stage_states`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StageStatesReadResult {
    /// Study stage index (0-based).
    pub stage_id: u32,
    /// Policy-graph node identity; [`STAGE_STATES_NODE_ID_SENTINEL`] when the
    /// field is absent from the buffer (a pre-`id:5` artifact).
    pub node_id: i32,
    /// Length of each state vector.
    pub state_dimension: u32,
    /// Number of states stored.
    pub count: u32,
    /// Flat data buffer (owned).
    pub data: Vec<f64>,
    /// Per-slot entity identity; empty when the field is absent from the buffer.
    pub entity_manifest: Vec<EntitySlot>,
}

/// Complete deserialized value-function artifact returned by [`crate::read_policy_checkpoint`].
#[derive(Debug, Clone)]
pub struct PolicyCheckpoint {
    /// Checkpoint manifest read from `manifest.bin`.
    pub metadata: CheckpointManifest,
    /// Per-pool affine-piece collections, sorted by pool id.
    pub stage_cuts: Vec<StageCutsReadResult>,
    /// Per-stage solver bases, sorted by `stage_id`.
    pub stage_bases: Vec<OwnedPolicyBasisRecord>,
    /// Per-stage visited states, sorted by `stage_id`.
    ///
    /// Empty when the artifact was written without visited states.
    pub stage_states: Vec<StageStatesReadResult>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use chrono::NaiveDate;

    use super::{
        ENTITY_SLOT_DATE_SENTINEL, EntitySlot, StateFamily, decode_slot_date, encode_slot_date,
    };

    #[test]
    fn state_family_codes_match_policy_fbs_entity_type() {
        assert_eq!(StateFamily::HydroStorage.code(), 0);
        assert_eq!(StateFamily::HydroInflowLag.code(), 1);
        assert_eq!(StateFamily::AnticipatedThermalState.code(), 2);
        assert_eq!(StateFamily::HydroTransitBucket.code(), 3);
    }

    #[test]
    fn state_family_from_code_round_trips_and_rejects_unknown() {
        for family in [
            StateFamily::HydroStorage,
            StateFamily::HydroInflowLag,
            StateFamily::AnticipatedThermalState,
            StateFamily::HydroTransitBucket,
        ] {
            assert_eq!(StateFamily::from_code(family.code()), Some(family));
        }
        assert_eq!(StateFamily::from_code(4), None);
        assert_eq!(StateFamily::from_code(u8::MAX), None);
    }

    #[test]
    fn entity_slot_family_reads_the_raw_byte() {
        let slot = EntitySlot::anticipated(7, 0, true);
        assert_eq!(slot.family(), Some(StateFamily::AnticipatedThermalState));
    }

    #[test]
    fn entity_slot_constructors_set_family_code_and_sentinel() {
        let storage = EntitySlot::storage(7, true);
        assert_eq!(storage.entity_type, StateFamily::HydroStorage.code());
        assert_eq!(storage.entity_id, 7);
        assert_eq!(storage.subindex, 0);
        assert!(storage.was_active);

        let inflow_lag = EntitySlot::inflow_lag(2, 3, false);
        assert_eq!(inflow_lag.entity_type, StateFamily::HydroInflowLag.code());
        assert_eq!(inflow_lag.entity_id, 2);
        assert_eq!(inflow_lag.subindex, 3);
        assert!(!inflow_lag.was_active);

        let transit_bucket = EntitySlot::transit_bucket(4, 5, true);
        assert_eq!(
            transit_bucket.entity_type,
            StateFamily::HydroTransitBucket.code()
        );
        assert_eq!(transit_bucket.entity_id, 4);
        assert_eq!(transit_bucket.subindex, 5);
        assert!(transit_bucket.was_active);

        let anticipated = EntitySlot::anticipated(6, 1, false);
        assert_eq!(
            anticipated.entity_type,
            StateFamily::AnticipatedThermalState.code()
        );
        assert_eq!(anticipated.entity_id, 6);
        assert_eq!(anticipated.subindex, 1);
        assert!(!anticipated.was_active);
    }

    #[test]
    fn entity_slot_constructors_leave_new_dates_at_sentinel() {
        for slot in [
            EntitySlot::storage(1, true),
            EntitySlot::inflow_lag(1, 1, true),
            EntitySlot::transit_bucket(1, 1, true),
            EntitySlot::anticipated(1, 0, true),
        ] {
            assert_eq!(slot.reference_date, ENTITY_SLOT_DATE_SENTINEL);
            assert_eq!(slot.interval_start, ENTITY_SLOT_DATE_SENTINEL);
            assert_eq!(slot.interval_end, ENTITY_SLOT_DATE_SENTINEL);
        }
    }

    #[test]
    fn entity_slot_with_interval_sets_both_endpoints() {
        let dated = EntitySlot::transit_bucket(9, 2, true).with_interval(20_311_201, 20_320_101);
        assert_eq!(dated.interval_start, 20_311_201);
        assert_eq!(dated.interval_end, 20_320_101);
        assert_eq!(dated.reference_date, ENTITY_SLOT_DATE_SENTINEL);
    }

    #[test]
    fn slot_date_codec_round_trips_and_rejects_non_dates() {
        for (year, month, day) in [(2031, 11, 10), (2024, 2, 29), (2030, 1, 1)] {
            let date = NaiveDate::from_ymd_opt(year, month, day).unwrap();
            assert_eq!(decode_slot_date(encode_slot_date(date)), Some(date));
        }
        for value in [i32::MIN, -1, 0, 20_240_230] {
            assert_eq!(decode_slot_date(value), None);
        }
    }
}
