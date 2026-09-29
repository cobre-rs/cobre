# Reserved Seams and Deferred Debt

> **Status:** Living register — tracks shipped state (what is reserved vs deferred today). Every entry is self-guarding; re-derive against the live tree before acting.

This document tracks two related but distinct things about the workspace's
non-obvious inert surfaces:

- The **reserved-seam register**: config fields, struct fields, and functions
  that are loaded, validated, and/or compiled, but have no production consumer
  today — each entered here with an **owner** (the subsystem responsible for
  wiring or retiring it) and a **consuming milestone** (the concrete condition
  that activates it). A candidate that cannot be given both is not registered;
  it is dead code to remove, not a seam to reserve. This operationalizes the
  project's "unwired config is reserved, not dead" rule — a claim is only
  trustworthy here if it is checkable, not merely asserted.
- The **deferred-debt register**: a separate, broader class of architectural
  debt (not limited to unwired seams) tracked alongside this one.

For known **structural/algorithmic** limitations of the node-graph engine
(as opposed to unwired config or code) — single-initial-node,
single-boundary-policy terminal cost (a per-leaf terminal future-cost function
needs multi-policy input), enumerated-selection restrictions — see
[`policy-graph-limitations.md`](policy-graph-limitations.md); this document
does not restate those.

A seam leaves this register the moment it is wired (delete the entry, wire the
reader) or the moment its milestone is ruled out (delete the seam itself, since
an unreachable milestone converts a reserved seam into dead code).

## Reserved-seam register

### `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`

**What it is.** `crates/cobre-io/src/config/training.rs` declares
`UpperBoundEvaluationConfig` (`enabled`, `initial_iteration`,
`interval_iterations`, and a nested `LipschitzConfig` with `mode`,
`fallback_value`, `scale_factor`) for a vertex-based inner-approximation upper
bound. The whole struct is loaded, schema-exported, and round-trips through
`config.json → upper_bound_evaluation`, but no field of it — not `mode`, not
`enabled` — is read anywhere in `crates/cobre-sddp`. This is a config-only stub
for a feature that has not been implemented, not merely one inert field inside
an otherwise-wired struct.

**Owner.** The vertex-based upper-bound-evaluation feature — currently a
config-surface stub with no solver-side implementation.

**Consuming milestone.** The architecture-and-debt-register audit that decides
whether to implement vertex-based inner approximation (wiring `enabled` /
`initial_iteration` / `interval_iterations` / `lipschitz.*` into the upper-bound
estimator) or retire the config surface as never-shipped. Until that
disposition is made, the fields stay reserved rather than removed, per the
"unwired config is reserved, not dead" rule — but this is the one entry in this
register whose milestone is a decision, not an implementation trigger, so it
is a stronger candidate for removal than the others here if that decision goes
against wiring it.

### Shared-memory communicator trait hierarchy (`cobre-comm`, `shared-memory` feature)

**What it is.** `crates/cobre-comm` defines a shared-memory provider abstraction —
`SharedMemoryProvider`, its `SharedRegion<T>` GAT, `LocalCommunicator`, the
`LocalCommKind` dispatch enum, and `HeapRegion<T>` — behind the `shared-memory`
feature, together with the live `MPI_Comm_split_type(SHARED)` call in
`FerrompiBackend::split_local`. No consumer outside `cobre-comm` uses any of it
yet: both the `LocalBackend` and the MPI `FerrompiBackend` currently resolve
`Region<T>` to the same per-rank `HeapRegion<T>` (a private `Vec<T>`), so the
seam is wired end-to-end but delivers no cross-rank shared memory until a
consumer and a real `MPI_Win_allocate_shared`-backed `SharedRegion` land.

**Owner.** The comm / HPC-distribution owner.

**Consuming milestone.** Intra-node deduplication of the two largest read-only
per-rank datasets — the historical/scenario library and the cut archive — so
ranks sharing a node hold one node-shared copy instead of one copy per rank,
backed by `MPI_Win_allocate_shared`. When the first such consumer lands,
`SharedRegion` is designed against ferrompi's real window allocation and the
`HeapRegion` fallback stops being the only resolution; until then the hierarchy
stays reserved rather than removed, per the "unwired config is reserved, not
dead" rule.

### Boundary-cut wire `graph_stage_id` (`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL`)

**What it is.** Every exported stage-cuts payload carries a `graph_stage_id`
field (`policy_export.rs`'s `build_stage_cuts_payloads`, falling back to
`STAGE_CUTS_GRAPH_STAGE_ID_SENTINEL` when a pool's `pool_stage` is out of
range), round-trips through the FlatBuffers wire encoding
(`cobre-io/src/output/policy/codec.rs`), and is exposed to Python callers
(`cobre-python/src/policy.rs`, `results.rs`). No code in `cobre-sddp` reads the
decoded value back off a loaded checkpoint any more — boundary-cut selection
reads `priced_state_date` instead, which the node-graph-to-calendar switchover
made the authoritative pool key.

**Owner.** The policy / boundary owner
(`crates/cobre-sddp/src/policy/policy_export.rs`, `policy_load.rs`;
`crates/cobre-io/src/output/policy/codec.rs`).

**Consuming milestone.** The first `cobre-sddp` reader that needs the
originating node-graph stage id independent of the calendar-derived
`priced_state_date` — for example a diagnostic cross-checking a loaded
checkpoint's graph shape, or a future selection mode disambiguating pools that
share one `priced_state_date` by graph position. Until such a reader lands the
field stays a write-only wire and Python-visible diagnostic surface.

## Verified NOT reserved

`historical_years` (`cobre_core::scenario::ScenarioSource`,
`crates/cobre-core/src/model/scenario.rs`) is consumed:
`crates/cobre-sddp/src/setup/{mod.rs,stochastic_pipeline.rs}` thread
`training_source.historical_years.as_ref()` / `simulation_source.historical_years.as_ref()`
into `discover_historical_windows`
(`crates/cobre-stochastic/src/sampling/window.rs`), which filters candidate
history years by the user pool, and the same `Option<&HistoricalYears>` is read
directly by `crates/cobre-stochastic/src/sampling/historical.rs`'s window
validation. It is deliberately absent from the register above; a prior
interface-review note calling it inert predates this wiring.

The hydro `storage_violation_below_cost` and `filling_target_violation_cost`
penalties (`HydroPenalties`, `crates/cobre-core/src/model/resolved/penalties.rs`)
are consumed for **filling-phase** hydros: `fill_filling_target_columns`
(`crates/cobre-sddp/src/lp/builder/columns.rs`) writes
`filling_target_violation_cost` as the objective coefficient on the `σ_fill`
target-shortfall slack at every Filling stage, and
`fill_filled_min_storage_floor_columns` writes `storage_violation_below_cost` on
the `σ^{v-}` operating-floor slack at every Operating stage of a filling hydro;
the paired soft `≥` rows live in `lp/builder/rows.rs`, and the penalty ordering
is validated (a filling-target penalty must exceed the max deficit cost). They
are unconsumed only for ordinary non-filling hydros, whose storage bounds are
HARD so no slack column exists — the case a prior note over-generalized to
"always 0". Consumed, not reserved: deliberately absent from the register above.

## Post-migration architecture audit

A judgment reading of the node-native engine after the graph, node axis, and
traversal machinery landed. It answers three questions: where each new
structure lives, whether each new seam has exactly one owner, and whether any
`stage`-keyed structure is residue where a node/pool key belongs. It is a
snapshot of the reading, not a mechanical gate — the mechanical checks live in
the "Audit-evidence" section below; re-derive both against the live tree rather
than trusting the prose as a frozen state.

### Module map

- **Graph type.** `NodeGraph` (one `NodeRuntime` per node) lives in
  `crates/cobre-sddp/src/setup/node_graph.rs`, alongside the typed node axis
  (`NodePos`, the dense position; `NodeId`, the declared id) and the enumerated
  walk plan (`EnumeratedPlan`).
- **Single construction dispatcher.** `build_node_graph` is the one entry point;
  it chooses `build_chain_node_graph` for an empty declared `nodes[]` and
  `build_declared_node_graph` otherwise, both producing the same representation
  consumed uniformly downstream.
- **Frontier walk / reverse-topological sweep.** `NodeGraph::stage_frontier`,
  `frontier_node`, and `any_stage_node` resolve a stage's nodes; the backward
  reverse-topological level partition is `NodeGraph::backward_cut_levels`. The
  forward root-to-leaf walk is `EnumeratedPlan::walk_path`.
- **`node → pool` map.** `NodeRuntime.pool_id`, assigned once during graph
  construction and read back via `NodeGraph::node_pool_ids`.
- **Visit records.** `VisitedStatesArchive` (one `NodeStates` per node) in
  `crates/cobre-sddp/src/training/visited_states.rs`.
- **Module homes still fit.** `training/` holds the passes (session, forward,
  backward, lower bound); `cut/` holds the cut substrate (pool, future-cost
  function, wire, sync, selection, dynamic selection, basis reconstruction);
  `workspace/` holds per-worker arenas and the captured basis. No module has
  accreted a node-work concern under a stale name: the graph machinery is homed
  in `setup/`, the traversal drivers in `training/forward` and
  `training/backward`, and each enumerated fork is a named sibling of its
  sampled counterpart, not an inline special case.

### One-owner checks over the new seams

Each seam below has exactly one construction site and one interpretation site;
no second site was found.

- **Node-graph home.** Constructed only through `build_node_graph`; interpreted
  uniformly — no chain-vs-tree shape predicate reaches training dispatch (the
  `is_chain` grep in the Audit-evidence section returns zero).
- **`node → pool` map.** `pool_id` is written once during graph build; read via
  `node_pool_ids` and consumed by `FutureCostFunction::new_per_pool` to size one
  pool per node.
- **Capacity function.** `pool_capacity` (`cut/fcf.rs`) has a single definition
  and a single caller (`FutureCostFunction::new_per_pool`).
- **Visit-record layout.** `VisitedStatesArchive` / `NodeStates` are owned
  solely by `training/visited_states.rs`.
- **Basis node tag.** `CapturedBasis.node_id` (a `NodeId`) is written once at
  capture and read once at apply (`reconstruct_basis`); a mismatch cold-starts
  rather than warm-starting, never a wrong answer.
- **Cut-wire header.** The cut record's version and per-record `node_id` are
  encoded and decoded only in `cut/wire.rs` (`CUT_WIRE_VERSION`); the
  captured-basis broadcast header (`BASIS_BROADCAST_WIRE_VERSION`) is encoded
  and decoded only by `CapturedBasis::to_broadcast_payload` /
  `try_from_broadcast_payload`. Each header has one encode and one decode owner.

### Stage-residue classification

- **Genuine per-stage (kept).** Stage LP templates (the LP structure is
  stage-shared because a node's state affects only the inflow realization, not
  the constraint structure), per-stage equipment geometry, per-stage cumulative
  discount factors, and the stage calendar / season cast. Each is correctly
  keyed by stage.
- **Residue (a stage index where a node/pool key belongs): none.** Cut pools are
  keyed by `pool_id`, not stage — `pool_stage[pool_id] == StageIdx(t)` holds only
  on a chain and diverges under branching. The chain reaches its values as the
  degenerate node graph (the empty-`nodes[]` path through `build_node_graph`),
  never via a shape fork; the architecture reads symmetric.

### Whole-lifecycle scope (beyond the node axis)

A full `cobre run` read (entrypoint → setup → policy → training → simulation →
outputs) found the largest structural debt outside the node axis, in the setup
config-projection layer and in the CLI/Python output orchestration. Both are
register items below, not node-axis findings, and neither is executed here:

- The `Config → StudyParams / ConstructionConfig / BroadcastConfig` projection
  and the CLI non-root stochastic-context reconstruction
  (`reconstruct_stochastic_context_non_root` /
  `rebuild_historical_library_non_root`), a rank-0 mirror that must stay
  bit-identical across a crate boundary.
- The output orchestration mirrored by hand across the CLI and Python crates.

## Deferred-debt register

Everything the plan deliberately did not finish. Each entry names an **owner**
(the role responsible for closing it) and a **trigger** (a durable condition
that makes the work worth doing — never a date or a release number). An item
that cannot be given both is not register-ready. Every symbol and path cited
resolves against the live tree.

### Enumerated interior-node outgoing-state exchange for multi-rank branching graphs

**What it is.** The enumerated forward populates a node's persisted outgoing
state only on the ranks whose assigned paths visit it, zero-filling every other
rank's slot; the replicated backward partitions a cut-generating node's
successor openings across all ranks regardless. This is sound only when every
cut-generating node lies on every root→leaf path (a deterministic trunk with a
terminal fan). A multi-rank enumerated run over a graph with interior branching
is therefore **hard-rejected before any solve**: `enumerated_requires_state_exchange`
(`setup/node_graph.rs`) returns the first offending interior node, and the
training-session constructor turns that into a validation error rather than
letting a rank cut against a zeroed incoming state. The deferred fix adds a
cross-rank exchange (or broadcast) of interior-node outgoing state so a rank
that never visits a node can still cut against its true incoming state.

**Owner.** The SDDP training-engine owner. A design decision is required first:
which rank's solve is canonical when a node is solved redundantly across ranks.

**Trigger.** Multi-rank enumerated training over interior-branching graphs is
required. Closing it also needs a real multi-rank MPI multi-level-branching
fixture that the in-process single-rank test harness cannot express.

### Enumerated backward observability gap

**What it is.** The enumerated backward driver (`run_enumerated_backward`)
returns an empty per-stage worker-statistics vector, so the solver-statistics
log carries no backward-phase rows under enumerated traversal. Cuts and bounds
are unaffected (correctness- and determinism-neutral).

**Owner.** The training owner.

**Trigger.** Enumerated-run backward telemetry / solver-profile output is
required.

### Predecessor-distinctness debug assertion

**What it is.** The parent lookup and parent-map construction carry a
debug-build assertion that a node has at most one predecessor; it is relaxed to
exclude sibling pairs, so it is vacuous at in-degree 1 (every node in a tree)
and bites only a future node reached by two or more edges.

**Owner.** The setup / SDDP owner.

**Trigger.** Recombination / dense-transition (DAG) graph work.

### Enumerated recombination seam

**What it is.** A node with in-degree ≥ 2 is hard-rejected under enumerated
selection at study setup (`reject_recombining_node_enumeration`), because the
engine reconstructs a node's incoming continuous state from a single-predecessor
map and would otherwise solve the join once under one arbitrarily chosen
parent's state. Full support needs per-prefix state reconstruction and a
prefix-based rewrite of the exact-bound path, with SDDP-specialist sign-off.
This seam and the single-predecessor restriction are documented in
[`policy-graph-limitations.md`](policy-graph-limitations.md).

**Owner.** The setup / SDDP owner.

**Trigger.** Recombining-graph support under exact enumeration.

### Oracle test-harness duplication

**What it is.** `crates/cobre-sddp/tests/extensive_form_oracle.rs` carries a
verbatim copy of the comparison harness in
`crates/cobre-sddp/tests/branching_value_oracle.rs` (the `close` tolerance
helper and its scaffolding) and pays a second static link of the solver,
against the test cost-discipline. Two resolution options for an owner pick:
promote the shared `close` / tolerance helpers into `cobre_sddp::test_support`,
or fold both fixtures into one test binary.

**Owner.** The test-infrastructure owner.

**Trigger.** The next consolidation of the branching oracle test suite.

### Cut-pool slot-index newtype

**What it is.** Cut-pool slot indices are a raw integer across many call sites
adjacent to the slot-identity basis-reconstruction hot path; a dedicated newtype
would make slot-vs-row confusion a compile error. Deferred because the change
touches many sites next to a protected hot path.

**Owner.** The cut-pool / training owner.

**Trigger.** The planned traversal-stride index rename lands in the same
neighborhood (a deliberate typed-index sweep of the cut-pool indices).

### Superseded cut-sync public methods — RESOLVED

**What it was.** Three methods on `CutSyncBuffers` — `sync_cuts`,
`pack_local_records`, `sync_packed_records` — were the legacy single-pool
exchange, superseded by the per-level batched `sync_level_records` but still
re-exported public API on a published crate, so removing them was a breaking
change; their test suite gave false "still used" confidence.

**Resolution.** The three methods and their tests are removed in a licensed
public-API break; `sync_level_records` is the sole cut-exchange path.

### Python-binding Rust tests invisible to CI

**What it is.** The Python-binding crate is excluded from the workspace, and its
CI job runs only the Python build plus pytest, so the crate's Rust `#[cfg(test)]`
modules are never compiled or run in CI. Fix: either wire a `cargo check --tests`
for the crate, or hoist a shared node-graph test-fixture builder into
`cobre_sddp::test_support` so those Rust tests live in a CI-visible crate.

**Owner.** The build / CI owner.

**Trigger.** Systemic — the next CI-configuration pass (a Rust test regression in
that crate would otherwise ship unseen).

### Backend-scoped parity-roster caveat

**What it is.** The opening-order determinism check and three of the
wire-exchange parity gates are exact-comparison only under the bit-exact
backend and are skipped or relaxed on the other LP backend, but the parity
roster does not state this backend scoping. A one-line clarification belongs in
the roster document. It is registered rather than applied here: applying it
would touch a second file beyond this register.

**Owner.** The parity / test owner.

**Trigger.** The next parity-roster revision.

### Horizon-type config field long-term fate

**What it is.** The policy-graph horizon-type field (`graph_type` in
`crates/cobre-io/src/stages.rs`) accepts only the finite-horizon value; the
cyclic value parses but is rejected as reserved during conversion. Under the
node-native engine the graph declaration itself is the structure, so the field
is a deletion candidate.

**Owner.** The setup / config owner.

**Trigger.** The next licensed input-format break.

### Reserved and consumed knobs (snapshot correction)

**What it is.** `LipschitzConfig.mode` (`crates/cobre-io/src/config/training.rs`)
is a single-valued, unconsumed string inside the vertex-based
upper-bound-evaluation config; it is already carried in the reserved-seam
register above, never swept, and that register owns its disposition. In
contrast, `historical_years` is **not** inert — it is consumed by the window
and historical samplers (see "Verified NOT reserved" above), correcting the
stale interface-review note that called it inert.

**Owner.** The upper-bound-evaluation feature owner (for the reserved knob).

**Trigger.** The decision to implement or retire vertex-based inner
approximation.

### Boundary-policy source-node

**What it is.** The boundary-policy config (`BoundaryPolicy`,
`crates/cobre-io/src/config/policy.rs`) addresses its source by the study's own
boundary date, has no source-node selector, shares one leaf pool
unconditionally, and rejects a multi-node source. It relocates once, under a
study-level boundary configuration.

**Owner.** The setup / config owner.

**Trigger.** That relocation (a study-level boundary redesign).

### Stage-configuration engine-scoping

**What it is.** The stage-configuration file carries several roles at once
(calendar, blocks, openings, seasons, risk, state-variable toggles,
discounting) and forces every future study kind to declare this algorithm's
uncertainty and risk axes. A first step is taken: the per-stage opening count
(`num_openings`) is conditionally required — required only for generated stages.
The full split — scoping the algorithm-specific axes out of the shared stage
file — is larger.

**Owner.** The setup / config owner.

**Trigger.** The full engine-scoping split of the stage file.

### Within-node weighted opening draw (chain-parity break)

**What it is.** Today a visited node draws one within-node opening uniformly over
its own opening sub-range. A future change replacing that uniform draw with an
inverse-CDF (probability-weighted) draw will select a different opening for the
same seed, breaking chain parity at that time. This numeric break is **separate**
from the input-format break already absorbed by the node-per-stage migration:
that migration's "one break" accounting does not cover it — the weighted-opening
draw introduces its own numeric break, stated here explicitly.

**Owner.** The SDDP / sampling owner.

**Trigger.** The weighted within-node opening feature landing.

### Anticipated-commitment per-block tension

**What it is.** The generic-constraint and bound validators forbid a block
argument on an anticipated thermal's commitment (an anticipated decision is a
stage-level scalar), which is in tension with per-block delivery. The tension is
parked.

**Owner.** The I/O constraints owner.

**Trigger.** Per-block delivery work on anticipated commitments.

### Pre-study-decided, post-study-delivered anticipated commitments (no carrier) — RESOLVED

**What it was.** A commitment decided at a **pre-study** stage that delivered
into a **post-study** stage had no carrier on the anticipated delivery axis:
the ring's slots keyed on delivery-target residue over in-study decision
stages, so a pre-study decider had no in-study stage to latch from and the
commitment could not be represented. `check_post_study_stages` hard-rejected
it as a `BusinessRuleViolation`, advising the user to shorten the lead so the
decision fell within the study horizon.

**Resolution.** The commitment is now representable as a **fixed post-horizon
commitment**: a declared constant that never enters the ring (no carrier is
needed — it is deliberately never a ring member), priced against the terminal
boundary FCF by a constant fold into the cut intercepts, and reported at its
real delivery date. The hard reject is retired; the input surface is the
existing `past_anticipated_commitments` windows extended past the study
horizon, validated by a two-sided tiling/envelope/commissioning matrix. The
durable homes are the live spec
[`anticipated-thermals-and-water-travel-time.md`](anticipated-thermals-and-water-travel-time.md)
and `.claude/rules/sddp.md`'s "Anticipated thermal commitments" contracts.

### Stage-calendar crate home

**What it is.** The stage-calendar / season-cast machinery (`StageCalendar`,
`season_cast`) currently lives in the stochastic crate, an in-plan extension of
the existing season-cast module. A relocation to a temporal module in the core
crate is registered as a structural follow-up.

**Owner.** The stochastic / temporal owner.

**Trigger.** When the season-machinery relocation is worth its scope.

### External-noise take/fill glue duplication

**What it is.** The mem-take/fill buffer glue around `fill_external_opening_noise`
is duplicated across the frozen-path backward worker
(`training/backward/by_node.rs`) and the DCS-path backward worker
(`training/backward/by_scenario.rs`). Extracting a shared helper crosses the
frozen-hot-path / DCS boundary that the basis-reconstruction module protects, so
it is an architectural call, not a drive-by. The adjacent backward-buffer
pre-allocation duplication across the two schedulers folds with the same
refactor.

**Owner.** The training owner.

**Trigger.** A deliberate frozen-hot-path / DCS-boundary refactor.

### Branching-engine housekeeping follow-ups

**What it is.** Three small deferred items from the branching work, each with its
own owner and trigger:

- A checkpoint-write regression observed on a fan-structured branching study.
  **Owner.** The training owner. **Trigger.** Reproduced when checkpointing a
  fan-structured branching study.
- The two-rank captured-basis-broadcast round-trip test
  (`broadcast_basis_cache`) belongs in a dedicated wire-format test home rather
  than its current location. **Owner.** The training / test owner. **Trigger.**
  The next MPI wire-format test reorganization.
- A design note describing an order/count-based cut-exchange header no longer
  matches the shipped self-describing, per-record node-tagged cut format;
  reconcile the note to the shipped format. **Owner.** The doc / comment owner.
  **Trigger.** The next cut-exchange wire-format touch.

### SDDP.jl-compatibility limit: multi-state first stage

**What it is.** A structural limit is deferred here: a multi-state / reserved-root
first stage — an initial probability distribution over several first-stage nodes. It
is documented in [`policy-graph-limitations.md`](policy-graph-limitations.md) and
cross-referenced here rather than restated.

The companion terminal-side item once bundled here — a per-terminal-state
continuation value — is **not** a deferred limit. The shared terminal pool holds one
boundary future-cost function evaluated at each leaf's own ending state, which is the
correct representation for the supported single-boundary-policy input (see
[`policy-graph-limitations.md`](policy-graph-limitations.md)). A distinct future-cost
function per leaf would need per-leaf boundary input — more than one boundary
policy — and that multi-policy regime is tracked by the **Boundary-policy
source-node** reserved-seam entry above, not restated here.

**Owner.** The SDDP engine owner.

**Trigger.** A model that begins from an uncertain initial regime (the multi-state
first stage).

### Retired input spellings (recorded as retired, not deferred)

**What it is.** Four legacy spellings were removed outright with no compatibility
alias: the two backward-scheduler method spellings (`trial_point`,
`opening_block`); the root-level forward-pass / scenario-count selection
aliases; the legacy scenario-count spelling of the per-stage opening count; and
the reclaimed cut-record wire slot (an id-4 field once named for a domination
count, now reused for the intercept). These are recorded as retired, not awaiting
a trigger.

**Owner.** The setup / config owner.

**Guard (not a trigger).** Any config using a removed spelling fails with an
unknown-field / unknown-variant deserialize error, pinned by the reject tests in
`crates/cobre-io/src/config/{training,simulation}.rs` and the FlatBuffers schema
conformance test — the invariant that stands in for a version snapshot.

### cobre-bridge and cobre-docs adoption of the integrated-productivity contract

**What it is.** Two follow-ups on external repositories (not part of this
tree) remain open, adopting the stored-energy / 2x2 scope x evaluator
productivity contract published in
`docs/design/hydro-productivity-and-stored-energy.md`:

- `cobre-bridge` must adopt the contract's "Bridge-facing deviations"
  section — author a security-curve constraint's coefficient as
  `integrated_accumulated_productivity`, keep flood-control and
  stored-volume ceilings as operative `HydroStageBounds` rows, and retire
  its per-plant basis-selection field — none of which has landed there yet.
  **Owner.** The `cobre-bridge` integration owner.
  **Trigger.** The next `cobre-bridge` release that authors or updates a
  security-curve constraint against this contract.
- `cobre-docs` methodology pages must document the integrated productivity /
  computed-parameter surface — the 2x2 scope x evaluator model, the
  quadrature invariant, and the physical/operative split — currently
  published only in this repository's own
  `docs/design/hydro-productivity-and-stored-energy.md`.
  **Owner.** The `cobre-docs` methodology owner.
  **Trigger.** The next `cobre-docs` methodology-page revision cycle that
  covers hydro productivity.

## Deferred-debt register — whole-lifecycle audit findings

Findings of the full `cobre run` lifecycle read, described by behavior. Each
names an owner and a trigger. The structural items are escalated (too large for
a behaviour-neutral consolidation); the byte-neutral consolidations were
executed in the consolidation work and are recorded here so a future audit does
not re-raise them.

### Structural — escalated (a dedicated follow-up redesign)

#### Setup config-projection sprawl + CLI non-root reconstruction

**What it is (partially addressed).** The run configuration is re-projected
through near-isomorphic in-memory structs kept in lockstep by hand, so one config
knob touches several sites. Two sub-parts are now closed: the local params /
construction-config projections are merged into one `StudyParams`, and the
domain-crate-owned stochastic-context builder now exists
(`build_stochastic_context_for_study`), so the CLI non-root path calls it instead
of hand-mirroring the rank-0 pipeline across the crate boundary — the
silent-MPI-vs-local-divergence hazard is retired. **What remains:** the local
`StudyParams` and the wire `BroadcastConfig` projection are still two structs kept
in step by hand (unifying them is the postcard non-self-describing serialization
boundary — the non-`Serialize` stopping-rule/scheduler mirrors, `usize`/`u32`, the
broadcast-only scenario-source fields); and the setup god-struct (`StudySetup`)
still fuses immutable inputs, mutable run-config, and produced output. Target: one
postcard-safe resolved-run config projection consumed by both the local and
non-root paths, and a lifecycle split of the god-struct. This remains the anchor
of the audit's lifecycle-modeling north star.

**Owner.** The architecture owner.

**Trigger.** A dedicated setup-layer redesign (the postcard tagged-enum
serialization boundary is solved once, there).

#### CLI/Python output orchestration hand-mirror

**What it is.** Every output file the CLI writes must also be written by the
Python bindings; the two write paths are hand-mirrored (which writers, in what
order, under what guards), held together only by the Python-parity rule plus
mirror comments, with no shared owner — confirmed across both training and
simulation outputs, including the census scenario-summary tuple-reshape copied
on both sides and pinned by a test that itself exists in both crates. Target:
hoist the shared "output set + guards" (the pattern the Python `*_if_any`
helpers already prove) into a crate both the CLI and Python depend on.

**Current state (2026-09-17).** The Python side now has one internal owner (`cobre.run.run`
drives the `Study` lifecycle), and a golden test runs the toy example through both entry
points and compares every output value under a literal wall-clock mask, with an
import-resolving parity gate in CI; the two content divergences that test exposed were fixed
rather than masked. Drift is caught, but the write orchestration is still two copies.

**Owner.** The architecture owner.

**Trigger.** The setup-layer redesign.

#### Untyped state-family primitive + colliding entity dictionaries — RESOLVED

**What it is (resolved 2026-08-22).** The policy state slot's `entity_type` was an
untyped small integer whose state-family dictionary lived downstream in the policy
writer (`cobre-sddp`), was independently re-declared once in
`crates/cobre-io/src/output/policy/checkpoint.rs`'s monotonicity check, and shared
its `ENTITY_TYPE_*` prefix with a second, overlapping physical-output-entity
dictionary (`crates/cobre-io/src/output/dictionary.rs`) — a grep hazard and a
type-safety gap (no live bug; the two dictionaries were used disjointly).

**Fix.** A typed `StateFamily` enum now lives in `cobre-io` next to `EntitySlot`
(`output/policy/records.rs`) — the Rust mirror of the `EntityType` enum in
`schemas/policy.fbs` — with `EntitySlot::family()` reading the raw byte. The
`cobre-sddp` `ENTITY_TYPE_*` constants are retired onto it; the reconcile report's
parallel `ReportFamily` enum collapsed to `Option<StateFamily>`; the checkpoint
duplicate is gone; and the physical-output dictionary was renamed `OUTPUT_ENTITY_*`.
The wire byte (`EntitySlot.entity_type: u8`) is unchanged, so the change is
byte-neutral, and the public Python-binding seam
(`reserve_boundary_inflow_lag_slots`) routes the family through the cobre-io type.
Residual (not debt): standalone integration-test files keep their own local
`const ENTITY_TYPE_* = N` wire-byte pins, which are self-contained and legitimately
pin the on-disk contract.

**Owner.** The I/O owner.

#### Cut-selection paradigm conflation

**What it is.** One cut-selection type (`CutSelectionStrategy`) fuses two
paradigms — periodic value-based selection and lazy per-solve dynamic selection —
but the dynamic variant does not honor the shared interface: it early-returns
empty from the value sweep, its real logic lives elsewhere, and the driver
carries an unreachable guard for it. Target: model the two paradigms as distinct
internal types unified only at the config boundary.

**Owner.** The training owner.

**Trigger.** A third selection paradigm, or the next cut-selection change.

#### MPI-ephemeral wire-version bytes on same-binary broadcast formats

**What it is.** Two broadcast formats — the cut wire (`CUT_WIRE_FORMAT_TAG`) and the
captured-basis payload (`BASIS_BROADCAST_FORMAT_TAG`) — carry a fixed leading tag
byte on an all-ranks broadcast where every rank runs the same binary, so a
cross-version mismatch cannot occur in one run; the byte is a corruption
tripwire, not a version negotiator. The simulation exchange (version-free,
length-prefixed all-gather) is the honest counter-example. A third such format —
a resolved-parameters broadcast envelope — previously fit this pattern but was
**removed outright as dead code** (it had no production caller). **Reframed
2026-08-22:** the two constants were renamed from `*_WIRE_VERSION` to
`*_FORMAT_TAG`, and the cut-wire module doc's version-compatibility framing (bump
discipline, no-compat-shim, redeploy-on-upgrade) was rewritten as a format-tag
guard. Residual (not debt): the two constants stay `pub` because the `mpi_wire.rs`
integration test consumes them, and the runtime reject diagnostics keep their
"version" wording, pinned by the reject tests.

**Owner.** The comm / I/O owner.

**Trigger.** The next wire-format change, if the `pub` exposure or the diagnostic
wording is revisited.

### Byte-neutral consolidations — already executed

Recorded so a future audit does not re-raise them. Each was byte-neutral with
the sacred-parity baselines green.

- **Sampled arms promoted to co-equal engines.** The forward and simulation
  passes now dispatch their sampled arm through a named engine (`run_sampled` in
  `training/forward_pass_state.rs`; `run_sampled_simulation` in
  `simulation/state.rs`), matching the backward pass's clean two-arm dispatcher —
  the original sampled path is no longer left inline.
- **Node-graph query surface as methods.** The queries that took a `&NodeGraph`
  first and answered questions about it are now methods on `impl NodeGraph`
  (`setup/node_graph.rs`), not free functions.
- **Write-partition helper.** The simulation writer's repeated
  create-dir/write/push skeleton is now one `write_partition` helper
  (`crates/cobre-io/src/output/simulation_writer.rs`).

**Owner.** The training and I/O owners (as executed). **Trigger.** None —
already done.

### Removed dead code — recorded as removed

**What it is.** The resolved-parameters MPI broadcast pair (a postcard
serialize/deserialize envelope with its own version byte) had zero production
callers — the resolved-parameters table is built per-rank and never broadcast —
and was **deleted outright** under the no-dead-code directive, together with its
public re-export, its wire-format checklist entry, and its reserved-seam
register entry. It is recorded here as removed, **not** as a reserved seam, so a
future audit does not resurrect it as either live or reserved.

**Owner.** The training owner. **Trigger.** None — removed.

**Guard.** The serialize / deserialize symbols no longer resolve anywhere in the
workspace.

### Organizational / quality follow-ups (low priority)

- **God-functions with weak extraction rationale.** The cut-management driver and
  the per-node backward compute function each inline separable sub-phases that
  extract cleanly as `&mut self` methods; the per-node successor reification is
  duplicated between the sampled and enumerated backward paths, and one shared
  reification extraction resolves both the god-function length and the copy.
  **Owner.** The training owner. **Trigger.** The next substantive backward-pass
  change.
- **Backward-scheduler buffer-prealloc duplication.** The two backward schedulers
  duplicate their per-worker buffer pre-allocation; a shared prepare-buffers
  helper folds it, adjacent to the external-noise glue dedup registered above.
  **Owner.** The training owner. **Trigger.** The same frozen-hot-path / DCS-boundary
  refactor.
- **Mega-file / inline-test-giant asymmetry.** The workspace module is a flat
  mega-file holding many per-worker arena structs, and the LP-builder
  entries/columns modules carry giant inline test modules while their sibling
  builder submodules use extracted test files. Split each into a directory module
  / sibling test file matching the crate's prevailing convention. **Owner.** The
  training owner. **Trigger.** Navigability-driven, low priority.
- **Policy-dir resolve triplication + naming split.** The policy-directory
  resolve-and-guard skeleton is repeated across the warm-start, resume, and
  simulation load sites; extract a shared resolver. Two naming conventions for the
  same borrowed/owned duality coexist; unify them. One manifest name is stale (it
  is a whole-comparison bundle carrying node/pool counts, not a per-stage record).
  **Owner.** The policy owner. **Trigger.** The next policy-load change.
- **Basis-reconstruct hardening.** Add the truncation debug-assertion at the
  reconstruction entry and drop the always-zero telemetry field; the positional
  demotion of excess basic cuts is a count-balance necessity, not a quality bug —
  measure the basis-consistency-failure ratio before investing in a
  metadata-driven demotion. **Owner.** The training owner. **Trigger.** A
  warm-start-quality investigation.
- **Shared solve-prep re-home (partially addressed).** The LP-**solve** entry has
  since moved to a neutral crate-level module: `run_stage_solve` / `StageInputs`
  live in `crates/cobre-sddp/src/solve/stage_solve.rs`, and every hot path
  (including `simulation/pipeline.rs`) imports them via `crate::stage_solve::…`, so
  the simulation→training inversion is gone _for the solve entry_. The solve-**prep**
  single owner (`StageSolvePrep`, `crates/cobre-sddp/src/training/stage_solve_prep.rs`)
  was **not** re-homed: `simulation/pipeline.rs` still imports it via
  `training::stage_solve_prep`, so the inversion **persists for the prep**. (`solve/`
  and `stage_solve_prep.rs` are two sequential phases of one pipeline —
  prep-then-solve — not duplicates.) A byte-neutral move of the prep to the neutral
  `solve/` home plus import retarget closes it. **Owner.** The training owner.
  **Trigger.** The simulation-dispatch symmetry refactor, or the next `solve/`-module
  touch.

### Post-lifecycle-walk findings (fresh pass, 2026-08-18)

The lifecycle walk above never covered the modules that landed with the later
branching / GNL / generic-constraint work (`solve/`, `hull/`, `lead_time/`,
`horizon_mode.rs`, `generic_constraint_echo.rs`, `validate_phases.rs`,
`simulation/enumerated.rs`, `training/backward/replicated.rs`). A fresh read
found those overwhelmingly clean (`hull/`, `lead_time/`, `horizon_mode.rs`'s
single-variant enum is a documented reserved seam, `generic_constraint_echo.rs`,
`config.rs`, `training/backward/replicated.rs`) plus two architecture items:

- **Prep-phase abstraction covered three of four phases; boundary check
  unmirrored — RESOLVED.** `crates/cobre-sddp/src/validate_phases.rs` `PrepPhase`
  documented itself as unifying the SDDP preparation steps while the enum had
  exactly three variants (`Config`, `Stochastic`, `HydroModels`) — a doc/code
  mismatch. The real fourth step — boundary-cut reconciliation — bypassed the
  shared `PrepPhase` / `prep_phase_metadata` abstraction with its own ad-hoc error
  formatting (`crates/cobre-cli/src/commands/validate.rs`) and had no equivalent
  in the Python binding. **Resolution.** `PrepPhase` carries a `Boundary` variant
  that both front ends route the boundary reject through (its own
  `prep_phase_metadata` row, `validate --json` error object included), the Python
  binding runs the boundary reconciliation as a validation phase of its own, and
  the enum doc states no variant count — see the fixed-items register below.
- **Enumerated forward and enumerated simulation duplicate their mid-level
  claim/scatter orchestration.** `simulation/enumerated.rs` and
  `training/forward/enumerated.rs` each reimplement the stage-synchronous
  claim/scatter skeleton; only the low-level primitives (`ClaimCursor` /
  `canonical_scatter`, `EnumeratedPlan` / `NodeGraph`) are shared. The primitive
  reuse mitigates the _kernels_ but not the _orchestration_, so a claim/scatter
  protocol change is a two-site edit held together only by mirror comments (same
  family as the backward reification copy in the god-function follow-up above).
  **Owner.** The training owner. **Trigger.** The enumerated broadcast-distribution
  migration (see [`enumerated-traversal-distribution.md`](enumerated-traversal-distribution.md))
  or the next claim/scatter change.

### Performance-debt follow-ups (first performance pass, 2026-08-18)

The lifecycle audit was architecture-only; this is a first hot-path allocation
pass against the "never allocate on hot paths — pre-allocate workspaces, reuse
buffers" rule. The codebase is overwhelmingly disciplined (near-universal
`mem::take` + reuse; no `Box<dyn>` on any hot path). Outcomes of the pass:

**Fixed (byte-neutral; determinism gates green).**

- **Per-iteration forward-stats aggregation buffers hoisted.**
  `TrainingSession::run_forward_phase` (`crates/cobre-sddp/src/training/session/mod.rs`)
  previously allocated two fresh `Vec`s plus a `.collect()` every training
  iteration to cross-rank-aggregate per-stage forward stats; they are now reused
  buffers on `IterationScratch` (`fwd_stats_pack_local` / `fwd_stats_pack_global` /
  `fwd_stats_unpacked`), resized-and-fully-overwritten each iteration. Telemetry
  only (no effect on cuts/bounds); the `solver_stats` roundtrip and the
  `opening_order_determinism` gate confirm neutrality.
- **Loop-invariant interior-node filter cached.** `run_cut_management` re-derived
  the interior cut-generating node set (`node_graph.nodes.iter_indexed().filter(…)`)
  every cut-selection cycle; the graph topology is fixed for the run, so it is now
  resolved once in `TrainingSession::new` and stored on `interior_cut_nodes`. The
  cached order is byte-identical to the recompute (same canonical `iter_indexed`
  order); the cut-selection determinism suite confirms neutrality. Residue (minor):
  the derivation is open-coded in the session constructor and copy-pasted by the
  test that pins it, rather than owned by a `NodeGraph` accessor beside the
  sibling predicate `backward_cut_levels` already owns — a one-owner follow-up.
  **Owner.** The training owner. **Trigger.** The next interior-node-set or
  cut-selection touch.

**Investigated and refuted (recorded so a future audit does not re-raise).**

- **Enumerated-simulation per-node capture is NOT a fixable allocation.** The
  per-node fresh buffer in `simulation/enumerated.rs` `enumerated_sim_stage_worker`
  is optimal, not debt: the `EnumeratedSimScratch::worker_captures` doc documents
  the deliberate move-scatter — each census arena entry needs its own buffer, so a
  move achieves exactly one allocation per arena buffer with zero copies. The
  training sibling's copy-reuse pays off only because its arena is reused across
  many iterations; the simulation sweep runs once per call, so copy-reuse would
  _add_ copies. (A first-pass write here proposed "fix" it; that was withdrawn on
  reading the design rationale.)

**Deferred (minor).**

- A fresh per-call `Vec` + per-stage `clone` in `run_enumerated_backward`
  (`backward_pass_state.rs`) is not a regression — the pre-existing
  `run_sampled_backward` has the identical shape — so folding both onto a recycled
  buffer is a symmetric follow-up, not escalated here. **Owner.** The training
  owner. **Trigger.** The next backward-scheduler scratch touch.

### Post-release fix-wave findings (2026-08-19)

A commit-by-commit architecture read of the bug-fix wave that shipped the
boundary-lag inference, the gap-rule-under-CVaR admission, the nested CVaR
upper bound, the outflow-diversion exclusion, and the projection-aware
intercept dot. The fixes are functionally correct and test-pinned; the debt
below is structural residue of their speed.

#### Boundary-derived setup context has no single owner — RESOLVED

**What it is (resolved 2026-08-22).** The depth-resolution rule was
single-owner (`boundary_policy_required_lag_depth`,
`crates/cobre-sddp/src/policy/policy_load.rs`), but the fold — given a case
directory and a `Config`, produce the effective depth — was open-coded, in
differing shapes, at every production setup entry point (CLI run, CLI
validate, Python run) plus a laxer test-helper copy. The resolved value rode
three relay carriers (`StudyParams` → `BroadcastConfig` → `ConstructionConfig`),
and `ConstructionConfig` held two facts derived from the same
`config.policy.boundary` filled by different owners at different times (a pure
boundary-present flag vs a patched-out-of-band inflow-lag depth) with nothing
guarding their agreement. The boundary checkpoint path join was open-coded at
every consumer — several carrying an identical warning comment, with no
`BoundaryPolicy` accessor owning the rule — and each entry point parsed the
checkpoint from disk twice (layout sizing, then boundary load).

**Fix.** One `BoundaryStateRequirements` value (`cobre-sddp` `setup`) owns both
boundary-derived facts — presence and inflow-lag depth — derived together, so
they cannot disagree; one resolver (`resolve_boundary_state_requirements`,
`crates/cobre-sddp/src/policy/policy_load.rs`) produces it, replacing all four
open-coded folds. It rides the same three carriers as a single field rather
than a scalar-plus-flag pair, and the constructed `StudySetup` retains it, so
the boundary-cut load path reads the depth off the setup instead of re-parsing
the checkpoint. `BoundaryPolicy::checkpoint_path` (`cobre-io`) owns the path
join at every consumer. Byte-neutral: the same depth reaches
`resolve_state_layout` and the same presence reaches the terminal-mask gate.
The chosen shape is an opaque owning struct behind accessors (it carries an
inflow-lag field today, but as the single owner every family flows through, not
a bare threaded scalar); a new externally-authored state family adds a field
and accessor here. The near-isomorphic carrier trio itself is unchanged —
collapsing it is the setup-layer redesign above.

**Owner.** The setup / config owner.

#### Boundary state-family coupling channels are per-family bespoke

**What it is.** The question "how does an externally-authored boundary
future-cost function's coupling on a given state family reach the study's state
space?" is answered by a different bespoke mechanism per family. The setup
channel is now unified — the resolved requirements ride the carriers as one
generic `BoundaryStateRequirements` (the entry above) — but the WRITER and
manifest channels are still family-specific: the inflow-lag family carries a
Python writer argument and per-cut keyed field, a family-specific manifest
decoder (`boundary_cut_lag_depth`), a family-specific widening
(`widen_lag_state_depth`), and a family-specific writer helper
(`reserve_boundary_inflow_lag_slots`). The anticipated family reserves
post-horizon lanes from study config with calendar fan-out reconciliation.
Transit buckets reserve from study arc topology with boundary-gated terminal
unmasking and have NO widening path — a boundary bucket coupling the study's
topology does not reserve is dropped during reconciliation. That drop is now
SURFACED (the reconciliation report's `superset_summary` names the dropping
family and its count, and `dropped_source_slots` carries every dropped slot's
own identity and interval; `policy.boundary.strict` rejects the load outright
instead), no longer silent, so the remaining transit limitation is only the
absent widening path — which is intentional: a study cannot fabricate a transit
arc it never declared, and rejecting would break a legitimate superset boundary
source. Each new family under this shape still repeats the per-family manifest
decoder + reservation slot-body. The generic frame half-exists: the checkpoint
manifest already self-describes every slot
(`entity_type`/`entity_id`/`subindex`/`reference_date`/`interval_start`/
`interval_end` — no format change needed) and the load-side rebind dispatch is
already family-generic in frame. The setup half of the target is DONE (one
`BoundaryStateRequirements` rides the config carriers, on the typed
state-family enum's vocabulary), and the writer's family-INDEPENDENT core is
now extracted (`splice_reserved_state_block` owns the prefix/reserved/tail
splice + keyed-coefficient placement + alignment guards). What REMAINS for a
second authored family is its own reserved slot-body constructor and keyed
per-cut coefficient field. Per-family reconciliation SEMANTICS (widen vs
calendar fan-out vs reject) are genuinely irreducible and stay per-family; the
debt is the missing per-family slot-body wiring, not the shared mechanism.

**Owner.** The policy / setup owner.

**Trigger.** Before the next state family gains external boundary authoring
(the anticipated post-study-commitment import is the concrete queued case), or
with the setup-layer redesign — whichever lands first.

#### Nested risk-adjusted upper bound is an override, not an estimator arm

**What it is.** `ForwardBound` is the named dispatch for the training upper
bound, yet the nested CVaR estimator lives outside it: `sync_forward` computes
the risk-neutral bound (one full `allgatherv` plus compensated reduction), then
`apply_nested_cvar_ub` discards that result and re-gathers with a second
`allgatherv` (`crates/cobre-sddp/src/training/forward/stats_aggregation.rs`).
`ForwardBound::Exact`'s rustdoc documents the external override — the enum
documents behavior it does not own — and `SimulationWeighting`, documented as
mirroring `ForwardBound`, no longer covers the estimator space. Three
duplicate-owner smells sit in the same code path: the recursion re-implements
`EnumeratedPlan::walk_path` (the documented single owner of the root→leaf
walk) without its length assertion; the rank-partition counts/displs
arithmetic gains another implementation beside `RankDistribution::actual_per_rank`
(the declared owner) and the copy in `cut_sync.rs`; and the per-path stage
stride is derived from a solver-statistics vector's length instead of the
declared `num_stages` owner. The recursion also allocates its full working
state fresh every iteration — topology vectors that are pure functions of the
`EnumeratedPlan`, plus a fresh `RiskMeasureScratch` per interior node while
the `_into` scratch form exists unused — against the hot-path pre-allocation
rule.

**Owner.** The training owner.

**Trigger.** The next stopping-rule, risk-measure, or upper-bound-estimator
change — promote to a `ForwardBound` arm consuming `EnumeratedPlan` +
`RankDistribution` with precomputed topology before a new variant lands on top.

#### Admission predicate duplicated between setup gate and session override

**What it is.** The setup rejecter (`reject_gap_under_nonuniform_risk`) and
the session override condition answer "uniform effective CVaR under enumerated
forwards" with independent code: the rejecter open-codes the uniformity scan
that `uniform_effective_measure` (`convergence/risk_measure.rs`) implements,
and effective-risk-aversion is implemented separately in the setup predicate,
the session's inline match, and `RiskMeasure::effective`. A divergence admits
a gap rule whose override does not fire — silently reverting to the
risk-neutral end-of-horizon bound and manifesting as a spurious bound
crossover, not an error.

**Owner.** The training owner.

**Trigger.** Same as the estimator-arm promotion above (one consolidation).

#### Outflow row-pair entries are hand-mirrored

**What it is.** After the diversion-exclusion fix made the minimum- and
maximum-outflow rows symmetric, their entry blocks in
`crates/cobre-sddp/src/lp/builder/entries.rs` are two copy-shaped loops
gathering the identical turbine-plus-spill expression, differing only in
row-family base, slack accessor, and slack sign — the "both rows bind
turbine+spill, neither couples diversion" invariant is held by a comment
across two sites. The sibling rows builder already emits the same row pair
from a descriptor array (`fill_operational_violation_rows`,
`lp/builder/rows.rs`), so the better shape exists one file over; the merge is
byte-neutral because the template sorts entries before CSC assembly.

**Owner.** The LP-builder owner.

**Trigger.** The next outflow-row or entries-builder touch.

#### Opening-outcome aggregation takes an untyped projection role

**What it is.** `accumulate_opening_outcome` / `write_opening_outcome`
(`crates/cobre-sddp/src/training/backward/outcome_aggregation.rs`) accept a
bare `&CutStateProjection` at a call site where two role-distinct projections
are live (the child pool's layout and the required cut-generating parent's
`SuccessorSpec.cut_state`); passing the wrong one compiles and diverges
exactly in the reduced-projection configuration the intercept fix addressed —
against the projection module's own compile-fail typed-role discipline. Fix:
pass `&SuccessorSpec`, or a role newtype.

**Owner.** The training owner.

**Trigger.** The next backward outcome-aggregation or successor-spec change.

#### Minor residues (same wave, small)

- `IterationScratch`'s upper-bound buffers (`ub_path_weights`,
  `ub_stage_costs`) are lazily grown on first use while the struct doc
  promises allocation in `new` — size them in `new` beside their pre-sized
  siblings. **Owner.** Training. **Trigger.** Next scratch touch.
- `CutStateProjection` exposes only the fused `dot_trial_state`; the
  cut-selection sweep open-codes the gather loop. Add a gather-only sibling
  and define the dot over it. **Owner.** Training. **Trigger.** Next
  projection-surface touch.
- The two gap-rule rejecters re-evaluate the same predicates and the rejection
  message a sampled-forwards study sees is call-order-dependent (documented as
  deliberate). Optional single-match consolidation in `admission_gate`.
  **Owner.** Setup. **Trigger.** Next admission-gate change.

### External-scenarios-authoritative deferrals (2026-08-24)

#### Deterministic (σ = 0) AR(p > 0) external inflow stays rejected

**What it is.** Under the `External` scheme the external scenario file is the
authoritative source of a class's realized values, σ = 0 included, for load, NCS,
and AR(0) inflow. An AR(p > 0) inflow at σ = 0 remains **rejected**: a deterministic
autoregressive series would have to equal the model's own deterministic PAR output
at every stage — a whole-trajectory constraint of marginal value that the loader
cannot compute upstream. Both the SDDP loader and `cobre.io.validate` reject it with
that reason (no "inversion is undefined" phrasing).

**Owner.** The stochastic/formulation owner.

**Trigger.** A real deck needing a deterministic AR(p > 0) external inflow — then
admit it by validating the values against the deterministic PAR recursion.

#### `LoadModel` conflates physical load with its stochastic model

**What it is.** `LoadModel` is both "this bus has load" (physical) and "here is its
noise model" (stochastic-stats-derived), unlike NCS which separates
`NonControllableSource` (physical) from `NcsModel` (stochastic). This conflation is
why load-noise membership had to be unified into one `System` authority
(`load_noise_member_bus_ids`) consumed by every site, rather than read off a physical
registry the way NCS is. Splitting the physical/stochastic roles is the deeper
structural fix.

**Owner.** The core data-model owner.

**Trigger.** A dedicated data-model plan; do not attempt inside a feature ticket
(the split ripples every `&[LoadModel]` borrow).

#### Cross-path static-RHS contract not yet in `.claude/rules/sddp.md`

**What it is.** The stage-0 lower-bound static LP RHS must read the same
`PrecomputedNormal` moment source the runtime reconstruction uses
(`load_models_from_normal`) — the load analogue of the "lower-bound evaluation must
patch NCS" contract. It is a Voice-1 doc comment on the owning symbols + pinned by
tests, but not yet mirrored into `.claude/rules/sddp.md`.

**Owner.** The SDDP-rules owner.

**Trigger.** Next `.claude/rules/sddp.md` edit — add the contract beside the NCS one.

### Cleared (recorded so a future audit does not re-raise)

The computed-parameter resolver is not a god-function (cohesive); the water
block-mode fill duplication is deliberate-by-design (two distinct LP
formulations); and the cut-pool substrate, the shared solve-prep, the training
session, the captured-basis codec, and the entity-family writers/extractors are
positive references, not debt.

The 2026-08-19 fix-wave read additionally cleared: the projection-aware
intercept fix is complete (no unfixed positional-zip sibling — the three gather
axes `global_state_index` / `outgoing_column` / `incoming_column` are correctly
distinguished by design); the `state_space` config removal left no residue
(survivors are the deliberate reject test and the CHANGELOG entry — remaining
`state_space` matches are the unrelated `lp/indexer/state_space.rs` name
collision); stopping-rule admission is single-owner and well-homed
(`admission_gate`, one call site, exhaustively-destructured predicates); the
infrastructure-genericity gate stays green across the wave's edits; and the
boundary-checkpoint slot-reservation authoring path
(`reserve_boundary_inflow_lag_slots`) is correctly homed in cobre-sddp beside
the canonical manifest builder, with the generic cobre-io writer unchanged —
cleared for execution and homing only; its family-specific mechanism shape is
the per-family-channel entry above, not cleared.

## Deferred-debt register — 2026-09 quality evaluation (core-io and stochastic stations)

Two of the evaluation's eleven stations have passed their owner gate: the descent through
`cobre-core` + `cobre-io` and the descent through `cobre-stochastic`. Every ratified finding was
re-derived against the tree at the `develop` merge that followed the gates, then ranked into the
tiers below. The finding ledger, its ranking and the per-finding evidence live in the evaluation's
own plan directory (tracked on the evaluation branch); this section is the behaviour-described
mirror. Each open entry names an owner and a trigger; each fixed entry is recorded so a future
audit does not re-raise it.

### Fixed — user-visible correctness (2026-09-11)

- **Per-stage curtailment-penalty overrides for non-controllable sources reach the LP objective.**
  The stage LP priced curtailment from the source's declaration-time constant; the resolved
  per-(source, stage) penalty table was written but never read. The column fill now reads it, on
  the same path as the hydro, line and bus fills.
- **Every bound-override family rejects a row naming an undeclared study stage.** Five families
  dropped such rows silently at resolution; the thermal family tested a `[0, n)` position while
  resolution keys rows by declared id, which mis-admitted gapped or 1-based id sets. One
  table-driven rule in `crates/cobre-io/src/validation/semantic/block_bounds.rs` now covers all six
  by declared-id set membership. The non-controllable-source family keeps its referential check.
- **An invalid `simulation.scenario_source` fails at load and under `cobre validate`.** Config
  loading validated only the training source; the CLI and Python validate paths mirrored that
  gap while `cobre run` failed later at setup. `validate_config` resolves both sources once.
- **Entity classes sampled out of sample draw independent noise streams.** Every class sampler
  was seeded from the same forward seed with no class tag, so a deck with two or more classes
  out of sample drew bit-identical noise for the k-th entity of each class under every noise
  method. The load and non-controllable-source classes now derive their seed from the root seed
  and their class tag; the inflow class keeps the root seed so inflow-only decks reproduce
  bit-for-bit.
- **Policy checkpoint writes are atomic and a rewrite cannot mix runs.** Payloads, the manifest
  and both dictionary CSVs go through the crate's atomic writer. A rewrite removes the previous
  manifest, then every previous payload file, before writing — the reader enumerates the payload
  directories, so a rerun into the same output directory with fewer pools or with states export
  off would otherwise read the earlier run's files back.

**Owner.** The `cobre-io` validation and output owners and the LP-builder owner (as executed).
**Trigger.** None — done.

### Fixed — forward-sampler hot path under out-of-sample QMC/LHS and wide correlation groups (2026-09-12)

- **No forward draw rebuilds scenario-invariant sampling state.** Under `scheme: out_of_sample`
  every Sobol draw rebuilt the direction matrix and scramble parameters on the heap, every Halton
  draw re-ran the prime sieve and rebuilt its scramble tables, and every LHS draw reshuffled the
  full stratification permutation set, quadratic in the scenario count per stage. Each driver now
  builds one table per entity class and per distinct (noise group, noise method) pair once per
  iteration and hands a shared reference down through the sample request; a draw reads its table.
  The direct generators survive only as test-only reference oracles that pin the precomputed
  paths bit-for-bit, and the table records the iteration and scenario count it was built for so a
  debug build fails loudly on a stale table.
- **The correlation applier is one code path with positions resolved at construction.** The
  decomposition takes the canonical entity order and class dimensions at build, parses each
  group's entity class into a closed enum, and resolves per-class positions once; the full-vector
  twin, the linear-scan fallback and the differential oracles that existed only to pin them are
  gone. A group of any width is correlated from caller-owned scratch that the per-worker scratch
  struct sizes at twice the noise dimension.
- **An out-of-sample class wider than the Sobol direction table is rejected when the tables are
  built.** Moving the Sobol construction ahead of the draw had turned the graceful dimension error
  into a panic; the table build is fallible and both drivers propagate the error.
- **A counting-allocator guard pins the draw.** One integration binary asserts zero heap
  allocations across Sobol, Halton and LHS stages with a correlation group wider than the stack
  fast path. It must be the only test in its binary, so it includes the shared fixture builders
  file directly rather than the aggregator module.

**Owner.** The `cobre-stochastic` sampling and tree-noise owners; the training and simulation
state structs own the tables (as executed). **Trigger.** None — done.

### Fixed — latent footguns, one change each (2026-09-14)

- **The builder owns canonical order.** `SystemBuilder::build` assigns each stage's index from its
  own sort instead of the loader; the three scenario model tables are validated as canonically
  ordered at construction (a new validation error; `System::with_scenario_models` is now fallible)
  rather than sorted, so every deck the loader emits is byte-identical. The canonical key is stated
  once, on the builder, and every resolver and parser doc points at it. Closing this exposed two
  pre-existing violations of the order contract: the pre-build lag-transition precompute in the
  inflow-seeding validation read the stage index the loader no longer writes (it now derives the
  window from slice position, which also corrects a latent over-skip when pre-study stages exist),
  and the partial-estimation path appended pre-study rows unsorted (now sorted). Both carry regression
  tests through the real parser and estimation entry points.
- **The input-file registry is keyed, not positional.** One enum keys the structural file table and
  the presence manifest; a reordering is a compile error or a registry-test failure, never a silent
  flag misassignment. The manifest exposes one read accessor.
- **One hydro penalty type.** The per-stage twin is removed; the resolved table stores the entity
  type and the sixteen-field copy is a move. The forward-hydro-production clause on the turbined-cost
  field now says it is not enforced by validation, which is true.
- **The sampler's class identity is typed end to end.** The factory carries the entity-class enum,
  the historical-replay gate matches a variant, and the load and non-controllable-source forward seeds
  derive from the enum's wire label so both pinned seed constants are unchanged. Each class's scheme
  and library are one per-class source inside the factory, with no historical variant for load and
  non-controllable sources; every missing-source diagnostic keeps its text and raise order.
- **The stage-0 derived seed travels as one aggregate** through the three standardizers and their two
  library builders, removing the adjacent same-typed slice hazard; every argument-count suppression
  stays with a rationale that is true.
- **The shared noise point spec names its key for the slot it fills** (a noise group on the forward
  path, a stage on the opening-tree path).
- **An unsupported forward noise method warns once per class at sampler construction**, naming the
  affected stages; the draw arms are silent fallbacks and allocate nothing.

**Owner.** The `cobre-core` builder owner, the `cobre-io` validation owner and the `cobre-stochastic`
sampling owner (as executed). **Trigger.** None — done.

### Fixed — a shared test-fixture surface per type owner (2026-09-15)

- **Every crate that owns shareable fixtures exposes them behind its own `test-support`
  feature.** The gate is `#[cfg(any(test, feature = "test-support"))]`, enabled by consumers as a
  dev-dependency feature, so a fixture has one definition where its type lives: the entity, stage
  and penalty builders (a spec struct with `Default` plus a `make_*` constructor, parameterised on
  the axes the former copies varied), the bit-exact scalar comparators and the numeric helpers in
  `cobre-core`; the Parquet and JSON writers, the minimal-case corpus, the validation-phase
  fixtures, the stats-parser template and the output-context fixtures in `cobre-io`; the
  opening-tree, season-map and inflow-model builders in `cobre-stochastic`. The `cobre-stochastic`
  integration binaries share one `tests/common` prelude that re-exports from those surfaces.
- **The struct-specific bit comparators are exhaustive by construction.** Each destructures both
  sides with a full field pattern and no rest, so a field added to a bounds struct fails to compile
  until the comparator names it; the scalar and `Option<f64>` comparison is one shared helper.
- **The shared Parquet extractors carry their own contract tests.** The happy path, the
  missing-column message and the wrong-type message are pinned once at the owner rather than
  incidentally in consumer test modules.
- **The order-invariance, reproducibility and golden-value pins relocated and none changed.** The
  declaration-order-invariance parser tests, the reproducibility suite, the sample-average golden
  value and the forward-sampler golden arrays pass with their constants unedited. The one declared
  coverage addition on the stats parsers is a zero-standard-deviation acceptance case for inflows;
  the out-of-range case is specific to availability factors and was recorded as not applying to the
  inflow and load parsers rather than copied to them.
- **Consolidation was a refactor, not a coverage change.** Every duplicated fixture that folded
  kept its callers' assertions; the test listing moves only by the declared additions and by the
  permutation-helper tests that the solver-linking test aggregator no longer inherits into every
  binary.

**What the convention has not reached.** `crates/cobre-sddp/tests/common/` still holds its own
fixture directory rather than collapsing into that crate's `test-support` surface, and
`crates/cobre-python` does not yet dev-depend on it; both remain the open half of the convention in
`docs/design/testing-architecture.md`. The `cobre-stochastic` in-`src` unit-test modules still keep
local identity-correlation builders where the integration binaries share one.

**Owner.** The `cobre-core`, `cobre-io` and `cobre-stochastic` owners (as executed).
**Trigger.** None — done.

### Fixed — the dead-surface sweep (2026-09-15)

- **Every public item the workspace ships has a production reader, or a recorded owner and
  milestone.** The unwired public error variants are gone: the validation variants no builder path
  produced, the cross-reference load-error variant no loading path produced, the stochastic-error
  variants no code path constructed, and the single-inhabitant PAR warning taxonomy whose one
  caller discarded it. The never-read output columns are gone: the per-iteration setup timings
  that reached no file, and the written-partition inventory together with the rank exchange that
  merged it. The unread projections are gone: the dictionary writer's configuration argument, the
  severity-defaulting method, the aggregate scenario-loading entry point, the case-relative
  scalar-parameter loader and raw season-map builder, the free postcard serializers, the dead
  penalties bundle field, the f32 vector decoder, the pipeline projection wrappers, the
  population-statistics arm of the Welford accumulator, the bare season-map forwarders and the
  sort-direction knob with one reachable setting. The reader-less bus adjacency topology is deleted
  with its constructor's unread bus argument.
- **The broadcast payload no longer carries content-determined derivations.** The cascade
  adjacency is rebuilt on receipt from the entity slices already in the payload, and a bit-equality
  round-trip guard pins that the rebuild is exact.
- **The read-side reader prologue has one owner.** Every Parquet input parser that maps open and
  build failures to the loading error opens through one crate-internal helper; the
  convergence-output readers, which map to the output error type or to an option, are named
  exclusions rather than absorbed into a wider helper.
- **The write-side ensure-parent-then-write sequence has one owner.** The parent-directory helper
  lives beside the atomic writers, its inline copies fold onto it, and an atomic batch writer
  performs the sequence for the writers whose shape matches; the writers that thread their own
  configuration or create the output directory themselves are named exclusions.
- **The copied extension extractors fold onto the shared pair**, so a required column missing from
  the hydro-geometry, energy-productivity or tailrace inputs reports the same wording every other
  tabular input already produced.
- **The load- and non-controllable-source factor resolvers are one generic routine** over a private
  kind marker, with the public entry points unchanged.
- **PAR parameter validation returns the fatal check directly**; the zero-standard-deviation rule,
  its message and its fields are unchanged.
- **The opening solve order is always descending by key**, ties broken by ascending canonical
  order — the same permutation every run produced before.

**Owner.** The `cobre-core`, `cobre-io` and `cobre-stochastic` owners (as executed).
**Trigger.** None — done.

### Fixed — documentation and rule-table drift (2026-09-15)

- **The uniform entity-table stride has one owner.** The resolved-bounds table indexes every
  non-thermal family through one private helper that carries the same stride assertion the
  thermal helper already had, so a layout change is one edit rather than a fourteen-site sweep.
- **The System wire payload serializes in content-determined order, stated once.** The last
  unordered map on the payload (the per-stage discount-rate overrides) is key-ordered, the rule
  lives on the `System` doc, and a test pins byte identity across insertion orders.
- **The semantic rule tables match the code.** Two rules the parse layer already enforces are
  retired from the semantic layer with their unreachable branches and tests; a ghost row for a
  field that no longer exists is retired; seven checks that ran without a row are tabulated;
  three drifted rows are reworded; and every dispatched check carries its rule number on its
  first doc line so a row greps to its implementation.
- **The bound-override family has one rule set.** Generic-constraint bounds join the per-family
  descriptor table and so get the per-column duplicate rule and the same error kind as the other
  six families (a deck-visible change recorded in the CHANGELOG); the referential module keeps
  only dangling-id and non-family-shaped checks, and the two non-controllable-source value checks
  that the parsers already enforced are gone.
- **The dangling-reference message has one owner.** A descriptor and one emit helper carry the
  same-shaped sites; the differently shaped ones stay explicit.
- **Every output schema is declared in one module and listed once.** The registry drives the
  axis-spelling gate, which now really covers the whole family, and the variables dictionary.
- **Each simulation entity family is declared once**, with its declared-predicate and batch
  adapter; directory creation and per-scenario writes iterate the same table, and the rule that a
  declared family's directory exists from construction while a partition needs a non-empty
  payload is stated once.
- **Docs tell the truth about ownership and layering.** The result-writer entry point documents
  what it writes and what the callers write; the stage lag-transition type no longer cites an
  engine-private function; the PAR module docs describe their layout without engine phase names.

**Not done here, by design.** The lag-transition type's crate home is a layering decision for the
alignment station, and consolidating the output orchestration into one owner is its own entry.

**Owner.** The `cobre-core`, `cobre-io` and `cobre-stochastic` owners (as executed).
**Trigger.** None — done.

### Deprioritized (recorded, not scheduled)

Setup-time performance items below the sweep threshold; the generalization-alignment holds, which
the alignment station adjudicates before any code; and the test-corpus sweep, which follows the
fixture surface above.

## Deferred-debt register — 2026-09 quality evaluation (remaining stations)

This section mirrors the fixed items from the 2026-09 quality evaluation's remaining
stations — the SDDP engine, the CLI/Python facade, the solver backends, the comm
layer and the build/CI surface. Every item was re-derived against the live tree
before it was fixed; the finding ledger and per-finding evidence live in the
evaluation's own plan directory, and this section is the behaviour-described mirror,
recorded so a future audit does not re-raise these items.

### Fixed — validate-time rejections replace mid-run and silent failures (2026-09-21)

- **The `--json` error vocabulary has one owner across both front ends.** One
  `LoadError`→kind map in `cobre-io` is called by the CLI and the Python
  `cobre.io.validate` binding, every CLI early return under `--json` routes through
  the shared `emit_validate_json` (carrying `error.phase` for `ParseError` /
  `SchemaError`), and the standalone `CaseValidationError` kind is retired with no
  alias — a documented `--json` contract change.
- **The boundary-preparation reject is a first-class preparation phase.**
  `PrepPhase` gains a `Boundary` variant, so both front ends route a boundary
  reject through the shared `PrepPhase` / `prep_phase_metadata` (its own metadata
  row) and `validate --json` emits the error object; the enum doc states no variant
  count.
- **The scalar-parameter table is a `StudySetup` constructor input and its
  resolution gaps fail loud.** The three `ResolvedParametersError` classes
  (`MissingSeason`, `PerStageBlockCoverage`, `MissingSpecificProductivity`) are
  raised at the LP-build / admission site, so both `cobre validate` and Python
  `cobre.io.validate` reject an unresolvable generic-constraint scalar parameter
  instead of exiting cleanly and failing later at run; `ResolvedParameters::get`
  stays infallible.
- **An enumerated-traversal study that also configures dynamic cut selection is
  rejected at setup.** A typed admission-gate arm refuses `Traversal::Enumerated`
  combined with dynamic cut selection beside the existing enumerated preconditions,
  with a named regression and a `.claude/rules/sddp.md` contract entry, rather than
  silently exercising an untested cut-eviction path.
- **A negative FPHA discretization count is rejected at the input boundary.**
  `cobre-io` validates the four count fields (`volume_discretization_points`,
  `turbine_discretization_points`, `spillage_discretization_points`,
  `max_planes_per_hydro`) non-negative when present, so a declared negative count
  fails validation instead of wrapping past the `< 2` / `< 1` grid guards into
  `build_grid`.

**Owner.** The `cobre-sddp`, `cobre-io`, `cobre-cli` and `cobre-python` owners (as
executed). **Trigger.** None — done.

### Fixed — one owner for each duplicated front-end computation (2026-09-21)

- **The run-phase plan is one owned value both front ends consume.** The engine
  answers the run-phase plan once and the CLI and Python entry points consume it,
  replacing the two non-equivalent simulate-arm gate copies; the per-front-end
  no-op rendering is kept.
- **The solver-stats log-to-totals fold has one home.** `solver_stats.rs` in
  `cobre-sddp` folds `&[SolverStatsLogEntry]` with the rank filter as an argument,
  called by both the CLI and Python; the `total_lp_solves` caveat is a doc line
  pinned by a named regression.
- **The simulation entity-family names are declared once.** The Python simulation
  readers iterate the `cobre-io` family declaration instead of a hand-kept copy, so
  no second enumeration exists to drift.
- **The convergence-output reader keys off the schema.** The Python convergence
  reader asserts its keys equal the declared schema fields rather than a fixed
  hand-listed column set.

**Owner.** The `cobre-sddp`, `cobre-cli` and `cobre-python` owners (as executed).
**Trigger.** None — done.

### Fixed — LP-builder anticipated-commitment fill has one walker (2026-09-21)

- **One ring-residue walker drives the anticipated-commitment fill.** A single
  `lp/builder` walker (via `ring_index`) drives the anticipated row fill, the column
  fill and `build_anticipated_slot_row_pos`, byte-neutral against every golden.
- **The anticipated-commitment LP columns resolve through typed accessors.** Typed
  `commitment_hold_incoming_col` / `commitment_hold_outgoing_col` resolvers on
  `StateSpace` replace the untyped column recompositions in
  `simulation/extraction.rs`, byte-neutral via an equivalence test.

**Owner.** The LP-builder and simulation owners (as executed). **Trigger.** None —
done.

### Fixed — public Rust-API surfaces removed in a licensed break (2026-09-21)

- **Unused and superseded public items are gone.** Removed in one licensed
  public-API break: `CutManagementConfig::warm_start_cuts` (with its two production
  literals, the `train_inner` reset and its tests); the `Col` / `Row` newtypes and
  their round-trip tests, dropped from the `lp/indexer` re-export; `FphaRowRange`
  and its smoke test; the `pub use policy::orchestration` crate-root re-export (with
  callers moved to the owning module path); the superseded single-pool cut-sync
  methods on `CutSyncBuffers`; and the CLP hot-start acquire half
  (`cobre_clp_mark_hot_start`, `cobre_clp_solve_from_hot_start` and their safe
  wrappers), with the release half kept. No deck, CLI output or Python package
  output is affected.

**Owner.** The `cobre-sddp`, `cobre-comm` and `cobre-solver` owners (as executed).
**Trigger.** None — done.

### Fixed — rustdoc and module-doc drift (2026-09-21)

- **The infrastructure-crate docs read in the generic register.** The `cobre-solver`
  doc sites are reworded to describe solver-handle properties without
  algorithm-specific names, and the crate README's HiGHS feature-gating contract is
  corrected (the `BasisStatus` mapping is unconditional).
- **The stale rustdoc lines are corrected.** The
  `CutManagementConfig::warm_start_cuts` line and the `FphaRowRange` /
  `BlockGrid::advance_fpha_base` lines are corrected before the surfaces they
  described are removed, and the `claim_scatter.rs` module-doc consumer list is
  corrected to name every importer.

**Owner.** The `cobre-solver` and `cobre-sddp` owners (as executed). **Trigger.**
None — done.

### Fixed — byte-neutral setup-path and output-path reductions (2026-09-21)

- **The setup-path scans and allocations are bounded.** The inflow-history and
  observation joins bucket by hydro id in one pass and borrow the contiguous
  subslice instead of copying per occurrence; the distance-matrix fill is symmetric
  with hoisted per-start buffers; the per-block reservation regrowth is removed so
  each branch reserves the exact product; the coverage-gated observation join walks
  one forward-sweep cursor; and `long_term_mean_inflow` bounds its scan to the total
  history rows once. Each is byte-neutral against the parity goldens.
- **The output writers allocate less.** The checkpoint write path consumes the
  serializer's finished bytes without a `to_vec()` copy; one lazily-initialised
  Arrow schema per output is cloned by the batch builders with `WriterProperties`
  resolved once; `delta_to_stats_row` carries its phase as a `&'static str`
  rather than an allocated `String`; and `build_iterations_columns` uses the
  builder-with-capacity idiom instead of intermediate vectors.
- **The Parquet writer configuration is frozen to internal constants.** The
  `ParquetWriterConfig` values are an internal constant set and the threaded
  `&ParquetWriterConfig` machinery is collapsed; no user-facing compression knob is
  exposed.

**Owner.** The `cobre-sddp`, `cobre-io` and `cobre-core` owners (as executed).
**Trigger.** None — done.

## Audit-evidence

The following mechanical checks were run against the tree at the time this
document was authored. Each command is regenerable — re-run it to get the
current state rather than trusting the prose below as a frozen count.

### Clippy, full feature set

```
cargo clippy --workspace --all-targets \
  --features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support" \
  -- -D warnings
```

Zero warnings, zero errors, across the full declared feature set (including
`mpi`, built against the local MPICH toolchain).

### `#[allow(...)]` census

Regenerate with:

```
grep -rn '#!\?\[allow(' crates/*/src --include='*.rs'
```

Every hit falls into one of three classes, and none is plan-dead-unconsumed:

- **Load-bearing.** Refactor-decision lints on `.claude/rules/comments.md` D4's
  closed list (`too_many_arguments`, `too_many_lines`, `type_complexity`,
  `dead_code`, `unused_*`) plus borrow-checker workarounds each carry a
  `// Rationale:` comment naming the non-obvious choice the lint would otherwise
  flag — the majority of the census. Numeric-cast lints
  (`cast_possible_truncation`, `cast_precision_loss`, `cast_sign_loss`,
  `cast_possible_wrap`), `needless_pass_by_value`, and the remaining pedantic
  openers (`struct_field_names`, `implicit_hasher`) sit outside that closed list:
  their suppressions are still load-bearing because CI's zero-warning bar needs
  them, but D4 mandates no rationale on them, so a bare opener there is a
  consistency preference rather than a rule violation.
- **Reserved-seam (Voice 4).** `dead_code` attributes each paired with a
  comment naming what will consume the item once a specific reader lands (the
  water travel-time topology and Lipschitz entries above are examples; several
  more of the same shape exist in `lp/builder/{layout,template}.rs`,
  `production/fpha_fitting/`, and `cobre-solver`'s FFI binding modules, the
  last being the standard `#![allow(dead_code)]` convention for a raw
  1:1-mapped C binding surface).
- **Symmetry-or-test-retention.** `unwrap_used` / `expect_used` / `panic` /
  `float_cmp` clusters on `#[cfg(test)]` modules (test code is exempt from the
  library's `unwrap_used = "deny"`), plus a handful of fields/functions kept
  for API symmetry and exercised only by unit tests (each with a `// Rationale:`
  naming the symmetric counterpart or the asserting test).

`#[allow(deprecated)]` sites (`crates/cobre-stochastic/src/{lib.rs,par/mod.rs,par/evaluate.rs}`)
are a fourth, narrower class tied to the deprecated-with-fallback surface;
verifying that surface is out of scope for this sweep (a dedicated ticket owns
the clean-break gate).

### `cargo machete`

```
cargo machete
cargo machete crates/cobre-python
```

Both report no unused dependencies.

### Graph-shape-predicate grep

```
grep -rn '\bis_chain\b' crates/cobre-sddp/src --include='*.rs'
grep -rn 'nodes\.is_empty()' crates/cobre-sddp/src --include='*.rs'
grep -rn 'graph\.is_none()\|graph\.is_some()' crates/cobre-sddp/src --include='*.rs'
```

`is_chain` returns zero hits — no such predicate exists; the engine is
node-native and does not special-case chains by name. The remaining hits:

- `policy/policy_load.rs` (`compare_graph_manifest_identity`,
  `resolve_warm_start_counts`) — policy-load validation and boundary-injection
  pool resolution, both one-time at load time, not per-iteration.
- `setup/node_graph.rs`'s `graph.nodes.is_empty()` — a one-time,
  setup-construction-time choice between `build_chain_node_graph` and the
  declared-graph builder, both producing the same `NodeGraph` representation
  the rest of the engine consumes uniformly afterward; not re-checked inside
  the forward/backward training loop.
- `setup/node_graph.rs`'s `frontier.next().is_none()` /
  `candidates.next().is_none()` / `parent[succ.child].is_none()` —
  `debug_assert!` invariant checks inside diagnostic helpers
  (`frontier_node`, `node_parent`, `build_parent_map`), not `if`-forks.
- `training/forward/enumerated.rs`'s `parent[node].is_none()` — root-vs-interior
  node detection inside the forward worker loop. This is universal graph-walking
  logic present in any DAG (every node either has a parent or is a root); it is
  not a chain-vs-tree shape fork and does not distinguish a chain study from a
  branching one.

None of these is an `is_chain`-style special case reaching training dispatch.
