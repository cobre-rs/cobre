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
- The **deferred-debt register**: known debt and deferred follow-ups, not
  limited to unwired seams — architectural, numerical or operational (for
  example log output repeated per process, or solver timing that decides
  whether a run completes) — tracked alongside this one.

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

### `policy.checkpointing.compress` and `policy.checkpointing.store_basis`

**What it is.** `crates/cobre-io/src/config/policy.rs` declares both keys on
`CheckpointingConfig`. Both are loaded, validated and schema-exported, and
nothing reads them. Periodic and final checkpoints are written uncompressed,
and they store bases whenever the run captured them. The other three
`policy.checkpointing` keys, `enabled`, `initial_iteration` and
`interval_iterations`, resolve through `Config::checkpoint_schedule` into the
schedule the periodic checkpoint writer consumes, so they are not reserved and
are absent from this register.

**Owner.** The training owner.

**Consuming milestone.** Periodic checkpointing needs smaller or cheaper writes
(`compress`), or a run needs checkpoints without stored bases (`store_basis`).

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
the paired soft `≥` rows live in `lp/builder/rows.rs`. They
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

### Python policy-write refusal still classified by message prefix

**What it is.** `cobre.write_policy_checkpoint` (`crates/cobre-python/src/policy.rs`)
reports its identity refusal, an `SddpError::PolicySoftwareMismatch`, as a message
with the `POLICY_VALIDATION_ERROR_PREFIX` prefix, and `message_prefix_to_pyerr`
(`crates/cobre-python/src/errors.rs`) turns that prefix into
`PolicyIncompatibleError`. That is the class the error's `ErrorClass` gives, so the
two routes agree today, but the class follows the text rather than the error.
Routing it through `ErrorSource::Sddp` would make the class independent of the
message and leave the prefix branch with no producer.

**Owner.** The Python bindings owner.

**Trigger.** A second refusal joins the writer, the writer's error text changes, or
the next change to the bindings' exception routing.

### Python raises SolverError for internal faults

**What it is.** Errors of class `ErrorClass::Internal` (communication failure,
wire-format mismatch, basis-shape mismatch) raise `cobre.errors.SolverError`, while
`cobre run` reports them as internal faults and `cobre.errors.InternalError` exists
for that meaning. Mapping them to `InternalError` changes a public exception class.

**Owner.** The Python bindings owner.

**Trigger.** A Python caller needs to tell a software or environment fault from an
LP failure, or the next revision of the exception hierarchy.

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

### Duplicate penalty override rows merge in file order

**What it is.** No validator rejects two rows with the same
`(entity_id, stage_id)` in a `constraints/penalty_overrides_*.parquet` file;
the module doc of `crates/cobre-io/src/constraints/penalty_overrides.rs` lists
that check as deferred. The parsers sort rows stably by that key, and
`resolve_penalties` (`crates/cobre-io/src/resolution/penalties.rs`) applies
them in that order, so for a field two such rows both set, the row later in the
file wins. A hydro row's symmetric evaporation or withdrawal cost also sets each
matching directional cost the row leaves unset, so a later row's symmetric
cost replaces an earlier row's directional cost. Reordering duplicate rows can
therefore change the resolved costs.

**Owner.** The input-validation owner, for a uniqueness rule on the override
key like the one the bound-override files have.

**Trigger.** A case is found with duplicate penalty override rows, or the
validation rule table next gains an input-uniqueness rule.

### Exported schema titles and `$defs` names are Rust type names

**What it is.** `cobre schema export` takes each schema's root `title` and
every `$defs` key from the Rust type that reads the file (`RawHydroFile`,
`RawHydro`, `TrainingConfig`), because `generate_schemas`
(`crates/cobre-io/src/schema.rs`) calls `schemars::schema_for!` on the
deserialization types. The descriptions are written for case authors; the
titles and `$ref` targets are not.

**Owner.** The cobre-io input-schema owner.

**Trigger.** An editor integration or schema consumer shows `title` or `$ref`
names to case authors.

### Python run reports a checkpoint write refusal as `CaseIoError`

**What it is.** Python run's training-output writer (`write_training_outputs`
in `crates/cobre-python/src/run.rs`) flattens checkpoint write errors into a
`POLICY_CHECKPOINT_ERROR_PREFIX` message. A write-time
`OutputError::ForeignEntry` therefore raises `CaseIoError` there, while the
CLI exits 1 and `cobre.write_policy_checkpoint` raises `ValidationError`.

**Owner.** The Python bindings owner.

**Trigger.** Python's training-output writes return typed errors.

### Energy-equivalent penalty ordering

**What it is.** The load-time penalty-ordering warnings
(`check_penalty_ordering`, `crates/cobre-io/src/validation/semantic/scenarios.rs`)
compare only costs that share a unit. The documented priority ordering
compares every cost as an energy-equivalent $/MWh, which turns a $/hm³
storage cost or a $/(m³/s·h) flow cost into $/MWh through the plant's
accumulated productivity. Validation runs before any productivity is known,
so the storage-violation and filling-target costs are not checked against
deficit, the flow-violation costs are not checked against deficit, and the
flow-versus-resource check compares the costs of different plants as raw
numbers. The check reads each plant's resolved costs, before any stage
override: stage-tier `penalty_overrides_hydro` values, which can reprice any
of these costs for a stage, are never ordering-checked. The `cobre-docs`
penalty-system and error-code pages still list the retired
storage-versus-deficit warnings and describe the checks as raw comparisons of
every tier.

**Owner.** The input-validation owner; the `cobre-docs` methodology owner for
the two pages.

**Trigger.** An energy-equivalent ordering check is requested, a setup
stage that already holds each plant's accumulated productivity gains a
validation pass, or a stage override inverts the ordering in a reported
study. For `cobre-docs`, the next revision of the penalty-system page.

### Writer path literals repeated by the output-file table

**What it is.** `OUTPUT_FILES` in `crates/cobre-io/src/output/file_registry.rs`
names the path of every registry Parquet file, but no writer reads it: each
joins its own path literal. The writers are in cobre-io (for example
`write_paths` and `write_dictionaries`), in `write_training_outputs`
(`crates/cobre-cli/src/commands/run/outputs.rs`) and its peer in
`crates/cobre-python/src/run.rs`, and in `export_stochastic_artifacts`
(`crates/cobre-sddp/src/policy/orchestration.rs`). Only the Hive
simulation-family rows are checked against their writer by a test, through
`simulation_family_subpaths()`; every other row was checked once, by symbol,
when the table was written. A writer path that changes without its row is not
caught.

**Owner.** The output data-model owner.

**Trigger.** A writer's output path changes, or a consumer of the exported
output registry checks the registry paths against a run.

### Per-phase forward-seed carriers

**What it is.** The out-of-sample root seed reaches `ForwardSamplerConfig::forward_seed`
from two carriers. The training phase reads `StochasticContext::forward_seed()`, which the
context stores at build time from the training scenario source and never reads itself. The
simulation phase reads `SimulationConfig::forward_seed`, which
`StudySetup::from_broadcast_params` passes to `resolve_phase_configs`. Every other per-phase
sampler input (the class schemes and the scenario libraries) rides on `TrainingContext`, set
by `training_ctx()` and `simulation_ctx()`. Moving both seeds onto `TrainingContext` and
dropping the context's stored seed touches every `TrainingContext` literal and every
`build_stochastic_context` call.

**Owner.** The training and simulation pipeline owner.

**Trigger.** A change adds another per-phase sampler input, or groups `TrainingContext`'s
per-phase fields; the two seeds then move with them.

### Split exit codes in the pre-training export and simulation-outcome reconciles

**What it is.** `run_pre_training` (`crates/cobre-cli/src/commands/run/setup.rs`)
and `run_simulation_phase` (`crates/cobre-cli/src/commands/run/simulation.rs`)
reconcile a bool through `cobre_sddp::reconcile_global_ok`. The failing rank
returns its own `CliError`, while every peer returns `CliError::Internal`
(exit 4). Under MPI each rank then aborts with its own code, and the launcher
reports whichever abort lands first. The post-training reconcile already
agrees the failing rank's code (`agree_post_write` in
`crates/cobre-cli/src/commands/run/graceful_stop.rs`, and
`CliError::for_peer_failure`). The fix is a code-carrying `Max` reduction for
both, with a peer error built from the agreed code and a message naming the
phase. The simulation outcome can fail on any subset of ranks, and its local
result combines the drain thread and `simulate()`, so it needs its own peer
message, and its peer-failure test changes with it.

**Owner.** The cobre-cli run-orchestration owner.

**Trigger.** The next change to either reconcile, or a job script or scheduler
that keys on the MPI launcher's exit code for a pre-training export or
simulation failure.

### Module-doc column tables that repeat the output column descriptions

**What it is.** The module docs of `crates/cobre-io/src/output/stochastic.rs`
and `crates/cobre-io/src/output/hydro_models.rs` carry column tables (column,
Parquet type, description) for the `stochastic/` Parquet exports and for
`hydro_models/fpha_hyperplanes.parquet`. `description_for` in
`crates/cobre-io/src/output/dictionary.rs` owns those column descriptions, and
the schema functions in `crates/cobre-io/src/output/schemas.rs` own the
types, so each table is a second copy that no test compares. Resolution
options for an owner pick: delete the tables and point the module docs at the
dictionary, or keep only what the dictionary does not state, such as row
order and the round trip with the matching input file.

**Owner.** The output data-model owner.

**Trigger.** A column of one of these schemas is added, renamed or redefined,
so that the same edit is needed in the dictionary and in a module doc.

### Training metadata `max_iterations` reports the first `iteration_limit` rule

**What it is.** `training/metadata.json` records `configuration.max_iterations`
(`MetadataConfiguration`, `crates/cobre-io/src/output/manifest.rs`) as the
limit of the first `iteration_limit` rule in `training.stopping_rules`
(`extract_max_iterations`, `crates/cobre-io/src/output/results_writer.rs`).
The run's iteration budget is the largest such limit
(`max_iterations_from_rules`, `crates/cobre-sddp/src/setup/mod.rs`), which
sizes the cut pool and ends the training loop. With one `iteration_limit`
rule the two agree; with several they can differ.

**Owner.** The output-contract owner.

**Trigger.** A supported case lists more than one `iteration_limit` rule, or
a consumer of `training/metadata.json` reads `max_iterations` as the run's
iteration budget.

### Slurm srun --mpi=pmix signal forwarding is unverified

**What it is.** Slurm `srun --mpi=pmix` signal forwarding is unverified; only
MPICH Hydra was probed. The MPI release README, written by the
`Package archive` step of `.github/workflows/release-mpi.yml`, tells users to
launch the ranks with `srun --mpi=pmix` so that `#SBATCH --signal=TERM@<lead>`
reaches them, but the graceful stop was reproduced only under `mpiexec`
(MPICH Hydra) with every rank signalled directly. No CI job launches ranks
with `srun`: the Slurm harness (`tests/slurm/run-tests.sh`) runs `mpiexec`,
inside `sbatch` jobs for its cluster cases, and has no `--signal` case.

**Owner.** The HPC build / deployment owner.

**Trigger.** A Slurm-launched graceful stop that behaves differently from the
local `mpiexec` reproduction, or the first CI job that launches ranks with
`srun`.

### Policy-path checks compare paths lexically

**What it is.** The output-directory and cleared-directory checks on
`policy.path` (`PolicyConfig::check_dir`, `crates/cobre-io/src/config/policy.rs`)
compare paths lexically. A symbolic link at the policy path itself is checked
through its recorded target. A `policy.path`, output directory or link target
can still reach the output directory or a cleared directory
through a symlinked parent component, or by another spelling of the same
directory. Such a path passes load-time validation. If that directory holds cobre's outputs,
the checkpoint writer refuses it with its foreign-entry error at its first
write. If the path lies inside a tree a run removes whole, the run-start clear
deletes the policy directory with that tree.

**Owner.** The cobre-io output owner.

**Trigger.** A user reports a symlinked output tree, or load-time
canonicalisation becomes possible without creating directories.

### Policy directory in a directory a run writes into but never clears

**What it is.** The cleared-directory guard covers only the directories a run
clears before writing (`cleared_output_dirs`, `crates/cobre-io/src/output/mod.rs`).
A `policy.path` naming a directory that cobre writes into but never clears,
such as `stochastic/`, `training/dictionaries/` or `training/timing/`, passes
both pre-training checks on a fresh output directory. The first checkpoint
write, periodic or final, then refuses it with the foreign-entry error, after
the training it was meant to save has run. Nothing is deleted, but the
training time is lost. A rerun into the same output directory is refused
before training, because the directory then holds cobre's files.

**Owner.** The cobre-io output owner.

**Trigger.** A user reports a training run lost to this refusal, or cobre-io
gains one registry of every path a run writes, from which this guard can take
the written directories as it takes the cleared ones.

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

#### Nested risk-adjusted upper bound is an override, not an estimator arm — RESOLVED

**What it was.** The nested CVaR upper bound was a session-side override that
discarded `sync_forward`'s risk-neutral result and gathered the forward costs a
second time. Around it sat duplicate owners: a second root→leaf walk without the
walk's length assertion, another copy of the rank-partition arithmetic, and a
path stride read from a solver-statistics vector. The recursion also rebuilt its
working state on every iteration, the tree topology and a CVaR-weight scratch per
interior node included.

**Fix.** The bound is the `ForwardBound::NestedRisk` arm of `sync_forward`
(`crates/cobre-sddp/src/training/forward/stats_aggregation.rs`): one gather and
one estimator dispatch. The recursion reads the tree from the precomputed
`NestedUbTopology`, whose walk shares `walk_leaf_to_root` with
`EnumeratedPlan::walk_path`; the partition from `cobre_comm::per_rank_counts`; and
the stride from the declared stage count. Its gather layout, gathered costs,
per-node buffers and CVaR-weight scratch live in a `NestedUbScratch` held by the
training session's iteration scratch, so the arm allocates only on its first
iteration. `SimulationWeighting` mirrors the risk-neutral `Statistical` and
`Exact` arms; the nested arm bounds training and has no simulation weighting
counterpart.

**Owner.** The training owner.

#### Sampled and exact forward upper bounds allocate their gather buffers every iteration

**What it is.** The `Statistical` and `Exact` arms of `sync_forward`
(`crates/cobre-sddp/src/training/forward/stats_aggregation.rs`) build the rank
partition (`per_rank_counts`), its displacements (`prefix_displs`) and the
gathered-cost buffer afresh on every training iteration, against the hot-path
pre-allocation rule. The `NestedRisk` arm keeps the same three buffers on a
persistent `NestedUbScratch`.

**Owner.** The training owner.

**Trigger.** The next change to the `Statistical` or `Exact` arm, or a forward-path
allocation pass: move the three buffers onto persistent scratch of the
`NestedRisk` arm's shape.

#### Admission predicate duplicated between setup gate and session override — RESOLVED

**What it was.** The setup rejecter (`reject_gap_under_nonuniform_risk`) and the
session's nested-bound selection answered "uniform effective CVaR under
enumerated forwards" with independent code. A divergence could admit a gap rule
whose nested bound never applied, silently reverting to the risk-neutral
end-of-horizon bound.

**Fix.** Both decide through `uniform_effective_measure`
(`crates/cobre-sddp/src/convergence/risk_measure.rs`), the owner of the
uniformity scan, and every effective-measure test routes through
`RiskMeasure::effective`, the owner of the `lambda > 0` rule.

**Owner.** The training owner.

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

## Deferred-debt register — 2026-10 fix-wave follow-ups

Follow-ups found during the 2026-10 fix wave and not scheduled in it. Each
entry names an owner and a trigger, as the first deferred-debt section
requires.

### Setup warnings print once per process under MPI

**What it is.** Setup runs on every rank, so a `tracing::warn!` emitted while
a study is loaded, fitted and prepared prints once per rank under
`mpiexec -n N`. Examples:

- the zero-turbine-capacity warning and the FPHA fit-deviation warning, both in
  `resolve_production_models_from_artifacts`
  (`crates/cobre-sddp/src/production/hydro_models/production.rs`);
- `warn_on_sub_stage_lead`;
- `warn_on_boundary_absent_post_study_delivery` (both in
  `crates/cobre-sddp/src/setup/mod.rs`).

The search
`rg -n 'tracing::warn!|\bwarn!\(' crates/cobre-sddp/src crates/cobre-io/src crates/cobre-stochastic/src`
finds the sites, about 30 of them. The exception is the policy-load warnings:
the warning callback that `check_policy_load` passes to `check_full_fcf_load`
(`crates/cobre-cli/src/commands/run/policy.rs`) prints only on rank 0.
`cobre-python` embeds no tracing subscriber, so a Python caller sees none of
these warnings. There are two rework paths:

- return the warning as data and print it on rank 0 at the call site, as the
  policy-load warning callback does. The zero-turbine-capacity warning can then be built
  from the `no_turbine_capacity` entries of the hydro-model summary
  (`NoTurbineCapacityHydro`), the same data that
  `training/hydro_models.json` records, so the file and the warning cannot
  disagree;
- a rank-aware tracing filter, which hides genuinely rank-local warnings unless
  `RUST_LOG` asks for them.

**Owner.** The setup / config owner.

**Trigger.** A multi-rank log read by a person or a parser that expects each
setup warning once, or a new setup warning being added (it should then take the
rank-0 path from the start).

### Solver retry wall-clock budgets decide whether a run completes

**What it is.** When a HiGHS solve fails, the backend escalates through a fixed
ladder of option levels (`retry_escalation` in
`crates/cobre-solver/src/backends/highs/retry.rs`). Each level is bounded by
iteration limits and by wall-clock budgets:

- the first-solve threshold in `solve_inner` (`solver.rs`);
- `phase1_wall_budget` and `phase2_wall_budget` per level;
- `overall_budget` for the whole ladder.

The budgets only decide whether a failed level is final or whether the ladder
stops, and a run that completes reaches the same first optimal level whatever
the timing. On an overloaded host, a level that would succeed can exceed its
budget. So the same inputs can complete on one host and abort on another, but
they never complete with different numbers.

**Owner.** The solver-backend owner.

**Trigger.** A run that aborts on a loaded or oversubscribed node and completes
on a quiet one, or a requirement that whether a run completes must not depend
on host load.

### PAR fitting stays within one resolution group

**What it is.** PAR fitting re-indexes seasons to cycle positions only on maps
with one resolution group (`CyclePositions::new` in
`crates/cobre-stochastic/src/par/fitting/cycle_positions.rs` returns `None`
otherwise). On a map with more than one resolution group (`SeasonCycles`
groups, for example monthly then quarterly), fitting keeps raw season ids.
`aggregate_observations_to_season` resolves a date's season through the stage
index first and the season map's `season_for_date` second, then keys the
observation by `observation_occurrence_year`. That function falls back to the
calendar year of the date when `season_for_date` resolves the date to another
season, so a coarse season reached through the stage index whose window a finer
season shadows keeps the calendar-year history key rather than the occurrence
year. Fitting across resolution groups is not supported.

**Owner.** The stochastic / temporal owner.

**Trigger.** A multi-resolution study that needs its coarse seasons fitted as
part of one cycle with the fine ones, or history for a coarse season that
crosses a year boundary.

### Simulation partition write failures do not withhold the phase marker

**What it is.** When a scenario's result partition cannot be written, the simulation logs the error, counts the scenario in `scenarios.failed` in `simulation/metadata.json` (it is not in `scenarios.completed`), and still writes `simulation/_SUCCESS`. The marker therefore means the phase finished writing, not that every scenario's partitions exist. The rework: fail the phase on any partition write failure, reconciled across ranks with the existing outcome flag before the root writes, so that no marker is written. That changes the outcome of runs that complete today with `failed > 0`.

**Owner.** The output-format owner.

**Trigger.** A consumer that reads `simulation/_SUCCESS` as "every scenario partition exists", or a reported run with `scenarios.failed > 0` whose missing partitions went unnoticed.

### Quarterly lag 1 differs between the downstream cascade and the cycle walk on single-resolution maps

**What it is.** On a single-resolution map with monthly then quarterly stages
(the ring fixtures), the two disagree:

- the downstream quarterly cascade (`derive_downstream_par_order` and its
  window in `crates/cobre-stochastic/src/par/lag_transition.rs`) gives the
  quarterly lag 1 as the previous quarter's aggregate;
- the season-cycle walk, PAR fitting and the in-study transition-lag statistics
  take the previous cycle member (the last month).

Both are deterministic. They define the quarter's predecessor differently.

**Owner.** The stochastic / temporal owner.

**Trigger.** A study on such a map whose quarterly lag-1 value must agree
between scenario generation and the fitted model, or a request to give the
quarter one predecessor definition.

### `next_season_period_window` advances `Custom` maps in id order

**What it is.** `next_season_period_window`
(`crates/cobre-stochastic/src/season_cast/mod.rs`) steps a `Custom` season map
to the next season id, not the next season in calendar order. Its callers are
the spillover in `crates/cobre-stochastic/src/par/lag_transition.rs` and the
history occurrence discovery in `crates/cobre-io/src/scenarios/estimation.rs`.
Every other season consumer walks the cycle (`SeasonCycles`).

**Owner.** The stochastic / temporal owner.

**Trigger.** A `Custom` map whose ids are not in calendar order reaching either
caller, or the next change to either caller.

### Three rules pick "the statistics of season s"

**What it is.** When several stages carry the same season with different
statistics (year-varying user statistics), three sites choose by different
deterministic rules:

- fitting (`build_pacf_stage_lookups` in
  `crates/cobre-stochastic/src/par/fitting/estimation.rs`, and
  `build_season_lookups` in `par/fitting/ar_coefficients.rs`) keeps the last
  entry in slice order;
- `check_std_ratio_divergence` (`crates/cobre-io/src/scenarios/estimation.rs`)
  keeps the first entry, and pairs consecutive seasons by ascending raw id, not
  by cycle order;
- the precompute lag fallback keeps the lowest stage id.

The public building blocks `estimate_ar_coefficients_with_season_map` and
`estimate_annual_seasonal_stats` still index by raw id when called directly.
Production reaches them only through the wrapped entry point. The candidate
rule is the lowest stage id, as precompute uses.

**Owner.** The stochastic / temporal owner.

**Trigger.** A study with year-varying seasonal statistics, or a request to make
the choice uniform.

### Month-length weights on sub-monthly history

**What it is.** `aggregate_observations_to_season`
(`crates/cobre-stochastic/src/par/aggregate.rs`) weights each entry of a
multi-entry bucket by `days_in_month(date)`, the length of the calendar month
containing the entry. For sub-monthly history in a bucket that spans two months
of different length, the weighted value is therefore not the plain mean of its
rows. For example, daily history of ISO week 2015-W09 weights six February days
by 28 and the March day by 31. The candidate fix is to weight each row by its
own span, which needs the row's end date in the observation tuple.

**Owner.** The stochastic / temporal owner.

**Trigger.** A study estimating from daily or other sub-monthly history.

### The downstream cascade's window assumes calendar quarters

**What it is.** The downstream cascade's window in
`crates/cobre-stochastic/src/par/lag_transition.rs` still makes calendar-quarter
assumptions. `compute_downstream_transitions` groups the window's months into
calendar quarters from each season definition's `month_start`, weights each
quarter by `quarter_hours`, and sizes the window as `downstream_par_order * 3`
monthly stages. The cascade's activation rule follows season spans, but the
window arithmetic is still calendar-quarter arithmetic.

**Owner.** The stochastic / temporal owner.

**Trigger.** A season map whose coarse seasons are not calendar quarters, or
whose fine seasons are not calendar months, reaching the cascade.

### Season grouping splits a level whose season lengths differ by more than the tolerance

**What it is.** `SeasonCycles` (`cobre_core::temporal`) groups seasons into
resolution levels, and seasons whose lengths differ by more than the 7-day
tolerance (`SUB_PERIOD_TOLERANCE_DAYS`) fall into different groups. A layered
map with unequal seasons in one intended level, for example monthly seasons plus
wet and dry seasons, is therefore split into several cycles, and each season's
predecessor is taken within its own group.

**Owner.** The core data-model owner.

**Trigger.** A layered season map whose unequal-length seasons must share one
cycle.

### `RankDistribution::new` recomputes the rank partition

**What it is.** `RankDistribution::new`
(`crates/cobre-sddp/src/training/session/rank_distribution.rs`) splits the
forward passes across ranks with its own base/remainder arithmetic.
`cobre_comm::per_rank_counts` is documented to own that partition. The two agree
today, and the module's own tests compare them.

**Owner.** The comm / HPC-distribution owner.

**Trigger.** Any change to how work is split across ranks. Both copies must
change together until `RankDistribution::new` calls `per_rank_counts`.

### Remaining sources of output difference across node layouts

**What it is.** Simulation outputs are required not to depend on node count,
rank count or rank placement. Two known sources could still break that:

- A build for a specific CPU. A build with `-C target-cpu` or `-march=native`,
  for the Rust code or for the vendored solvers' C/C++ builds, can produce
  different floating-point results on heterogeneous nodes. No such flag is set
  today; `.cargo/config.toml` fixes the target features.
- The CLP reset fallback. When allocating a fresh CLP model fails during a
  reset, `reset_solver_state`
  (`crates/cobre-solver/src/backends/clp/interface.rs`) keeps the old model,
  whose stale pricing state can change later solves.

**Owner.** The build / CI owner (build flags) and the solver-backend owner (the
CLP reset).

**Trigger.** Running one study across nodes with different CPUs, a build that
sets a CPU-specific flag, or a reported CLP allocation failure during a reset.

### Minor residues (2026-10 fix wave)

- **D30's unread `recent_observations` window.**
  `examples/deterministic/d30-multi-resolution-monthly-quarterly/initial_conditions.json`
  keeps a `recent_observations` window that no lag reads, since lag seasons
  follow the season cycle. **Owner.** The stochastic / temporal owner.
  **Trigger.** The next edit to that case.
- **Ring tests' assertion wording.** Assertion messages in
  `crates/cobre-sddp/tests/forward_sampler_integration.rs`,
  `crates/cobre-sddp/src/setup/stochastic_pipeline.rs` and
  `crates/cobre-stochastic/src/par/lag_transition.rs` still say "crosses
  season_id >= 12". The fixtures do cross id 12, so the messages are literally
  true, but the cascade no longer activates on an id threshold. **Owner.** The
  training / test owner. **Trigger.** The next edit to those tests.
- **`.venv` at the repo root is not ignored.** `CONTRIBUTING.md` § "Testing
  cobre-python" creates `.venv` at the repo root, and `.gitignore` ignores
  `.venv-mpi-smoke/` and `.venv.claude/` but not `.venv`. **Owner.** The
  build / CI owner. **Trigger.** The next `.gitignore` edit.
- **Test-helper copies.** `crates/cobre-cli/tests/cli_validate.rs` keeps private
  `cobre` and `write_file` helpers that `crates/cobre-cli/tests/common/mod.rs`
  exports. Recursive directory-copy helpers are repeated across test suites:
  `copy_dir_recursive` (`crates/cobre-cli/tests/common/mod.rs`,
  `crates/cobre-sddp/tests/common/mod.rs`,
  `crates/cobre-sddp/tests/cut_basis.rs`), `copy_recursive`
  (`crates/cobre-sddp/tests/deterministic.rs`), `copy_case_dir_into`
  (`crates/cobre-io/tests/integration.rs`) and `copy_dir_all` (the tests in
  `crates/cobre-python/src/run.rs`). **Owner.** The test-infrastructure owner.
  **Trigger.** A behaviour difference between two copies, or the next test that
  needs the same helper.
- **Unit notation in rustdoc that the schemas do not export.**
  `HydroEnergyProductivityRow`
  (`crates/cobre-io/src/extensions/hydro_energy_productivity.rs`) keeps the
  escaped-bracket unit notation the exported schemas no longer use. The error
  table in `parse_correlation`'s rustdoc
  (`crates/cobre-io/src/scenarios/correlation.rs`) escapes brackets inside a
  code span, so rustdoc shows the backslashes. **Owner.** The doc / comment
  owner. **Trigger.** The next edit to either doc comment, or a `JsonSchema`
  derive on `HydroEnergyProductivityRow`.
- **The anticipated-lane record's stage field.** The rustdoc of
  `AnticipatedLaneWriteRecord::stage_id`
  (`crates/cobre-io/src/output/simulation_writer.rs`) calls it a "Stage index
  (0-based)" while the field is named `stage_id`. Align the name or the doc
  once the value it holds is confirmed. **Owner.** The output-format owner.
  **Trigger.** The next edit to the anticipated-lane writer, or a consumer
  reading the column as a stage id.
- **JSON conversion split across modules.** `json_value_to_py` lives in
  `crates/cobre-python/src/results.rs`, while its inverse `py_to_json_value`
  lives in `crates/cobre-python/src/convert.rs`. **Owner.** The Python bindings
  owner. **Trigger.** The next edit to either conversion.
- **Reserved checkpoint keys describe effects nothing implements.** The
  schema-visible docs of `compress` ("Compress checkpoint files.") and
  `store_basis` ("Include LP basis in checkpoints for warm-start.") on
  `CheckpointingConfig` (`crates/cobre-io/src/config/policy.rs`, exported to
  `schemas/config.schema.json`) state effects that no code performs. Both keys
  are reserved. **Owner.** The training owner. **Trigger.** Either key being
  wired, or the next edit to those doc comments.

### The Python policy writer accepts a foreign checkpoint's identity

**What it is.** The load gate (`validate_policy_load` in
`crates/cobre-sddp/src/policy/policy_load.rs`) refuses any checkpoint whose
recorded writer is not exactly `SoftwareIdentity::THIS_BUILD`. The Python
binding has a two-call bypass. `cobre.results.load_policy` returns a foreign
checkpoint's cuts and metadata as plain dicts, `metadata_to_py` carrying the
`software` and `software_version` keys
(`crates/cobre-python/src/results.rs`). `cobre.write_policy_checkpoint` then
writes them back: `PyPolicyCheckpointMetadata`
(`crates/cobre-python/src/policy.rs`) does not declare the identity keys, so
extraction drops them, and `impl From<PyPolicyCheckpointMetadata> for CheckpointManifest`
stamps this build's `SOFTWARE_NAME` and `SOFTWARE_VERSION`. The result passes
the gate. The writer's docstring, the stub in
`crates/cobre-python/python/cobre/__init__.pyi` and `.claude/rules/sddp.md` all
state that identity keys are ignored.

The rework: the metadata struct declares the three identity keys and its
conversion becomes fallible; one shared check in `cobre-sddp`, called first by
`validate_policy_load` and by the writer, refuses metadata that names a writer
other than this build, surfacing as `PolicyIncompatibleError`. Metadata that
names no writer keeps being stamped with this build, which is how an external
tool authors a boundary policy. The three documents above change with it.

**Owner.** The Python bindings owner.

**Trigger.** A script that round-trips a policy through `load_policy` and
`write_policy_checkpoint`, or the next change to the writer's metadata handling.

### cobre-bridge accepts any cobre-python patch release

**What it is.** A policy loads only in the exact software and version that wrote
it. cobre-bridge's DECOMP conversion writes a terminal boundary policy through
`cobre.write_policy_checkpoint`, and cobre loads it later, yet the bridge's
`cobre-python` dependency in its `pyproject.toml` and `uv.lock` (a separate
repository) is a range over one minor release. A resolver can therefore install
a patch release whose checkpoints another patch release of cobre refuses. The
bridge's minimum-version constant and its packaging tests treat the dependency
as a floor. The rework is an exact `==` pin kept equal to that constant by the
packaging tests, with the floor and lockstep wording in the bridge's
`CONTRIBUTING.md` and `CLAUDE.md` and a changelog line updated to match. Every
cobre patch release then needs a matching bridge release, by design.

**Owner.** The build / CI owner.

**Trigger.** A release that pairs cobre-bridge with a new cobre-python, or a
report of a boundary policy refused for a software-version mismatch.

### MPI load refusals do not exit with the refusal's own code

**What it is.** When rank 0 refuses at load (a config refusal, a `policy.path`
guard, a missing iteration limit, a stochastic refusal), the error reaches the
other ranks through `broadcast_value`'s length-0 sentinel
(`crates/cobre-cli/src/commands/broadcast.rs`), called from
`broadcast_and_build_setup` (`crates/cobre-cli/src/commands/run/setup.rs`).
Every peer turns the sentinel into `CliError::Internal` ("rank 0 signaled
broadcast failure (length 0)", followed by the report-a-bug hint). `execute`
aborts each rank with its own code, and the launcher reports the first abort it
receives, which is a peer's. A refusal that exits 1 single-process therefore
exits 4 under `mpiexec -n 2` with MPICH. Rank 0 renders its own error only in
`execute`, after the collective, so its text also races the peers' aborts, both
for a load refusal and for a failed rank-0 write after `agree_post_write`
(`crates/cobre-cli/src/commands/run/graceful_stop.rs`).

The rework: in the failure branch only, rank 0 sends its exit code in the
length broadcast that already runs; peers return `CliError::for_peer_failure`
with that code and a load-phase message (the constructor takes only the code
today and its message names the write phase); rank 0 prints its diagnostic
before the collective that carries its failure, and a flag stops `execute` from
printing it again. The success path gains no collective. The entry
"Split exit codes in the pre-training export and simulation-outcome
reconciles" covers the two reconciles that still split exit codes.

**Owner.** The cobre-cli run-orchestration owner.

**Trigger.** A job script or scheduler that keys on the MPI launcher's exit code
for a load refusal, or the next change to `broadcast_value` or the load
broadcasts.

### Python Ctrl-C and SIGTERM do not stop training at an iteration boundary

**What it is.** Three linked gaps in `cobre-python`:

- `Study.train` runs `train_native` under `py.detach`
  (`crates/cobre-python/src/study.rs`). The streaming drain thread in
  `crates/cobre-python/src/run.rs` calls `py.check_signals()`, but CPython runs
  Python signal handlers only on the main thread, so that call never raises. A
  Ctrl-C is recorded by CPython's C-level handler and surfaces as
  `KeyboardInterrupt` only when the whole call returns, after every iteration
  and the simulation.
- `run_via_study` leaves on a captured callback error right after
  `train_native` returns, before the skipped-simulation writes. A signal stop on
  `cobre.run.run` with a configured simulation therefore writes no skipped
  simulation metadata and marker, which the CLI's stop path writes.
- No code in `crates/cobre-python` touches SIGTERM, so the process dies by the
  default action wherever training is, and a scheduler's time-limit SIGTERM loses
  every iteration since the last periodic checkpoint.

The rework: the calling (main) thread services signals while training runs on a
`std::thread::scope` worker, looping over the iteration channel and
`check_signals()`; an exception from a Python signal handler, or a
`KeyboardInterrupt` or `SystemExit` from `on_iteration`, raises the Signal level
of the shared stop flag and is re-raised only after every artifact, the skipped
simulation's partial metadata and marker included; a second SIGINT while a stop
is pending restores the default handler and re-raises it. When SIGTERM is at its
default on the main thread, a scoped guard installs a handler that raises the
same level, stays through the skipped-simulation writes on a signal stop, drains
`check_signals()` before it is removed, and re-delivers a pending SIGTERM after
the artifacts. A user SIGTERM handler is left in place, and a call from a
non-main thread installs nothing.

That rework covers training only. `Study.simulate()` and the simulation phase
of `cobre.run.run` run without signal servicing, so a signal received during
simulation is acted on only when the call returns.

**Owner.** The Python bindings owner.

**Trigger.** A report that Ctrl-C or a scheduler SIGTERM does not stop a Python
run at an iteration boundary, a request to interrupt a long Python simulation,
or simulation gaining a stop point at which partial results are written, or the
next change to the streaming drain, `train_native` or `run_via_study`.

### External-library refusals name loop positions instead of ids

**What it is.** `validate_external_library`
(`crates/cobre-stochastic/src/sampling/external.rs`) has three refusals that
print loop counters. V3.3 (a study stage with no rows) and V3.4 (a row count
that is not a whole number of scenarios) print the 0-based study-stage
position. V3.7 (a non-finite standardized value) prints the stage position, the
scenario slot and the 0-based entity position. None of these appears in
`scenarios/external_*_scenarios.parquet`: a study whose stage ids start at 1 and
whose hydro ids are plant codes such as 66 or 156 gets `stage 2, scenario 4,
entity 1`. The entity id is in reach (`entity_ids[entity_idx]`, already a
parameter). The scenario slot is the file's `scenario_id` itself, so only its
label changes. The stage id is not in reach, because the function receives only
`n_stages: usize`; every production caller holds the study-stage slice it
standardized against, so the fix replaces `n_stages` with that slice. V3.2 and
V3.5 already print the entity id.

**Owner.** The stochastic / temporal owner.

**Trigger.** A report of a V3.3, V3.4 or V3.7 refusal whose position a user
cannot map to their files, or the next change to `validate_external_library`.

### Documentation corrections in rustdoc, release text and recordings

Small corrections that change no behaviour, one bullet each.

- **Anticipated-commitment ring sizing.** The `commit_out` field doc of
  `StateSpace` (`crates/cobre-sddp/src/lp/indexer/state_space.rs`) calls a slot
  `k >= k_i` padding. Slots are keyed ring-axis-modular, so the padding is a
  slot no decision stage latches, and a plant with a short lead cycles through
  every slot. The docs of `StateSpace::k_max` and `anticipated_lead_stages`
  define the depth as the maximum `lead_stages`, but `k_max` comes from
  `AnticipatedResolution::ring_size` (the larger of the anchored depth and the
  longest `lead_stages`), and a `lead_time_hours` plant has no `lead_stages`.
  The `k_max` formula in `docs/design/anticipated-thermals-and-water-travel-time.md`
  and the "Ring depth sizing" contract in `.claude/rules/sddp.md` omit the
  `ring_size` widening, and the doc comment and inline comment of
  `state_to_lp_column_commit_out_identity_multi_plant_heterogeneous_k` in
  `state_space.rs` repeat the padding idea.
  **Trigger.** The next edit to the ring-sizing code or to any of those texts.
- **Retired spec links in cobre-solver rustdoc.** `types.rs`, `trait_def.rs`
  and `freeze.rs` under `crates/cobre-solver/src` link `src/specs/` pages of the
  methodology repository that no longer exist and cite their sections in plain
  text. One fact lives only on a retired page: the dual-sign normalization
  convention stated on `LpSolution`, which the solver conformance tests already
  pin. The replacement text is a solver-interface statement with no
  algorithm-specific term, because the crate is infrastructure.
  **Trigger.** The next edit to those three files.
- **Retired spec links in cobre-core rustdoc.** `temporal.rs`, `scenario.rs`
  and `horizon.rs` under `crates/cobre-core/src/model` link retired spec pages
  through relative `.md` paths, carry section-number banners such as
  `// Block (SS12.2)`, and `HorizonGraph::annual_discount_rate` cites a retired
  "validation rule 7" (the cyclic-discount check in `cobre-io` is labelled "Rule
  3"). A topic with a live page on the documentation site is linked once, from
  the module doc or the type that owns it; the rest are deleted.
  **Trigger.** The next edit to those three files.
- **Release README launch command.** The README that the "Package archive" step
  of `.github/workflows/release-mpi.yml` writes tells the user to launch with
  `srun --mpi=pmi2`, then says in Troubleshooting that `srun --mpi=pmi2` fails
  with "pmijobid missing in fullinit command" on SLURM 24.05. The binary links
  against the MPICH ABI, so `srun --mpi=pmix` works only when Slurm has its PMIx
  plugin and the cluster's MPI runtime was built with PMIx. The fix recommends
  `srun --mpi=pmix` with that precondition, keeps `mpiexec` as the fallback and
  mentions pmi2 only in Troubleshooting. `scripts/ci/check-no-plan-leaks.sh`
  and `scripts/ci/check_doc_voice.py` do not scan the workflow, so the text is
  checked by hand. **Trigger.** A cluster report of the pmi2 failure, or the next
  edit to that README.
- **Errata in the released `CHANGELOG.md` section.** The `[0.17.0]` entries
  misstate CLI-observable points, to be amended in place:
  - the historical opening-tree entries name a study that uses historical
    sampling, but the trigger is a stage whose `sampling_method` is
    `historical_residuals`, under any forward scheme;
  - "no complete historical window" was already refused before that release, so
    it does not belong in the list of newly refused cases;
  - the single-season shift and the study stages' year offsets are described
    imprecisely;
  - the passthrough fix says "first operating plant", where the target is the
    first plant below that is not `PreFilling`;
  - the discount fix's wording of the future-cost cascade is imprecise, and it
    omits post-study deliveries, the upper bound, simulation costs and
    `anticipated_thermal_cost`;
  - the stored-basis refusal omits that changing a stage's block mode or block
    count changes the basis dimensions, so warm-start, resume and
    simulation-only loads of an older policy are refused while a boundary-cut
    load, which reads no stored basis, is unaffected.

  Points visible only to Python callers are left out, since the changelog lists
  what a `cobre` CLI user observes. **Trigger.** A user report that a released
  entry misdescribes behaviour, or the next edit to the `[0.17.0]` section.

- **Recording GIFs.** The tapes in `recordings/` now run against the current
  config schema, but the committed GIFs in `recordings/` were rendered earlier
  and show an older banner and earlier CLI output. `recordings/generate.sh`
  regenerates them, and `recordings/setup.sh` installs its tools.
  **Trigger.** A release, or a CLI text change that makes the GIFs visibly
  stale.

**Owner.** The doc / comment owner.

**Trigger.** The next edit to any file a bullet names, or a bullet's own
trigger, whichever comes first.

### The Python bindings crate is not linted by CI

**What it is.** `cobre-python` is excluded from the Cargo workspace (`exclude`
in the root `Cargo.toml`), so the CI clippy job's `cargo clippy --workspace`
never lints it, although the format check and the bindings crate's Rust tests
run in CI. The missing step is
`cargo clippy --manifest-path crates/cobre-python/Cargo.toml --all-targets -- -D warnings`
in the `python` job of `.github/workflows/ci.yml`. That job's toolchain step
installs no clippy component today, and PyO3's build script needs an
interpreter on `PATH`, which the job's virtualenv provides. One matrix leg
suffices, because the crate builds against the stable ABI.

**Owner.** The build / CI owner.

**Trigger.** A clippy regression in the bindings crate shipping unseen, or the
next CI-configuration pass.

### The output-file registry is not exported

**What it is.** `OUTPUT_FILES` in
`crates/cobre-io/src/output/file_registry.rs` owns each output file's path,
layout, format, write phase and backing schema, and the module carries an
`expect(dead_code)` because only tests read it. `schemas/` holds only input
schemas, so documentation and tooling cannot read the output layout from the
repository. The rework serializes the registry, with each backing schema's
columns, into a committed `outputs.json` under `schemas/` that `export_schemas`
(`crates/cobre-io/src/schema.rs`) writes, so `cobre schema export` and
`cobre.schema.export` produce it with no new Python code. The file is
deterministic (no timestamp and no version) and the CI `schemas` job's drift
check covers it. `test_export_schemas_writes_all_files_as_valid_json` asserts
that the written count equals the number of generated input schemas and changes
with the new file.

**Owner.** The output data-model owner.

**Trigger.** A consumer, such as the documentation site or an external tool,
needs the output layout in machine-readable form, or the registry gains a
reader that makes its `expect(dead_code)` lapse.

### Validation rules have no machine-readable registry

**What it is.** `RULES` in `crates/cobre-io/src/validation/rules.rs` is a
single-owner table that the loading layers' emitters reference, so a
diagnostic's kind and severity cannot drift from it. The other validation
checks have no such table. The historical-library checks V2.1 to V2.9 in
`crates/cobre-stochastic/src/sampling/historical.rs` are split across two
functions, each with its own hand-copied "Checks performed" rustdoc table:
`check_historical_structure` holds V2.1 and V2.9, and
`validate_historical_library` holds V2.3, V2.5 and V2.6, with V2.2, V2.4 and
V2.7 as asserts. Only some of the messages carry their code. The preparation
and policy-load checks of `cobre validate` take their kind from string
literals in `prep_phase_metadata`
(`crates/cobre-sddp/src/validate_phases.rs`), while the individual refusals
are free-text `SddpError::Validation` messages with no id. Nothing aggregates
the tables, and there is no `cobre validate --list-rules --json` or
`cobre.schema.list_rules`.

The rework gives `cobre-stochastic` and `cobre-sddp` each a rule table with an
id, layer, kind, severity and summary, referenced by their emitters. An
aggregator in `cobre-sddp`, the lowest crate that sees all three tables,
normalizes them into `{format_version, rules: [{id, layer, kind, severity,
summary}]}` and checks that the layer names agree across the tables. The CLI
prints that object under `--list-rules --json`, following the `--list` flag of
`cobre init`, and Python returns its `rules` member as a list of dicts, so the
two outputs cannot diverge. Rule ids are opaque strings, never renumbered or
reused, and the summaries in infrastructure crates carry no algorithm-specific
term.

**Owner.** The input-validation owner.

**Trigger.** A consumer, such as the documentation site, an editor integration
or a test, needs the rule list, or a validation rule is added in
`cobre-stochastic` or `cobre-sddp` (it should then enter a table from the
start).

### A constant inside a parenthesized group gets a generic parse error

**What it is.** The generic-constraint expression parser rejects a bare
constant inside a parenthesized group, as the grammar in the module doc of
`crates/cobre-io/src/constraints/generic.rs` states. The refusal is the generic
token error from `parse_single_term`, reached through `parse_group_terms` (for
example "expected '*' after coefficient 73, got RParen"). It does not say that
a group cannot hold a constant.

**Owner.** The input-validation owner.

**Trigger.** A user report of that message, or the next change to the
expression parser's messages.

### Validation messages hand-type their input-file labels

**What it is.** Validation findings name the input file they concern by a path
literal typed at each emitting site, for example the file label in
`validate_variable_ref_entity` (`crates/cobre-io/src/validation/referential.rs`).
The private `INPUT_FILES` table in `crates/cobre-io/src/validation/structural.rs`
already pairs each input file with its relative path. A label looked up from it
could not drift from the real path; a hand-typed one can.

**Owner.** The input-validation owner.

**Trigger.** The next wrong file label in a validation message, or the next
change that adds validation emitters.

### No validator checks that block hours sum to the stage duration

**What it is.** Three docs state that a stage's block hours sum to its
duration: `Block::duration_hours` and `Stage::blocks`
(`crates/cobre-core/src/model/temporal.rs`), and the module doc of
`crates/cobre-io/src/stages.rs`, which defers the check to the semantic layer.
No validator performs it, so a stage whose blocks do not cover its date span
loads, and `Stage::total_hours` returns the block sum. Either the check is
added or the docs state the rule as a convention.

**Owner.** The input-validation owner.

**Trigger.** A case whose block hours differ from its stage spans, or the next
change to stage validation.

### A non-root hydro-model preprocessing failure exits as an internal error

**What it is.** Under MPI, ranks other than rank 0 rebuild the hydro production
models themselves (`prepare_hydro_models` in
`crates/cobre-cli/src/commands/run/setup.rs`). A failure there is mapped by
hand to `CliError::Internal` (exit 4), bypassing `CliError::from`, which
classifies the same error on rank 0. Rank 0 has already built the same models,
so the failure is unexpected. When it happens, it is reported as a software
fault whatever its cause.

**Owner.** The cobre-cli run-orchestration owner.

**Trigger.** A preprocessing failure seen on a non-root rank, or the next
change to non-root reconstruction in that file.

### Exit codes can differ by rank after a coordinated training failure

**What it is.** When training fails on one rank under MPI, every rank stops
together (`reconcile_error_flag` in
`crates/cobre-sddp/src/training/rank_reconcile.rs`), but each rank then maps
its own error (`train_then_simulate` in
`crates/cobre-cli/src/commands/run/mod.rs`). The failing rank exits with its
error's code, and its peers, which hold a communication error, exit 4. The
launcher then reports whichever rank exits first. The graceful-stop path
agrees one exit code on every rank through `CliError::for_peer_failure`; the
training-failure path does not.

**Owner.** The cobre-cli run-orchestration owner.

**Trigger.** A multi-rank run whose launcher reports exit 4 for an input error,
or the next change to the training-failure path.

### The run-error hint does not carry --output

**What it is.** When `cobre run` fails, its hint (`format_error` in
`crates/cobre-cli/src/error.rs`) suggests running `cobre validate <CASE_DIR>`.
After a run with `--output`, that suggestion validates against the default
output directory, so a refusal that depends on the chosen directory may not
reproduce. `cobre validate` accepts the same `--output` flag. The hint text is
pinned by tests, so a change to it updates those pins.

**Owner.** The cobre-cli run-orchestration owner.

**Trigger.** A user following the hint after a run with `--output`, or the next
change to the hint text.

### cobre validate --json omits input-validation warnings

**What it is.** `cobre validate --json` prints one JSON object
(`ValidateBoundaryOutput`, written by `emit_validate_json` in
`crates/cobre-cli/src/commands/validate.rs`), which has no warnings field. The
input-validation warnings the human report prints are therefore absent from the
JSON output, while the Python `validate` result reports them
(`build_warnings_list` in `crates/cobre-python/src/io.rs`).

**Owner.** The cobre-cli owner.

**Trigger.** A script that needs the warnings from `--json`, or the next change
to the `--json` object.

### Validate's construction-failure kind depends on whether a boundary policy is configured

**What it is.** `cobre validate --json` and Python's validate report a
study-construction failure as `StudySetupError` on a case without a boundary
policy. On a case with a boundary policy, they report the same failure,
including a scalar-parameter gap, as `BoundaryReconciliationError`
(`PrepPhase` and `prep_phase_metadata` in
`crates/cobre-sddp/src/validate_phases.rs`). The kinds were kept so that no
case changed the kind it reported.

**Owner.** The setup / config owner.

**Trigger.** A consumer that classifies validate failures by kind across cases
with and without a boundary policy, or the next change to the `--json` kinds.

### A failed final checkpoint write hides an earlier failed periodic write

**What it is.** When a periodic checkpoint write fails
(`write_periodic_checkpoint` in
`crates/cobre-sddp/src/training/session/mod.rs`, which raises
`SddpError::CheckpointWrite` with the iteration) and the final checkpoint write
(`write_training_outputs` in `crates/cobre-cli/src/commands/run/outputs.rs`)
then fails as well, the CLI reports the final write's error first, because
`train_then_simulate` returns the output-write result before it returns
`training.error`. The message therefore does not name the iteration whose
periodic write failed.

**Owner.** The training owner.

**Trigger.** A run whose periodic checkpoint write fails, or the next change
to checkpoint error reporting.

### The external scheme requires a seed it never reads

**What it is.** The loader requires `scenario_source.seed` whenever a class
uses the `out_of_sample` or the `external` scheme
(`crates/cobre-io/src/config/mod.rs`). External selection draws from the
constant `EXTERNAL_SELECTION_BASE_SEED`
(`crates/cobre-stochastic/src/sampling/class_sampler.rs`), so in a study whose
only non-in-sample classes are external, the required seed is never read.
Dropping the requirement for `external` is backward-compatible.

**Owner.** The setup / config owner.

**Trigger.** An external-only study that has to carry an unused seed, or the
next change to the seed requirement.

### The exported schemas do not encode value-conditional requirements as schema constructs

**What it is.** Two load requirements depend on another field's value, and the
exported JSON schemas carry both only as description prose, not as conditional
schema constructs:

- the payload field each `kind` of a generic parameter needs
  (`schemas/generic_parameters.schema.json`);
- the seed each sampling scheme needs in `scenario_source`
  (`RawScenarioSourceConfig`; `schemas/config.schema.json`).

The loader enforces both, so a file that passes schema validation can still be
refused at load.

**Owner.** The cobre-io input-schema owner.

**Trigger.** A schema consumer (an editor or a generator) that needs these
requirements, or the next change to either entry shape.

### A pinned test asserts warm and cold costs are bit-identical

**What it is.** `enumerated_census_pool_fill_warms_previously_cold_leaves`
(`crates/cobre-sddp/tests/simulation_integration.rs`) asserts with
`assert_eq!` that per-scenario costs are bit-identical between the
warm-started and the cold run, and `.claude/rules/sddp.md` names that
bit-identity as the pin of the pool-fill basis path. The determinism contract
does not include cross-algorithm equivalence: a warm-started solve may report
a different, equally valid optimal vertex, whose cost can differ in the last
bits. The assertion holds today, but a solver or basis change that keeps the
contract could break it.

**Owner.** The SDDP-rules owner.

**Trigger.** That assertion failing after a solver, basis or LP-layout change,
or the next edit to the test or to its paragraph in `.claude/rules/sddp.md`.

### Plain-text citations of retired methodology sections

**What it is.** Rustdoc and comments in several crates still cite sections of
the retired methodology specification in plain text: section numbers such as
`SS5.1` or `§15`, and page names such as "Solver Abstraction" or
`internal-structures.md`. Those pages no longer exist on the docs site, and
because the citations are not links, no link check finds them. Examples:

- the HiGHS backend (`crates/cobre-solver/src/backends/highs/`), one of them in
  the `// SAFETY:` comment on `unsafe impl Send for HighsSolver`;
- `crates/cobre-io/src/output/schemas.rs`;
- `crates/cobre-comm/src/ferrompi.rs`;
- `crates/cobre-core/src/constraints/initial_conditions.rs` and
  `crates/cobre-core/src/constraints/generic_constraint.rs`.

The entry "Documentation corrections in rustdoc, release text and recordings"
covers the solver-interface and the `cobre-core` model files. The test
fixtures labelled by retired section numbers (`SS1.1`, `SS5 row 1`) in
`crates/cobre-solver/tests/` and the backends' `tests.rs` are labels, not
pointers. Find the candidates with
`rg -n '\bSS[0-9]|§[0-9]|Solver Abstraction|HiGHS Implementation|Solver Workspaces' crates`.
Not every hit is a retired-spec citation, so classify each one when editing.

**Owner.** The doc / comment owner.

**Trigger.** The next edit to a file the search finds (remove the citation
there), or a reader following one to a missing page.

### A cobre-core private doc links a serde-gated type

**What it is.** A doc comment on a private field of `ResolvedBounds` in
`crates/cobre-core/src/model/resolved/bounds.rs` links
``[`ResolvedBoundsWire`]``, which exists only under `#[cfg(feature = "serde")]`.
`cargo doc -p cobre-core --document-private-items` without `--features serde`
therefore reports a broken intra-doc link. CI passes only because its docs
builds enable the features.

**Owner.** The core data-model owner.

**Trigger.** A docs build of the crate without the `serde` feature, or the next
edit to that doc comment.

### Crate READMEs mirror enumerations that drift

**What it is.** Two READMEs restate lists their crates own:

- `crates/cobre-sddp/README.md`: the feature-flag table, which has no
  `test-support` row, and the `CutSelectionStrategy` list;
- `crates/cobre-io/README.md`: the `Config` section table and the stopping-rule
  variant list.

Each copy drifts as the code changes, as the error-variant tables did before
they were replaced by pointers to the enums' rustdoc.

**Owner.** The doc / comment owner.

**Trigger.** The next edit to either README, or the next change to one of the
mirrored lists.

### Docs-site pages to revise after the 2026-10 fix wave

**What it is.** The docs site (the `cobre-docs` repository) still describes
behaviour that this repository changed in the 2026-10 fix wave:

- the error-codes reference and the policy-management page quote the policy
  refusal's remedy and the checkpoint `format_version` refusal as they read
  before the wave;
- the error-codes reference lists a `PolicyIncompatible` value as reserved
  although the variant no longer exists, and says `WarmStartIncompatible` and
  `ResumeIncompatible` are reserved and never emitted, while `cobre validate`
  now reports both;
- no page documents the `.staging` and `.previous` directories that the
  checkpoint writer keeps beside the policy directory.

The penalty pages are covered by the entry "Energy-equivalent penalty
ordering".

**Owner.** The `cobre-docs` methodology owner.

**Trigger.** The next docs-site revision against this repository.

### recordings/setup.sh does not install the recording host's fonts and browser libraries

**What it is.** The tapes set `FontFamily "JetBrains Mono"`, and vhs renders
through a headless Chromium that it downloads on first use.
`recordings/setup.sh` installs vhs, ttyd and ffmpeg only. It installs neither
the font nor the shared libraries the downloaded Chromium needs. On a fresh
host, the GIFs therefore render in a fallback font, or the render fails until
those are installed by hand.

**Owner.** The build / CI owner.

**Trigger.** The next GIF regeneration on a fresh host, or the next edit to
`recordings/setup.sh`.

### The doc guards do not scan docs/design/

**What it is.** `scripts/ci/check-no-plan-leaks.sh` scans crate sources, tests
and benches, `CHANGELOG.md` and `README.md`. `scripts/ci/check-doc-paths.sh`
and `scripts/ci/check_doc_voice.py` scan root-level documents. None reads
`docs/design/`, so plan identifiers, dead repo-relative paths and stale symbols
in the design documents, this register included, are caught only by hand.

**Owner.** The build / CI owner.

**Trigger.** A plan identifier or a dead path found in `docs/design/`, or the
next change to one of those scripts.

### check_schemas.sh writes its diff inside a directory it compares

**What it is.** `scripts/ci/check_schemas.sh` runs `diff -ruN` between the
committed `schemas/` and a freshly exported temporary directory, and writes the
output to `drift.diff` inside that same temporary directory. The output file is
therefore part of the tree being compared while the comparison runs. The check
passes today, but its result depends on `drift.diff` still being empty when
`diff` reaches it.

**Owner.** The build / CI owner.

**Trigger.** The next edit to that script, or a drift report that lists
`drift.diff` itself.

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
