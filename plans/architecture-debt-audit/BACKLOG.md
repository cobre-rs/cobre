# Cobre Architecture Debt Backlog

A living backlog of architectural smells and technical debt, built by walking the
full lifecycle of a `cobre run` command from `main()` to the last output write.

- **Method**: station-by-station descent through the real call tree (not a fan-out).
- **Scope**: architecture-only — god modules/functions/structs, bad/missing
  abstractions, asymmetric ownership, leaky crate boundaries, coupling,
  duplication. **Out of scope** (by hard rule): reserved/unwired config, load-bearing
  contract comments, determinism contracts, perf micro-optimization.
- **Status**: read-only investigation. No repo mutation outside this gitignored file.

Baseline: a136840d4f2ea137f685f0af6dac04254b983b60 (pinned 2026-09-05)
Ledger: this file is the sole home of finding IDs (`CD-` / `PD-` / `OD-` / `TD-`).
Workers return JSON only; the main session is the sole writer. The 2026-09 quality
evaluation writes exactly two tracked surfaces: the ID-free mirror
`docs/design/reserved-seams-and-deferred-debt.md` and this directory
(`plans/architecture-debt-audit/`, tracked on the evaluation branch so its work
sessions carry git evidence); no crate, docs, script, CI or schema file is touched
(amends the Status bullet above).
Protocol bound: 234.781 s (median of 3 timed runs after one warm-up, layout `4t`,
deck `~/git/cobre-bridge/example/cobre_reduzido_2`; measured 2026-09-06 at the
pinned baseline on the profiling-profile binary — see `measurements/CAL/`).
Protocol bound (enumerated): 32.589 s (layout `2t`, deck
`~/git/cobre-bridge/example/cobre-mar-26-rv2-reduced`, owner-limited to two workers;
see `measurements/CAL-ENUM/`).
Only those decks at those worker budgets are sanctioned: `--threads 4` (`4t`) or
`mpiexec -n 2 … --threads 2` (`2x2`) on the sampled deck, `--threads 2` (`2t`) on
the enumerated deck. A run past 3x its bound is killed and its claim tagged
`UNMEASURED`; reasons are `timeout-3x`, `unexercised-path`, `mpi-unavailable`.
Perf fix-shapes stay byte-neutral.

## Milestones

Strict precedence: `0a` < `0b` < `1`. `neutral` advances no phase and is ordered by
dependency only. Sequencing principle (pull, don't push; a second consumer proves a seam
before the data model breaks): plans/generalizing/beyond-sddp-generalization.md V.0.

| Milestone | Meaning | Precedes | Source |
| --- | --- | --- | --- |
| `0a` | Engine seam; `study` config block + admission gate; shared output orchestration in cobre-io; rank-0-executes MPI; byte-identical for SDDP | `0b` | beyond-sddp-generalization.md V.1 (split resolved by D12) |
| `0b` | Carve `cobre-model` from the engine-neutral part of cobre-sddp `lp/` | `1` | beyond-sddp-generalization.md V.1 / IV.2 (split resolved by D12) |
| `1` | Purify the data model: stochastic off System/Stage, training_event out of cobre-core, StageTemplate shed, case v2 + bit-for-bit shim | — | beyond-sddp-generalization.md V.2 |
| `neutral` | Advances no phase | — | — |

Triggers: `gnl-import` — the GNL anticipated-coupling import. Wave 3 (boundary frame)
MUST land before it (Execution-waves table, Wave 3 row, 2026-08-22 section). SATISFIED
2026-08-22 by the Wave-3 execution (`44e72b76`, continuation `9db2cbdf`); the roadmap
must not re-open this gate.

---

## Legend

**Category (under-/mis-structure)** — `god-fn` · `god-struct` · `god-module` ·
`bad-abstraction` · `missing-seam` · `asymmetry` · `leaky-boundary` · `coupling` ·
`duplication`

**Category (over-structure, `OD-` class)** — `speculative-generality` ·
`premature-abstraction` · `needless-indirection` · `new-noun` ·
`over-parameterization` · `redundant-wrapper`. These flag EXCESS structure —
abstraction, indirection, or generality not paid for by a present consumer — the
inverse of the under-/mis-structure smells above. A documented reserved seam
(`docs/design/reserved-seams-and-deferred-debt.md`, the "Unwired config is
reserved, not dead" hard rule, or the sanctioned `#[allow(...)]` census classes)
is NOT over-engineering and is excluded by construction.

**Severity**

- **A — Structural**: a bad/missing abstraction or god-thing that actively makes
  changes error-prone or spreads across the codebase. Highest ROI.
- **B — Local**: a real smell confined to one area; bounded blast radius.
- **C — Cosmetic**: naming, minor asymmetry, low ROI.

**Effort**: S (<½ day) · M (1–3 days) · L (>3 days / multi-crate)

**Confidence**: high (verified in-file) · med (pattern seen, needs a deeper pass) ·
low (hypothesis, flagged for later verification)

**Finding ID**: `CD-NNN` (Cobre architecture Debt), `PD-NNN` (Performance debt),
`OD-NNN` (Over-engineering debt) — each assigned in discovery order within its class.

---

## ★ North-star design note — model input data by lifecycle (workspace-wide)

Owner-ratified 2026-08-06 as the guiding fix-shape for the setup layer **and every
analogous input-data structure in the workspace**. The same conflation recurs wherever a
large "everything the X needs" struct is built from `Config` + parsed inputs — treat this
as a general principle, not a one-off cleanup.

**Diagnosis — two recurring smells:**

1. **Lifecycle conflation.** One struct fuses three lifecycles that change for different
   reasons: immutable **Inputs**, mutable **Config** (run knobs set/tweaked before a run),
   and produced **Output** (state the run itself grows).
2. **Config-projection sprawl.** `Config` is re-projected through 3–4 near-isomorphic
   structs — one per code path (`StudyParams` ≈ `ConstructionConfig` ≈ `BroadcastConfig`)
   — kept in lockstep by hand; each new knob touches ~7 sites and risks silent MPI-vs-local
   divergence.

**Target model:**

- **One `ResolvedRunConfig`** = the single `Config` projection, made postcard-safe _at its
  own boundary_ (solve the `#[serde(tag)]`-enum problem once, there). MPI broadcasts this
  struct; local builds it directly. Both feed construction. (Kills the sprawl.)
- **Split the god-struct by lifecycle**: `Model` (immutable inputs) + `RunParams` (mutable
  config; setter walls evaporate) + `PolicyState`/output (produced by the run).
- **Finish sub-struct extraction symmetrically** inside `Model` — never dissolve a transient
  grouping struct into loose fields.
- **Domain crate owns construction** — no crate-boundary reconstruction mirror.
- **Queries are methods on their type**, not scattered free functions.

**Principle: model by lifecycle, not by "everything the X needs."** Each struct then has
exactly one reason to change.

Anchors findings: **CD-001, CD-003, CD-004, CD-005, CD-006.** Apply the same lens when
auditing every other input-data struct downstream in this traversal.

---

## Traversal map & station status

| #   | Station               | Entry symbol                                                          | Primary files                                                                              | Status                           |
| --- | --------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------- |
| 0   | Entrypoint & dispatch | `main` → `run::execute` → `execute_inner`                             | `cobre-cli/main.rs`, `run/mod.rs`                                                          | ✅ surveyed                      |
| 1   | Setup & broadcast     | `broadcast_and_build_setup`, `setup_communicator`, `run_pre_training` | `run/setup.rs`, `cobre-sddp/setup/`, `lp/builder/`                                         | ✅ complete (1a–1c); CD-001..008 |
| 2   | Policy apply/load     | `apply_training_policy`, `load_policy_for_simulation`                 | `run/policy.rs`, `policy/policy_load.rs`                                                   | ✅ complete; CD-009..011         |
| 3   | Training              | `run_training_phase`                                                  | `run/training.rs`, `training/{session,forward,backward,lower_bound}`, `cut/`, `workspace/` | ✅ complete; CD-012..022         |
| 4   | Simulation            | `run_simulation_phase`                                                | `run/simulation.rs`, `simulation/{pipeline,enumerated,extraction,state}`                   | ✅ complete; CD-027              |
| 5   | Outputs               | `write_training_outputs`                                              | `run/outputs.rs`, `cobre-io` writers, Python parity                                        | ✅ complete; CD-024..026         |

---

## Findings

<!-- Appended station-by-station. Newest station's findings at the bottom. -->

### Station 0 — Entrypoint & dispatch

_Clean. `main()` and `execute_inner` are thin orchestrators delegating to submodules.
No findings; recorded as the baseline "this is what good looks like here."_

### Station 1a — CLI load / broadcast / non-root reconstruction (`cobre-cli/.../run/setup.rs`)

Theme: the MPI root/non-root asymmetry has leaked domain logic up into the CLI crate.

**CD-001 · Sev A · leaky-boundary + duplication · effort L · confidence high**
The CLI crate hand-rolls a mirror of the rank-0 stochastic pipeline.
`reconstruct_stochastic_context_non_root` (`run/setup.rs:405`) and the 110-line
`rebuild_historical_library_non_root` (`run/setup.rs:467`) re-derive from scratch what
rank 0 gets from `prepare_stochastic`: PAR build (`:494`), window discovery (`:501`),
seed derivation (`:547`), lag transitions (`:542`), standardization (`:558`) — reaching
into deep `cobre-stochastic` internals (imports `:52-56`). Must stay bit-for-bit
identical to the rank-0 path for MPI reproducibility, yet lives across a crate boundary
from the code it mirrors. Evidence in the code's own comments: `:474` "Mirrors
`prepare_stochastic`", `:523-525` a non-monthly misrouting trap already hit, `:1161-1168`
the mirror is "unreachable from cobre-sddp's test scope" so 3 tests in 2 crates pin the
two paths instead of shared code.
Direction: one domain-crate entry point that builds the stochastic context,
parameterized by reuse-artifacts (root) vs re-read-from-disk (non-root), so the paths
cannot drift. Related: CD-002.

**CD-002 · Sev B · bad-abstraction · effort S–M · confidence high**
Positional 10-tuple straddling three concerns in `broadcast_and_build_setup`
(`run/setup.rs:262-327`): raw broadcast inputs + root-only pass-throughs + `load_err`,
with a parallel `(None×10)` non-root arm (`:326`) and `(None×9, Some(e))` error arm
(`:312-323`). Field add = edit three positional arms, no compiler help on ordering.
Direction: named `RootLoadArtifacts` struct, `Option` on non-root. Shrinks once CD-001
lands.

**CD-003 · Sev B · duplication + missing-seam · effort M · confidence high**
`BroadcastConfig` → `ConstructionConfig` is a ~22-field manual copy in `build_study_setup`
(`run/setup.rs:583-626`), with `.take()` mutating the "broadcast" struct mid-flight
(`:591-594`). Third parallel representation of run config: `Config` (cobre-io) →
`BroadcastConfig` (cobre-cli) → `ConstructionConfig` (cobre-sddp). Broadcast→Construction
are two in-memory structs kept in lockstep by hand; every new config knob is a silent
4-hop thread.
Direction: collapse the Broadcast→Construction hop (builder / `From` / share one struct
post-broadcast).

_Positive contrast (record as the local "good pattern"): `run_pre_training` /
`run_root_exports` (`run/setup.rs:628-759`) — clean rank-0 extraction, deliberate
reconcile-before-barrier (`:659-667`)._

### Station 1b — StudySetup construction (`cobre-sddp/setup/`)

**CD-004 · Sev B (A-risk) · duplication · effort M–L · confidence high**
Two parallel `Config → ConstructionConfig` projection pipelines. In-process path:
`StudyParams::from_config` (`params.rs:127`) → `into_construction_config` (`params.rs:297`).
MPI path: `BroadcastConfig::from_config` (`broadcast.rs:84`) → hand-assembled in
`build_study_setup` (`run/setup.rs:583`). `params.rs:61-64` acknowledges the twin. One
config knob = ~7 edit sites; the two must agree field-for-field or MPI diverges silently
from local (the expensive reproducibility bug class → A-risk). Structural root beneath
CD-003. Owner confirms the split exists because postcard cannot serialize `Config`'s
`#[serde(tag)]` enums.
Direction: a single postcard-serializable projection of `Config` that is itself broadcast
on the wire and consumed by both paths — collapsing `StudyParams` + `BroadcastConfig` into
one. The tagged-enum breakage is handled once at that projection's boundary. See the
"Setup-layer modeling proposal" note below.

**CD-005 · Sev B · god-struct + asymmetry · effort M · confidence med**
`StudySetup` (`setup/mod.rs:125-284`) has a half-finished sub-struct extraction and
conflates two roles. (1) Partial extraction: 5 cohesive NCS fields are left loose
(`:138-161`) even though they are built as one `NcsEntityData` group (`:699`), then
destructured (`:595-602`) and re-splatted as loose fields (`:643-647`) — the grouping is
deliberately dissolved at the struct boundary; the anticipated cluster is loose the same
way. (2) Role conflation: the doc says "All precomputed study state" yet it also holds
resolved run-config (`backward_profile`, `forward_profile`, `backward_scheduler`,
`hardest_first_claim_order`, `loop_params`, `events`, `policy_path`).
Caveat: NCS fields feed hot-path `StageContext` borrows + carry the D15 patch-identity
contract → any fix must be byte-neutral vs parity goldens. Likely overlaps an existing
"StudySetup Sub-Structs" effort.
Direction: finish the symmetric extraction; consider splitting precomputed-model from
resolved-run-params.

_Watch (not a finding): `from_broadcast_params` (`setup/mod.rs:344-691`, ~340 lines) is at
the edge of god-function but defended (D4) and mostly delegates to well-named helpers;
CD-005's re-splat is part of its length._

### Station 1b-sweep — remaining setup submodules (`node_graph`, `stochastic_pipeline`, `params`, `accessors`, `orchestration`)

**CD-006 · Sev C · asymmetry · effort S · confidence high**
`NodeGraph`'s query surface is scattered as free functions. Exactly one query is a method
(`node_pool_ids`, `node_graph.rs:133`); ~10 functions that also take `&NodeGraph` first and
answer questions about it are free functions: `frontier_node`, `any_stage_node`,
`node_parent`, `node_opening_range`, `node_pinned_scenario`, `stage_frontier`,
`build_parent_map`, `max_successor_outcome_count`, `backward_cut_levels`, `pool_cut_stride`,
`forward_solve_counts`. Two call styles for one receiver; hurts discoverability and feeds the
`nodes[0]`-as-root bug class (the correct `frontier_node` accessor is an overlookable free
fn). Direction: move `&NodeGraph`-querying fns into `impl NodeGraph`; leave true
constructors/algorithms (`build_*`, `assemble_outcome_weights`) free.

_Sharpens existing findings (no new entry):_

- CD-004: `StudyParams` (19 fields) ≈ `ConstructionConfig` (20 fields) — near-isomorphic;
  three ~20-field structs 90% identical.
- CD-005: `accessors.rs:53-84` is a wall of single-field setters (`set_budget`,
  `set_risk_measures`, `set_scheduler`, …), each mutating a resolved-run-config field
  embedded in `StudySetup` — the setter wall exists _only because_ run-config lives in the
  data struct. Nudges the "split model vs run-params" direction from optional → recommended.

_Corrected earlier read (recorded for honesty): `build_declared_node_graph`
(`node_graph.rs:230-395`, ~165 lines) is NOT a god-function — a strictly linear,
single-responsibility build dense with CVaR-order/pool-sharing contracts; splitting fragments
correctness. `NodeGraph` itself is a lean 5-field struct. Minor: `ncs_stochastic_dormant_for_test`
(`accessors.rs:158`) is a `_for_test` `pub fn` on the shipped impl — wants a `#[cfg(test)]`
glance._

### Station 1c — LP builder entries/columns (`lp/builder/{entries,columns}.rs`)

_Reframe — NOT god-files. `entries.rs` = ~1596 production + ~7746 test lines; `columns.rs` =
~1256 production + ~7094 test. Production is a well-factored family of concern-specific
`fill_*` fns (30 / 36). The 5 largest read in full are cohesive: `fill_load_balance_entries`
(`entries.rs:968`, physical KCL aggregation), `fill_generic_constraint_entries` (`:1240`,
delegates per-variable dispatch to `resolve_variable_ref`), `fill_anticipated_columns`
(`columns.rs:546`, delivery-anchored). **This is the "good pattern" reference for the setup
layer.** Size tracks entity-count, not god-ness._

**CD-007 · Sev C · asymmetry · effort M · confidence high**
Inline `#[cfg(test)]` modules dominate both files (~7.7k / ~7.1k lines, 83–85%), while sibling
builder submodules in the same dir — `template/tests.rs` (4700), `layout/tests.rs` (2979) —
use extracted sibling test files. Two conventions in one module dir; the inline giants are the
sole reason these read as god-files. Direction: promote to directory modules
`entries/{mod.rs,tests.rs}` + `columns/{mod.rs,tests.rs}`. Behavior-neutral, low-priority,
serves navigability. Minor sibling asymmetry: `fill_ncs_load_balance_entries` is its own fn
while 6 other entity contributions are inline in `fill_load_balance_entries`.

**CD-008 · RETRACTED (was Sev B duplication) — verified deliberate-by-design (WONTFIX)**
Initial read flagged `fill_parallel_water_entries` (`entries.rs:225`) vs
`fill_chronological_water_entries` (`:503`) as a ~70% duplicated skeleton. Owner-prompted
verification: the overlap is superficial (same entity-iteration order) but masks two genuinely
different LP formulations —
(1) rows: one per hydro (stage) vs one per (hydro, block);
(2) storage: stage V_in/V_out (`col_storage_in_start`) vs per-block boundaries
(`block_storage_col`, `Boundary::from_index`) — different LP variables;
(3) inflow-noise/slack/evap scale `zeta` (stage-level) vs `tau_k` (per-block) — different
variable granularity;
(4) different arc-release helper (`fill_arc_release_block_entries` vs `_chrono_`) and bucket
delivery (single vs `arrival_density`-distributed).
A shared skeleton needs ~7 mode-injection points and risks leaking one mode's assumptions into
the other (concept-mixing). Genuinely-shared pieces (`fill_prefilling_shortcircuit`,
`transit_bucket_ring`, cell iteration) are already extracted. Recorded retracted so a future
audit does not re-raise it.

---

## Station 1 — summary

Setup layer complete. Net: **7 live findings** (CD-001..007) + 1 retraction (CD-008). The
headline debt is the ★ north-star (lifecycle conflation + config-projection sprawl,
CD-001/003/004/005) — a workspace-wide pattern, not a local cleanup. The LP builder itself
(Station 1c) is the positive reference. Severities: 1×A (CD-001), 4×B (CD-002/003/004/005),
2×C (CD-006/007).

### Station 2 — Policy apply/load (`run/policy.rs`, `policy/policy_load.rs`)

_Reframe — a clean, well-designed station (a second positive reference). `policy_load.rs` =
~711 prod + ~1639 test. Exemplary type-state: `validate_policy_load<K: PolicyLoadKind>`
(`policy_load.rs:225`) is the single validation entry; `PolicyLoadProof<K>` is an unforgeable
credential (`FutureCostFunction::new_with_warm_start` accepts only `PolicyLoadProof<FullFcf>`);
`ValidatedBoundaryCuts` newtype gates `inject_boundary_cuts`. The write-borrows/read-owns
`Owned*` split is architecturally sound (residual: 4× field duplication across payload/read
pairs — watch-only, low-ROI). MPI note: policy load reads uniformly on all ranks from the
shared FS with ZERO duplicated reconstruction → CD-001's badness is **localized to setup**,
not pervasive._

**CD-009 · Sev C · duplication · effort S · confidence high**
The `policy_dir` resolve + `!exists()` guard is repeated 3× — `run/policy.rs:151` (WarmStart),
`:180` (Resume), `:273` (`load_policy_for_simulation`) — differing only in the trailing
"cannot X without Y" clause. Direction: extract `resolve_policy_dir(ctx, setup, missing_hint)
-> Result<PathBuf>`. Deliberately NOT merging the WarmStart/Resume match arms (they diverge on
Resume's iteration logic; explicit arms read clearer). Deep-read correction: the skeleton pass
saw only 2 arms; the 3rd site (simulation) is what tipped it from "borderline" to "log it".

**CD-010 · Sev B · leaky-boundary + missing-type + naming-collision · effort M · confidence high**
`EntitySlot.entity_type` (defined in cobre-io) is an untyped `u8` whose STATE-family dictionary
(`HYDRO_STORAGE=0, HYDRO_INFLOW_LAG=1, ANTICIPATED_THERMAL_STATE=2, HYDRO_TRANSIT_BUCKET=3`)
lives DOWNSTREAM in `policy_export.rs:25-31` (the writer), and the reader `policy_load.rs:26,510`
reaches across to import `pub(crate) ENTITY_TYPE_HYDRO_INFLOW_LAG` to special-case the one
family with a declared-depth config (`boundary_cut_lag_depth`, needed vs
`state_space.inflow_lag_depth`). Plus a SECOND colliding dictionary:
`cobre-io/output/dictionary.rs:31-38` `ENTITY_TYPE_HYDRO=0..HYDRO_UNIT_GROUP=8` as `i8`
(physical output entities) — same prefix, overlapping values (`=1` is THERMAL there,
HYDRO_INFLOW_LAG here), different type. No live bug (disjoint use) but a grep-hazard +
type-safety gap. North-star instance (untyped primitive where a modeled type belongs).
Direction: a typed `StateFamily` enum owned in cobre-io next to `EntitySlot` + `EntitySlot::family()`;
disambiguate the two dictionaries (`OUTPUT_ENTITY_*` vs `STATE_FAMILY_*`).

**CD-011 · Sev C · naming asymmetry · effort S · confidence high**
Two conventions for the identical borrowed/owned duality: `Owned*` prefix for records
(`PolicyCutRecord<'a>`/`OwnedPolicyCutRecord`, `PolicyBasisRecord`/`OwnedPolicyBasisRecord`) vs
`*Payload`/`*ReadResult` suffix for containers (`StageCutsPayload`/`StageCutsReadResult`,
`StageStatesPayload`/`StageStatesReadResult`) — and inverted from std (borrowed gets the plain
name). Plus `PolicyStageManifest` (`policy_load.rs:142`) is a stale name: it's the
whole-comparison validation bundle (carries `num_stages`, `n_pools`, `graph` — node/pool
concepts post-node-migration), not a per-stage record; only its `slots` come from one
(terminal) stage. Direction: one convention; rename to `PolicyLoadManifest`/`PolicyStateManifest`.

_Method lesson (recorded): skeleton reads under-serve — CD-009's 3rd site, CD-010's
cross-crate leak, and CD-011's naming split were all invisible until the definitions were read
in full. Station 3 onward: line-deep from the start, no skeleton-only verdicts._

---

## Station 2 — summary

Policy layer complete. Net: **3 findings** (CD-009 C, CD-010 B, CD-011 C). The layer is
otherwise a positive reference (type-state proof pattern). CD-010 is a north-star instance
(untyped primitive → modeled type). Running total: **10 live findings** (CD-001..007, 009..011)

- 1 retraction (CD-008). Severities: 1×A, 5×B, 4×C.

---

## Station 3 — GLOBAL CAVEAT (epic-14 active)

Everything in Station 3+ lives in **cobre-sddp**, which is **actively changing** under the
epic-14 work (not just `enumerated.rs` — the whole crate). Treat every Station 3+ finding as a
**snapshot**: verify against the live tree before acting; some may be resolved or moved by
in-flight work. Investigation stays strictly read-only.

### Station 3a — Session & training loop (in progress)

**★ Positive — the north-star pattern already exists in this layer (de-risks CD-005):**

- **Two `train`s.** `StudySetup::train` (method, `setup/orchestration.rs`) is a thin adapter
  over the god-struct that delegates to the free `train<S,C>(solver, config, fcf, stage_ctx,
training_ctx, …)` (`training/training.rs:327`) — which takes **Config** (`TrainingConfig`) +
  borrowed **Model** views (`stage_ctx`/`training_ctx`) + the **Output** (`fcf`) as _separate
  explicit params_, NOT the god-struct. The algorithm already consumes inputs/config/output
  separated; CD-005's conflation is confined to construction/storage, and the free `train`'s
  signature IS the target shape.
- **`TrainingSession` (`session/mod.rs:101`) is the exemplar.** Its ~18 fields are explicitly
  lifecycle-grouped with section comments: Borrowed inputs (`solver`/`fcf`/`stage_ctx`/
  `training_ctx`/`comm`) · Config (`config`) · Runtime handles · Rank math · Per-run scratch
  (sub-structs) · Forward scratch · Backward scratch · Result accumulators (`results`). Output
  (`fcf` mut-borrow + `results`) separated from config from inputs. The codebase KNOWS this
  pattern (TrainingSession, free `train`, policy type-state) — **StudySetup is the outlier**.

**CD-012 · Sev C · god-fn (weak rationale) · effort M · confidence high**
`run_cut_management` (`session/mod.rs:937`, ~217 lines, `too_many_lines`-allowed) inlines three
separable sub-phases: cut-selection (`:944-1117`, ~173 lines), budget enforcement
(`:1119-1153`), then a call to the already-extracted `freeze_active_cuts_into_templates`. The
selection block is self-contained (resolve root node → per-pool records → project archive
states into pool slot space → apply deactivations → trim archive) and extracts cleanly into a
`run_cut_selection(&mut self, iteration) -> Option<SelState>` method, with budget reading the
returned `(sel_state, record_by_pool)`. The stated rationale ("splitting would pass every field
individually") defends against FREE-function extraction — a category error, since the natural
refactor is `&mut self` methods (direct field access, thread just 2 values). Correctness-dense
(nodes[0]-as-root care `:953-959`, `:1124-1126`) → effort M. Affirmed cohesive (NOT flagged):
`run_iteration` (`:413`, 141 lines) is a clean linear phase-ordering driver.

_Entry watch-items (Sev C, not yet promoted): `run_training_phase` (`run/training.rs:59`, 195
lines) interleaves run / MPI-stats-aggregate / summary-assemble / print; dual stats-DTO
assignment (`GlobalTrainingStats` → both `TrainingSummary` print + `MetadataTrainingSolveStats`
persist, field-by-field)._

### Station 3a — memory-architecture design note (not a CD finding)

The training memory model is **per-pool/per-node right-sized end-to-end** and scales correctly
to node-native / branching / DECOMP — a strength, not debt.

- **Cut pools** — `FutureCostFunction::new_per_pool` (`cut/fcf.rs:146`) sizes each pool by its
  OWN `visit_bounds[p]`: `pool_capacity(wsc, max_iterations, visit_bound[p])`, EXACT for
  enumerated (`pool_cut_stride`), upper-bound capped at `forward_passes` per pool for sampled.
  Small node → small pool. `BasisStore` (`max_local_fwd × num_nodes`) is per-node too.
- **Only max-sized buffer** — per-WORKER reusable scratch (`slot_increments`, DCS scratch;
  `session/mod.rs:172-177`, `:231-234`), sized to the largest pool a mobile worker may claim.
  `O(n_workers × largest_pool)`, bounded by core count — negligible vs the `Σ_pools` storage.
  Not a scaling wall.
- **Arena model** — allocate-once in `TrainingSession::new`, reuse across iterations (no
  hot-path alloc); results moved out in `finalize`.

Two watch-axes remain (deliberate tradeoffs, NOT debt):

- **B — monotonic growth**: pools are append-only (deactivation toggles bounds, never frees —
  required for slot-identity basis reconstruction). Per-pool memory grows
  `O(max_iterations × visit_bound[p])`; the long-run ceiling is total cuts GENERATED, not
  active. No better design exists under the current warm-start contract.
- **C — partial-borrow friction**: the one-big-owned-buffer `TrainingSession` forces manual
  borrow-splitting (`ForwardPassInputs::from_session_fields`). Minor, inherent to Rust.

_Retracted en route (owner-caught): an earlier "worst-case-uniform sizing over-allocates on
heterogeneous graphs" concern — FALSE. The cut pools are per-pool sized; only per-worker scratch
is max-sized, and that's core-bounded. Pre-allocation scales to branching precisely because
per-pool counts (enumerated) / bounds (sampled) are known ahead._

### Station 3a — COMPLETE

Net: **1 finding (CD-012, Sev C)** + entry watch-items + the memory design note above. The
session layer is a POSITIVE reference (`TrainingSession` = north-star exemplar); the only smell
is CD-012's inline selection block.

### Station 3b — Forward pass (`forward_pass_state.rs`, `forward/`)

**CD-013 · Sev B · asymmetry (retrofitted-variant class) · effort S · confidence high**
`ForwardPassState::run` (`forward_pass_state.rs:329`, 184 lines) dispatches its traversal match
asymmetrically: `Enumerated(plan) => self.run_enumerated(...)` (a clean extracted method) but the
`Sampled` arm is a ~122-line inline block (`:386-508`: partition records → resize worker stats →
apply profile → parallel `run_forward_worker` dispatch → `post_process_worker_results`). No
borrow-checker reason (the `Sampled` pattern binds nothing; the block doesn't borrow the taken
`traversal` — the `:379-382` comment notes only Enumerated's `plan` does). Extract
`run_sampled(&mut self, inputs, &sampler)` → `run` becomes a 2-line dispatcher mirroring
`run_enumerated`. Analogous to CD-006 (sibling treatment asymmetry).

_Watch (Sev C): minor setup duplication between the Sampled block and `run_enumerated` —
forward-profile application, `worker_timing_buf` reset, and `terminal_has_boundary_cuts`
resolution appear in both (`:435-449` vs `:536-555`); a `prepare_workspaces_for_forward` helper
would absorb it. Borderline._

**Affirmed cohesive (NOT flagged, line-read in full):**

- `run_forward_worker` (`:774`, 206 lines) — per-worker hot kernel (stages × scenarios), dense
  with determinism/ordering contracts (reset-then-reload `:842-858`, per-trajectory node walk,
  lag seeding, chain-parity advance). Length tracks the algorithm.
- `run_forward_stage` (`forward/stage_solve.rs:41`, 211 lines) — per-stage solve; frozen-vs-DCS
  branch, warm-start basis keyed by NODE (`:146-149`, sddp.md contract), theta zeroing, lag
  accumulation, basis-capture-frozen-only. The two solve arms share only `fill_unscaled` (~2
  lines) — no false-DRY.
- `post_process_worker_results` (`:638`, 117 lines) — linear cross-worker aggregation in
  canonical order.

**Good patterns:** `ForwardWorkerParams`/`StageKey` grouping structs (arg-count + borrow
management); pervasive `mem::take` allocation reuse; `StageSolvePrep::run` as the single shared
solve-prep.

### Station 3b — verdict

Forward pass is well-factored — a positive-leaning area like 3a. Net: **1 finding (CD-013, Sev
C)** + 1 watch. The delegation chain run → worker → stage → StageSolvePrep is clean. (3c
`enumerated.rs` DEFERRED — most epic-14-active file.)

### ★ DEBT CLASS — retrofitted-variant asymmetry (owner-elevated, Sev B)

**Pattern:** when a 2nd (or Nth) variant is retrofitted onto a mode axis, the ORIGINAL
implementation must be PROMOTED to a co-equal named sibling — never left inline as "the default
when we don't branch to the new one." The dispatcher then does nothing but dispatch. Leaving the
original inline is a structural/**extensibility** debt (not cosmetic): the day a 3rd variant
lands you either extract-under-pressure or bolt on another inline arm, and the "N co-equal
modes" mental model is obscured. As the project fans out into branching/DECOMP variants this
compounds.

**Fix-shape (the codebase's own correct examples — copy these):**

- ✅ **Backward scheduler** (`backward_pass_state.rs:1526`): `use_by_node` →
  `process_stage_backward_by_node` vs `process_by_scenario_backward`, BOTH extracted; the
  by-node arm's extra inline code is genuinely by-node-specific setup (block ordering), not the
  original inline. Symmetric. THE MODEL.
- ✅ **DCS solve** (`backward/by_scenario.rs:113`): `solve_opening` → `Frozen => solve_frozen` /
  `Lazy => solve_lazy`, co-equal methods. Symmetric.

**Confirmed debt instance:**

- ❌ **CD-013** — forward `run`: `Enumerated => run_enumerated` (method) vs `Sampled => inline`
  ~122-line block. The sampled original was never promoted. Fix: extract `run_sampled` so `run`
  is a 2-line dispatcher.
- ❌ **CD-015** — backward `compute_one_backward_node`: by_node's post-dispatch aggregation is
  extracted (`by_node_finish`) while by_scenario's — the ORIGINAL scheduler — is inline
  (`:1624-1664`). Class instance #2. Fix: extract `commit_by_scenario_cuts` (also relieves CD-014).

**Still to scan for this class:** simulation (Station 4 — sampled vs enumerated/census is the
prime next suspect); policy/output write vs read paths. **Checked-and-clear:** LP-builder water
block modes (parallel/chronological both extracted — CD-008 retracted); cut-selection strategies
(trait dispatch, symmetric); frozen/DCS forward-stage arms (both delegate).

### Station 3d — Backward pass (`backward_pass_state.rs`, `backward/`)

_Trustworthy snapshot: owner confirms the epic-14 remnant is only raw-index → newtype renaming;
structure is stable._

**Affirmed clean/cohesive (line-read in full):**

- `BackwardPassState::run` (`:500`, 160) — orchestrator; reverse-topological level sweep;
  defensive rank-consistency check (`:575-598`, allreduce Min/Max on `n_workers_local`).
- `run_one_backward_level` (`:1210`, 177) — per-level driver; owns the sddp.md per-level-exchange
  contract (ONE state `allgatherv` + ONE batched cut exchange per level); mem-swap scratch reuse.
- `process_stage_backward` (`:1825`, 149) — by_scenario worker kernel (buffer pre-alloc →
  per-worker `par_iter` → `process_by_scenario_backward`).
- `process_stage_backward_by_node` (`backward/by_node.rs:131`, 268) — by_node claim-loop kernel;
  large-but-cohesive (claim → child-run → opening), dense with canonical-ω / warm-start-only /
  per-child-pool contracts.
- Small correctness leaves well-factored: `duals_extraction` (extract_duals_from_view 29),
  `outcome_aggregation` (write_opening_outcome 35), `lp_setup` (patch_opening_bounds 41),
  `state_exchange` (exchange 43), `by_scenario` (`solve_opening` symmetric dispatch).

**CD-014 · Sev C · god-fn (weak rationale) · effort M · confidence high**
`compute_one_backward_node` (`:1399`, 290 lines, biggest method in crate) — extractable
sub-phases: successor-outcome reification (`:1438-1507`) + by_scenario aggregation (`:1624-1664`).
Same weak-rationale-vs-method-extraction pattern as CD-012 (`too_many_lines` rationale defends
against FREE-fn extraction; natural fix is a `&mut BackwardPassState` method).

**CD-015 · Sev B · retrofitted-variant asymmetry (class instance #2) · effort S · confidence high**
Inside `compute_one_backward_node`, by_node's aggregation is extracted (`by_node_finish`) while
by_scenario's — the ORIGINAL scheduler — is inline (`:1624-1664`). Fix: extract
`commit_by_scenario_cuts` — converges with CD-014 (one extraction resolves both). See ★ DEBT CLASS.

**CD-016 · Sev C · duplication · effort S · confidence high**
The two backward scheduler workers duplicate ~25 lines of per-worker `backward_accum` buffer
pre-allocation: `process_stage_backward` (`:1862-1892`) and `process_stage_backward_by_node`
(`by_node.rs:183-214`) both resize `outcomes`/`slot_increments`/`metadata_sync_contribution`/
`per_opening_stats` identically (each then adds its own extras — by_node's `block_pivot_*`,
by_scenario's `agg_arena`). Extract `prepare_shared_backward_buffers(ws, n_openings, cut_n_state, pop)`.

### Station 3d — verdict

Backward pass is well-structured: clean orchestrator + level driver, cohesive worker kernels,
SYMMETRIC scheduler dispatch. Net: **3 findings (CD-014 C, CD-015 B, CD-016 C)**. Density
concentrated in `compute_one_backward_node` + the worker kernels, as expected for the
cut-generation heart.

### Station 3e — Lower bound & solve-prep (`lower_bound.rs`, `stage_solve_prep.rs`)

_Clean station — **0 findings**. The most important positive reference of the audit._

**★ `StageSolvePrep::run` (`stage_solve_prep.rs:85`, 122 lines) is the divergence ANTI-DOTE** —
the exemplar CD-001 and the retrofitted-variant class should be refactored toward. A SINGLE
OWNER the forward / backward / LB / simulation solve sites all route through, so they patch the
LP identically (D15 "patch NCS identically"). Per-call variation points are modeled as TYPED
ENUMS (`LoadNoise::Present/Absent`, `InflowNoise::Transform/PreBuilt`), never booleans; the
NCS-patch and commitment-reconcile are DELIBERATELY not variation points (derived internally) —
its own doc: "An opt-in hook is what let four call sites silently lose it." This is exactly how
CD-001's non-root reconstruction and the config-projection sprawl SHOULD be modeled: one owner,
typed variation points, no opt-out for the invariants.

**Affirmed cohesive (line-read):**

- `evaluate_lower_bound` (`lower_bound.rs:436`, 62) — clean orchestrator; rank-0 runs the loop;
  reconcile-before-broadcast MPI pattern.
- `lb_evaluate_stage_0` (`:173`, 145) — cohesive per-opening loop (truncation precompute prologue
  - per-opening solve **via `StageSolvePrep::run`**), so the LB can't diverge from forward/backward
    on NCS patching (D15). Defended per comments.md §12.
- `lb_aggregate_and_broadcast` (`:393`) — good HARD length-check guard (objectives vs weights,
  `:405-412`), not a debug assert.
- `assemble_outcome_weights` (`:356`) is a thin delegating WRAPPER to `node_graph`'s single owner
  (`:361`) — not a duplicate; single-owner contract honored. (Minor: two fns share the name — a
  deliberate local alias.)

### Station 3e — verdict

Clean, well-architected. **0 findings.** `StageSolvePrep` is the positive model the whole audit
points toward — one owner, typed variation points, invariants that can't be opted out of.

### Station 3f — Cut pool, FCF & rows (`cut/{pool,fcf,row,row_map,basis_reconstruct}.rs`)

_Clean station — **0 findings**. Well-engineered cut-pool substrate._

- **`CutPool`** (`pool.rs:58`, ~14 fields) — a well-designed SoA append-only data structure:
  flat `coefficients` + parallel `intercepts`/`metadata`/`active`; `cached_active_count`
  maintained incrementally (O(1) `active_count`); `set_active` the single activity primitive
  `deactivate`/`apply_updates`/`replace_selection` build on (each keeps the cache consistent,
  atomically). `add_cut` embodies slot-identity + double-insert guard. Biggest method 96 lines
  (warm-start ctor). No god-struct/fn.
- **`FutureCostFunction`** (`fcf.rs:61`, 3 fields) — clean facade over `Vec<CutPool>`; per-pool
  ops delegate to `CutPool`.
- **`row.rs`** — focused cut-row builders (`push_cut_row` owns the Benders sign negation; alloc
  vs `_into` pairs for buffer reuse). **`row_map.rs`** — small `CutRowMap` slot→row mapping.
  **`basis_reconstruct.rs`** — focused; the two entry points `reconstruct_basis` (frozen) /
  `reconstruct_basis_uniform_basic` (DCS) are BOTH named siblings → **retrofitted-variant class
  checked-and-clear** (the DCS variant was extracted co-equally).

### Station 3f — verdict

Clean, well-engineered substrate. **0 findings.** Training + cut-pool layers remain
positive-leaning; structural debt stays in setup/input (Stations 1–2).

**CD-017 · Sev C · quality + hardening (measure-first) · effort S · confidence high** — `cut/basis_reconstruct.rs`
Owner-requested deep-dive on warm-start basis quality. **No correctness bugs** — the
BASIC→LOWER demotion (`:314-323`) can only produce a slower (rejected/repaired) basis, never a
wrong answer (HiGHS cold-starts a bad basis, CLP repairs it). Three items:

1. **Latent hazard (add assert):** `reconstruct_col_statuses` (`:208-210`) silently TRUNCATES
   when `stored.col_status.len() > target.num_cols` (premise #1, `:37`, "assumed and never
   verified"). A violation surfaces downstream as a confusing `BasisShapeMismatch` _deficit_,
   masking the cause. Add `debug_assert_eq!(stored.basis.col_status.len(), target.num_cols)` at entry.
2. **Dead field (YAGNI):** `ReconstructionStats::new_tight` (`:100-104`) is "always zero … kept
   for telemetry stability" — a field that can never be non-zero.
3. **Best-effort lever (measure first):** preserved cuts + template rows/cols get OPTIMAL status
   (copied from the stored basis); NEW cuts get BASIC — a **count-balance necessity, not a quality
   choice** (`:15-20`: seeding LOWER would deficit) — then `demote-excess-from-newest`, which
   correctly targets highest-slot ≈ newest ≈ likely-to-bind cuts. The `CutMetadata` binding
   history (`active_count`, `last_active_iter`) is NEVER consulted — demotion is purely positional.
   A metadata-driven demotion is the principled version, but position already correlates with
   binding-likelihood, so the gain is likely marginal. **Before optimizing, check the
   `basis_consistency_failures / basis_offered` ratio (SolverStatsDelta):** low → basis quality is
   fine, leave it; high → invest, and track warm-vs-cold simplex iters/iteration to quantify.

### Station 3g — Cut selection & DCS (`cut/cut_selection.rs`, `cut/dcs.rs`)

**CD-018 · Sev C · bad-abstraction (paradigm conflation) · effort M · confidence high**
`CutSelectionStrategy` (enum, `cut_selection.rs:144`) conflates TWO selection paradigms under
one type: periodic value-based (Level1/Lml1/Dominated — all share `select_for_stage`'s
gemm-based value sweep + `apply_column_rule`) and lazy-solve DCS (Dynamic). The Dynamic variant
does NOT honor the enum's implied `select_for_stage` interface — it EARLY-RETURNS empty
(`:320-326`) and its real logic lives in `dcs.rs::lazy_solve_preloaded`. Symptoms: the `if let
Dynamic = self { return empty }` guard + the `unreachable!` for Dynamic in `run_cut_management`
(should_run always false) + `DcsParams::from_strategy` extracting DCS params back out of the
variant. Config-unification (one user-facing enum) is the rationale, but the internal dispatch
pays with an opt-out variant. Direction: model the two paradigms as distinct internal types
(value-based enum + DCS its own), unified at the config boundary via `From`. Related to the
retrofitted-variant class (DCS is the variant that doesn't fit the abstraction) — different flavor.

**Affirmed cohesive (line-read):**

- `select_for_stage` (`:312`, ~138) — value-based selection kernel (eligibility → parallel
  m-block gemm value sweep + `apply_column_rule` → reactivation/deactivation). Cohesive apart
  from the Dynamic early-return.
- `score_violated_candidates` (`dcs.rs:231`, ~108) — DCS candidate-scoring (unscale → gemm cut
  values → violations → sort D5 → top-nadic). Cohesive.
- `lazy_solve_preloaded` (`dcs.rs:589`, ~133) — DCS lazy-solve loop (continue/fresh setup →
  score+append+re-solve until no violations → TC fallback preserving exactness). Uses the DCS
  basis path + the same cross-node-reuse rejection as the frozen path. Cohesive.

_Watch (Sev C, minor): `lazy_solve_preloaded`'s TC fallback (`:688-719`) duplicates the
score+append+solve sequence from the lazy loop — a special "last iteration at nadic=∞". Borderline._

### Station 3g — verdict

Well-factored kernels; **1 finding (CD-018, Sev C)** — the CutSelectionStrategy paradigm
conflation. The DCS lazy-solve machinery is cohesive and correctness-dense (exactness preserved).

### Station 3h — Cut sync & wire (`cut/cut_sync.rs`, `cut/wire.rs`)

**CD-019 · Sev C · dead-code / retrofit-remnant · effort S · confidence med**
`sync_cuts` (`cut_sync.rs:250`, 129 lines) + `sync_packed_records` (`:474`, 124 lines) — ~253
lines of `pub fn` production methods with NO production caller: every call site is in
`cut_sync.rs`'s own test module (past the test boundary at 886), and no other file references
them. They appear to be the legacy SINGLE-pool cut sync, superseded by the node-native
MULTI-pool `sync_level_records` (`:598`) when the node engine landed — left in place with no
`#[allow(dead_code)]`/reserved annotation. Their test suite gives false "it's used" confidence.
Retrofitted-variant class, **dead-code flavor** (the original wasn't removed OR promoted — it was
orphaned). Per CLAUDE.md's "leave no dead code" bar: verify not an external API, then remove (if
legacy) or annotate reserved-with-rationale + tighten `pub`→`pub(crate)`. Confidence med pending
owner confirmation it's not a reserved seam.

**Affirmed cohesive + E6-clean:**

- `sync_level_records` (`:598`, 191) — the live per-level batched exchange (pack via shared
  `pack_pool_into` → per-pool count `allgatherv` → validate → byte `allgatherv` → deserialize +
  insert). Dense DEFENSIVE MPI code (overflow / corrupt-count guards throughout,
  chain-byte-identical degeneracy). Cohesive.
- `wire.rs` — the E6 dual-owned cut format: `CUT_WIRE_VERSION = 2`, explicit version-reject, no
  compat shim. **E6 contract HONORED** — round-trip + reject-wrong-version tests present
  (`deserialize_cut_rejects_wrong_version_byte` `:750`, `deserialize_cut_rejects_wrong_version`
  `:797`). Checked-and-clear.

### Station 3h — verdict

Live path (`sync_level_records`) + wire codec are well-engineered and E6-compliant. **1 finding
(CD-019, Sev C med)** — the orphaned legacy single-pool sync methods.

**CD-020 · Sev C · over-engineering + over-exposure + doc-drift · effort S · confidence high** — `cut/wire.rs`
Owner question — does MPI cut-wiring support multiple formats? **No**: `cut::wire` supports
exactly ONE format (v2); `deserialize_cut` REJECTS any other version with NO decode path. The
version byte is a reject-GUARD, not a multi-format dispatcher; all sync methods use the one
`cut::wire` codec. But the version byte is questionable for its context:

- **`cut::wire` is MPI-ONLY** — cobre-io's disk format (policy checkpoint) is FlatBuffers
  (`policy/codec.rs`), NOT `cut::wire`; nothing persists it. Confirmed: cobre-io never references it.
- In one MPI run every rank is the SAME binary → same `CUT_WIRE_VERSION`, so a cross-rank
  version MISMATCH is IMPOSSIBLE. As a cross-version-compat mechanism the byte guards a scenario
  that cannot occur.
- Its real value is a per-record **corruption/alignment tripwire** (a misaligned `allgatherv`
  deserialize lands on non-v2 bytes → reject), complementing `sync_level_records`' count guards —
  legitimate defensive use, but that intent reads more honestly as a fixed MAGIC/format-tag than
  a "version" with bump discipline + "no compat shim" framing implying non-existent cross-version
  semantics.

Sub-issues: **over-exposure** — `cut::wire` fns are publicly re-exported (`lib.rs:112`) for an
MPI-internal format; **doc-drift** — comments.md E6 (`:536`) says cut_sync "serialises via
`cut::wire` wire-version 1" while code is `CUT_WIRE_VERSION = 2`. Direction: reframe as a fixed
MAGIC/format-tag OR document a future persisted-cut intent (Voice 4); tighten the `pub` re-export
to `pub(crate)` unless externally consumed; fix the E6 "1"→"2" drift. **No correctness bug** —
single format, correctly guarded.

### Station 3i — Workspace & captured basis (`workspace/workspace.rs`)

**CD-021 · Sev C · large-module / organizational-asymmetry · effort M · confidence high**
`workspace.rs` is 1148 prod lines holding NINE distinct structs — `CapturedBasis`,
`WorkspaceSizing`, `BackwardAccumulators`, `ByNodeScratch`, `ScratchBuffers`, `SolverWorkspace`,
`WorkspacePool`, `BasisStore`, `BasisStoreSliceMut`. Cohesive (all per-worker arena machinery)
but a flat mega-file, ASYMMETRIC with the crate's directory-module convention (`setup/`,
`lp/builder/`, `training/session/`, `cut/` are all split dirs); `workspace/` already has `mod.rs`

- `context.rs` siblings. Split by concern (`captured_basis.rs`, `accumulators.rs`, `scratch.rs`,
  `pool.rs`, `basis_store.rs`). Low priority, navigability-only.

**CD-020 gains a 2nd instance:** `BASIS_BROADCAST_WIRE_VERSION = 2` (`workspace.rs:75`) is a
SECOND MPI-ephemeral format carrying disk-style versioning — same pattern as `cut::wire`
(broadcast, same-binary, cross-version-mismatch impossible; the version byte's real value is
corruption-detection framed as versioning). CD-020 is a PATTERN across both MPI wire formats.

**Affirmed EXEMPLARY (positive reference for wire codecs):** the `CapturedBasis`
`to_broadcast_payload`/`try_from_broadcast_payload` pair (`:152`, `:214`, ~117-line decode) —
symmetric encode/decode (doc lists the layout on both sides), EXHAUSTIVE bounds-checking on every
read (truncation → descriptive `Validation` naming stage + expected/have), version+reject,
single-owner discriminant mapping (`BasisStatus::to_discriminant_code`, injective so CLP
`Superbasic`/`Fixed` round-trip). NO hidden bugs — every read guarded. Minor defensive-consistency
nit: unchecked `cursor + len` in bounds checks vs `cut_sync`'s `checked_add` (a theoretical
overflow on a corrupt length, unreachable since the broadcast source is the trusted same-binary
rank 0).

### Station 3i — verdict

Well-organized structs + an exemplary wire codec. **1 finding (CD-021, Sev C) + CD-020 2nd
instance.**

---

## ★ STATION 3 COMPLETE — Training

9 sub-stations (3a–3i; 3c `enumerated.rs` deferred — epic-14). **10 findings** (CD-012..021) +
the memory-architecture design note (3a). Verdict: **the training subsystem is well-architected.**
Positive references discovered: `TrainingSession` (lifecycle-grouped struct), `StageSolvePrep`
(single-owner solve-prep, the divergence anti-dote), `CutPool` (SoA append-only), the
`CapturedBasis` codec (exhaustively-checked). Debt is mostly cosmetic/organizational (9×C) +
the retrofitted-variant class (CD-013/015 B, CD-018/019 C). Severity within St.3: 2×B (CD-013,
CD-015), 8×C. **No Sev A** in training — the structural debt (Sev A/B) is concentrated in
setup/input (Stations 1–2), exactly as the north-star predicted.

Running total (through St.3): **20 live findings + 1 retraction.** Severity: **1×A, 7×B, 12×C.**

---

## ★ Station 3 — RE-VERIFICATION (post epic-14/15, plan 75/83)

**Why:** the working tree is a LIVE moving target — agents edit + `git reset` (reflog
HEAD@{4}/{5}); epic-15 (tickets 087–089) landed a node-native **enumerated backward fork**
(+495 prod lines in `backward_pass_state.rs`; +103 in `cut_sync.rs`; +96 in `session/mod.rs`).
My Station 3d backward audit read a transient pre-fork state. All St.3 findings re-verified
against the current committed tree (8903937c).

**RESULT: every Station 3 finding HOLDS — none invalidated.** Refs shifted; CD-012 & CD-014
grew; CD-013 sharpened; one NEW finding (CD-022).

| Finding                                     | Status               | Current ref / note                                                                                                               |
| ------------------------------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| CD-012 run_cut_management                   | HOLDS (worse)        | now ~277 lines (was 244), `session/mod.rs:960`                                                                                   |
| CD-013 forward sampled-inline               | HOLDS, **SHARPENED** | `forward_pass_state.rs` run@338, sampled inline @395                                                                             |
| CD-014 compute_one_backward_node            | HOLDS                | now `:1674`, ~288 lines                                                                                                          |
| CD-015 by_scenario-inline vs by_node_finish | HOLDS                | `:1897-1938` vs `by_node_finish:1860`                                                                                            |
| CD-016 backward buffer prealloc dup         | HOLDS                | both workers still prealloc the same `backward_accum` buffers                                                                    |
| CD-017 basis_reconstruct                    | HOLDS                | ±3 lines                                                                                                                         |
| CD-018 CutSelectionStrategy conflation      | HOLDS                | +40 lines, structure intact                                                                                                      |
| CD-019 dead sync methods                    | HOLDS                | `sync_cuts`/`sync_packed_records` STILL no production caller (the +103 went to `sync_level_records`/hygiene, not a resurrection) |
| CD-020 MPI-ephemeral versioning             | HOLDS                | 2 instances (`cut::wire` + `CapturedBasis`)                                                                                      |
| CD-021 workspace.rs mega-file               | HOLDS                | +28 lines                                                                                                                        |

**CD-013 SHARPENED (Sev B):** epic-15 gave the BACKWARD pass the exact symmetric fix CD-013
prescribes — `BackwardPassState::run` (`:492-501`) is a clean 2-arm dispatcher over extracted
`run_sampled_backward` (`:523`, the PROMOTED old level-driver) + `run_enumerated_backward`
(`:713`). Forward `run` still inlines its sampled arm → now a **direct in-crate inconsistency**:
backward = model, forward = outlier. Fix = make forward `run` mirror backward `run`.

**CD-022 · Sev B · duplication (converges with CD-014) · effort M · confidence high** — NEW (epic-15 fork)
The successor-outcome reification (loop over `node_graph.successors` → `build_delta_cut_row_batch_into`
→ `SuccessorEntry` → `SuccessorOutcomes`, ~66 lines) is DUPLICATED between the sampled path
(`compute_one_backward_node:1708-1777`) and the enumerated path (`run_enumerated_backward:768-834`).
Deliberate — the fork's own rationale (`:706-711`) copied rather than shared "because the sampled
path must stay byte-frozen." A 2-site sync burden: any reification change (e.g. a new
`SuccessorEntry` field) must touch both. **Converges with CD-014**: extracting
`reify_successor_outcomes(&mut self, inputs, node_pos, successor_stage) -> SuccessorOutcomes` as a
shared helper de-god-fns `compute_one_backward_node` (CD-014) AND removes this copy (CD-022) —
one byte-neutral extraction verified against the sampled parity goldens, two findings.

**POSITIVE — the fork validates our prescription:** `run_sampled_backward`/`run_enumerated_backward`
is the CORRECT retrofitted-variant handling (both extracted, clean dispatcher, old sampled path
PROMOTED not left inline). The codebase itself chose the pattern CD-013 recommends. The enumerated
driver is a legitimate fork (persisted-state read + replicated per-node solve + no MPI exchange),
not a false-DRY; its only debt is the reification copy (CD-022).

**New sddp.md branching contracts** now pin the branching backward: "Joint risk applied once over
the flattened successor×opening vector" + "the branching backward integrates every successor
exhaustively" (child-0 collapse forbidden) — oracled by `joint_cvar_differs_from_nested_per_child`
and `water_binding_external_fan_final_lb_matches_extensive_form`.

Running total: **21 live findings + 1 retraction.** Severity: **1×A, 8×B, 12×C.**

---

## ★ Station 3 — epic-16 review (shared-primitive hoist + census-sim landing)

**Trigger:** epic-16 "noticed a gap in the simulation definitions" and extracted behaviour
into common homes usable by **both** training and simulation. Two changes: a committed hoist
(`f527e3f9`) + uncommitted census-simulation work. Owner asked whether Station 3 needs review.

**VERDICT: every Station 3 finding HOLDS; none invalidated; no training file behind
CD-012..022 was touched.** Scope was narrow and is fully re-verified.

**What the hoist (`f527e3f9`) moved (all 100%-similarity, byte-neutral relocations):**

| Primitive                                         | From                                                         | To (neutral home)                                               | Station-3 impact                                                                   |
| ------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `claim_scatter` (ClaimCursor + canonical_scatter) | `training/claim_scatter.rs`                                  | crate-root `claim_scatter.rs` (`pub(crate)`)                    | none (was not behind a finding)                                                    |
| `AccumSnapshot`                                   | `training/forward/enumerated.rs`                             | `stochastic/noise.rs` (beside `accumulate_and_shift_lag_state`) | none; clean home (cobre-sddp-internal module, not the infra crate)                 |
| root→leaf walk                                    | free `walk_root_to_leaf` in `training/forward/enumerated.rs` | **method** `EnumeratedPlan::walk_path` (`node_graph.rs`)        | **reinforces CD-006** (free-fn → method-on-its-type is exactly CD-006's direction) |

- **CD-006 HOLDS, mildly reinforced.** The NodeGraph free-fn list (`frontier_node`, `node_parent`,
  … still imported as free fns in `enumerated.rs`) is untouched; but the hoist _chose the method
  form_ for `walk_path`, validating CD-006's prescription in a neighbouring type.
- **CD-012..022 untouched.** `forward_pass_state.rs`, `backward_pass_state.rs`, `session/mod.rs`,
  `cut/*`, `workspace/*` appear in neither diff — line refs stable since the last re-verification.
- **Deferred 3c (`training/forward/enumerated.rs`)** shrank −103 lines (walk + AccumSnapshot
  hoisted out). Still unaudited by design; no recorded finding affected.

**★ The hoist is a POSITIVE — record it as the correct layering discipline.** It relocated shared
primitives to neutral homes _before_ the second consumer (census sim) arrived, avoiding a
`simulation→training` layering inversion by construction. This is the disciplined move CD-001 was
the violation of — the codebase applying the audit's own north-star pro-actively.

**CD-023 · Sev C (B-risk: declared-undesirable inversion) · leaky-boundary / incomplete-hoist ·
effort M · confidence high** — DIRECTION RATIFIED (Read A)
The hoist established a principle it did **not** finish applying. `StageSolvePrep`
(`training/stage_solve_prep.rs`) — the Station 3e ★ single-owner solve-prep that forward /
backward / LB **and simulation** all route through — still lives under `training/`, yet
`simulation/pipeline.rs:50` imports `training::stage_solve_prep` (**pre-existing**: present at
HEAD, not introduced by epic-16). By the hoist commit's own stated rule ("avoid a
simulation→training layering inversion"), this IS that inversion — and `stage_solve_prep` is a
_more_ general shared primitive (used by every solve site, not just enumerated) than the three
that were hoisted, so the rationale applies a fortiori. Fix-shape = the hoist's own: byte-neutral
move to a neutral home (e.g. crate-root `stage_solve_prep.rs` or a `solve_prep` module) + retarget
6 import sites (5 training + 1 simulation) + tests. **Direction (ratified 2026-08-07 — Read A):**
`simulation → training` IS the inversion the hoist commit set out to eliminate; `stage_solve_prep`
is the residual it left behind (a _more_ general shared primitive than the three hoisted, so the
rationale applies a fortiori). Fix = the hoist's own shape: byte-neutral move to a neutral home +
import retarget. The rejected alternative (Read B: "simulation is a legitimate downstream consumer
of training machinery") was declined for consistency with the principle the team declared in
`f527e3f9` — the codebase has already taken the position that this dependency direction is an
inversion. **Sequencing:** land alongside/after the CD-013-class simulation-dispatch symmetry work
(Station 4) so the sim-side solve-prep call site is retargeted once, not twice.

**Sharpened Station 4 open threads (NOT locked — simulation is mid-flight / uncommitted):**

- **Retrofitted-variant class has MATERIALIZED in simulation.** The new `simulation/enumerated.rs`
  is a co-equal extracted census engine (like epic-15's `run_enumerated_backward` — the correct
  shape), and `orchestration.rs::simulate` now dispatches via the same `Traversal::resolve`.
  **Station 4 must verify** the _sampled_ simulation path was promoted symmetrically (not left
  inline) and the `simulate` dispatcher is clean — the CD-013 question, now in simulation.
- **CD-022-scale cross-module mirror.** `simulation/enumerated.rs` doc: "mirrors
  `training::forward::enumerated::run_enumerated_forward` **step-for-step**." A whole engine
  mirrored across the training/sim boundary = a larger CD-022. **Mitigant already in place:** it
  _reuses_ the hoisted primitives (claim_scatter / AccumSnapshot / walk_path) rather than copying
  them, so the mirror is partial (shared kernels, forked orchestration). Evaluate the residual
  copy once the engine stabilises.

Running total (through the epic-16 review): **22 live findings + 1 retraction.**
Severity: **1×A, 8×B, 13×C.** (CD-023 added, Sev C.)

---

## Station 5 — Outputs (`run/outputs.rs`, cobre-io writers, cobre-python parity)

Done OUT OF ORDER: Station 4 (Simulation) deferred (epic-16 mid-flight); Station 5 audited
while its targets are stable (confirmed not in any epic-16 diff). Return to Station 4 after
epic-16 stabilises, re-checking the Station 4↔5 output-coupling then (did the census engine add
outputs Station 5 must mirror in Python parity?).

**Verdict: the outputs layer is largely well-architected.** `outputs.rs` (212 lines) is a thin
orchestrator (positive baseline); the cobre-io writers are a legit WRITER-FAMILY (per-entity
`*WriteRecord` + `build_*_batch`, size tracks entity count — the LP-builder pattern, NOT a
god-module); Python already practices single-owner discipline internally (the `_if_any` helpers).
The real debt is one structural item — the ★ north-star, again.

**↩︎ CLEARED (not a CD finding) — `resolve_computed` is NOT a god-fn.** The Station-2 "~154-line
god-fn candidate" flag is retracted. The real function is `resolved_parameters.rs:323→417`
(**~94 lines**), cohesive (uniform id-extract → early-return the 2 stage-invariant variants →
stage-varying loop for the other 5), carrying two correct load-bearing contracts (the
`ReferenceVolume` retired-source-of-truth guard `:378`, the `ReferenceTurbine` StageId-keying
contract `:387`). Both the Station-2 estimate (~154) and this session's awk (~221) were inflated
skeleton artifacts — the awk "next fn" skipped two column-0 `pub fn` postcard helpers and a
doctest, landing on a test helper. The no-skeleton lesson, reconfirmed.

**CD-025 · Sev B · duplication + missing-seam (★ north-star instance) · effort M · confidence high** — OWNER-AGREED
Training-output orchestration is HAND-MIRRORED across the CLI/Python crate boundary:
`write_training_outputs` (`cobre-cli/.../run/outputs.rs:57`) and `write_training_artifacts`
(`cobre-python/src/run.rs:478`) call the same cobre-io writers (`write_checkpoint` with an
identical `CheckpointParams`, `write_solver_stats`, `write_row_selection_records`,
`write_training_results`) with the same guards — kept aligned only by the Python-parity HARD RULE

- mirror comments, with NO shared owner. The tell: Python factored the fpha/evap/deviation writes
  into shared `_if_any` helpers (`run.rs:554/574/597`) "so both Python sites emit it identically" —
  but they live IN cobre-python, so the CLI can't reach them and RE-INLINES its own copies
  (`outputs.rs:89-121`). CD-023's incomplete-hoist shape across a different boundary. Direction:
  hoist the shared "which outputs + guards" helpers to a crate BOTH already depend on (cobre-sddp or
  cobre-io); the `_if_any` helpers prove the shape. The Python-parity hard rule existing AT ALL is
  the evidence this drifts. North-star: model the output-set as one owner with typed variation
  points (error type, progress sink, `OutputContext` source), not two hand-synced copies.

**CD-026 · Sev C · god-fn by repetition / duplication · effort S–M · confidence high** — OWNER-AGREED
`SimulationParquetWriter::write_scenario` (`cobre-io/.../simulation_writer.rs:627`, ~247 lines)
repeats one write-partition skeleton 12× (costs, hydros, hydro_bus_generation, thermals,
exchanges, buses, pumping, contracts, non_controllables, inflow_lags, in_transit,
generic_violations): each block is the IDENTICAL `any-nonempty` guard → `create_dir_all` →
`sum n` → `build_<entity>_batch` → `write_parquet_atomic` → `push` partition-path. Only a
`(field, dir-name, builder, extra-args)` tuple varies; the ~8-line write-tail is verbatim ×12.
Extract `write_partition(subpath, suffix, Option<batch>)` owning the create_dir/write/push tail;
keep the per-entity batch construction (genuinely varying args). Collapses ~18 lines/entity → ~3
and drops the per-new-output-entity edit tax here. (Distinct from the LP `fill_*` family, which is
NOT duplication — each entity's LP contribution genuinely differs; here the write skeleton is
identical.)

**CD-024 · Sev C · unwired reserved seam / false-live wire format (CD-019 flavor) · effort S · confidence high (unwired) / owner (disposition)** — INVESTIGATED, DISPOSITION RATIFIED
`serialize_/deserialize_resolved_parameters` (`resolved_parameters.rs:477/529`, the postcard
MPI-broadcast envelope) have ZERO production callers anywhere in the workspace (exhaustive grep:
only the lib.rs re-export + doctests + tests). Deep trace: `StudySetup::new` delegates to the
single `from_broadcast_params` constructor (both ranks), which builds `ResolvedParameters`
LOCALLY per-rank via `build_energy_and_templates`→`build_resolved_parameters` (`setup/mod.rs:876`),
beside `energy_conversion`'s explicit "Per-rank, never broadcast" annotation. Git: the
introduction commit ("wire ResolvedParameters through LP build AND add broadcast") added the two
`pub fn` + only doc/test callers — NEVER a production broadcast site; no caller was ever added or
removed across all three touching commits. So it is a SPECULATIVE seam, never wired — not
dead-then-orphaned (unlike CD-019's superseded `sync_cuts`). Cheap per-rank table → no perf motive
to ever broadcast. Symptom shared with CD-019: an E6-registered "dual-owned wire format" + version
byte signalling a live format that serves no one (false-liveness). ALSO the CD-020 pattern
(MPI-ephemeral versioning). Disposition (owner): safe-now = `pub`→`pub(crate)` (MPI-internal, zero
external consumers) + fix the E6-registry "live format" framing; then EITHER remove (if per-rank
derivation is permanent) OR keep as a genuine Voice-4 reserved seam annotated with its sole
plausible activator — the CD-001/004 north-star setup redesign ("rank-0 builds `ResolvedRunConfig`,
broadcasts it"). **DISPOSITION RATIFIED (2026-08-07):** `pub`→`pub(crate)` + keep as a Voice-4
reserved seam tied to the CD-001/004 redesign as its sole activator + fix the E6-registry
"live format" framing; remove outright if the CD-001/004 redesign is ruled out.

**CD-020 gains a 3rd instance:** `RESOLVED_PARAMETERS_WIRE_VERSION = 2`
(`resolved_parameters.rs:428`) — MPI-broadcast-only versioning (same-binary, cross-version
mismatch impossible in one run; the byte is really a corruption tripwire framed as versioning),
identical pattern to `cut::wire` + `CapturedBasis`. CD-020 is now confirmed across THREE
MPI-ephemeral formats — a systematic pattern, not three coincidences.

_Minor (not a separate ID): `simulation_writer.rs` is 1905 prod lines in one flat file (5× the
next writer) — a CD-021-neighbor organizational option (`simulation_writer/{records,writer,
batches}.rs`), but it is correctly homed (one file per output type) and cohesive. Noted, not
raised._

### Station 5 — verdict

Net: **3 findings (CD-024 C, CD-025 B, CD-026 C)** + 1 cleared candidate (`resolve_computed`) +
CD-020's 3rd instance. Headline is **CD-025** — the CLI/Python output hand-mirror, the ★ north-star
duplication class landing at the outputs boundary exactly as predicted. Otherwise the layer is
a positive reference (thin CLI orchestrator, writer-family, Python's internal `_if_any`
single-owner discipline).

Running total (through Station 5): **25 live findings + 1 retraction.** Severity: **1×A, 9×B, 15×C.**

---

## Station 4 — Simulation (`simulation/{state,pipeline,enumerated,extraction,...}`, `run/simulation.rs`)

Audited after epic-16's census engine committed (`0ce061f6`/`fc8d8b99`/`2a9b878f`; remainder is
tests only; working tree clean, re-verified). Deep code-level read of the dispatch, both engines,
the shared solve kernel, extraction, and the MPI exchange.

**Verdict: the simulation layer is well-architected — one Sev-B finding (the retrofitted-variant
asymmetry, exactly the predicted suspect) + a confirmed CD-025 extension; everything else clean or
positive.**

**CD-027 · Sev B · retrofitted-variant asymmetry (class instance #3) · effort S · confidence high**
`SimulationState::run` (`simulation/state.rs:224`, ~164 lines) dispatches its traversal match
ASYMMETRICALLY: `Traversal::Sampled` is a ~61-line INLINE block (`:299-360`: assign_scenarios →
resolve root_node → build `SimWorkerParams` → `par_iter_mut` over workspaces → `run_worker_scenarios`
→ concat + sort-assert), while `Traversal::Enumerated` (`:361-372`) validates `k == n_scenarios`
then delegates to the extracted `run_enumerated_simulation`. The retrofitted census variant was
promoted to a co-equal engine (`enumerated.rs`); the ORIGINAL sampled path was never promoted —
left inline. IDENTICAL to CD-013 (forward) and to what epic-15 FIXED in the backward pass. Fix:
extract `run_sampled_simulation(inputs, frozen_templates, &sampler)` so the match is a clean 2-arm
dispatcher mirroring backward `run` — this ALSO de-bloats `run` to setup + dispatch (same
convergence as CD-013). See ★ DEBT CLASS.

**CD-025 EXTENDS to simulation outputs (not a new ID — broader scope).** The census
`scenario_summary` write is hand-mirrored CLI↔Python verbatim: the tuple-reshape
`(id, cost, prob) → (id, prob, cost)` + `write_scenario_summary` is copied between
`cli/.../outputs.rs:197` and `python/run.rs:722`, pinned by a test
(`simulation_weighting_census_underivable_from_sampled_traversal`) that itself exists in BOTH
crates (`cli/.../simulation.rs:521`, `python/run.rs:2388`) — two test copies pinning two code
copies, no shared owner. CD-025's blast radius is confirmed as BOTH training and simulation output
orchestration; the fix (hoist the shared output helpers to cobre-sddp/cobre-io) covers both.

**Positives / checked-and-clear:**

- **CD-020 CLEAN in simulation (positive contrast).** The sim MPI exchange (`aggregation.rs:111`,
  `run/simulation.rs:321-460`) is a version-free raw `allgatherv` of length-prefixed primitive
  buffers — NO wire-version byte / postcard envelope. Exactly the honest approach CD-020 argues the
  three MPI-ephemeral-versioned formats should adopt. The sim exchange is the counter-example.
- **CD-022-scale cross-module mirror MITIGATED (positive).** `enumerated.rs` "mirrors
  `training::forward::enumerated` step-for-step" only STRUCTURALLY (the 3-phase mark→sweep→re-expand
  shape); it REUSES the hoisted primitives (`ClaimCursor`/`canonical_scatter`, `EnumeratedPlan::walk_path`,
  `AccumSnapshot`) and the pipeline.rs solve/extract/dispatch kernel (`solve_simulation_stage`,
  `extract_sim_stage_result`, `dispatch_scenario_result`) — no logic copy-paste. The epic-16 hoist
  did its job; unlike the backward CD-022 (which duplicated the reification loop), there is no
  residual copy here.
- **CD-023 sim-side CONFIRMED.** `solve_simulation_stage:405` routes through `StageSolvePrep::run`
  (`training::stage_solve_prep`) — the shared single-owner solve-prep the forward/backward/LB use.
  This is the live sim→training dependency CD-023 flags for re-homing; its fix must keep the shared
  owner reachable from simulation.
- **`solve_simulation_stage` (`pipeline.rs:372`, ~190) affirmed cohesive** — per-stage solve kernel,
  analog of the affirmed `run_forward_stage`; DCS-vs-frozen branch, shared solve-prep, DCS lazy path
  shared with backward, lag accumulation via the hoisted primitive. Length tracks the algorithm.
- **`extraction.rs` (1958 prod) is a legit EXTRACTION-FAMILY** — per-entity `extract_*` fns
  (hydros/thermals/exchanges/buses/contracts/non_controllables/pumping/…), size tracks entity count
  (the LP-builder/writer pattern), linear `extract_stage_result_with_lookups` assembler. NOT a
  god-module. _Minor (not a separate ID, same as `simulation_writer.rs`): a flat 1958-line file, a
  CD-021-neighbor organizational option, but correctly homed (one file per simulation concern)._
- **`simulate` (`pipeline.rs:988`) + `StudySetup::simulate` (`orchestration.rs:228`)** are thin
  shims delegating to `SimulationState::run` — clean.

### Station 4 — verdict

Net: **1 new finding (CD-027, Sev B)** + CD-025's simulation-side extension. The layer is otherwise
a positive reference: shared solve kernel, census engine reusing the hoisted primitives (the
epic-16 hoist validated), version-free MPI exchange, extraction-family. The ★ retrofitted-variant
class is now confirmed SYSTEMATIC across all three passes — forward (CD-013, open), backward (fixed
by epic-15, the model), simulation (CD-027, open) — CD-013 + CD-027 share one proven fix-shape.

Running total (audit COMPLETE, Stations 0–5): **26 live findings + 1 retraction.**
Severity: **1×A, 10×B, 15×C.**

---

## ★ AUDIT COMPLETE — full `cobre run` lifecycle walked (2026-08-07)

All six stations covered: entrypoint · setup · policy · training · **simulation** · outputs. **26
live findings + 1 retraction** (1×A, 10×B, 15×C). The station-by-station co-navigation is done.

**The two structural themes held all the way through:**

1. **★ North-star (lifecycle conflation + config/orchestration duplication):** the only Sev A
   (CD-001) + most Sev B (CD-003/004/005/010/025) — setup/input config-projection sprawl AND the
   CLI/Python output hand-mirror (CD-025, spanning training + simulation). One principle:
   model by lifecycle, one owner per fact, typed variation points. See [[project_input_data_lifecycle_modeling]].
2. **★ Retrofitted-variant asymmetry:** CD-013 (forward), CD-027 (simulation) open; backward fixed
   by epic-15 (the model); CD-015/018/019 the training-side flavors; CD-023 (incomplete hoist);
   CD-024 (unwired seam). Systematic — a variant retrofitted without promoting/removing the original.

**Positive references discovered (refactor TOWARD these):** `StageSolvePrep` (single-owner solve-prep,
the divergence anti-dote), `TrainingSession` (lifecycle-grouped struct), `CutPool` (SoA append-only),
the `CapturedBasis` codec (exhaustively-checked), the epic-15 backward fork + epic-16 primitive hoist
(correct retrofitted-variant handling), the LP-builder / writer / extraction entity-families.

**CD-020 is a systematic pattern** across 3 MPI-ephemeral-versioned formats (`cut::wire`,
`CapturedBasis`, `resolved_parameters`) — while the simulation exchange (version-free allgatherv) is
the honest counter-example. Consider one decision: reframe all three as fixed MAGIC/format-tags.

**Next step is prioritization, not more discovery:** rank the 26 findings into a fix roadmap
(the north-star cluster is the highest-ROI structural work; CD-013+CD-027 are one cheap proven
sweep). Backlog is the durable artifact; re-verify any finding against then-current code before
acting (the tree moved twice mid-audit — Stations 3 & 4 both needed re-verification).

---

## Epic-12 integration (2026-08-07 — owner-directed)

The findings are routed into the rung1-tree **epic-12 consolidation** (a debt-audit epic:
ticket-041 = architecture audit + deferred-debt register, ticket-038 = dead-code +
reserved-seam register). Epic-12 is PENDING (runs last, after epic-16), so its ticket specs
were safe to augment. Owner decisions: **augment the ticket specs** (not a side hand-off) +
**register all, execute only the trivial byte-neutral subset**. Epic-12's own doctrine
("escalate, do not squeeze"; behaviour-neutral) is why the large items are registered, not
executed here.

**Edits made to the plan (spec `.md` files only — the peer-managed
`.implementation-state.json` + README dependency graph are flagged, not touched):**

- **ticket-041** — Requirement A gained a whole-lifecycle audit-scope bullet (extends the
  node-axis read to the setup config layer, its blind spot); Requirement B2 registers every
  unique finding with owner + trigger, dispositioned [follow-up plan] (CD-001/003/004/005
  north-star, CD-025, CD-010, CD-018, CD-020-pattern + the Sev-C register-only set) vs
  [execute-eligible] (CD-013/027/006/026); a matching acceptance criterion added.
- **ticket-038** — reserved-seam register gains CD-024 (resolved-parameters broadcast
  helpers): `pub(crate)` + register tied to the setup redesign, or remove; fix the E6
  checklist framing.
- **ticket-098 (NEW)** — executes the four byte-neutral consolidations (CD-013/027/006/026);
  byte-neutral AC, sacred-parity green, backward `run` as the model.
- **00-epic-overview** — lists ticket-098 + the audit-sourced scope extension.

**Overlap already in epic-12's register (our findings confirmed, added nothing):** CD-019
(`CutSyncBuffers` API), cobre-python CI-invisibility, the external-noise glue dedup (CD-016
neighborhood), the cut-wire-header one-owner check (CD-020 neighborhood).

**The north-star cluster (CD-001/003/004/005) + CD-025/010/018 do NOT execute in epic-12** —
registered with a "dedicated post-0.14 setup-layer redesign plan" trigger. That plan is the
natural next artifact if/when the owner greenlights it.

**Scheduler wiring:** ticket-098 is registered in `.implementation-state.json` (pending, refined,
readiness estimate 0.96 mirroring ticket-038's profile — author estimate, re-scoreable by the
pipeline). Numbered 098 (next free; 043 was already epic-02's stage-id ticket). By numeric sort it
lands LAST in epic-12 (after 042/050) — dependency-valid (needs only 038), but a
`ticket_reorder_overrides` entry would be needed to run it before 041 finalizes / 042's CHANGELOG.
Still open: the README dependency graph (deps: 038, blocks: none) is not yet updated; and confirm
ticket-098 persists after the active epic-16 peer writes (a cached full-file writer could overwrite
it — re-add if so).

---

## Audit status — COMPLETE (2026-08-07)

All six stations complete (entrypoint · setup · policy · training · simulation · outputs) plus the
epic-16 review. **26 live findings + 1 retraction** (1×A, 10×B, 15×C). Full close-out — themes,
positive references, and the CD-020 pattern — is in the **★ AUDIT COMPLETE** section directly above.

**Discovery phase is done; next is prioritization, not more walking.** The suggested roadmap order:

1. **★ North-star cluster (highest structural ROI):** CD-001/003/004/005 (setup config-projection
   sprawl + lifecycle split) and CD-025 (CLI/Python output hand-mirror, training + simulation). One
   principle, one owner per fact — the Sev A + most Sev B collapse together here.
2. **Retrofitted-variant cheap sweep:** CD-013 (forward) + CD-027 (simulation) share one proven
   fix-shape (epic-15's backward fork) — promote each sampled original to a co-equal engine. CD-023
   (re-home `stage_solve_prep`) lands with CD-027's sim-dispatch work (retarget once).
3. **Pattern decisions (one call each):** CD-020 (reframe the 3 MPI-ephemeral version bytes as
   MAGIC tags), CD-024 (`pub(crate)` + reserved-seam-or-remove), CD-010 (typed `StateFamily`).
4. **Local cleanups (Sev C):** CD-006/007/009/011/012/014/016/017/018/019/021/026.

**Resume protocol (for the fix phase):** the tree is a live target under the active plan; before
acting on ANY finding, re-verify it against the then-current code — findings are snapshots (the
tree moved mid-audit twice; Stations 3 and 4 both required re-verification, and every finding held).

---

## ★ POST-v0.14.1 RECONCILIATION (2026-08-18 — `develop` @ HEAD `225f46a9`)

Re-verified every finding against the current tree (~105 commits past the audit baseline
`8903937c`). Each verdict below carries current-tree evidence; re-derive before acting.

### Resolved since the audit (5) — byte-neutral consolidations that landed

These match the "Byte-neutral consolidations — already executed" section of
`docs/design/reserved-seams-and-deferred-debt.md`; confirmed present in the tree:

| Finding                                         | Resolution (current-tree evidence)                                                                                                                                                                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CD-006** (NodeGraph query free-fns)           | RESOLVED — all now `impl NodeGraph` methods (`setup/node_graph.rs`: `frontier_node:1133`, `node_parent:1198`, `backward_cut_levels:774`, `stage_frontier:1120`, `max_successor_outcome_count:759`, `build_parent_map:1265`, …). Zero free-fn defs. |
| **CD-013** (forward sampled arm inline)         | RESOLVED — `ForwardPassState::run` (`training/forward_pass_state.rs:339`) is a clean 2-arm dispatcher over `run_enumerated` + extracted `run_sampled:411`, mirroring the backward `run`.                                                           |
| **CD-024** (resolved-parameters broadcast pair) | RESOLVED — deleted outright as dead code (zero production callers); `serialize_/deserialize_resolved_parameters` + `RESOLVED_PARAMETERS_WIRE_VERSION` no longer resolve anywhere.                                                                  |
| **CD-026** (`write_scenario` 12× repeat)        | RESOLVED — one `write_partition` helper owns the create-dir/write/push tail (`cobre-io/output/simulation_writer.rs:685`).                                                                                                                          |
| **CD-027** (simulation sampled arm inline)      | RESOLVED — `run_sampled_simulation` extracted (`simulation/state.rs:372`); `run` is a clean dispatcher over it + `run_enumerated_simulation`.                                                                                                      |

Running total after resolutions: **21 live architecture findings + 1 retraction** (1×A, 8×B, 12×C).

### Still HOLD (16 + CD-008 retracted) — refs refreshed to current tree

CD-001 (`run/setup.rs:424`/`486` non-root mirror), CD-002 (`broadcast_and_build_setup:274`
positional tuple — output now a named `LoadBroadcastResult`, input tuple unchanged),
CD-003 (`build_study_setup:602`), CD-004 (three projections live; `BroadcastConfig::from_config`
relocated to `cobre-cli/.../commands/broadcast.rs:149` — doc's `broadcast.rs:84` was stale),
CD-005 (`StudySetup` `setup/mod.rs:135` still fuses inputs/config/output + setter wall
`accessors.rs:56`+), CD-007 (`entries.rs`/`columns.rs` inline-test giants, both grew),
CD-009 (policy-dir resolve triplicated `run/policy.rs:167`/`196`/`317`), CD-010 (`entity_type: u8`
untyped + colliding dictionaries; file moved to `cobre-io/.../output/policy/records.rs:31`),
CD-011 (naming split + stale `PolicyStageManifest`), CD-012 (`run_cut_management`
`session/mod.rs:1015`), CD-014 (`compute_one_backward_node` `backward_pass_state.rs:1727`),
CD-015 (by_scenario aggregation inline vs `by_node_finish`), CD-016 (backward buffer-prealloc dup),
CD-017 (`basis_reconstruct.rs` missing truncation assert + always-zero `new_tight`),
CD-018 (`CutSelectionStrategy` Dynamic early-return `cut_selection.rs:320`),
CD-019 (dead `sync_cuts`/`pack_local_records`/`sync_packed_records` — now formally registered),
CD-020 (**now 2** MPI-ephemeral version bytes, not 3 — `resolved_parameters` removed;
`CUT_WIRE_VERSION`, `BASIS_BROADCAST_WIRE_VERSION` remain),
CD-021 (`workspace.rs` flat mega-file, 9 structs), CD-022 (backward reification copy, 2 sites),
CD-025 (CLI/Python output hand-mirror — blast radius **widened** to also mirror
`generic_constraint_echo`). CD-008 remains RETRACTED.

**CD-023 — PARTIALLY addressed (HOLDS for the prep).** A new neutral `solve/` module now owns the
LP-**solve** entry: `run_stage_solve` / `StageInputs` live in `solve/stage_solve.rs` and every hot
path (incl. `simulation/pipeline.rs`) imports them via `crate::stage_solve::…`, so the
sim→training inversion is gone _for the solve entry_. But the solve-**prep** single owner
(`StageSolvePrep`) still lives in `training/stage_solve_prep.rs:71` and `simulation/pipeline.rs:50`
still imports it via `training::stage_solve_prep`, so the inversion CD-023 named **persists for the
prep**. (`solve/` and `training/stage_solve_prep.rs` are two sequential phases of one pipeline —
prep-then-solve — not duplicates.)

### New architecture findings (fresh pass on modules that landed since the audit)

The original walk never covered `solve/`, `hull/`, `lead_time/`, `horizon_mode.rs`,
`generic_constraint_echo.rs`, `validate_phases.rs`, `simulation/enumerated.rs`, or
`training/backward/replicated.rs`. A fresh read of those found `hull/`, `lead_time/`,
`horizon_mode.rs` (single-variant enum = documented reserved seam), `generic_constraint_echo.rs`,
`config.rs`, and `training/backward/replicated.rs` **clean**, plus:

**CD-028 · Sev B · duplication (retrofitted-variant / cross-module mirror) · effort M · confidence high**
The enumerated forward and enumerated simulation reimplement the **mid-level claim/scatter
orchestration skeleton** independently — `simulation/enumerated.rs` (`run_sweep:292`,
`enumerated_sim_stage_worker:161`) vs `training/forward/enumerated.rs` (stage loop in
`run_enumerated_forward:531`, `enumerated_stage_worker:713`). Only the low-level primitives are
shared (`ClaimCursor`/`canonical_scatter` from `claim_scatter.rs`, `EnumeratedPlan`/`NodeGraph`).
Doc comments admit the mirror ("mirrors X step-for-step") but nothing enforces the two skeletons
stay in sync — a future claim/scatter-protocol fix must be hand-applied in both. This SHARPENS the
Station-4 "CD-022-scale mirror MITIGATED" note: the primitive reuse mitigates the _kernels_, but
the _orchestration_ remains a 2-site copy. (Same family as CD-022, which is the backward
reification copy.)

**CD-029 · Sev B · leaky-boundary / missing-seam + doc-drift + Python-parity gap · effort M · confidence high**
`validate_phases.rs:20` `PrepPhase` doc claims it unifies "one of the **four** SDDP preparation
steps," but the enum has exactly **three** variants (`Config`, `Stochastic`, `HydroModels`). The
real 4th step — boundary-cut reconciliation — bypasses the shared `PrepPhase`/`prep_phase_metadata`
abstraction entirely (`cobre-cli/.../commands/validate.rs:255-287`, its own ad-hoc
`format_boundary_error`) and has **zero equivalent** in `cobre-python/src/io.rs` (grep-confirmed:
no boundary references). Two defects in one: the abstraction oversells a "shared validation-phase"
contract that covers 3 of 4 phases, and the 4th escapes the Python-parity discipline. Fix: fold the
boundary check into `PrepPhase` (or correct the doc to "three") and mirror it in the Python binding.

**CD-030 · Sev C · duplication (minor) · effort S · confidence high**
`mark_own_paths` (`simulation/enumerated.rs:123-140`) reimplements the same ~10-line path-marking
algorithm inline in `training/forward/enumerated.rs:502-513`. Low complexity, low drift risk;
folds naturally with the CD-028 orchestration-dedup.

### ★ NEW — Performance-debt pass (the audit's declared blind spot)

The audit above was **architecture-only** (perf micro-optimization was out of scope, per the Scope
note at the top). This is a first performance-debt pass, scoped to the hot paths
(`training/forward/`, `training/backward/`, `training/training.rs`, `simulation/pipeline.rs`,
`training/lower_bound.rs` + the new solve/enumerated kernels) against the hard rule "**never
allocate on hot paths — pre-allocate workspaces, reuse buffers**." The pass found the codebase
overwhelmingly disciplined (near-universal `mem::take`+reuse; no `Box<dyn>` anywhere in scope) with
four exceptions:

**PD-001 · Sev B · hot-path-allocation · effort S · confidence high** — `simulation/enumerated.rs:259-272`
`enumerated_sim_stage_worker` heap-allocates a fresh `out_state` Vec (`ws.current_state[..n_state].to_vec()`)
and a fresh `AccumSnapshot::default()` on **every claimed node** in the census sweep, whereas the
training sibling `training/forward/enumerated.rs:824` reuses the slot via `clear()` +
`extend_from_slice`. Root cause: `EnumeratedSimScratch::new:94` initializes `arena` to `None`s,
never pre-sized (nothing to extend into). Runs once per distinct tree node the rank owns (thousands
for a real census) inside a rayon-parallel claim loop → allocator churn/contention. Fix: pre-size
the arena and reuse per-slot buffers, mirroring `EnumeratedForwardScratch::ensure_sized`.
**→ REFUTED (2026-08-18) — NOT debt; do not fix.** The `EnumeratedSimScratch::worker_captures`
doc documents the move-scatter as deliberate and optimal for a one-shot census: each arena entry
needs its own buffer, so a move gives exactly one allocation per arena buffer with zero copies;
the training sibling's copy-reuse pays off only because its arena is reused across many iterations,
whereas the simulation sweep runs once, so copy-reuse would ADD copies. The `to_vec()`/`::default()`
are immediately moved into the arena — the very "one allocation per arena buffer" the doc describes.
The confidence-high flag over-weighted the training-sibling comparison and under-weighted the
one-shot rationale.

**PD-002 · Sev B · hot-path-allocation · effort S · confidence high** — `training/session/mod.rs:773,780,793`
`TrainingSession::run_forward_phase` makes three fresh heap allocations **every training iteration**
to cross-rank-aggregate per-stage forward stats: `local_buf = vec![0.0; …]`, `global_buf = vec![…]`,
and `.collect::<Vec<SolverStatsDelta>>()`. Every other scratch in the same function
(`IterationScratch`, `ub_stage_costs`, `ub_path_weights`) is deliberately hoisted and reused; this
one is not, with no `#[allow]`/rationale. The `LbEvalScratch` pattern in
`.claude/architecture-rules.md` (rule #9) is the existing template. Fix: hoist the two buffers into
per-run scratch, reuse across iterations.
**→ RESOLVED (2026-08-18).** Hoisted to `IterationScratch` (`fwd_stats_pack_local` /
`fwd_stats_pack_global` / `fwd_stats_unpacked`), resized-and-fully-overwritten each iteration.
Byte-neutral (telemetry only, no effect on cuts/bounds); `solver_stats` roundtrip +
`opening_order_determinism` gate green.

**PD-003 · Sev C · hot-path-recompute · effort S · confidence med** — `training/session/mod.rs:1069`
`run_cut_management` re-derives the loop-invariant `interior_nodes` filter over the whole node graph
every cut-selection cycle (topology never changes mid-run), plus fresh `deactivations`/`record_by_pool`
Vecs. Cost gated by `cut_selection.check_frequency` (bites only at `=1`). Fix: cache the interior-node
set once in `TrainingSession::new`.
**→ RESOLVED (2026-08-18).** The `interior_nodes` filter is cached once in `TrainingSession::new`
(`interior_cut_nodes` field) and read by `run_cut_management`; byte-identical canonical order, so the
cut-selection determinism suite stays green. (The per-call `deactivations`/`record_by_pool` working
Vecs are genuine per-cycle data, not loop-invariant — left as-is.)

**PD-004 · Sev C · allocation (low impact) · effort S · confidence high(exists)/low(matters)** —
`training/backward_pass_state.rs:919-931` `run_enumerated_backward` builds a fresh
`Vec<(usize, Vec<StageWorkerOpeningDelta>)>` + a per-stage `SolverStatsDelta::clone()` every `run()`
call, not `mem::take`-recycled. NOT a regression — the pre-existing `run_sampled_backward` has the
identical shape; folding both onto a recycled buffer is the fix if measured to matter.
**→ DEFERRED (2026-08-18).** Left as-is: fixing only the enumerated path would diverge it from the
identical-shaped sampled path; a symmetric fold is a follow-up if a profile shows it matters.

### Reconciliation total (2026-08-18)

**24 live architecture findings + 1 retraction** (1×A, 10×B, 13×C) — down 5 (resolved), up 3
(CD-028/029/030) from the audit's 26. **Performance-debt pass: PD-002 + PD-003 fixed (byte-neutral,
determinism gates green); PD-001 refuted (deliberate one-shot move-scatter, not debt); PD-004
deferred (symmetric with the sampled path).** These counts are a snapshot; re-verify before acting.

---

## ★ POST-v0.14.3 RECONCILIATION + v0.14.1 FIX-WAVE REVIEW (2026-08-19 — `main` @ `a3dbecc5` = v0.14.3)

**Scope.** (1) The two releases cut after the 2026-08-18 pass — v0.14.2 (`8f112a02` perf,
`e7b68ba3` register reconcile) and v0.14.3 (`87175432` boundary inflow-lag slot reservation);
(2) a dedicated commit-by-commit architecture review of the **v0.14.1 bug-fix wave**
(`14e5ba2a`, `5775b1e2`, `5b3cbede`, `c56db6ba`, `599c1239`, `c61bed74`, `140610e5`) — in-tree
at the 2026-08-18 reconciliation but never read as diffs (that pass re-verified old findings +
swept new modules). The wave review confirms the rush hypothesis: **8 new architecture findings

- 1 new performance finding**, concentrated on the two feature-fix commits (gap-under-CVaR and
  inflow-lag inference); the two pure LP fixes and the projection fix are near-clean.

**Baseline for the next reconciliation: `main` @ `a3dbecc5`.** `origin/develop` (5cea0bbf) is
content-identical (same tree `bbb74e75`); main contains it.

### Verified against HEAD (record corrections)

- **PD-002 / PD-003 fix commit identified: `8f112a02` (shipped v0.14.2).** Verified in-tree:
  `fwd_stats_pack_local`/`fwd_stats_pack_global`/`fwd_stats_unpacked` on `IterationScratch`
  (`iteration_scratch.rs:68-73`), clear+resize+overwrite reuse in `run_forward_phase`
  (`session/mod.rs:732`); `interior_cut_nodes` (`session/mod.rs:131`) computed once in
  `TrainingSession::new` (`:398-405`), read by `run_cut_management` (`:1090`).
- **CD-012 anchor refreshed**: `run_cut_management` now `session/mod.rs:1036-1288` (~253 lines,
  was ~277 — the cached filter shaved the inline derivation). Still a god-fn; HOLDS.
- **Spot-checked HOLD at HEAD**: CD-001 (`run/setup.rs:424`/`486`), CD-019 (zero production
  callers; only 2 integration tests), CD-025 (both hand-mirrors incl. the scenario_summary
  tuple-reshape `outputs.rs:211-218` / `run.rs:739-745`), CD-029 ("four"-vs-three doc + boundary
  bypass + no Python check), CD-023 (prep-side inversion `pipeline.rs:50-51`).
- **CD-010 blast radius GREW + fix shape constrained** (v0.14.3): `reserve_boundary_inflow_lag_slots`
  (`policy_export.rs:310`) is a new production consumer of the raw `ENTITY_TYPE_*` u8 dictionary
  AND a public export consumed cross-crate by cobre-python (`policy.rs:19,306`) — the untyped
  semantics now cross a public seam. Census of consumers: `build_stage_entity_manifest`,
  `reserve_boundary_inflow_lag_slots` (policy_export.rs), `boundary_cut_lag_depth` +
  `decode_pool_anticipated_months` (policy_load.rs), four sites in `reconcile.rs`, and —
  previously unlisted — **`cobre-io/src/output/policy/checkpoint.rs:30` independently re-declares
  `ENTITY_TYPE_HYDRO_TRANSIT_BUCKET` from `policy.fbs`**. Since cobre-io cannot import cobre-sddp,
  the typed `StateFamily` enum must live in cobre-io next to `EntitySlot` (committed register
  updated with both facts). CD-010 HOLDS, sharpened.
- **CI-invisibility debt unchanged**: `87175432` added no Rust `#[cfg(test)]` to cobre-python;
  its Rust tests live in the workspace crate (CI-visible) + pytest (CI-visible).

### v0.14.2 (`8f112a02`) — reviewed: clean, two minor residues (no new IDs)

- Stale/duplicated test derivation: `session/mod.rs:2729-2737` copy-pastes the interior-node
  filter ("computed exactly as run_cut_management does" — no longer true; it lives in
  `TrainingSession::new`). Can silently diverge from the cached set.
- One-owner nit: the derivation is open-coded in the constructor; `NodeGraph` already owns the
  sibling predicate (`backward_cut_levels`, `node_graph.rs:774`). A `NodeGraph::interior_cut_nodes()`
  accessor gives the constructor and the test one owner. Both registered in the committed
  register (perf-section residue, owner: training, trigger: next interior-node/cut-selection touch).

### v0.14.3 (`87175432`) — reviewed: execution clean; design verdict REVISED (owner review, see CD-039)

Commit-local execution is correct: `reserve_boundary_inflow_lag_slots` lives in cobre-sddp
beside `build_stage_entity_manifest` (the canonical lag-major/1-based-subindex owner), NOT in
the generic cobre-io writer; cobre-io unchanged (existing callers byte-identical); unplaceable
terms fail loudly; the Python side (`build_stage_cuts_data`) converts and delegates with no
duplicated layout logic. **But the initial "architecturally CLEAN" verdict was scoped too
narrowly** — it measured the commit against local criteria and missed the mechanism-design
axis the owner flagged on review: the fix is a family-SPECIFIC channel (a lag-only scalar +
lag-only keyed field + lag-only helper) for a family-GENERIC problem. That is CD-039 below;
the CD-010 sharpening above stands.

### v0.14.1 fix-wave — NEW findings

**CD-031 · Sev B (A-risk) · missing-seam + duplication (★ north-star instance) · effort M · confidence high**
The boundary-derived setup context has no single owner. The depth-resolution RULE is clean and
single-owner (`boundary_policy_required_lag_depth` / `resolve_effective_inflow_lag_depth`,
`policy_load.rs:554-609`), but the FOLD — "given a case dir and a Config, produce the depth" —
is open-coded in three different shapes across the three production entry points
(`run/setup.rs:129-138` if-let + progress line; `validate.rs:386-389` bare match;
`cobre-python/run.rs:906-910` match + map_err) plus a fourth, laxer copy in the test helper
(`tests/common/mod.rs:143-152`). The value then rides three relay carriers
(`StudyParams.inflow_lag_depth` — always `None` out of `from_config`; `BroadcastConfig.inflow_lag_depth`
— always `None`, patched out-of-band at `run/setup.rs:141`; `ConstructionConfig.inflow_lag_depth`),
and `ConstructionConfig` now carries TWO facts derived from the same `config.policy.boundary`
filled by different owners at different times (`boundary_present` pure vs `inflow_lag_depth`
patched, needs I/O) with nothing guarding their agreement. A forgetful new entry point gets
`boundary_present=true, inflow_lag_depth=None` — reproducing the dropped-lag-coupling class this
wave fixed (whether the load-path reconciliation catches it loudly is unverified — check when
scheduling). Subsumes two sibling smells with the same fix: the `case_dir.join(&bp.path)`
checkpoint-path join open-coded at SEVEN sites, three carrying an identical two-line warning
comment (`validate.rs:195-197`, `policy.rs:237-239`, `run.rs:1215-1217`; no
`BoundaryPolicy::checkpoint_path` accessor exists); and the checkpoint being read+parsed from
disk TWICE per entry point (layout sizing then boundary load), so the layout depth and the load
guard's depth are computed independently rather than being the same value by construction.
SHARPENS CD-004/CD-003: `ConstructionConfig` (21 fields) now has three independent field-by-field
assemblers plus two structurally "always empty, then patched" fields (`scalar_parameters`,
`inflow_lag_depth`) whose patch obligation is documented only in a test comment
(`tests/common/mod.rs:136-139`). Direction: one `resolve_boundary_context(case_dir, &config) ->
BoundaryContext { present, inflow_lag_depth }` resolver owned by the domain crate; a
`BoundaryPolicy::checkpoint_path(case_dir)` accessor in cobre-io; `ConstructionConfig` holds the
sub-struct; resolve once, thread everywhere. Lands naturally with the north-star setup redesign
(CD-001/003/004/005) — same fix principle, one owner per fact.

**CD-032 · Sev B · retrofitted-variant asymmetry (class instance #4) + coupling · effort M · confidence high**
The nested CVaR upper bound (`c56db6ba`) is a session-side OVERRIDE, not a `ForwardBound` arm.
`ForwardBound` is the named UB-estimator dispatch (`Statistical` | `Exact`), yet the nested
estimator lives outside it: `sync_forward` computes the risk-neutral bound (full `allgatherv` +
compensated reduction), then `apply_nested_cvar_ub` (`session/mod.rs:890-933`) DISCARDS
`global_ub_mean` and recomputes via a SECOND `allgatherv` (`stats_aggregation.rs:207`).
`ForwardBound::Exact`'s own rustdoc documents the external override (`:35-37`, `:153-156`) — the
enum documents behavior it does not own; the dispatch condition is split (session tests
`path_weights()` at `:860-873`, re-tests `Traversal::Enumerated` at `:907`); and
`SimulationWeighting` — documented as "mirroring `ForwardBound`" — no longer covers the estimator
space. Three sibling smells fold into the same fix: (a) `nested_ub_recursion` re-implements
`EnumeratedPlan::walk_path` verbatim (`stats_aggregation.rs:247-262` vs `node_graph.rs:1440-1457`
— the documented "single owner of this walk"), dropping its length debug-assert; (b) the
rank-partition counts/displs arithmetic gains a FOURTH owner (`stats_aggregation.rs:196-204`,
line-for-line the copy 90 lines above; `RankDistribution::actual_per_rank` is the declared owner;
`cut_sync.rs:182-193` the third) — four implementations of one wire-partition rule every
`allgatherv` must agree on; (c) the per-path stage stride is derived from
`forward_result.stage_stats.len()` (a solver-statistics vector, `session/mod.rs:915-922`) instead
of the declared owner `self.ranks.num_stages` used ten times in the same file — correct today,
silently mis-strides if `stage_stats` is ever resized. Direction: promote to
`ForwardBound::NestedRisk` consuming `&EnumeratedPlan` + `RankDistribution` — one gather, one
match arm, one walk owner, one stride owner. Converges with PD-005.

**CD-033 · Sev B · duplication + coupling · effort S · confidence high**
Two independent implementations of the "nested UB applies" predicate must agree or the gap rule
silently reverts to the risk-neutral end-of-horizon bound `c56db6ba` eliminated (manifesting as
a spurious LB/UB crossover, not an error). Uniformity: `reject_gap_under_nonuniform_risk`
(`setup/mod.rs:2140-2156`) open-codes what `uniform_effective_measure`
(`risk_measure.rs:215-243`) implements — both added by the SAME commit (`5b3cbede`).
Risk-aversion: implemented three times (`is_effective_non_expectation` `setup/mod.rs:2211`;
inline `matches!` `session/mod.rs:912`; `RiskMeasure::effective` — whose doc admits the mirror).
Direction: rejecter calls `uniform_effective_measure`; express `is_effective_non_expectation`
over `RiskMeasure::effective`.

**CD-034 · Sev B · asymmetry + duplication · effort S · confidence high**
The min/max outflow entry blocks (`entries.rs:1499-1533`) are two hand-mirrored `for blk` loops
after `5775b1e2` made them semantically symmetric — identical `q + s` gathers, differing only in
row-family base, slack accessor, and slack sign; the "both bind `q + s`, neither couples `d`"
invariant is enforced by a comment across two sites. The sibling rows file already uses the
correct shape — `rows.rs:461-477` emits the SAME row pair from a two-element descriptor array.
Byte-neutral merge (template.rs sorts entries before CSC assembly). Direction: gather once per
`(h, blk)`, emit both rows from a `(row_start, slack_col, sign)` table.

**CD-035 · Sev B · missing-seam (typed-role gap) · effort S–M · confidence high**
`write_opening_outcome` / `accumulate_opening_outcome` take a bare `&CutStateProjection`
(`outcome_aggregation.rs:20-30,101-122`) at a call site where TWO role-distinct projections are
live (`child_cut_layout` at `backward_pass_state.rs:799` vs the required cut-generating parent's
`succ_spec.cut_state` at `:854`) — passing the wrong one compiles and differs exactly in the
reduced-projection case `599c1239` fixed. Against the module's own discipline
(`CutStateProjection` carries compile_fail doctests to prevent role swaps; sddp.md names
parent-pool-vs-leaf-pool confusion as a contract). Direction: pass `&SuccessorSpec`, or a
`CutGeneratingProjection<'a>` newtype.

**CD-036 · Sev C · asymmetry (doc-contract drift) · effort S · confidence high**
`IterationScratch.ub_path_weights`/`ub_stage_costs` are `Vec::new()` + grown on first use
(`iteration_scratch.rs:24-26,56-61,182-183`) while the struct doc promises "allocated once in
`new`" and the sibling `records` is pre-sized. First-iteration reallocations only. Fix: size in
`new` from `max_local_fwd`/`num_stages`; `extend` over `push`.

**CD-037 · Sev C · asymmetry · effort S · confidence high**
The two gap-rule rejecters (`setup/mod.rs:2081-2088,2111-2116,2172-2182`) each re-evaluate
`rule_is_gap` + `training_enumerated`, and which rejection message a sampled-forwards study sees
depends on call order (documented as deliberate at `:2092-2095`). Readability only. Optional fix:
one `match (gap_present, enumerated, risk_averse)` in `admission_gate`.

**CD-038 · Sev C · duplication (missing primitive sibling) · effort S · confidence high**
`CutStateProjection` exposes only the fused `dot_trial_state`; the cut-selection value sweep
needs gather-only and open-codes the gather loop (`session/mod.rs:1113-1122`) — correctly (via
`global_state_index`), so this is loop-shape duplication, not a hazard. Fix: add
`gather_trial_state(&self, x_hat, out)`; define `dot_trial_state` over it.

**PD-005 · Sev B · hot-path-allocation + recompute · effort S–M · confidence high**
`nested_ub_recursion` (`stats_aggregation.rs:239-296`) runs once per iteration on the forward
path and allocates ALL working state fresh: three `vec![…; n_nodes]`, `children` (one inner Vec
per non-root node), `roots`, `order` + sort, `value`, two child buffers — although only
`node_cost` changes between iterations; `children`/`node_stage`/`node_prob`/`roots`/`order` are
pure functions of the `EnumeratedPlan`, resolvable once at training start. Each interior node's
`evaluate_risk` additionally builds a fresh `RiskMeasureScratch` per call while the `_into`
scratch form exists, is public, and is unused here — the scratch struct's own doc says its
purpose is "the allocation is paid once". The adjacent session comment records the opposite
(correct) decision for the sibling path-weights quantity. Fix: precompute topology alongside
`EnumeratedPlan` (or on `IterationScratch`), thread a `RiskMeasureScratch`, call
`compute_cvar_weights_from_costs_into`. Folds naturally into CD-032's `ForwardBound::NestedRisk`
promotion.

### Owner design review (2026-08-19, post-reconciliation) — NEW finding

**CD-039 · Sev B · bad-abstraction + missing-seam (per-family bespoke boundary-state channels) · effort M–L · confidence high — OWNER-RAISED**
The question "how does an externally-authored boundary FCF's coupling on state family X reach
the study's state space?" is answered by a DIFFERENT bespoke mechanism per family, and the
v0.14.1/v0.14.3 lag work added the second such channel in a family-specific shape rather than
generalizing the frame:

- **Inflow-lag** (the new channel): a family-specific scalar threaded through 19 files — the
  Python API arg + per-cut `inflow_lag_coefficients` keyed field, `StudySetup::new_with_inflow_lag_depth`,
  the three config-projection carriers (`StudyParams`/`BroadcastConfig`/`ConstructionConfig`),
  per-entry-point folds, a family-specific manifest decoder (`boundary_cut_lag_depth`,
  `policy_load.rs:554`), a family-specific widening (`widen_lag_state_depth`,
  `stochastic_pipeline.rs`), and a family-specific writer helper (`reserve_boundary_inflow_lag_slots`).
- **Anticipated** (pre-existing bespoke channel #1): its own manifest decoder
  (`decode_pool_anticipated_months`, `policy_load.rs:931`) for source-stage auto-resolution;
  reservation from STUDY config (post-horizon lanes) + calendar fan-out reconciliation.
- **Transit buckets** (latent gap): reservation from study arc topology only; boundary-gated
  terminal unmasking; NO widening/inference path exists — `resolve_target_slot`
  (`reconcile.rs:277-285`) is target-driven, so boundary bucket coupling the study's topology
  does not reserve is silently never consulted: the C17 silent-drop class, latent in another
  family. An external boundary authoring bucket coupling would also force the caller to
  hand-fabricate cobre's canonical bucket-slot layout — the exact problem `87175432` solved
  for lags only.

Recurrence is concrete, not hypothetical: bridge-authored anticipated coupling (GNL post-study
commitments) is roadmapped; each new family under the current shape repeats: API arg + keyed
cut field + reservation helper + manifest decoder + relay through all three carriers + entry-point
folds. The generic mechanism HALF-EXISTS: the manifest is already the generic metadata (slots
self-describe by `(entity_type, entity_id, subindex, delivery_date)`; no format change needed)
and the load-side rebind is already family-generic in frame (one `resolve_target_slot` dispatch,
per-family rules, one `FamilyTally`). What is family-specific where it should be generic:
(a) the setup channel — replace the lag scalar with ONE `BoundaryStateRequirements` derived
once from the manifest (per-family summary: lag depth, anticipated months, bucket identities),
riding the carriers as a single struct; (b) the writer channel — parameterize
`reserve_boundary_inflow_lag_slots`'s internals (leading-block detection, canonical insertion,
keyed placement — ~90% family-independent) by a typed family + slot-key constructor, with the
per-cut keyed coefficients becoming family-keyed rather than a lag-named field. Per-family
RECONCILIATION semantics (widen vs calendar fan-out vs reject) are genuinely irreducible physics
decisions and stay per-family — the debt is the missing shared frame, not that families differ.

Prerequisite/converges: **CD-010** (typed `StateFamily` in cobre-io) is the vocabulary this
frame needs — it gets PULLED EARLIER by this finding, from a standalone pattern call into the
prerequisite of the boundary-channel redesign. **CORRECTS CD-031's fix-shape**: the boundary
resolver must produce the family-generic `BoundaryStateRequirements`, NOT a
`BoundaryContext { present, inflow_lag_depth }` — a lag-specific field in the new abstraction
would enshrine exactly this debt. Do the frame BEFORE the second family's authoring need lands
(the GNL/anticipated import), or that work pays the full bespoke-channel tax again.

### Fix-wave CLEAN verdicts (checked-and-clear — do not re-raise)

- **Projection-fix completeness (`599c1239`)**: no unfixed sibling of the positional-zip bug —
  both intercept sites route through `dot_trial_state`; DCS scoring gathers via `outgoing_column`
  with a dimension assert; the selection sweep uses `global_state_index`; the three gather axes
  (`global_state_index` archive / `outgoing_column` LP-primal / `incoming_column` extraction) are
  correctly distinguished by design. Only residues: CD-035 (typed role) + CD-038 (gather-only sibling).
- **Stopping-rule admission is single-owner and well-homed**: `admission_gate` called once from
  `from_broadcast_params`, named rejecters, exhaustively-destructured predicates; no duplicate
  admission logic in cobre-io or the session. Residues: CD-033 + CD-037 only.
- **`state_space` removal (`140610e5`) is residue-free**: survivors are the deliberate reject
  test + the CHANGELOG entry; schema/docs/config plumbing all clean; remaining `state_space`
  matches are the unrelated `lp/indexer/state_space.rs` name collision.
- **Infra-crate genericity holds** (`check-infra-genericity.sh` green; the wave's infra edits are
  generic doc wording). **No `#[allow]` without rationale, no `Box<dyn>`, no bool-where-enum.**
  **CHANGELOG entries behavior-described, no plan-structure leakage.** **The new broadcast field
  has a round-trip test** (`broadcast_config_carries_inflow_lag_depth`).
- **Considered and rejected**: the nested recursion's naive per-node weighted sum vs the sibling
  Neumaier-compensated exact bound — fixed order on every rank, handful of terms per node;
  rank-invariance unaffected. Not registered.

### Reconciliation total (2026-08-19)

**33 live architecture findings + 1 retraction** (1×A, 16×B, 16×C) — up 8 (CD-031..038) from
2026-08-18's 24 via the fix-wave review, plus 1 (CD-039) from the owner's design review of the
v0.14.3 verdict. Performance register: **PD-005 open (B)**; PD-004 deferred; PD-002/003 fixed
(v0.14.2); PD-001 refuted. Severity calibration note: the fix-wave reviewer rated CD-031/032/033
and PD-005 as Sev A; they are recorded at B / B(A-risk) to stay calibrated with the existing
scale (CD-004's config-sprawl precedent = "B (A-risk)"; retrofitted-variant class = B). These
counts are a snapshot; re-verify before acting.

### ★ Scheduling roadmap (refreshed 2026-08-19 — ready to schedule)

1. **★ North-star cluster** (highest structural ROI, one principle — model by lifecycle, one
   owner per fact): CD-001/003/004/005 (setup config-projection sprawl + lifecycle split) **now
   including CD-031 + CD-039** (the boundary resolver, whose output must be the family-generic
   `BoundaryStateRequirements` — CD-039 corrects CD-031's fix-shape) and CD-025 (CLI/Python
   output hand-mirror, training + simulation). **CD-010 (typed `StateFamily` in cobre-io) is
   pulled up from tier 4 as this cluster's prerequisite vocabulary.** The reserved
   CD-024-successor seam (resolved-parameters broadcast) activates or dies here too. Sequencing
   constraint: land the CD-039 frame BEFORE the bridge's anticipated-coupling authoring need
   (GNL post-study commitments) arrives.
2. **Nested-UB consolidation sweep** (new, cheap-to-medium, one code region):
   CD-032 (+ its three folded sub-items) + CD-033 + PD-005 + CD-036 — one
   `ForwardBound::NestedRisk` promotion with precomputed topology resolves the retrofit
   asymmetry, the predicate split, the hot-path allocations, and the scratch-contract drift
   together. Do this BEFORE the next risk-measure or stopping-rule feature lands on top.
3. **Retrofitted-variant & typed-seam sweep** (proven fix-shapes): CD-015/022 (backward
   reification extraction), CD-018 (selection-paradigm split), CD-035 (typed projection role),
   CD-034 (outflow descriptor table), CD-023 (re-home solve-prep with the sim-dispatch touch).
4. **Pattern decisions (one call each)**: CD-020 (2 MPI-ephemeral version bytes → MAGIC tags),
   CD-029 (fold boundary check into `PrepPhase` + Python parity call). (CD-010 moved to tier 1
   as the CD-039 prerequisite — scope includes the checkpoint.rs duplicate constant + the
   public cobre-python seam.)
5. **Local cleanups (Sev C, batchable)**: CD-002/007/009/011/012/014/016/017/019/021/026/028/030/
   036/037/038 + the two v0.14.2 residues (NodeGraph accessor + test derivation).

**Resume protocol unchanged**: findings are snapshots — re-verify each against the then-current
tree before acting. Baseline: `main` @ `a3dbecc5` (v0.14.3).

### Post-study-stages unification update (2026-08-21, v0.15.0)

**CD-039 scope narrows — the boundary-channel frame now covers two families, not three
mechanisms (owner decision D4).** The post-study-stages unification retired the parallel
post-study "lane" subsystem — `initial_conditions.future_anticipated_deliveries[]` →
`resolve_commitment_hold_windows` → the `n_commitment` / `CommitmentHoldAddress` state block —
and routes every post-study anticipated delivery through the SAME anticipated-commitment ring
as an in-study delivery, over a delivery axis extended across the post-study calendar
(`post_study_stages.json`'s `thermal_bounds[]`). So the anticipated family no longer reaches the
state space through two distinct carriers (the ring for in-study deliveries, the lane block for
post-study ones); the anticipated coupling collapses to ONE ring channel. The frame CD-039's
generic redesign must generalize is now **inflow-lag + the unified anticipated ring — two
families** — rather than the three separate mechanisms the finding enumerated; the transit-bucket
latent gap is unchanged. Per owner decision D4 the generic `BoundaryStateRequirements` redesign
(CD-039) still lands AFTER this work: this unification simplifies the anticipated channel that
redesign must cover, it does not perform the redesign. CD-039's finding text above is the
pre-unification snapshot; this note records the post-unification scope.

**CD-025 unchanged — no extension recorded.** The new runtime Python-parity guard for the
`anticipated_lanes/` partition (`crates/cobre-python/tests/test_anticipated_lanes_output_parity.py`)
adds a test only (no `filesToBeModified`): both surfaces already reach the partition through the
shared `SimulationParquetWriter` / `ScenarioWritePayload` path via the single
`extract_commitment_lanes` in cobre-sddp, and the writer audit found no production divergence, so
no CLI/Python hand-mirror was added or changed. CD-025's blast radius is unchanged by this work.

**RESOLVED — the post-study row-parity guard is now active, not post-release debt.** The plan's
doc-closure ticket originally assumed no on-disk post-study deck would ship, so
`test_cli_python_anticipated_lanes_row_parity` would keep skipping and the guard would be recorded
here as never-exercised post-release debt. That is now stale: a representative post-study deck
ships as `examples/deterministic/d55-post-study-anticipated-lanes/` (declares
`post_study_stages.json` with a `thermal_bounds[]` cell whose anticipated lead reaches a post-study
delivery), so `_require_post_study_case()` no longer skips and the guard runs and passes
end-to-end (compiled CLI vs `cobre.run.run`, element-wise row comparison). No open post-release
debt entry is owed. Residual (cleanup, not debt): that test module's docstring still reads as if
the deck is a pending forward dependency — a stale-comment fix for whoever next touches the file.

---

## ★ POST-v0.15.0 RECONCILIATION + OVER-ENGINEERING PASS (2026-08-22 — `feat/anticipated-fixed-post-horizon-commitments` @ HEAD `b80d9e62`)

**Scope.** (1) Re-verified every live `CD-`/`PD-` finding against the current tree — 36 commits
past the 2026-08-19 baseline `a3dbecc5` (v0.14.3), on top of the v0.15.0 post-study-stages
unification and this branch's fixed-post-horizon-commitments work (31 tickets touching `setup/`,
`lp/`, `io/` output+policy, and the anticipated-commitment ring). (2) A **first over-engineering
pass** — the inverse lens to the whole audit above, which hunts too-LITTLE / wrong structure; this
pass hunts too-MUCH structure and opens the `OD-` class. Documented reserved seams
(`docs/design/reserved-seams-and-deferred-debt.md`, the "Unwired config is reserved" hard rule, and
the sanctioned `#[allow(...)]` census classes) were excluded by construction; every `OD-` finding
below was independently re-verified at HEAD before recording. **Baseline for the next
reconciliation: `feat/anticipated-fixed-post-horizon-commitments` @ `b80d9e62`.** Counts are a
snapshot; re-verify before acting.

### Re-confirmed RESOLVED (7) — the prior fixes are intact at HEAD

CD-006 (NodeGraph query methods — six symbols still `impl NodeGraph`, call sites use method syntax),
CD-013 (`ForwardPassState::run` 2-arm dispatcher, `forward_pass_state.rs:339`), CD-024
(resolved-parameters broadcast pair — `serialize_/deserialize_resolved_parameters` +
`RESOLVED_PARAMETERS_WIRE_VERSION` resolve nowhere), CD-026 (`write_partition` helper,
`simulation_writer.rs:676` — now called **14×**, up from 12×: the branch routed
`anticipated_lanes`/`in_transit` through the same helper, extending the fix), CD-027
(`SimulationState::run` dispatcher over extracted `run_sampled_simulation`/`run_enumerated_simulation`),
PD-002 (`fwd_stats_pack_local/global/unpacked` on `IterationScratch`, clear+resize+overwrite reuse),
PD-003 (`interior_cut_nodes` cached in `TrainingSession::new`, read by `run_cut_management`).

### HOLDS-SHARPENED (3) — still present, blast radius grew on this branch

**CD-010 (untyped `entity_type: u8` state-family dictionary) — GREW again.** The 2026-08-19 note's
census stopped at "four sites in `reconcile.rs`". The fixed-post-horizon-commitments work built
`policy/reconcile.rs` into the **primary** consumer of the raw dictionary: **14** non-test
`ENTITY_TYPE_*` references now span **two** dispatch matches — `resolve_target_slot`'s `RebindOp`
dispatch and a **new** `report_family`/`ReportFamily` classifier that independently reinvents part
of the missing family vocabulary. Still no `StateFamily` type anywhere in the workspace
(`grep StateFamily crates/` → zero) and no `EntitySlot::family()`; the `checkpoint.rs:30` duplicate
`ENTITY_TYPE_HYDRO_TRANSIT_BUCKET` const and the colliding `output/dictionary.rs` i8 table are both
unchanged. The typed `StateFamily`-in-cobre-io prerequisite (tier-1 of the roadmap) is now overdue —
a second family-classifier landed while it was pending.

**CD-007 (`entries.rs`/`columns.rs` inline-test giants) — GREW.** `entries.rs` is now **10,079 lines
(~1,668 prod / ~8,411 test, ~83% test)**; `columns.rs` **9,026 lines (~1,314 prod / ~7,712 test,
~85% test)** — 459 and 752 lines changed respectively over the 36 commits; the extracted sibling
test files grew in lockstep (`template/tests.rs` 4,700→5,333, `layout/tests.rs` 2,979→3,565). No
`entries/{mod,tests}.rs` directory split executed. Same asymmetry, bigger god-files.

**CD-039 (per-family bespoke boundary-state channels) — NEW third mechanism on this branch.** The
2026-08-21 addendum narrowed CD-039 to two families (inflow-lag + the unified anticipated ring),
confirmed here: `resolve_commitment_hold_windows`/`CommitmentHoldAddress`/`n_commitment`/
`future_anticipated_deliveries` are gone from production (sole survivor is a retired-field rejection
test). Both surviving channels are unchanged in shape (`boundary_cut_lag_depth`,
`reserve_boundary_inflow_lag_slots`, `widen_lag_state_depth` for lag; `decode_pool_anticipated_months`
for the ring). **But the fixed-post-horizon-commitments work added a THIRD boundary-coupling shape:**
`build_boundary_fold` (`policy/reconcile.rs:227`) folds a class-4 fixed post-horizon window's declared
MW as a **constant into each cut's raw intercept** (see `sddp.md` "The ring axis" / anticipated
thermal-commitment contract). It is a constant-fold, not a state-family carrier, so it does not add a
fourth *state* channel — but it is a fourth bespoke path by which an externally-authored boundary
couples the study, exactly the proliferation CD-039 warns about. The generic
`BoundaryStateRequirements` frame (tier-1) is now guarding against a moving target: land it before the
next boundary-coupling family, not after.

### Still HOLD (30 architecture + PD-005) — refs refreshed to `b80d9e62`

CD-001 (`run/setup.rs:424` non-root mirror), CD-002 (`broadcast_and_build_setup:274` 10-tuple input),
CD-003 (`build_study_setup:602-646`, 21-field `ConstructionConfig` literal + 4 mid-flight `.take()`),
CD-004 (`StudyParams::from_config` `params.rs:129` + MPI `BroadcastConfig` hand-assembly; field counts
19/21/24 confirmed), CD-005 (`StudySetup` `setup/mod.rs:131` lifecycle conflation + setter wall;
struct field count unchanged at 35 — the branch swapped the retired lane field
`commitment_window_delivery_dates` for `extended_delivery_anchors` in the same slot, not additive),
CD-009 (policy-dir resolve triplicated, `run/policy.rs:167-176`+), CD-011 (`PolicyStageManifest`
`policy_load.rs:153` naming split), CD-012 (`run_cut_management` `session/mod.rs:1034-1286`, 253 lines —
still a god-fn), CD-014 (`compute_one_backward_node` `backward_pass_state.rs:1727-2015`, 289 lines),
CD-015 (ByScenario inline aggregation `backward_pass_state.rs:1950-1991`), CD-016 (per-worker
prealloc dup `backward_pass_state.rs:2060-2111`), CD-017 (`reconstruct_col_statuses`
`basis_reconstruct.rs:205` — still no truncation `debug_assert`), CD-018 (`CutSelectionStrategy`
`cut_selection.rs:144`), CD-019 (`sync_cuts` `cut_sync.rs:250` — zero production callers), CD-020 (2
MPI-ephemeral wire-version bytes: `CUT_WIRE_VERSION` `wire.rs:52`, `BASIS_BROADCAST_WIRE_VERSION`),
CD-021 (`workspace.rs:44` flat mega-file), CD-022 (backward reification copy, `backward_pass_state.rs`
sampled+enumerated), CD-023 (solve-**prep** inversion persists — `StageSolvePrep`
`training/stage_solve_prep.rs:71` still imported by `simulation/pipeline.rs`), CD-025 (CLI/Python
output hand-mirror — `outputs.rs:59` comments still say "mirror on the Python side"), CD-028
(enumerated forward-vs-sim orchestration mirror, `simulation/enumerated.rs` module doc admits it),
CD-029 (`PrepPhase` doc "four" vs three variants `validate_phases.rs:20` + boundary bypass + no
Python parity), CD-030 (`mark_own_paths` dup `simulation/enumerated.rs:123-139`), CD-031
(boundary-context fold triplication, `run/setup.rs:129-141`+), CD-032 (nested CVaR UB session
override `apply_nested_cvar_ub` `session/mod.rs:899-931`, not a `ForwardBound` arm), CD-033
(`reject_gap_under_nonuniform_risk` `setup/mod.rs:2139` open-codes uniformity), CD-034
(`fill_operational_violation_entries` `entries.rs:1473` min/max outflow hand-mirror), CD-035
(`accumulate_opening_outcome` bare `&CutStateProjection` role hazard,
`training/backward/outcome_aggregation.rs:20`), CD-036 (`IterationScratch.ub_*` doc-vs-alloc drift
`iteration_scratch.rs:22-26`), CD-037 (gap-rule rejecter order-dependence `setup/mod.rs:2080`),
CD-038 (no `gather_trial_state` sibling to `dot_trial_state`
`lp/indexer/cut_state_projection.rs:175`). **PD-005** (nested_ub_recursion hot-path allocation,
`training/forward/stats_aggregation.rs:230-307` — allocates ~10 working vectors fresh every
iteration) HOLDS, open.

### Performance register (unchanged in kind)

PD-005 open (B); PD-004 **re-confirmed DEFERRED** (`run_enumerated_backward`
`backward_pass_state.rs:919-928` fresh `Vec` + per-stage `delta.clone()`, still symmetric with
`run_sampled_backward:632`; the "fixing one diverges it from the other" premise holds — no symmetric
fold landed); PD-002/PD-003 fixed (v0.14.2); PD-001 **re-confirmed REFUTED** (the deliberate one-shot
move-scatter `simulation/enumerated.rs:258-273`; the `worker_captures` rationale doc is intact — one
allocation per arena buffer, zero copies, correct for a one-shot census).

### ★ NEW DEBT CLASS — Over-engineering (`OD-`)

First pass; the inverse lens to the audit above. Each finding was re-verified at HEAD by grepping the
claimed call/read sites and checking `docs/design/reserved-seams-and-deferred-debt.md` — a documented
reserved seam is not over-engineering. **9 findings (3×B, 6×C).**

**OD-001 · Sev B · speculative-generality · effort M · confidence high** — The whole shared-memory
trait hierarchy in `cobre-comm` has **zero consumers** outside the crate itself. `SharedMemoryProvider`
(`traits.rs:372`), `SharedRegion<T>` (a GAT `type Region<T: CommData>`), `LocalCommunicator`, the
`LocalCommKind` dispatch enum, and `HeapRegion<T>` — plus a live `MPI_Comm_split_type(SHARED)` call in
`FerrompiBackend::split_local` — exist only behind the `shared-memory` feature. `grep
'SharedMemoryProvider|split_local|create_shared_region' crates/` is empty outside `cobre-comm/src`;
every call site is the crate's own tests. **Both** impls (`LocalBackend` and the "real" MPI
`FerrompiBackend`) resolve `Region<T>` to the same `HeapRegion<T>` — a private per-rank `Vec<T>` — so
even the MPI build delivers no true shared memory; the module doc admits it. Not in the reserved-seams
register (grep-confirmed): the `traits.rs` "deferred until a downstream consumer exists" note is a
circular non-milestone, and by the register's own admission bar (a candidate with no named owner AND
concrete consuming milestone "is dead code to remove, not a seam to reserve") this fails the
reserved-seam test. **OWNER DECISION** (flag-only, not removed here — the `shared-memory` feature is in
the CI feature set): either register it as a reserved seam with a real owner + activating milestone, or
delete the feature and its five types and design `SharedRegion` against ferrompi's actual
`MPI_Win_allocate_shared` when an intra-node-shared-memory consumer lands.

**OD-002 · Sev B · premature-abstraction · effort S · confidence high** — `resolve_production_models`
(`production/hydro_models/production.rs:77`) and `resolve_evaporation_models`
(`evaporation.rs:70`) are `pub` `case_dir`-taking wrappers with **zero callers** anywhere in the
workspace (grep for the call form is empty; only defs, `pub use` re-exports, and doc-links appear).
Every real consumer already holds a parsed `CaseArtifacts` and calls the `_from_artifacts` sibling
directly. They speculatively mirror the crate-level `prepare_hydro_models`/`_from_artifacts` split one
layer down — but that outer split has dozens of real callers, this inner one has none. Being `pub`
(not `pub(crate)`) suppresses the `dead_code` lint, so the speculative surface ships silently. No
`// Rationale:`/Voice-4 comment names a future reader. Fix: delete both; re-add a `case_dir` entry
point next to the first real caller if one ever needs this granularity (YAGNI).

**OD-003 · Sev B · needless-indirection · effort S · confidence high** — `run_forward_pass` +
`ForwardPassBatch` (`training/forward/mod.rs:110-125`, `:189-233`) are a test-only shim shipping as
**public production surface**. `run_forward_pass`'s own doc says production callers use
`TrainingSession::run_forward_phase` and bypass this shim; all ~14 call sites are in-crate/integration
tests (the one non-test `use` is a doc-link import in `stats_aggregation.rs`). `ForwardPassBatch`
copies 5 fields straight into 5 of `ForwardPassInputs`'s 13 with no added invariant. Worse, lacking an
`IterationScratch` it **re-derives `terminal_has_boundary_cuts` independently**
(`mod.rs:208-215`, `fcf.pools[...].warm_start_count > 0`) — a second implementation of a fact the
session's priming bake already computes, free to silently drift. Fix: gate behind
`#[cfg(any(test, feature = "test-support"))]` (the pattern `StudySetup::training_ctx()` already uses),
or delete it and have tests construct `ForwardPassState`/`ForwardPassInputs` directly as
`TrainingSession` does.

**OD-004 · Sev C · speculative-generality · effort S · confidence high** — `FphaVisit`
(`lp/builder/fpha_cursor.rs`) carries two fields, `local_idx` and `plane_idx`, that **no consumer ever
reads** (grep for `.local_idx`/`.plane_idx` on the type across prod + tests → zero). They are
`#[allow(dead_code)]` with the rationale "rounds out the visit's identity" — a completeness narration,
not a reserved-seam owner/milestone and not the sanctioned symmetry-or-test-retention class (no test
reads them either). Per `.claude/rules/comments.md` a `dead_code` allow needs a D4 rationale, not
narration. Fix: drop both fields (the 4-field struct behaves identically;
`fill_fpha_entries`/`fill_fpha_rows` resolve everything through `cell`/`cell_local`/`row`).

**OD-005 · Sev C · needless-indirection · effort S · confidence high** —
`build_fpha_deviation_point_rows` (`production/hydro_models/export.rs:95-99`) is a one-line accessor —
body is literally `&result.fpha_deviation_point_rows` on an already-`pub` field; its own doc calls it
"A pass-through". Its two callers (`outputs.rs:114`, `cobre-python/run.rs:646`) could read the field
directly. It is the odd one out among `build_evaporation_model_rows`/`build_deviation_summary`, which do
real work. The one load-bearing fact in its doc ("already in canonical order, do not re-sort") belongs
on the field/resolver, not on a wrapper fn. Fix: delete; read the field directly.

**OD-006 · Sev C · needless-indirection · effort S · confidence high** — `CutActivityUpdates`
(`cut/cut_selection.rs:121-129`) exposes `deactivation_indices()`/`reactivation_indices()`, each a pure
`self.<field>.clone()` on an **already-`pub`** field. Every production consumer bypasses the methods and
reads the fields directly (`cut/pool.rs::apply_updates`, `session/mod.rs:1139-1140`); the accessors are
called only from the file's own `#[cfg(test)]` module (~55 deactivation / ~14 reactivation) plus one
integration test. They add no invariant or validation — only extra API surface only tests use. Fix:
delete both; tests read `.updates`/`.reactivations` directly as production already does.

**OD-007 · Sev C · over-parameterization · effort S · confidence high** — `validate_external_library`
(`stochastic/sampling/external.rs:547`) is generic over the hasher (`<S: BuildHasher>` on
`&HashSet<EntityId, S>`), but all three call sites (`setup/scenario_libraries.rs:158,223,273`) and the
doctest pass a default-`RandomState` `HashSet`; no caller supplies a non-default hasher. The parameter
buys polymorphism nothing exercises. Declaration-order invariance is carried by the ordered
`entity_ids: &[EntityId]` slice, not this membership set, so the hasher is irrelevant to reproducibility.
Fix: take `&HashSet<EntityId>`; re-add the type param when a caller needs one.

**OD-008 · Sev C · redundant-wrapper · effort S · confidence med** — `MethodologyConfig`
(`setup/methodology_config.rs`, whole 15-line file) exists solely to group two fields, `horizon` and
`inflow_method`. Every read site accesses `.methodology.horizon` or `.methodology.inflow_method`
individually (`accessors.rs`, `orchestration.rs`, `backward_pass_state.rs`); the struct is never passed
or matched as a unit outside its own constructor literal. Its "no `Default`, prevents misconfiguration"
rationale doesn't hold — `StudySetup` itself has no `Default` and uses the same one-shot literal. Fix:
hoist the two fields onto `StudySetup` directly (like the adjacent `downstream_par_order` /
`hydro_min_storage_hm3` fields); every call site loses one hop.

**OD-009 · Sev C · redundant-wrapper (policy refinement, not a violation) · effort S · confidence high**
— Several production fields/functions ship as `#[allow(dead_code)]` "for symmetry, exercised only by
tests": `RuntimeHandles.export_states` (`session/runtime.rs:11-18` — production reads
`config.events.export_states` directly at `session/mod.rs:263`), `RankDistribution.my_rank`
(`rank_distribution.rs:9-22`), and `ExchangeBuffers::new` (`state_exchange.rs:116-146` — a
uniform-distribution special case of `with_actual_counts`, called only from tests). These are the
sanctioned **"Symmetry-or-test-retention"** `#[allow(...)]` class
(`reserved-seams-and-deferred-debt.md` §`#[allow(...)]` census), so this is **not a violation** — but
the over-engineering lens notes the cleaner form: `#[cfg(test)]`-gating (or a `#[cfg(test)]` inherent
impl) achieves the same test-symmetry **without** shipping dead code in the release binary, which
`#[allow(dead_code)]` does. Owner-optional refinement of the sanctioned class, batchable with the Sev-C
cleanups; recorded so the pattern is tracked, not to force a change.

### Self-accounting — this branch's own contribution to the debt

The fixed-post-horizon-commitments work (which this record's own author helped land) is a net add to
two live findings: it made `reconcile.rs` the primary consumer of the untyped `entity_type` dictionary
and added a second family-classifier (**CD-010 sharpened**), and it introduced `build_boundary_fold` as
a third bespoke boundary-coupling shape (**CD-039 sharpened**). Both are consequences of the
still-pending tier-1 prerequisites (typed `StateFamily`; the generic `BoundaryStateRequirements`
frame) — the feature shipped correctly on top of the missing frame, as designed, and enlarged the frame's
eventual scope. This is the expected cost the roadmap's sequencing constraint ("land the frame BEFORE
the next boundary family") was written to avoid; it is now one family further behind.

### Reconciliation total (2026-08-22)

**33 live architecture findings + 1 retraction** (1×A, 16×B, 16×C) — unchanged in count from
2026-08-19 (7 previously-resolved re-confirmed; 3 sharpened: CD-007/010/039; no CD resolved or added
this pass). **Performance register:** PD-005 open (B), PD-004 deferred, PD-002/003 fixed, PD-001
refuted. **NEW `OD-` class: 9 over-engineering findings** (OD-001..009; 3×B, 6×C), all re-verified at
HEAD, reserved seams excluded. Grand total tracked: **33 CD + 9 OD live + PD register.** The
scheduling roadmap (2026-08-19) is unchanged in shape; the tier-1 typed-`StateFamily` /
`BoundaryStateRequirements` cluster is now **more** overdue (CD-010/CD-039 grew on this branch), and the
Sev-C cleanup tier gains OD-004..009. OD-001/002/003 (the Sev-B over-engineering findings) are
independent one-shot removals/gates, schedulable any time. Counts are a snapshot; re-verify each
against the then-current tree before acting.

---

## ★ OWNER RATIFICATION + EXECUTION WAVES (2026-08-22 — post-reconciliation scheduling)

Three owner decisions ratified in review of the 2026-08-22 reconciliation; the 2026-08-19 tier
roadmap is refined into the dependency/deadline-ordered execution waves below. All waves start
from merged `main` (the fixed-post-horizon branch's PR is pending at ratification time).

### Ratified decisions

- **OD-001 → KEEP-RESERVED (owner-ratified 2026-08-22).** The shared-memory trait hierarchy
  stays. Disposition: add the reserved-seam register entry
  (`docs/design/reserved-seams-and-deferred-debt.md`) with owner + activating milestone =
  **intra-node scenario-library sharing AND intra-node cut-archive sharing** (the long-term
  goal: one copy per node instead of per rank for the two largest read-only per-rank
  datasets), with `SharedRegion` to be designed against ferrompi's `MPI_Win_allocate_shared`
  when the first consumer lands. The finding CLOSES when the register entry lands (Wave 0);
  until then the hierarchy is reserved, not dead.
- **CD-020 → RATIFIED: reframe as MAGIC tags — keep the bytes (owner-ratified 2026-08-22).**
  `CUT_WIRE_VERSION` and `BASIS_BROADCAST_WIRE_VERSION` stay as leading bytes, reframed as
  fixed MAGIC/format tags whose sole job is the corruption/misalignment tripwire: rename the
  constants, drop the bump-discipline + "no compat shim" framing, fix the E6 doc drift
  ("wire-version 1" → the tag), tighten the `pub` re-export to `pub(crate)`. NO wire change.
  Removal (matching the sim exchange's version-free style) was considered and DECLINED — it
  discards a free defensive check. Wave 0.
- **Wave 4 (setup lifecycle redesign) → PLAN GREENLIT (owner-ratified 2026-08-22).**
  CD-001/003/004/005 (+ CD-002 shrink, CD-024-successor seam activate-or-die) get a
  dedicated plan. Sequencing: author the plan AFTER Wave 3 lands, so it targets the
  post-frame carriers rather than baking in pre-Wave-3 shapes (findings are snapshots; the
  boundary frame reshapes the very carriers the redesign must collapse).

### Execution waves (refines the 2026-08-19 tiers; ordered by dependency + deadline)

| Wave | Content | Effort | Gate / notes |
| ---- | ------- | ------ | ------------ |
| **0** | Owner-decision execution + dead-surface sweep: OD-001 register entry; CD-020 MAGIC reframe; removals/gates OD-002/003/004/005/006/007/008, CD-019, CD-017 (assert + drop `new_tight`), CD-036 | all S (~2–3 days) | No parity risk (dead/test-only surfaces); OD-003 gates via the `test-support` pattern |
| **1** | CD-010: typed `StateFamily` in cobre-io + `EntitySlot::family()`; absorb the `checkpoint.rs:30` duplicate const; disambiguate the `OUTPUT_ENTITY_*` collision; cover the public cobre-python seam | S–M | Byte-neutral (wire stays u8); regen `schemas/` if schemars-visible |
| **2** | Nested-UB consolidation: CD-032 (+ its 3 folded sub-items) + CD-033 + PD-005 | S–M, one region | `ForwardBound::NestedRisk` promotion; land before the next risk-measure/stopping-rule feature; order-flexible vs Wave 3 |
| **3** | Boundary frame: CD-031 + CD-039 — family-generic `BoundaryStateRequirements` resolver, `BoundaryPolicy::checkpoint_path`, parameterized reservation helper | M–L | ⏰ MUST land before the GNL anticipated-coupling import; depends on Wave 1 |
| **4** | Setup lifecycle redesign: CD-001 (Sev A non-root mirror) + CD-003/004 (`ResolvedRunConfig` single projection) + CD-005 (Model/RunParams/output split) + CD-002; CD-024-successor seam activates or dies here | L, dedicated plan | MPI byte-repro bar: parity goldens, rank-invariance, `mpiexec -n 1/2` repro |
| **5** | Output single owner: CD-025 hoist (shared output helpers reachable by CLI + Python) + CD-029 fold (`PrepPhase` boundary check + Python parity) | M | Python-parity tests; parallelizable with Waves 2–4 (disjoint files) |
| **6** | Retrofit/typed-seam sweep: CD-015+022 (one extraction, relieves CD-014), CD-018, CD-035, CD-034, CD-023 prep re-home | S–M each | Proven fix-shapes; byte-neutral vs sacred parity goldens |
| **7** | C-tier batch: CD-007/009/011/012/014-remnant/016/021/028/030/037/038 + OD-009 (owner-optional) + the two v0.14.2 residues | opportunistic | Batch with adjacent feature work; no dedicated push |

**Scheduling refinements vs the 2026-08-19 tiers (recorded):** CD-025 is demoted out of the
tier-1 cluster into its own wave — it shares the principle (one owner per fact) but has zero
code dependency on the setup redesign, so it must not be hostage to the L-effort plan. CD-029
moves from "pattern decisions" into Wave 5 (same Python-parity seam, same fix session).
Wave 3 deliberately precedes Wave 4: the frame rides the existing three carriers **as one
struct** (CD-039's own prescription), so the later carrier collapse relocates one field
instead of N scalars; the GNL deadline drives the frame, nothing drives the redesign's date.

**Do-not-touch list (unchanged):** CD-008 (retracted), PD-001 (refuted), PD-004 (deferred
pending a profile), the sanctioned reserved-seam census. **Resume protocol unchanged:**
findings are snapshots — re-verify each against the then-current tree before acting.

### Wave 0 — executed (2026-08-22, branch `chore/arch-debt-wave-0` off `b80d9e62`)

Dead-surface sweep + ratified decisions applied. Net −91 lines across 29 files; every finding
re-verified against the live tree before acting.

**Applied (behavior-neutral):**

- **OD-002** — deleted the zero-caller `resolve_production_models` / `resolve_evaporation_models`
  case-dir wrappers; relocated their `# Errors` tables and the evaporation linearization doc onto
  the surviving `_from_artifacts` siblings; re-pointed every intra-doc link; dropped now-unused
  `Path` / `load_artifacts_for_hydro_models` imports.
- **OD-004** — dropped `FphaVisit::{local_idx, plane_idx}` (no consumer).
- **OD-005** — inlined `build_fpha_deviation_point_rows` at its two cross-crate callers (CLI +
  Python) as `…fpha_deviation_point_rows.as_slice()`; relocated the "canonical order, do not
  re-sort" contract onto the `PrepareHydroModelsResult` field.
- **OD-006** — removed `CutActivityUpdates::{deactivation_indices, reactivation_indices}` and
  their dedicated test; call sites read `.updates` / `.reactivations` directly.
- **OD-007** — dropped the unexercised `<S: BuildHasher>` genericity from `validate_external_library`.
- **OD-008** — hoisted `MethodologyConfig`'s two fields (`horizon`, `inflow_method`) onto
  `StudySetup`; deleted the wrapper struct + its module.
- **OD-003** — gated the test-only `run_forward_pass` / `ForwardPassBatch` shim behind
  `#[cfg(any(test, feature = "test-support"))]` (its sole integration consumer already requires
  `test-support`); the doc-link import in `stats_aggregation.rs` gated identically. CI's doc job
  runs with `test-support` (NON_SOLVER_FEATURES), so the `[run_forward_pass]` intra-doc links
  still resolve.
- **CD-017** — added a truncation guard to `reconstruct_col_statuses` + dropped the always-zero
  `ReconstructionStats::new_tight`. DEVIATION: used `debug_assert!(stored ≤ target)`, NOT the
  backlog's literal `debug_assert_eq!` — the function documents "padded if wider", so a
  legitimately wider target LP is valid and `_eq!` would false-fire; `≤` encodes the real hazard
  (stored wider than target → silent truncation).
- **CD-036** — pre-allocate `IterationScratch.ub_{path_weights,stage_costs}` via
  `Vec::with_capacity` (not length): removes the first-iteration realloc while keeping
  `len()==0`/empty on the sampled path (honoring the field-level "empty on sampled" contract).

**Ratified decisions applied:**

- **CD-020** — renamed `CUT_WIRE_VERSION` → `CUT_WIRE_FORMAT_TAG` and
  `BASIS_BROADCAST_WIRE_VERSION` → `BASIS_BROADCAST_FORMAT_TAG` (value unchanged at `2` →
  byte-neutral on the wire); reframed the cut-wire "Version compatibility" module doc as a
  format-tag corruption-tripwire guard; fixed the E6-checklist drift in `comments.md`
  ("wire-version 1" → the format tag); updated this register's entry. Residual (recorded, not
  debt): the reject-diagnostic runtime strings keep their "version" wording (pinned by the reject
  tests), and the constants stay `pub` (the `mpi_wire.rs` integration test consumes them).
- **OD-001** — KEEP-RESERVED: added the reserved-seam register entry for the `cobre-comm`
  shared-memory trait hierarchy; activating milestone = intra-node scenario-library + cut-archive
  sharing.

**Deferred with rationale (NO code change):**

- **CD-019** (`sync_cuts` / `pack_local_records` / `sync_packed_records`) — NOT removed. The
  committed register (`### Superseded cut-sync public methods`) defers removal to "the next
  licensed public-API break": these are re-exported public API on a published crate, so removing
  them now is an out-of-band breaking change, and the "unwired config is reserved, not dead" hard
  rule forbids removing a registered item without owner sign-off. The Wave-0 "removal" label is
  overridden by the committed register. Already registered; left in place.

**Verification (all green):** `cargo fmt --all` + cobre-python fmt-check; `cargo check --workspace
--features "mpi numa shared-memory serde schema slow-tests flatc-conformance test-support"
--all-targets`; the same with `cargo clippy … -- -D warnings`; `RUSTDOCFLAGS=-Dwarnings cargo doc
--workspace --no-deps …`; cobre-python `cargo check --manifest-path`; and targeted `nextest` over
the touched lib units + `cut_basis` + `mpi_wire` (determinism gates included) + `deterministic`.
No parity baseline moved.

### Wave 1 — executed (2026-08-22, CD-010 typed `StateFamily`)

Introduced the typed state-family vocabulary in cobre-io and migrated every production consumer
off the scattered untyped `entity_type: u8` dictionary. Byte-neutral (the wire byte
`EntitySlot.entity_type: u8` is unchanged; `schemas/policy.fbs` untouched — `StateFamily` mirrors
the existing `EntityType` enum).

**cobre-io (the new single owner):**

- `output/policy/records.rs` — new `pub enum StateFamily` (Rust mirror of `EntityType`:
  HydroStorage=0, HydroInflowLag=1, AnticipatedThermalState=2, HydroTransitBucket=3) with
  `code()` / `from_code()`, plus `EntitySlot::family()`; a contract unit test pins the
  discriminants to the schema.
- Re-exported through `output/policy/mod.rs` and crate-root `lib.rs` (`cobre_io::StateFamily`).
- `output/policy/checkpoint.rs` — deleted the duplicated `ENTITY_TYPE_HYDRO_TRANSIT_BUCKET` const;
  the monotonicity check uses `slot.family()`, and its "only this family, subindex is
  delivery-ordered" rationale moved onto `check_transit_bucket_monotonicity`.
- `output/dictionary.rs` — renamed the colliding i8 physical-output dictionary `ENTITY_TYPE_*` →
  `OUTPUT_ENTITY_*` (file-local disambiguation).

**cobre-sddp (consumers):**

- `policy/policy_export.rs` — removed the 4 `pub(crate) const ENTITY_TYPE_*` (the downstream
  dictionary); writers/readers use `StateFamily::X.code()` / `.family()`.
- `policy/policy_load.rs` — import retargeted to `cobre_io::StateFamily`.
- `policy/reconcile.rs` — `resolve_target_slot` now matches `slot.family()`; the parallel private
  `ReportFamily` enum + its `report_family` classifier were ELIMINATED, collapsed into
  `Option<StateFamily>` (None = other-identity) across `tally_mut` / `classify_op` /
  `build_reconciliation_report`.

**Residual (not debt):** the 5 standalone integration-test files keep their own local
`const ENTITY_TYPE_* = N` wire-byte pins — self-contained, legitimately pinning the on-disk byte
contract; not migrated.

**Verification (all green):** `cargo fmt --all`; `cargo check --workspace` full-MPI-features
--all-targets; clippy -D warnings; the infra-genericity gate (`check-infra-genericity.sh` — cobre-io
stays algorithm-generic); `cargo doc` -D warnings; targeted `nextest` over all cobre-io tests +
the cobre-sddp policy/reconcile units + the boundary/anticipated integration suites. No parity
baseline moved. The committed register's `#### Untyped state-family primitive…` entry is marked
RESOLVED.

### Wave 2 — executed (2026-08-22, nested-UB consolidation: CD-032 + CD-033 + PD-005)

Consolidated the enumerated-forward upper-bound machinery. Byte-neutral — the nested bound's
value, gather order, and rank-invariance are preserved (a parity hash + the CVaR determinism gates
stay green).

- **CD-032** — the nested risk-adjusted upper bound was a session-side override
  (`apply_nested_cvar_ub`) that discarded `sync_forward`'s risk-neutral result and did a SECOND
  `allgatherv`. It is now a first-class `ForwardBound::NestedRisk` arm of `sync_forward` (one
  gather, one estimator dispatch); `enumerated_nested_ub` + `apply_nested_cvar_ub` deleted.
  Sub-items: the stride reads the declared `RankDistribution::num_stages` (not `stage_stats.len()`);
  the recursion's root→leaf walk gained the length assert `EnumeratedPlan::walk_path` carries.
- **CD-033** — the "nested UB applies" predicate is single-owned: `is_effective_non_expectation`
  and the session both route through `RiskMeasure::effective` (sole owner of the `lambda > 0`
  rule), and `reject_gap_under_nonuniform_risk` decides via `uniform_effective_measure`, so the
  admission gate cannot drift from the bound the session applies.
- **PD-005** — `nested_ub_recursion` reuses one `RiskMeasureScratch` across every interior node's
  risk evaluation via the new `RiskMeasure::evaluate_risk_into`, removing the per-node CVaR-weight
  allocation; the now-dead allocating `compute_cvar_weights_from_costs` wrapper removed.

**Scoped-down (noted residuals, not done):** PD-005's once-per-iteration TOPOLOGY precompute
(children/roots/order are pure functions of `EnumeratedPlan`, cacheable at training start) — the
per-node scratch alloc was the dominant one and is fixed; the topology recompute is a lower-value
follow-up. CD-032 sub-item (a)'s full walk-dedup (routing `nested_ub_recursion` through
`EnumeratedPlan::walk_path`) was NOT done — `nested_ub_recursion` keeps decomposed args so its
decomposed-arg unit tests stay intact; the dropped length-assert half IS restored. CD-032 sub-item
(b): folding `enumerated_nested_ub` into `sync_forward` removed ONE of the four counts/displs
copies; the `cut_sync` + `RankDistribution::actual_per_rank` unification remains.

**sddp.md updated:** the "enumerated CVaR upper bound is NESTED" contract now references
`ForwardBound::NestedRisk` / `sync_forward` (the deleted `enumerated_nested_ub` /
`apply_nested_cvar_ub` pointers removed).

**Verification (all green):** fmt; `cargo check --workspace` full-MPI --all-targets; clippy
-D warnings; `cargo doc` -D warnings; and the pinned tests —
`nested_ub_recursion_is_nested_not_end_of_horizon`, the `enumerated_cvar_gap` end-to-end module
(`uniform_cvar_gap_admits_trains_and_brackets`), all 8 `admission_gate_*`, the mpi_wire CVaR
determinism gates, and a declaration-order parity hash. No parity baseline moved.

### Residuals closed (2026-08-22) — Wave-0/1/2 follow-through

Every scope-down flagged in Waves 0–2 is now closed; no open residual remains (CD-019 is a
deliberate register-gated deferral, not a residual).

- **Wave-1 (CD-010) test-file migration** — the 5 standalone integration-test files migrated off
  their local `const ENTITY_TYPE_* = N` wire-byte consts to `cobre_io::StateFamily` (12 consts).
  The wire bytes stay pinned by `state_family_codes_match_policy_fbs_entity_type`.
- **Wave-2 PD-005 topology precompute** — `NestedUbTopology` (children / roots / valuation order /
  node marginals / stage) is precomputed ONCE on `EnumeratedPlan` (`Box`ed to keep
  `Traversal::Enumerated` lean); `nested_ub_recursion` reads it and fills only the per-iteration
  realized costs. Byte-neutral (representative-path cost lookup preserves the idempotent per-node
  value); the per-iteration `children: Vec<Vec<NodePos>>` rebuild is gone.
- **Wave-2 CD-032 sub-item (a)** (walk-dedup) — the leaf→root walk is a single free fn
  `walk_leaf_to_root`; `EnumeratedPlan::walk_path` + `NestedUbTopology::new` both drive off it.
- **Wave-2 CD-032 sub-item (b)** (counts/displs) — `cobre_comm::{per_rank_counts, prefix_displs}`
  single-own the allgatherv rank-partition arithmetic; `sync_forward`, `CutSyncBuffers`, and
  `RankDistribution::actual_per_rank` all use them (was 3 copies).

**Surfaced + fixed a latent Wave-0 clippy failure:** OD-007's non-generic `&HashSet<EntityId>`
trips `clippy::implicit_hasher` under `-D warnings`; Waves 0/1/2 clippy greens were cache-masked
(cobre-stochastic was never re-linted until the cobre-comm change here invalidated its cache).
Fixed with `#[allow(clippy::implicit_hasher)]` + a rationale (preserves OD-007's non-generic form).
Lesson: incremental `cargo clippy` can false-green; the fresh CI clippy is authoritative.

**Verification (all green):** fmt; `cargo check --workspace` full-MPI --all-targets; clippy
-D warnings (fresh recompile of the changed crates); `cargo doc` -D warnings; and the nested-UB
unit tests, `enumerated_cvar_gap`, the mpi_wire CVaR determinism gates, and the parity hashes.

### Wave 3 — executed (2026-08-22, boundary frame CD-031 + CD-039 setup-half; commit `44e72b76`)

**Scope (owner-ratified):** setup-channel frame only. CD-031 in full + CD-039's setup half.
The writer channel + transit-bucket gap stay reserved seams (see below). Object shape =
opaque struct + accessors (not a family-keyed map/enum), owner-chosen with the lag-field
tension surfaced.

**CD-031 (RESOLVED).** New `BoundaryStateRequirements { present, inflow_lag_depth }` in
cobre-sddp `setup/params.rs` — opaque, accessors `is_present()`/`inflow_lag_depth()`,
constructors `none()`/`present(depth)` (`present(0)` = present-no-depth). One resolver
`resolve_boundary_state_requirements(case_dir, &config)` (policy_load.rs) replaces all four
open-coded folds (CLI run/validate, Python run, test helper). `resolve_effective_inflow_lag_depth`
RETIRED (its `then_some` folded into `present`); `boundary_policy_required_lag_depth` kept as the
depth rule. `BoundaryPolicy::checkpoint_path(case_dir)` accessor added in cobre-io — owns all 7
`case_dir.join(&bp.path)` joins (3 identical warning comments hoisted into its doc). The 3 carriers
(`StudyParams`/`BroadcastConfig`/`ConstructionConfig`) hold ONE `boundary` field (was
`inflow_lag_depth` + `boundary_present`), so the two co-derived facts are consistent by
construction. `StudySetup` gained a `boundary_requirements` field + accessor; the boundary-cut LOAD
path (`apply_training_policy`, python `apply_policy`) reads the reserved depth off the setup instead
of re-parsing the checkpoint (the double-read smell). `new_with_inflow_lag_depth` →
`new_with_boundary_requirements(…, BoundaryStateRequirements)`.

**CD-039 (setup-half RESOLVED; writer + transit still open).** The setup channel is now one generic
`BoundaryStateRequirements` riding the carriers. REMAINING per the reserved-seams doc: (a) the WRITER
channel — family-parameterize `reserve_boundary_inflow_lag_slots` (leading-block detect / canonical
insert / keyed placement, ~90% family-independent) + family-key the per-cut coefficient field; (b)
the transit-bucket C17 silent-drop (`resolve_transit_bucket` → `RebindOp::Zero`); (c) the
near-isomorphic carrier-trio collapse (= Wave 4 setup redesign). Trigger unchanged: before the next
externally-authored boundary family (GNL anticipated import) or with the setup redesign.

**Byte-neutrality gotcha (recorded in memory):** bundling `present`+`depth` makes `from_config` emit
`none()` (the resolver owns presence), so `StudySetup::new` re-derives presence from
`config.policy.boundary.is_some()` (→ `present(0)`, depth unresolved — no case_dir), and the test
helper (renamed `boundary_inflow_lag_depth`→`boundary_requirements`) returns `present(0)` (NOT
`none()`) for a configured-but-unreadable fake checkpoint. Two `boundary_present`-gate water-bucket
tests (deterministic + hydro_sim) caught both regressions — they ARE the byte-neutrality proof.

**Verification (all green):** fmt --check; `cargo check --workspace` full-MPI --all-targets; fresh
clippy -D warnings; `cargo doc` -D warnings; `check-infra-genericity.sh` + the doc/comment integrity
gates + `check_doc_voice.py`; `cargo check` cobre-python (excluded from workspace; 5 pre-existing
Wave-0/OD-003 featureless warnings, unrelated); the full 6401-test workspace sweep; and the byte-
neutrality gates — every parity hash (d02/d06/d30/d34/d41…), mpi_wire determinism (opening-order,
by-node, retry-armed, hardest-first), and the boundary/anticipated suites. Committed reserved-seams
doc updated (CD-031 entry → RESOLVED, CD-039 entry → setup-half done). NOT pushed. No live boundary
example deck exists, so the MPI wire reshape is covered by the postcard round-trip + rank0-vs-non-root
reconstruction unit tests + the 2-rank-stub determinism gates, not a live mpiexec.

### Wave 3 continuation — executed (2026-08-22, close the deferred boundary/setup debt; commits `1702b2a0`→`9db2cbdf`)

**Scope (owner-ratified via AskUserQuestion after 3-agent recon):** carrier-trio = bounded
(StudyParams/ConstructionConfig merge + CD-001 non-root hoist), NOT the full wire-collapse —
that is postcard-serialization-boundary-blocked and stays Wave 4. Writer = extract the generic
core only (full family-parameterization blocked: anticipated `delivery_date` needs resolver
context, no 2nd authoring consumer exists). Transit-C17 warning + Wave-0 residual land regardless.

**Phase 1 — transit-C17 (`1702b2a0`).** `warn_dropped_source_couplings` (policy_load.rs) surfaces a
dropped boundary SOURCE coupling (a slot the study's topology omits) as a per-family load-time
warning through the existing `on_warning` sink. KEY: the drop is family-blind (`dropped_source_positions`),
no §7 contract addresses the source-drop direction, rejecting would break a legit superset source,
and no widening path can fabricate an undeclared arc — so the fix is to surface, not reject/widen.
Non-breaking (load still succeeds). New pin `boundary_injection_dropped_source_transit_coupling_warns_and_loads`.

**Phase 2 — writer core (`273f801f`).** `splice_reserved_state_block` (policy_export.rs) owns the
family-independent prefix/reserved/tail splice + keyed-coefficient placement + alignment guard;
`reserve_boundary_inflow_lag_slots` keeps its inflow-lag anchor + slot-body + reject and delegates.
Byte-identical (all 7 reserve_* units + Python round-trip unchanged). Anticipated/transit authoring
still unsupported (no side-channel field; anticipated slot `delivery_date` is resolver-derived).

**Phase 3 — StudyParams/ConstructionConfig merge (`4075c4e8`).** `ConstructionConfig` +
`into_construction_config` DELETED; `StudyParams` is the single local projection. `from_config`
captures `export_states` from `config.exports.states` (drops the false stub); `scalar_parameters`
stays late-bound (disk artifacts). Kept StudyParams (67 refs) over ConstructionConfig (22 refs) —
fewer edits, same outcome. `BroadcastConfig` wire projection UNTOUCHED (the postcard boundary =
Wave 4). Net −103 lines. Python `set_export_states` dropped (redundant once from_config captures it).

**Phase 4 — CD-001 non-root hoist (`b051c410`, the Sev-A).** `build_stochastic_context_for_study`
(cobre-sddp stochastic_pipeline.rs) is the single owner both `prepare_stochastic` (rank 0, after
estimation + user-tree load) and the CLI non-root path call. `rebuild_historical_library_non_root`
(the line-for-line cross-crate mirror) + its ~260-line test fixture DELETED (widening covered by
`build_opening_tree_library_widens_max_order_to_declared_depth`). `user_tree` + `external_scenario_counts`
are the rank-varying params (rank 0 loads/computes; non-root gets the wire tree + None counts). Net
−490 lines. Byte-neutral: `derived_inflow_seeds_rank_invariant`, `non_root_opening_tree_matches_rank_0`,
`opening_order_determinism`, every parity hash green.

**Phase 5 — Wave-0 residual (`18c175ab`).** `training/forward/mod.rs` shim imports (Sender,
TrainingEvent, ActiveProfile, solver traits, context/workspace/trajectory types) cfg-gated
`#[cfg(any(test, feature = "test-support"))]` to mirror the `run_forward_pass`/`ForwardPassBatch`
usage → the cobre-python featureless build is now WARNING-CLEAN (closes the OD-003 gating imperfection).

**Verification (all green):** fmt --check; fresh clippy -Dwarnings (full MPI feature set, --all-targets);
doc -Dwarnings; check-infra-genericity + doc-paths/comment-refs/comment-line-refs/doc-voice gates;
cobre-python featureless check (warning-clean); 6401-test workspace sweep; parity + mpi_wire
byte-neutrality (62 tests). Reserved-seams doc updated (`9db2cbdf`): setup-sprawl entry → 2/3 sub-parts
closed; per-family entry → transit surfaced + writer core extracted. NOT pushed (branch = 11 commits).

**REMAINING (Wave 4, needs its own /plan):** BroadcastConfig↔StudyParams wire-projection unification
(the postcard tagged-enum serialization-boundary decision) + StudySetup god-struct lifecycle split.
Writer's 2nd-family slot-body constructor + keyed per-cut field (+ anticipated `delivery_date`) stay a
reserved seam until a concrete 2nd authoring family lands.

## ★ POST-PLAN UPDATE — boundary-policy self-describing reconciliation (2026-08-23, feat branch @ HEAD `4f2ca640`, NOT pushed)

The `boundary-policy-self-describing-reconciliation` plan (8 tickets / 3 epics, commits
`ce25796c`→`4f2ca640`, stacked on the Wave-3-continuation tip `9db2cbdf`) landed. It is NEW
architectural work born from two owner considerations — make the cut `.bin` self-contained for
boundary reconciliation, and make a NEWAVE-like→DECOMP-like boundary transition reconcilable from
within cobre — NOT a scheduled debt wave, so it **resolves no tracked CD**. Recorded here for the
current-state picture + self-accounting.

**What landed:**

- `ce25796c` — relax the boundary `state_dimension` gate (`PolicyLoadKind::CHECK_STATE_DIMENSION`,
  `false` for `BoundaryInjection`); a differing-dimension boundary load defers to per-slot
  reconciliation (+ an empty-manifest fallback guard so an unverifiable, un-reconcilable load
  rejects cleanly rather than panicking the fixed-length cut-pool copy). Enables NEWAVE→DECOMP.
- `c3869088` — `StageCuts` gains `cost_scale_factor`/`node_id`/`graph_stage_id` (additive ids
  8/9/10); the boundary load reads its study-global facts from the resolved pool's own
  `cuts/<pool>.bin`; clean-break reject of a pre-change `.bin`; a `node_id == -1` shared-pool reject
  (a boundary source must be a single-node terminal pool).
- `32fa7b15` — retire `metadata.json` onto a new `CheckpointManifest` FlatBuffers root
  (`manifest.bin`, a 1:1 swap — no net new file); FullFcf source cost-scale reads the per-pool
  `.bin` via `checkpoint_terminal_cost_scale_factor`; `PolicyCheckpointMetadata` deleted (serde
  derives dropped through `PolicyCheckpoint`); a shared `test_support::checkpoint_metadata` builder
  absorbs the ~19-site constructor ripple; dead `resolve_warm_start_counts` removed; Python parity
  (`load_policy` surfaces the metadata field-by-field).
- `4318e57d` — §7 `.claude/rules/sddp.md` policy-load contract completeness (the epic-2/3
  self-describing reads + three rejects) + reserved-seam annotation on the Legacy cost-scale branch.
- `4f2ca640` — rationale the PRE-EXISTING `leading_in_study` `#[allow(dead_code)]` (from the
  fixed-post-horizon workstream `b7d37237`, a different branch of work) so the merge-base-diffed
  `check-allow-rationale` gate is green — the branch full CI bar is now clean.

**Debt impact:**

- **No tracked CD resolved** — the plan's subject was not a backlog finding; it is new capability.
- **CD-039 (per-family boundary channels) — UNCHANGED / adjacent.** The plan made the boundary READ
  path fully self-describing (every family's facts now read from the per-pool `.bin`; `metadata.json`
  off the boundary path entirely), but did NOT touch CD-039's open writer-half — the
  family-parameterized slot-RESERVATION channel for a 2nd authoring family stays the reserved seam.
  The manifest's `cost_scale_factor`/`node_id`/`graph_stage_id` are per-pool scalar provenance, not
  the per-family slot-body authoring channel CD-039 tracks.
- **Station 2 (policy/checkpoint layer) stays a POSITIVE reference** — the type-state proof pattern is
  intact and now cleaner: the boundary path reads zero `metadata.json` fields, a single
  `cuts/<pool>.bin` is self-describing, and `manifest.bin` is a version-gated FlatBuffers root with no
  hand-editable JSON to silently desync from the `.bin`s.
- **Minor cleanups (not tracked CDs):** dead `resolve_warm_start_counts` deleted (no
  `#[allow(dead_code)]` left behind); the ~19-site `PolicyCheckpointMetadata` test-construction ripple
  collapsed to one `test_support::checkpoint_metadata` builder — the mitigation ticket-003 flagged,
  relieving the test-construction-fan-out concern for this type (CD-007-adjacent).

**Self-accounting (this plan's own contribution to the debt — kept minimal):**

- **New dual-owned wire format `CheckpointManifest`** (`manifest.bin`) — properly E6-covered
  (round-trip + stale-version reject + a None-path regression test in
  `flatbuffers_schema_conformance.rs`; the E6 table gained a `CheckpointManifest` note). Not debt.
- **New documented reserved seam:** `rescale_cut_records_for_load`'s `None`/Legacy branch +
  `LEGACY_COST_SCALE_FACTOR` are now unreachable from both front ends (both hard-reject a `None`
  cost-scale before rescale) — annotated as a reserved direct-library-caller seam, NOT removed
  (it is `pub`-exported; reserved-not-dead rule). Documented, not silent.
- No new CD / OD / PD.

**Net:** totals UNCHANGED by this plan (resolves none, adds none). **Wave 4 scope UNCHANGED** — the
plan touched the policy/checkpoint layer, not the setup config-projection carriers
(`StudyParams`/`BroadcastConfig`) or the `StudySetup` god-struct; the BroadcastConfig↔StudyParams wire
unification + lifecycle split remain Wave 4. The prior Sev-A headline (CD-001) was already closed by
the Wave-3 continuation; the largest live cluster remains the Wave-4 setup redesign (CD-002/003/004/005).

**Verification:** all 8 tickets guardian-verified + quality-scored (mean quality 0.99); 3
epic-boundary + 1 plan-level simplify/review passes (plan-level review PASS_WITH_WARNINGS, both
warnings closed); byte-neutral throughout — `tests/fixtures/` never re-baselined, parity + mpi_wire
reproduced at every boundary; full fast-gate suite green (infra-genericity, plan-leak,
comment/doc/voice, allow-rationale, python-parity) + `cargo doc -D warnings` + fresh full-feature
`clippy -D warnings`. NOT pushed.

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — core-io

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — stochastic

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — solver-comm

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — cli-python

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — build-ci

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — test-corpus

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — generalization-alignment

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — performance-sweep

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — reconciliation

_(no entries yet)_

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — unified-roadmap

_(no entries yet)_
