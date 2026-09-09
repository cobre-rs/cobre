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

| Milestone | Meaning                                                                                                                                   | Precedes | Source                                                           |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------- |
| `0a`      | Engine seam; `study` config block + admission gate; shared output orchestration in cobre-io; rank-0-executes MPI; byte-identical for SDDP | `0b`     | beyond-sddp-generalization.md V.1 (split resolved by D12)        |
| `0b`      | Carve `cobre-model` from the engine-neutral part of cobre-sddp `lp/`                                                                      | `1`      | beyond-sddp-generalization.md V.1 / IV.2 (split resolved by D12) |
| `1`       | Purify the data model: stochastic off System/Stage, training_event out of cobre-core, StageTemplate shed, case v2 + bit-for-bit shim      | —        | beyond-sddp-generalization.md V.2                                |
| `neutral` | Advances no phase                                                                                                                         | —        | —                                                                |

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
fourth _state_ channel — but it is a fourth bespoke path by which an externally-authored boundary
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

| Wave  | Content                                                                                                                                                                                                      | Effort            | Gate / notes                                                                                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **0** | Owner-decision execution + dead-surface sweep: OD-001 register entry; CD-020 MAGIC reframe; removals/gates OD-002/003/004/005/006/007/008, CD-019, CD-017 (assert + drop `new_tight`), CD-036                | all S (~2–3 days) | No parity risk (dead/test-only surfaces); OD-003 gates via the `test-support` pattern                                   |
| **1** | CD-010: typed `StateFamily` in cobre-io + `EntitySlot::family()`; absorb the `checkpoint.rs:30` duplicate const; disambiguate the `OUTPUT_ENTITY_*` collision; cover the public cobre-python seam            | S–M               | Byte-neutral (wire stays u8); regen `schemas/` if schemars-visible                                                      |
| **2** | Nested-UB consolidation: CD-032 (+ its 3 folded sub-items) + CD-033 + PD-005                                                                                                                                 | S–M, one region   | `ForwardBound::NestedRisk` promotion; land before the next risk-measure/stopping-rule feature; order-flexible vs Wave 3 |
| **3** | Boundary frame: CD-031 + CD-039 — family-generic `BoundaryStateRequirements` resolver, `BoundaryPolicy::checkpoint_path`, parameterized reservation helper                                                   | M–L               | ⏰ MUST land before the GNL anticipated-coupling import; depends on Wave 1                                              |
| **4** | Setup lifecycle redesign: CD-001 (Sev A non-root mirror) + CD-003/004 (`ResolvedRunConfig` single projection) + CD-005 (Model/RunParams/output split) + CD-002; CD-024-successor seam activates or dies here | L, dedicated plan | MPI byte-repro bar: parity goldens, rank-invariance, `mpiexec -n 1/2` repro                                             |
| **5** | Output single owner: CD-025 hoist (shared output helpers reachable by CLI + Python) + CD-029 fold (`PrepPhase` boundary check + Python parity)                                                               | M                 | Python-parity tests; parallelizable with Waves 2–4 (disjoint files)                                                     |
| **6** | Retrofit/typed-seam sweep: CD-015+022 (one extraction, relieves CD-014), CD-018, CD-035, CD-034, CD-023 prep re-home                                                                                         | S–M each          | Proven fix-shapes; byte-neutral vs sacred parity goldens                                                                |
| **7** | C-tier batch: CD-007/009/011/012/014-remnant/016/021/028/030/037/038 + OD-009 (owner-optional) + the two v0.14.2 residues                                                                                    | opportunistic     | Batch with adjacent feature work; no dedicated push                                                                     |

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

**Station.** cobre-core + cobre-io. **Method.** Four lenses over sub-stations A (cobre-core) and B/C/D (cobre-io input, config+validation, output).

**Baseline.** `a136840d4f2ea137f685f0af6dac04254b983b60` (pinned). Read-only station: no tracked file was modified.
**Method.** Four lenses over sub-stations A (cobre-core) and B/C/D (cobre-io input, config+validation, output); 81 attacker candidates, 81 defended, 77 confirmed, 4 dismissed.

### Architecture findings

**CD-040 · Sev B · asymmetry · effort M · confidence high**
HydroPenalties (entities/hydro.rs:58) and HydroStagePenalties (model/resolved/penalties.rs:37) are structurally identical 16-f64 types joined by an unguarded positional field-copy in cobre-io (resolution/penalties.rs:397-419) where any two-line transposition compiles silently; the HydroPenaltyOverrides third declaration is justified by its distinct Option cascade semantics and is not part of the defect.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/entities/hydro.rs::HydroPenalties`, `crates/cobre-core/src/model/resolved/penalties.rs::HydroStagePenalties`, `crates/cobre-core/src/model/penalty.rs::HydroPenaltyOverrides`
- **Evidence:** All three structs declare exactly 16 fields.
- **Fix-shape:** Collapse the two identical structs into one declaration owned by the entity module, and have the resolved per-(hydro, stage) table store that single type rather than a twin. The override struct stays distinct because its `Option` semantics differ, but it should be derived from the same field list rather than restated, so a new penalty column is added once.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-041 · Sev B · asymmetry · effort M · confidence high**
DisconnectedBus (error.rs:65) and InvalidPenalty (error.rs:72) are emitted by no production/validation path and both carry a false 'Emitted by cobre-io validation' doc line while cobre-io imports ValidationError zero times; DisconnectedBus additionally has a discarded buses builder parameter (network.rs:90-91) behind a TODO. Narrower than 'constructed nowhere', because DisconnectedBus is constructed in the test_error_trait test at error.rs:196.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/error.rs::ValidationError`, `crates/cobre-core/src/error.rs:72`, `crates/cobre-core/src/topology/network.rs::build`
- **Evidence:** Outside its own declaration and Display arm, `DisconnectedBus` appears only in a TODO comment and `InvalidPenalty` appears nowhere at all — neither variant is ever constructed in production or test code.
- **Fix-shape:** Decide whether the disconnected-bus rule is wanted, then make the code say so. If it is wanted, implement it where the topology is already being built — that is the one place with every entity family in hand — and drop the parameter's discard;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-042 · Sev C · asymmetry · effort S · confidence high**
The uniform-stride flat-index arithmetic is repeated at 14 sites across hydro/line/pumping/contract with no shared helper and, unlike thermal, no stride debug_assert: a DRY/symmetry and guard-asymmetry gap only. It is not a correctness bug (all four share the bounds-checked n_stages stride) and the thermal_cell_index helper is justified by thermal's genuinely distinct padded stride.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/resolved/bounds.rs::thermal_cell_index`, `crates/cobre-core/src/model/resolved/bounds.rs::ResolvedBounds`
- **Evidence:** The flat cell-index invariant is owned once for the thermal table by `thermal_cell_index`, which also carries a debug assertion protecting its stride.
- **Fix-shape:** Give the four remaining families the same treatment the thermal table already has: one private index helper per family, or a single shared helper taking the stride, so the layout rule is stated once and each accessor reads a named call rather than repeating the multiply-add. This is a mechanical, behaviour-preserving change confined to one file, and it makes a future stride change for any family a one-line edit instead of a fourteen-site sweep.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-043 · Sev B · asymmetry · effort M · confidence high**
SystemBuilder::build re-sorts stages by id (builder.rs:364) but neither reassigns nor validates Stage.index (only cobre-io stages.rs:811-815 writes it; validate.rs has no check), so a non-cobre-io producer can silently supply an index disagreeing with the post-sort slot: a latent footgun, not a live wrong result on the cobre-io path (stable sort of already-indexed input).

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/temporal.rs::Stage`, `crates/cobre-core/src/system/builder.rs::build`, `crates/cobre-core/src/model/temporal/stage_key.rs::StudyPos`
- **Evidence:** `Stage.index` is documented as the position in the canonical-ordered stage vector.
- **Fix-shape:** Move the assignment to the sort. The builder already establishes the canonical stage order, so it should reassign each stage's position immediately after sorting, making the field a derived value that cannot disagree with its slot and letting the cobre-io parser drop its own loop.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-044 · Sev B · asymmetry · effort M · confidence high**
StageLagTransition (temporal.rs:175) has zero cobre-core consumers (all 4 occurrences are its own declaration/comments) and its field doc references the L3 pub(crate) symbol accumulate_and_shift_lag_state (noise.rs:232) that no cobre-core reader can resolve; the strictly-defensible residue is this zero-consumer plus broken-L0-doc-contract pair, leaving physical relocation to Epic-9 layering adjudication.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/temporal.rs::StageLagTransition`, `crates/cobre-core/src/model/temporal.rs:202`
- **Evidence:** All four cobre-core occurrences are the declaration itself and its own doc/section comment — the type has no cobre-core consumer.
- **Fix-shape:** Relocate the type to the crate that owns the uncertainty representation, cobre-stochastic, alongside the PAR machinery that is its only L1 consumer, and re-express its field docs against that crate's own kernel rather than against an engine-private function. Nothing in cobre-core reads it, so the move is a pure re-home plus an import change in the two consuming crates.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-2 (cross-reference; verdict travels to Epic 9).

**CD-045 · Sev B · asymmetry · effort M · confidence high**
The three *_models accessors (mod.rs:432/438/466) promise canonical order that no write path enforces (setters silent at builder.rs:229/237/244, build never sorts them) and with_scenario_models (mod.rs:571) replaces inflow_models post-construction with no sort/validation, while L1's PrecomputedPar::build depends on that order; the four 'raw' tables carry an explicit setter precondition and are honest delegation, so they fall outside the defect.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/system/builder.rs::build`, `crates/cobre-core/src/system/builder.rs::inflow_models`, `crates/cobre-core/src/system/mod.rs::inflow_models`, `crates/cobre-core/src/system/mod.rs::with_scenario_models`
- **Evidence:** `build` sorts nine collections (seven operational families via `sort_canonical`, plus stages and generic constraints).
- **Fix-shape:** Make the L0 owner own the invariant it advertises. Either the builder sorts these seven tables into their documented canonical key the way it already sorts the other nine, or it validates them as sorted and returns a validation error otherwise — the second is cheaper and preserves the current cobre-io behaviour byte-for-byte, since cobre-io already emits them sorted.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).

**CD-046 · Sev C · asymmetry · effort M · confidence high**
The wire-reproducibility rationale in the System serde(skip) comment (mod.rs:64) is enforced by three unrelated bespoke mechanisms yet six HashMap fields on HorizonGraph/CascadeTopology/NetworkTopology serialize unguarded as non-skipped SystemRepr fields (mod.rs:157/158/160), an inconsistency with no single owner and no guard test; explicitly NOT a live wrong result today (single-serialize-then-broadcast plus a value-equality round-trip guard).

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/system/mod.rs::System`, `crates/cobre-core/src/model/horizon.rs::HorizonGraph`, `crates/cobre-core/src/topology/cascade.rs::CascadeTopology`, `crates/cobre-core/src/topology/network.rs::NetworkTopology`
- **Evidence:** cobre-core states the same rule three times and enforces it three different ways: `serde(skip)` on the seven index maps with an explicit wire-reproducibility rationale, a hand-written `Serialize` that sorts its composite keys, and a `BTreeMap` chosen over `HashMap` for the correlation profiles.
- **Fix-shape:** Give the rule one owner instead of three restatements. State once, in the crate root or the system module, that anything reachable from the System payload must serialize in a content-determined order, and satisfy it uniformly — the cheapest route is switching these six fields to an ordered map, since all six are keyed by an already-`Ord` entity id or stage id and none is on a hot path where the lookup cost would matter.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-047 · Sev C · asymmetry · effort S · confidence high**
Six numeric extractors in extensions/{hydro_geometry,hydro_energy_productivity,tailrace_curves}.rs re-implement parquet_helpers' extract_required_int32/float64 (byte-identical to each other, three-line delta vs the shared exports), and the File::open->try_new->build->rows prologue recurs 28x across 19 files uncovered by parquet_helpers; the evaporation_models.rs:158 string extractor is NOT part of the defect (a genuine gap -- parquet_helpers offers no Utf8 extractor, documented at :157).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/extensions/hydro_geometry.rs::extract_int32_column`, `crates/cobre-io/src/extensions/hydro_energy_productivity.rs::extract_int32_column`, `crates/cobre-io/src/extensions/tailrace_curves.rs::extract_int32_column`, `crates/cobre-io/src/constraints/bounds.rs::parse_line_bounds`, `crates/cobre-io/src/scenarios/inflow_stats.rs::parse_inflow_seasonal_stats`, `crates/cobre-io/src/parquet_helpers.rs::extract_required_int32`
- **Evidence:** The md5 line is over the 23-line bodies of `extract_int32_column` sliced from the three extension files with sed;
- **Fix-shape:** Extend `parquet_helpers.rs` past column extraction to cover the reader itself: one helper that takes a path and returns the batch reader with the three error mappings applied, so each parser opens with a single call and keeps only its own column reads and row loop. Delete the six copied extractors in `extensions/` in favour of the shared pair, accepting the one-word change in the missing-column message or reconciling the two spellings first.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-048 · Sev B · asymmetry · effort M · confidence high**
bounds.rs:24 and penalties.rs:23 state `sorted by ID` for entity families that all carry operational_start_date, contradicting the enforced `(operational_start_date, id)` canonical order (builder.rs sort_canonical, pipeline.rs:289) on the publicly reachable resolve_bounds/resolve_penalties surface -- a doc-contract-vs-enforced-contract drift, not a live miscompute; generic_bounds.rs:15's `sorted by ID` is correct (no date axis) and the three `must be sorted` spellings (ncs_bounds/load_factors/ncs_factors) are underspecified rather than wrong, so `two wrong statements` is the exact residue.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/resolution/bounds.rs::BoundsEntitySlices`, `crates/cobre-io/src/resolution/penalties.rs::PenaltiesEntitySlices`, `crates/cobre-io/src/resolution/ncs_bounds.rs::resolve_ncs_bounds`, `crates/cobre-io/src/resolution/load_factors.rs::resolve_load_factors`, `crates/cobre-io/src/resolution/ncs_factors.rs::resolve_ncs_factors`, `crates/cobre-io/src/resolution/group_bounds.rs::resolve_hydro_unit_group_bounds`, `crates/cobre-io/src/lib.rs:130`
- **Evidence:** Seven resolvers share one precondition — slice position becomes the table's entity index — but state it four different ways.
- **Fix-shape:** Give the canonical key one owner and make every resolver doc point at it instead of restating it. cobre-core already owns the ordering in `sort_canonical`;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-049 · Sev C · asymmetry · effort S · confidence high**
The two resolvers share an identical algorithm skeleton and five parallel tests collapsible to one generic routine parameterised over the id accessor and destination table; residual differences are confined to the entity/entry/output-table types, local names, and one incidental `usize::try_from` spelling (load_factors.rs:59 vs ncs_factors.rs:61). `Same function twice` over-reaches (they are two monomorphizations over distinct output-table types, not literal copies), but the duplication is real and has two live consumers.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/resolution/load_factors.rs::resolve_load_factors`, `crates/cobre-io/src/resolution/ncs_factors.rs::resolve_ncs_factors`
- **Evidence:** A full `diff -u` of the two files shows no structural divergence anywhere.
- **Fix-shape:** Collapse the two into one resolution routine parameterised over the entity id accessor and the destination table, keeping `resolve_load_factors` and `resolve_ncs_factors` as thin named entry points so the call sites in the pipeline stay readable and the two output types remain distinct. Two present consumers make this a fold of existing duplication, not a speculative seam, so it does not trip the one-consumer-abstraction rule.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-050 · Sev B · asymmetry · effort M · confidence high**
Exactly two semantic branches -- block-duration>0 (semantic/stages.rs:88-103) and CVaR alpha/lambda range (:104-131), rules 4-5 in the mod.rs:83-84 table -- are unreachable for any pipeline-loaded deck because parse-layer validate_block_hours (stages.rs:623) and the CVaR-range half of validate_risk_measure (:661) reject such decks first and validate_schema bails before pipeline.rs:89; this is a duplicated-rule/doc-table drift against the retired-rule-42 convention, NOT a correctness gap (the rules ARE enforced at parse time), and the parser's unrecognized-risk-measure-string check (stages.rs:665) has no semantic counterpart and is excluded from the redundancy.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/stages.rs::validate_block_hours`, `crates/cobre-io/src/stages.rs::validate_risk_measure`, `crates/cobre-io/src/stages.rs::validate_raw_stages`, `crates/cobre-io/src/stages.rs::convert_stages`, `crates/cobre-io/src/validation/semantic/stages.rs::check_stage_structure`
- **Evidence:** The Layer-5b rule table in `validation/semantic/mod.rs:83-84` claims rules 4 and 5 — block duration positive, CVaR alpha in (0,1] and lambda in [0,1] — as the semantic layer's own, sourced from stages.json.
- **Fix-shape:** Decide which layer owns each of the two rules and delete the other copy, following the retired-rule-42 precedent already recorded in the semantic module doc. If the parse layer keeps them, strike rules 4 and 5 from the Layer-5b table with a note that stages.rs owns them, delete the two unreachable branches from `check_stage_structure`, and delete or relabel the two unit tests so they stop reading as coverage of a live rule.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-2 (cross-reference; verdict travels to Epic 9).

**CD-051 · Sev B · asymmetry · effort M · confidence high**
The scenario-source admission rules are enforced only lazily inside the accessor and cobre-io's own load pipeline (`run_pipeline`) never rejects an invalid config, the nine let-else/`.ok()` swallows resting on a false premise comment at scenarios.rs:459 that `validate_config` does not establish; but rejection is not lost end-to-end because each consumer (validate.rs:382, setup/mod.rs:385, cobre-python) re-propagates, so the defect is a mislocated/duplicated admission gate, not a production silent-accept.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/config/mod.rs::validate_scenario_source_cfg`, `crates/cobre-io/src/config/mod.rs::validate_openings_cfg`, `crates/cobre-io/src/config/mod.rs::validate_config`, `crates/cobre-io/src/validation/semantic/scenarios.rs::check_external_scheme_has_files`
- **Evidence:** Five config admission rules — historical-scheme restricted to the inflow class, seed required once any class leaves in-sample, `openings` only under `training`, historical year range ordered, `historical_years` only with a historical class — live in `validate_scenario_source_cfg` / `validate_openings_cfg`, whose only caller is `convert_scenario_source_config` at :227-228, itself reachable only from the two `Config::*_scenario_source` accessors.
- **Fix-shape:** Move the admission decision to the point where the config is admitted, not to whoever happens to read it. Resolve both scenario sources once inside the Layer-2 config gate, report their failures into the validation context alongside every other layer's findings, and hand the already-resolved values to the semantic rules so those rules take a resolved value rather than a fallible accessor.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-7 (cross-reference; verdict travels to Epic 9).

**CD-052 · Sev B · asymmetry · effort M · confidence high**
Confirmed narrowly: `LoadError::CrossReferenceError` is a dead variant (zero production producers, findings routed through `ConstraintError`) retaining two cobre-python consumer arms, and the overlapping line/hydro-filling predicate pairs can drift undetected because only cobre-io's copy fires in the production pipeline, not that cobre-core's builder validation is itself redundant (it legitimately guards cobre-core's public builder for direct/test constructors).

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/referential.rs::validate_referential_integrity`, `crates/cobre-io/src/validation/referential.rs::check_line_references`, `crates/cobre-io/src/validation/semantic/hydro.rs::check_filling_guards`, `crates/cobre-io/src/error.rs::LoadError`
- **Evidence:** Six entity cross-reference rules exist in matched pairs, one per crate, and the predicates coincide: cobre-core `validate_line_refs` checks `line.source_bus_id`/`target_bus_id` against the bus registry, cobre-io `check_line_references` checks the same two fields against the same set.
- **Fix-shape:** Name one owner per invariant class. The cross-reference and filling predicates are paradigm-neutral entity invariants, so either they live beside the entity in cobre-core and cobre-io's Layer 3 delegates to them, or cobre-io stays the sole owner and the cobre-core builder copies are deleted as unreachable — but not both, and the choice should be recorded once rather than settled per rule.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-053 · Sev B · asymmetry · effort M · confidence high**
Confirmed as-scoped to two concrete divergences: the block-id-range defect is `BusinessRuleViolation` in block_bounds.rs but `InvalidValue` in referential.rs, and the duplicate-row rule keys per-column in Layer 5a versus per-row in Layer 3 so a disjoint-column duplicate is legal for six families and rejected for generic constraint bounds; plus the NCS negative-value check sits in the reference-only Layer 3 module.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/referential.rs::check_generic_constraint_bounds_validity`, `crates/cobre-io/src/validation/semantic/block_bounds.rs::check_bound_block_id_range`, `crates/cobre-io/src/validation/semantic/block_bounds.rs::check_duplicate_bound_rows`, `crates/cobre-io/src/validation/referential.rs::check_ncs_bounds_and_factors`
- **Evidence:** Two rules of the bound-override family are implemented twice.
- **Fix-shape:** Give the bound-override family one home and one rule set. The natural landing zone is the Layer 5a module that already models the family generically: extend its per-family descriptor table to cover generic constraint bounds and NCS bounds, and delete the Layer 3 copies, leaving Layer 3 with the dangling-id checks that its own header claims.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-054 · Sev B · asymmetry · effort M · confidence high**
Confirmed narrowly: the uniform flat 'field is in id-set' reference blocks and the 20 copies of the `s.id >= 0` study-stage predicate are open-coded despite in-crate precedents (`FamilyMeta` table, `StageIdResolver`), and the header comment stands in for one message-template definition, conceding that Option-valued, nested, and per-plant-scoped references are not mechanically table-collapsible, so the residue is redundant idiom restatement rather than a single table replacing all 48 blocks.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/referential.rs::check_scenario_references`, `crates/cobre-io/src/validation/referential.rs::check_bounds_references`, `crates/cobre-io/src/validation/semantic/block_bounds.rs::FamilyMeta`
- **Evidence:** The check-and-emit shape for a dangling reference — index the rows, test membership in an id set, push an `InvalidReference` entry whose message reads "<RowType>[i] references non-existent <Entity> <id> via field '<field>'" — is written out 48 times in one file, spread over 13 functions;
- **Fix-shape:** Align the referential module to the pattern its sibling already proves rather than propagating the open-coded one. A descriptor carrying the row label, source file, target entity name and field name, plus one emit helper taking a descriptor and an id set, collapses the bulk of the 48 blocks and makes the message template a single definition instead of a header comment describing 48 copies.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-055 · Sev C · asymmetry · effort S · confidence high**
Confirmed narrowly on the concrete drift: registry row 26 asserts a live semantic rule for `simulation.sampling_scheme.type`, a field `deny_unknown_fields` now rejects at parse-time with a different ErrorKind, while curated retirements elsewhere prove the table is maintained, so the demonstrated defect is this one drifted row, the unbound prose registry being the mechanism rather than a second proven drift.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/mod.rs:105`, `crates/cobre-io/src/validation/semantic/mod.rs::validate_semantic_stages_penalties_scenarios`, `crates/cobre-io/src/config/simulation.rs::SimulationConfig`
- **Evidence:** The only index of the numbered semantic rules is the doc table in `semantic/mod.rs`, and nothing links a table row to the function that implements it: rules live in twelve sibling modules, are wired through two hand-maintained dispatch lists, and are described in a third place.
- **Fix-shape:** Bind rule identity to code so the registry cannot drift silently. The cheapest version keeps the table but attaches each rule number to its implementing function as a doc anchor, so a removed rule leaves a dangling reference the doc build notices.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-056 · Sev B · asymmetry · effort M · confidence high**
Confirmed narrowly: the stage-axis out-of-horizon validator is missing for the five block-eligible non-thermal bound families (hydro/line/pumping/contract/hydro_unit_group), whose out-of-horizon rows are silently dropped by resolve_bounds, whereas thermal (5a) and NCS (Layer 3, different ErrorKind) reject hard, so the fix is to add the five missing checks; thermal's guard is legitimately family-specific (padded resolution region), making this a coverage/consistency gap rather than a claim that the two existing rules are misplaced.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/thermal.rs::check_thermal_bounds_override_stage_range`, `crates/cobre-io/src/validation/semantic/block_bounds.rs:98`, `crates/cobre-io/src/validation/referential.rs::check_ncs_bounds_and_factors`
- **Evidence:** A stage-axis rule for bound-override rows exists for exactly two of the seven families: `thermal_bounds` (Layer 5a, `BusinessRuleViolation`) and `ncs_bounds` (Layer 3, `InvalidReference`).
- **Fix-shape:** Give the stage axis the same table-driven treatment the block axis already has, so one rule covers every bound family instead of two families having bespoke rules and five having none. It belongs beside the block-axis rule in the Layer 5a family module, keyed off the same per-family descriptor and the same study-stage set, and it should reuse the crate's stage resolver rather than a fresh id-non-negative scan.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-057 · Sev B (A-risk) · asymmetry · effort M · confidence high**
Confirmed narrowly: the positional `FILE_ENTRIES` to `manifest_fields_mut` zip is guarded only by an equal-length assertion that cannot detect a same-arity reordering, so swapping two entries silently misassigns presence flags; the `ParsedData`/schema.rs list is a third parallel restatement of the file set but keyed by name (a DRY/fan-out concern), not part of the positional join.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/structural.rs::manifest_fields_mut`, `crates/cobre-io/src/validation/structural.rs::FileManifest`, `crates/cobre-io/src/validation/structural.rs::validate_structure`, `crates/cobre-io/src/validation/schema.rs::ParsedData`
- **Evidence:** One input file is spelled in four places that must agree by hand: its path string in `FILE_ENTRIES` (43 entries, plus the one `struct FileEntry` declaration the count of 44 includes), its `bool` field on `FileManifest` (43 fields), its slot in the fixed-length 43-element mutable-reference array, and its path string again plus its field on `ParsedData` in `validation/schema.rs`.
- **Fix-shape:** Collapse the parallel lists to one keyed registry so a file is declared once and looked up by name rather than by ordinal. The manifest becomes a lookup keyed on the registry's own entry identity instead of a 43-field struct plus a 43-slot array, which removes the positional join and the hand-kept field order along with it.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Re-raise-of:** CD-031 (precedent citation only — this entry cites the boundary-context config-projection-sprawl calibration precedent, it does not re-raise CD-031)

**CD-058 · Sev B · asymmetry · effort M · confidence high**
The crash-safety hole is specifically the in-place (O_TRUNC) overwrites of manifest.bin at checkpoint.rs:244 and of entities.csv/variables.csv in dictionary.rs, which can leave a truncated file replacing the previous good one; the cuts/basis/states .bin payload writes are covered by the manifest-last commit-signal design and are not an independent crash-safety hole.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/atomic.rs::write_bytes_atomic`, `crates/cobre-io/src/output/policy/checkpoint.rs::write_policy_checkpoint`, `crates/cobre-io/src/output/policy/checkpoint.rs:242`, `crates/cobre-io/src/output/dictionary.rs::write_entities_csv`, `crates/cobre-io/src/output/dictionary.rs::write_variables_csv`
- **Evidence:** atomic.rs's module doc opens 'Single owner of the write-side crash-safety contract: write to {path}.tmp, flush explicitly (never via Drop), then rename', and twelve of the thirteen writer modules import it.
- **Fix-shape:** Route every remaining output write through output/atomic.rs. For the policy artifact, serialize each payload to bytes as it already does and hand the buffer to write_bytes_atomic instead of std::fs::write, keeping manifest.bin last so the commit-signal ordering is unchanged;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-059 · Sev B · asymmetry · effort M · confidence high**
The confirmed defect is narrowly the false module-doc contract at output/mod.rs:6-8 (write_results does not 'write all output artifacts' and does not mirror load_case) and the resulting undocumented CLI/Python hand-mirror; it does not establish that write_results must be expanded to orchestrate every artifact - that consolidation is the 0a design choice, not part of the present defect.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/mod.rs:6`, `crates/cobre-io/src/output/results_writer.rs::write_results`, `crates/cobre-io/src/output/results_writer.rs::write_training_results`
- **Evidence:** write_results at results_writer.rs:166 calls write_training_results and write_simulation_results only;
- **Fix-shape:** This is the cobre-io-side owner shape for CD-025 rather than a new duplication finding, and it should attach to that entry's fix shape. Two things belong in cobre-io: first, correct the contract statement now — either write_results genuinely orchestrates the full artifact set, or the mod.rs doc and the function name stop claiming it does, because a false ownership claim in the module doc is what lets the CLI/Python twin drift unnoticed.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-060 · Sev B · asymmetry · effort M · confidence high**
The load-bearing, concretely-defective residue is the over-broad self-description of the two enumerations versus their partial hand-maintained lists - chiefly the axis-spelling gate's 'a later file cannot reintroduce a variant without failing this one test' asserted over a 21-of-34 subset; the wider 'no single owner' framing is the shape of the fix, not itself the proven defect.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/schemas.rs::costs_schema`, `crates/cobre-io/src/output/dictionary.rs::variables_csv_schemas`, `crates/cobre-io/src/output/schemas.rs::one_spelling_per_axis_across_every_output_schema`, `crates/cobre-io/src/output/stochastic.rs::noise_openings_schema`, `crates/cobre-io/src/output/hydro_models.rs::fpha_hyperplanes_schema`, `crates/cobre-io/src/output/dictionary.rs::bounds_schema`
- **Evidence:** `python3 - <<'EOF'` — schemas.rs opens with '//! Arrow schema definitions for all Parquet output files' yet 9 of the 34 output schemas are declared privately in three sibling writers.
- **Fix-shape:** Give the output-schema family one owner and derive both consumers from it. Move the nine sibling-declared schemas into output/schemas.rs alongside the twenty-five already there, then replace the three hand-maintained lists with a single crate-internal table that pairs each output file's relative path with its schema function.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**CD-061 · Sev B · asymmetry · effort M · confidence high**
The confirmed defect is the register/oracle-coverage gap only: the Parquet convergence/timing/row_selection schemas and IterationRecord carry training-loop column names that Part-I I.3-6 does not name and the word-boundary genericity gate deliberately cannot see; it is expressly NOT an enforced-contract violation (the ratified sddp/SDDP/Benders/standalone-cut tokens are absent) and requires no rename now - only widening the I.3-6 disposition.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/schemas.rs::convergence_schema`, `crates/cobre-io/src/output/schemas.rs::iteration_timing_schema`, `crates/cobre-io/src/output/schemas.rs::row_selection_schema`, `crates/cobre-io/src/output/mod.rs::IterationRecord`, `scripts/ci/check-infra-genericity.sh:79`
- **Evidence:** 45 of the 237 output columns declared in schemas.rs are training-loop vocabulary: convergence_schema (:365) carries cuts_added / cuts_removed / cuts_active / time_forward_ms / time_backward_ms / forward_passes / lower_bound / upper_bound / gap_percent, iteration_timing_schema (:398) carries forward_wall_ms / backward_wall_ms / cut_selection_ms / cut_sync_ms / cut_batch_build_ms / bwd_* / fwd_* / lazy_scoring_ms, and row_selection_schema (:516) writes training/cut_selection/iterations.parquet with five cuts_* columns.
- **Fix-shape:** Record this as the second half of Part-I item 6 rather than fixing it now, and do not build a generic output-schema trait for one engine — that would trip the one-consumer-abstraction trigger. The roadmap-consistent shape under Milestone 0a is that cobre-io keeps the mechanics it is good at (atomic write, Parquet properties, Hive partitioning, the dictionary) while the row type and its column list are supplied by the engine that produces them, so a second engine brings its own convergence-equivalent table instead of reusing SDDP's column names.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-6 (cross-reference; verdict travels to Epic 9).

**CD-062 · Sev B · asymmetry · effort M · confidence high**
The confirmed defect is the unreconciled duplication plus the contradictory documented contract for the empty-family case (new() creates a directory for a system-declared-but-payload-empty family, which write_scenario's doc says cannot exist), with new()'s create_dir_all otherwise redundant against write_partition's own create_dir_all; the two predicate sets are each locally sensible and this is not a runtime data-corruption bug.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/simulation_writer.rs::new`, `crates/cobre-io/src/output/simulation_writer.rs::write_scenario`, `crates/cobre-io/src/output/simulation_writer.rs::write_partition`
- **Evidence:** `sed -n '556,566p;586,600p;686,700p' crates/cobre-io/src/output/simulation_writer.rs` — The entity-family list is written twice in one file with two different predicates.
- **Fix-shape:** Make one place own the entity-family table. Declare each family once as a row pairing its directory subpath, its system-side predicate, its schema and its batch builder, then have new() iterate that table for directory creation and write_scenario iterate the same table for the per-family write, so the two predicates become one and a fourteenth family is one row rather than three edits.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Re-raise-of:** byte-neutral-consolidations-already-executed, NOT a re-raise — the mirror records the create-dir/write/push write_partition helper as done; this entry addresses a different defect at the same call site (see the diagnosis above)

**CD-063 · Sev C · asymmetry · effort S · confidence high**
The concrete defensible defect is the cross-domain mis-homing of ensure_parent_dir inside stochastic.rs (imported by three unrelated writers) together with its two verbatim open-coded copies in scaling_report.rs:24 and provenance.rs:26; the 'ten identical prologues' is the broader repetition a fix would fold, not itself the load-bearing residue.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/parquet_helpers.rs::extract_required_date32`, `crates/cobre-io/src/output/stochastic.rs::ensure_parent_dir`, `crates/cobre-io/src/output/fixed_delivery.rs::write_fixed_delivery`, `crates/cobre-io/src/output/scaling_report.rs::write_scaling_report`, `crates/cobre-io/src/output/provenance.rs::write_provenance_report`
- **Evidence:** parquet_helpers.rs is the read side's answer to per-parser open-coding: six typed extract helpers, crate-internal, imported by every Parquet parser.
- **Fix-shape:** Add a write-side counterpart to parquet_helpers.rs beside output/atomic.rs, so the directory has one owner for the mechanics of getting a batch onto disk. Move ensure_parent_dir there out of the stochastic domain writer, fold the two open-coded copies in scaling_report.rs and provenance.rs onto it, and add one helper that takes a target path and a RecordBatch and performs the ensure-parent / default-config / atomic-write sequence, so the ten prologues collapse to one call each and the domain writers keep only their batch builders.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

### Performance findings

**PD-006 · Sev C · allocation · effort S · confidence high**
Confirmed as a loader/study-setup-path inefficiency only (not any hot path in training/forward, training/backward, or simulation/pipeline): canonical_calendar_days rebuilds a compile-time-constant 366-entry Vec on each call, and is_multi_resolution is re-swept once per raw stage over the immutable season_map. Narrowed: the per-stage sweep is O(1) for Monthly/Weekly maps (the cycle_type early-return at temporal.rs:487), so the recomputation cost is non-trivial only for Custom maps, where it is 366 * seasons per stage.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/temporal.rs::canonical_calendar_days`, `crates/cobre-core/src/model/temporal.rs::is_multi_resolution`, `crates/cobre-core/src/model/temporal.rs::span_days`, `crates/cobre-core/src/model/temporal.rs::resolution_level_of`
- **Evidence:** `canonical_calendar_days` builds a fresh 366-entry `Vec<(u32, u32)>` on the heap and is reached from two places: the `Custom` arm of `span_days` (:390, itself reached through `resolution_level_of` at :502) and `is_multi_resolution` (:489).
- **Fix-shape:** Two independent moves. First, stop rebuilding the constant: express the 366-day canonical calendar as a compile-time constant array or a process-lifetime lazily-initialised static, and have both readers borrow it instead of receiving an owned vector — the function's own doc already states the sequence is year-independent, so nothing observable changes.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-007 · Sev B · allocation · effort M · confidence high**
Confirmed narrowly: all five discarding call sites lie on one-time setup/validation paths (PAR lag-transition build, season-cast coverage, cobre-io travel-time semantic validation, SDDP bucket-topology setup), none on a declared hot path. The single site whose allocation cost is worse than O(1) per call is check_horizon_inertness (travel_time.rs:332-334), which makes O(N) predicate-only calls each allocating an O(remaining-stages) vector, i.e. O(N^2) allocation for what is only an emptiness test.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/temporal/overlap.rs::window_period_overlaps`, `crates/cobre-core/src/model/temporal/overlap.rs:41`, `crates/cobre-core/src/model/temporal/overlap.rs:55`
- **Evidence:** The only entry point returns an owned `Vec<f64>`, and its accumulator is built with `Vec::new()` (:41) rather than reserving against `stage_lengths_hours.len()`, which is known before the loop starts.
- **Fix-shape:** Keep the vector-returning function as the multi-period answer, and give the same module two narrower entry points beside it that the discarding callers can use: a scalar single-period overlap that returns the intersected hours for one period without touching the heap, and a reach predicate or depth count that answers how far a window extends by walking periods and returning a boolean or an index instead of materialising the per-period series. All three should share one internal walk so the overlap arithmetic stays single-owner and the existing bit-exactness tests keep covering it.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-008 · Sev B · allocation · effort M · confidence high**
Confirmed, narrowed to a one-time study-setup MPI broadcast payload (not any per-iteration hot-path cost) and with the title's count corrected: the skipped-and-rebuilt siblings are the seven entity index maps plus stage_index (eight, not 'three fields above'). The defensible residue is that cascade+network's five HashMaps are transmitted on the wire despite being pure, content-determined derivations of the seven entity slices already serialized ahead of them in the same struct, and thus locally reconstructible in rebuild_indices.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/system/mod.rs::System`, `crates/cobre-core/src/system/mod.rs::SystemRepr`, `crates/cobre-core/src/system/mod.rs::rebuild_indices`, `crates/cobre-core/src/topology/cascade.rs::CascadeTopology`, `crates/cobre-core/src/topology/network.rs::NetworkTopology`
- **Evidence:** The eight `HashMap` index fields carry `serde(skip)` and are rebuilt by `rebuild_indices` from `From<SystemRepr>` at :221;
- **Fix-shape:** Give `cascade` and `network` the same treatment their sibling index maps already have: mark both fields skipped on the wire, drop them from the deserialize shadow struct, and extend the existing rebuild step that `From<SystemRepr>` already calls so it reconstructs both topologies from the deserialized entity slices alongside the seven entity indices and the stage index. The rebuild is safe to make unconditional because both builders are content-determined: the cascade's topological order is drawn from a min-heap keyed on the raw entity id (so it does not inherit the surrounding map's iteration order), and every upstream and per-bus list is explicitly sorted by id before the builder returns.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-009 · Sev B · allocation · effort M · confidence high**
The projection join at estimation.rs:468-479 is O(occurrences x windows) per hydro — quadratic in unbounded historical record depth — plus a per-occurrence throwaway `Vec<RealizedWindow>` copy of what is a contiguous subslice of the already-sorted, provably-disjoint `windows`; the second anchor (check_std_ratio_divergence:842) is a bounded (hydros x seasons) scan that does not scale with record depth and is not part of the surviving defect.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/scenarios/estimation.rs::resolve_coverage_gated_observations`, `crates/cobre-io/src/scenarios/estimation.rs::check_std_ratio_divergence`
- **Evidence:** The inner loop walks every one of a hydro's history windows for every season occurrence of that hydro, and both counts grow linearly with the depth of the historical record, so the comparison count is quadratic in record depth per hydro.
- **Fix-shape:** Replace the nested filter with a single forward sweep that advances one cursor through the hydro's window slice as it advances through the occurrence list, since both are already ascending and the windows are already proven disjoint. Hand `cast` the resulting contiguous subslice of the existing window list by borrow instead of building a per-occurrence owned vector, which removes the allocation and the copy entirely.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-010 · Sev B · allocation · effort M · confidence high**
The nested O(hydros x [inflow_history + recent_observations] rows) filter in merged_windows_for_hydro (181/185) is paid once per hydro inside each of check_slot_coverage (244) and check_inprogress_partial_coverage (339); the ADDITIONAL cross-rule duplicate construction of the same per-hydro merged map only occurs when both rules' preconditions hold at once (inflow_ar_coefficients non-empty and l_state > 0), not on every deck.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/inflow_seeding.rs::merged_windows_for_hydro`, `crates/cobre-io/src/validation/semantic/inflow_seeding.rs::check_slot_coverage`, `crates/cobre-io/src/validation/semantic/inflow_seeding.rs::check_inprogress_partial_coverage`
- **Evidence:** The helper at 181 scans `data.inflow_history` in full and keeps the rows matching one hydro id (line 185), collects them into a fresh `Vec<RealizedWindow>`, does the same over `recent_observations`, and returns a third vector from `merge_layered_windows`.
- **Fix-shape:** Bucket the history rows and the recent observations by hydro id in one pass at the top of the inflow-seeding entry point, merge each bucket once, and pass the resulting per-hydro map into both the slot-coverage rule and the in-progress-coverage rule. That removes the nested scan and the duplicate construction together.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-011 · Sev B · allocation · effort M · confidence high**
check_prefix_coherence (846-857) re-walks stages 0..=sn per transition and re-does identical cell comparisons for any two transitions sharing (source column cn, target column cm, source stage depth sn); the redundancy is real specifically in the no-disagreement case (find_map+break bounds the walk once a stage disagrees), and de-duplicating to once-per-pair alters per-edge warning attribution unless the per-edge loop is retained with a decided-pair short-circuit.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/scenarios.rs::check_prefix_coherence`, `crates/cobre-io/src/validation/semantic/scenarios.rs:853`
- **Evidence:** `sed -n '846,857p' crates/cobre-io/src/validation/semantic/scenarios.rs` — The comparison depends only on the class, the two column indices `cn` and `cm`, and the source stage depth `sn`;
- **Fix-shape:** Compare each distinct column pair once rather than once per edge. Collect the distinct pairs with their maximum source stage depth and one representative edge before the comparison, then walk each pair's prefix a single time and attribute the first disagreement to its representative edge.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-012 · Sev C · allocation · effort S · confidence high**
On a chain-dialect deck (data.stages.policy_graph.nodes empty, guard at 620) extract_class's cells map retains f64 values no rule reads (its only value reader check_prefix_coherence at 855 is skipped) while the map keys are still used for in-build duplicate detection; the wider 'four full-table structures per class' is narrowed because inflow_sample_rows is inflow-class-only (693-694) and union_by_stage is bounded by distinct scenario_ids per stage rather than the full row count.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/scenarios.rs::extract_class`, `crates/cobre-io/src/validation/semantic/scenarios.rs::check_external_library_coherence`
- **Evidence:** Every external row drives four inserts inside the single row loop: a `cells` hash insert (line 665), a `union_by_stage` set insert (line 684), an `entity_scen` set insert (lines 685-688), and, for the inflow class, a full row copy pushed onto `inflow_sample_rows` (line 694).
- **Fix-shape:** Build only what a rule will read. On a deck with no declared node list the class extraction can carry a key-only set for duplicate detection and skip the values entirely, since the value-carrying map has no other reader.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-013 · Sev C · allocation · effort S · confidence high**
The identical stage_index build (season.rs:141, season.rs:264, scenarios.rs:1012) and the identical partition_point predecessor lookup over data.inflow_history are triplicated under the shared estimation-active predicate; the narrowed residue is that shared index+lookup only, NOT a fully-shared season resolution (site 1 adds an .or_else season_for_date fallback, site 3 resolves to stage-occurrence + cast() coverage gating rather than a season), and the three do not always co-fire (site 2 requires non-External inflow scheme, site 3's index needs season_map present).

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/season.rs::check_observation_season_alignment`, `crates/cobre-io/src/validation/semantic/season.rs::check_season_observation_coverage`, `crates/cobre-io/src/validation/semantic/scenarios.rs::check_estimation_prerequisites`
- **Evidence:** Three sites declare the identical index type, build it from the same stage filter, then run the identical predecessor lookup on the same table.
- **Fix-shape:** Resolve each history row's season and stage occurrence once. Lift the index build and the per-row predecessor lookup into a single helper that returns, per history row, its resolved season and stage position, evaluate the estimation-active predicate once alongside it, and let the three rules consume that shared result to fill their own counters.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-014 · Sev B · allocation · effort M · confidence high**
slot_occupying_classes (357) full-scans up to three external tables plus a fresh HashSet per call, and check_realization_rules invokes it once per node (308 inside for node in nodes) so nodes sharing a stage recompute an identical stage-keyed result — that per-node redundancy is the solid defect; call sites 667/797 recompute once per DISTINCT staged stage (staged is already deduped), not per node, and the sibling ClassExternal.raw_c (536/716) holds the same count in one pass but reusing it is a cross-module (scenarios.rs->stages.rs) share, not a free local one.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/stages.rs::slot_occupying_classes`, `crates/cobre-io/src/validation/semantic/stages.rs::check_realization_rules`, `crates/cobre-io/src/validation/semantic/stages.rs::check_num_openings_declaration`, `crates/cobre-io/src/validation/semantic/stages.rs::check_sampling_method_meaningfulness`, `crates/cobre-io/src/validation/semantic/scenarios.rs::ClassExternal`
- **Evidence:** The helper body at 365-397 filters `data.external_scenarios`, `data.external_load_scenarios` and `data.external_ncs_scenarios` in full for a single `stage_id`, then `distinct_count` (line 403) collects the surviving scenario ids into a fresh `HashSet<i32>`.
- **Fix-shape:** Compute the per-stage slot-occupancy vector once per validation run, before any node or stage loop, with a single pass over each external table that accumulates a distinct-scenario-id count per resolved stage index, and have all three call sites index that vector instead of recounting. The value already has an owner one module over: the external-library coherence check builds exactly this vector in one pass and keeps it on its per-class record.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-015 · Sev C · allocation · effort S · confidence high**
The internal checkpoint write path pays an avoidable full-buffer memcpy: each serializer's finished_data().to_vec() (codec.rs:284/333/365/455) is copied only for fs::write in checkpoint.rs:221/227/237/244 to borrow it as a slice and drop it, but the copy occurs at checkpoint cadence and the owned-Vec return stays justified for the pub API's doctest/external callers.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs::serialize_stage_cuts`, `crates/cobre-io/src/output/policy/checkpoint.rs::write_policy_checkpoint`
- **Evidence:** All four serializers end by copying the builder's finished bytes into a new owned vector.
- **Fix-shape:** Let the write path consume the builder's bytes without an intervening owned copy: either give the checkpoint writer serialize-and-write entry points that hand the builder's finished slice straight to the file write, or have the serializers surrender the builder's own buffer instead of copying out of it. Keep the current owning signatures available if external callers need a standalone buffer, so the copy is paid only by callers that genuinely want ownership rather than by the one production path that writes and drops.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-6 (cross-reference; verdict travels to Epic 9).

**PD-016 · Sev C · allocation · effort S · confidence high**
Each build_*_batch reconstructs its run-invariant Arrow Field list (with fresh column-name Strings) once per scenario (simulation_writer.rs:1076; schemas.rs:10-53) and write_parquet_atomic rebuilds WriterProperties per file (atomic.rs:109-114): a bounded, data-volume-independent per-scenario/per-file allocation, not a row-count-scaling cost.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/simulation_writer.rs::build_costs_batch`, `crates/cobre-io/src/output/schemas.rs::costs_schema`, `crates/cobre-io/src/output/atomic.rs::write_parquet_atomic`
- **Evidence:** Each of the fourteen `build_*_batch` functions in `simulation_writer.rs` opens with `let schema = Arc::new(<entity>_schema());`, and each `*_schema()` in `schemas.rs` constructs its `Field` list from scratch (`costs_schema` at crates/cobre-io/src/output/schemas.rs:22-53 builds twenty-nine of them).
- **Fix-shape:** Give each output schema a single lazily initialized shared instance that the batch builders clone the handle of rather than the contents, so the field list and its column-name strings are constructed once per process instead of once per scenario. For the writer properties, resolve them once where the `ParquetWriterConfig` is already stored on the writer and pass the resolved value into the atomic write helper, rather than rebuilding them from the same config inside every file write.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-017 · Sev B · allocation · effort M · confidence high**
The partitions_written inventory reaches no output file (SimulationMetadata at manifest.rs:434-462 has no field for it and write_simulation_results at results_writer.rs:133-155 never reads it), so its cross-run retention, merge clone-and-sort (mod.rs:431-435), and MPI allgatherv (simulation.rs:305-357) are unconsumed work; the per-partition format! allocation itself is negligible.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/simulation_writer.rs::write_partition`, `crates/cobre-io/src/output/mod.rs::SimulationOutput`, `crates/cobre-io/src/output/mod.rs::merge`
- **Evidence:** `write_partition` pushes a freshly formatted path string (`format!("simulation/{subpath}/{suffix}/data.parquet")` at crates/cobre-io/src/output/simulation_writer.rs:691) on every partition write, and `write_scenario` writes up to fourteen partitions per scenario.
- **Fix-shape:** Decide first whether any consumer still needs the partition inventory; the evidence says none does.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Re-raise-of:** byte-neutral-consolidations-already-executed, NOT a re-raise — the mirror records the create-dir/write/push write_partition helper as done; this entry addresses a different defect at the same call site (see the diagnosis above)

**PD-018 · Sev B · allocation · effort M · confidence high**
On the one-shot output-conversion path (cli outputs.rs / python run.rs, not the hot push loop) delta_to_stats_row allocates a String per log entry for phase (solver_stats.rs:293) though the value is a closed four-item vocabulary an enum/&'static str field would carry allocation-free; the owned-Vec<u64> histogram clone is conceded as a defensible cross-Python-boundary DTO choice.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/solver_stats_writer.rs::SolverStatsRow`, `crates/cobre-io/src/output/solver_stats_writer.rs::SolverStatsRow`, `crates/cobre-io/src/output/solver_stats_writer.rs::build_retry_histogram_batch`
- **Evidence:** The producer's log entry deliberately stores `phase` as `&'static str` and its own doc at crates/cobre-sddp/src/solver_stats.rs:231-232 states the reason is 'to avoid per-entry heap allocation on the hot push path'.
- **Fix-shape:** Change the row's phase field to a borrowed or enumerated phase so the closed vocabulary travels without an allocation, and let the row reference the producer's histogram rather than owning a copy of it, since the writer's only use of it is to fold it into the aggregation map. If a borrow is undesirable across the crate boundary, the alternative shape is to have the writer accept the producer's log slice directly and do the phase-name and histogram handling internally, which removes the intermediate `Vec<SolverStatsRow>` entirely.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**PD-019 · Sev C · allocation · effort S · confidence high**
build_iterations_columns (solver_stats_writer.rs:88-176) alone builds 18 scalar columns via <Array>::from(iter.collect::<Vec<..>>()), paying one redundant intermediate Vec allocation + copy per column that a pre-sized Builder::with_capacity+append (the idiom every sibling writer uses) would avoid, a one-shot iterations.parquet write cost, not a hot-path cost.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/solver_stats_writer.rs::build_iterations_columns`
- **Evidence:** Counting only production code (the scan stops at the inline test module), every other Parquet writer in the crate builds its columns exclusively through `Builder::with_capacity` and per-row `append_value`, with zero intermediate collects.
- **Fix-shape:** Rewrite the column construction to the idiom the other four writers already use: allocate one typed Arrow builder per column with the row count as its capacity, then walk the row slice once appending each column's value or null in the same loop. That removes the eighteen intermediate vectors and the duplicate copy, and it makes the writer read the same way as its siblings so a future column addition follows one pattern rather than two.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

### Over-engineering findings

**OD-010 · Sev C · speculative-generality · effort S · confidence high**
InvalidPenalty is constructed nowhere at all (not even a test) and carries no reserving TODO, and both it and DisconnectedBus bear a doc comment falsely attributing emission to cobre-io validation, which raises ErrorKind (validation/mod.rs:56) not ValidationError; DisconnectedBus is at least test-constructed (error.rs:196) and reserved by the network.rs:90 TODO, so for it only the false attribution -- not its existence -- is the defect.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/error.rs::ValidationError`, `crates/cobre-core/src/error.rs:65`, `crates/cobre-core/src/error.rs:72`
- **Evidence:** Across the whole workspace the only occurrences of the two variants are their declaration, their Display arm, one construction inside error.rs's own test module at :196, and a TODO comment.
- **Fix-shape:** Remove the two never-constructed variants together with their Display arms, the test that only exists to construct one of them, and the TODO that promises one of them; the enum then describes exactly the failures the builder can report.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-011 · Sev B · speculative-generality · effort M · confidence high**
The single missing wire is at columns.rs:1136: the NCS curtailment objective is sourced from the stage-invariant entity field ncs.curtailment_cost instead of ResolvedPenalties::ncs_penalties(ncs_idx, stage_idx), making the NCS resolved axis write-only in production while its three siblings are read at columns.rs:347/687/722; the resolution and write path (incl. the penalty_overrides_ncs override at resolution/penalties.rs:387) is fully functional -- only the LP read is absent.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/resolved/penalties.rs::ncs_penalties`, `crates/cobre-core/src/model/resolved/penalties.rs::ncs_penalties_mut`, `crates/cobre-core/src/model/resolved/penalties.rs::NcsStagePenalties`, `crates/cobre-core/src/model/resolved/penalties.rs::ResolvedPenalties`
- **Evidence:** Every remaining call of the read accessor `ncs_penalties` sits past the `#[cfg(test)]` marker of its file (cobre-core line 416 is past 328;
- **Fix-shape:** Decide the axis one way and make the code say so. Either wire it: have the NCS column builder take its objective coefficient from the resolved per-(ncs, stage) cell the way the hydro, line and bus column builders already take theirs, which makes the declared stage override effective and puts all four penalty families on one read path.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Reviewer rating:** A — downgraded here because defender narrowed the claim to a single missing LP read at `columns.rs:1136`; the resolve and write path (including the `penalty_overrides_ncs` override) is functional, so the blast radius is one call site, not a spreading structural gap.

**OD-012 · Sev C · speculative-generality · effort S · confidence high**
The genuinely unconsumed public surface is the population-statistics arm: population ci_95_half_width and the count accessor have only test callers, and the population variance/std_dev are public entry points whose sole non-test use is internal delegation within a population branch no production path reaches (the one consumer uses the sample arm exclusively); variance and sample_variance are conceded to be live internal delegates, not deletable outright.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/stats/welford.rs::WelfordAccumulator`, `crates/cobre-core/src/stats/welford.rs::variance`, `crates/cobre-core/src/stats/welford.rs::std_dev`, `crates/cobre-core/src/stats/welford.rs::sample_variance`, `crates/cobre-core/src/stats/welford.rs::ci_95_half_width`, `crates/cobre-core/src/stats/welford.rs::count`
- **Evidence:** The only production consumer of the accumulator is the forward-pass statistics aggregator, and it calls new, update, mean, sample_std_dev and sample_ci_95_half_width.
- **Fix-shape:** Keep the arm the single consumer uses and drop the mirrored one. Remove the population variance, population standard deviation, population confidence half-width and the sample variance accessor, keeping sample standard deviation and sample confidence half-width as the surface the aggregator reads, and fold their internal delegation into the two survivors.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-013 · Sev B · speculative-generality · effort M · confidence high**
The load-bearing residue is the build-time plus broadcast-wire cost of a reader-less structure: the System.network field lacks the serde(skip) its seven sibling index fields carry, so NetworkTopology is serialized into every postcard System payload (mirrored in SystemRepr) despite zero production reader -- the only callers are three sites in cobre-core/tests/integration.rs and the unit test at system/mod.rs:970.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/topology/network.rs::NetworkTopology`, `crates/cobre-core/src/topology/network.rs::build`, `crates/cobre-core/src/topology/network.rs:91`, `crates/cobre-core/src/system/mod.rs:87`, `crates/cobre-core/src/system/mod.rs::network`
- **Evidence:** The first count is zero: no file outside crates/cobre-core mentions `NetworkTopology`, `BusGenerators`, `BusLineConnection`, `BusLoads`, or calls `System::network()` anywhere in the workspace, cobre-python included.
- **Fix-shape:** Give it an owner and a consuming milestone or delete it, which is the register's own admission rule for an inert surface. The plausible owner is the reserved power-flow vertical, since bus-to-line, bus-to-generator and bus-to-load adjacency is exactly what a network formulation would pull;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).

**OD-014 · Sev C · speculative-generality · effort S · confidence high**
The four helper FUNCTIONS (not the Broadcast* mirror types, which setup.rs:288 genuinely consumes) are absent from the production MPI path; serialize_system/deserialize_system retain a single integration round-trip assertion (integration.rs:989-995) as their only non-file consumer and serialize_parameters/deserialize_parameters have none beyond their own doctests, so the defensible residue is that the four functions duplicate the encoding cli/broadcast.rs:419/:456 open-codes, not that the whole broadcast module is unused.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/broadcast.rs::serialize_system`, `crates/cobre-io/src/broadcast.rs::deserialize_system`, `crates/cobre-io/src/broadcast.rs::serialize_parameters`, `crates/cobre-io/src/broadcast.rs::deserialize_parameters`
- **Evidence:** All four are `pub` and crate-root re-exported.
- **Fix-shape:** Delete the four helper functions and their crate-root re-exports, keeping the `Broadcast*` mirror types and their `From` conversions, which the CLI genuinely consumes. Rewrite the module doc so the seam it describes is the one that exists - the mirror types plus the generic value broadcaster in the CLI - rather than a usage example built on the deleted helpers, and keep the integration round-trip assertion by expressing it as a direct postcard round-trip over `System`, so the guarantee that `System`'s `Deserialize` rebuilds its lookup indices stays pinned by a test.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-015 · Sev C · speculative-generality · effort S · confidence high**
Both are pub speculative surfaces with no non-test consumer (only crate-root re-exports plus their own #[cfg(test)] assertions and doctests) - the title zero-consumer claim holds strictly for non-test callers; load_scalar_parameters_json specifically is the OD-002 shape (schema.rs:544 open-codes the constraints/generic_parameters.json join it wraps), while build_season_stage_map is NOT a duplicate of the production season-map owner but a distinct raw stage_id->season_id builder that nothing calls (resolve_stage_seasons at residual_derivation.rs:219 produces dense ordinals instead).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/extensions/scalar_parameters.rs::load_scalar_parameters_json`, `crates/cobre-io/src/stages.rs::build_season_stage_map`
- **Evidence:** Every hit outside the two defining files is a `pub use` re-export;
- **Fix-shape:** Delete both functions together with their crate-root and module re-exports. If a case-relative scalar-parameter entry point is wanted, re-add it beside the first real caller and route the schema-validation site through it, so a single owner holds the `constraints/generic_parameters.json` path literal - the same collapse already executed for the boundary checkpoint path accessor.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-016 · Sev B · speculative-generality · effort M · confidence high**
The zero-non-test-consumer claim holds exactly (only lib.rs:134/137 re-exports; three in-file tests plus a no_run doctest exercise it); the load-bearing residue is the by-construction divergence at mod.rs:397-403 (residual derivation gated on manifest.stages_json) versus the unconditional pipeline.rs:224 call, so the second public carrier can leave residual_std_ratio = 1.0 unresolved - but the eleven per-file load_* helpers it wraps are NOT dead (schema.rs consumes them), so the confirmed defect is the aggregate orchestrator plus ScenarioData only.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/scenarios/mod.rs::load_scenarios`, `crates/cobre-io/src/scenarios/mod.rs::ScenarioData`, `crates/cobre-io/src/pipeline.rs:225`
- **Evidence:** Outside its own file the only two hits for either symbol are the crate-root `pub use` lines: no production caller, no integration-test caller and no Python-binding caller exists anywhere in the workspace.
- **Fix-shape:** Delete `load_scenarios` and `ScenarioData` together with their crate-root re-exports, and re-home the three inline tests that exercise them onto the assembly surface production actually uses. If an aggregate entry point is wanted as a supported library convenience rather than deleted, it must stop being a second assembly implementation: have it consume the same parsed artifacts the pipeline consumes so the residual-ratio derivation cannot be manifest-gated in one path and unconditional in the other, and drop the gate entirely.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).

**OD-017 · Sev C · speculative-generality · effort S · confidence high**
default_severity (validation/mod.rs:95) has no caller outside its own unit test AND its BusinessRuleViolation->Error classification contradicts the sole BusinessRuleViolation emission (add_warning at season.rs:175-177); the defensible residue is an uncalled, already-divergent parallel severity table, conceding the title's 'purports to describe the call sites' framing since the method's doc only claims a per-kind default, never a mirror of the emission sites.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/mod.rs::default_severity`, `crates/cobre-io/src/validation/semantic/season.rs:177`
- **Evidence:** The only references to `default_severity` anywhere in crates/ are its own declaration and the four assertions of its own unit test at validation/mod.rs:419-429;
- **Fix-shape:** Delete the method and its tautological unit test, leaving `add_error` / `add_warning` as the single owner of severity. If a per-kind default is actually wanted, invert the direction instead of deleting: make the table the one that decides, by routing every diagnostic through a single `add` entry point that consults the kind, and turn the `season.rs` warning into a deliberate documented override rather than a silent divergence.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-018 · Sev B · speculative-generality · effort M · confidence high**
ParsedData.penalties (schema.rs:75) has zero data.penalties reads workspace-wide and its rationale's saving is false (Layer-5 reads hydro.penalties at scenarios.rs:179, not the bundle); the scalar_parameters #[allow(dead_code)] (schema.rs:113) is merely REDUNDANT because the field is read at pipeline.rs:96 and moved at :244, so — narrowing the title — only its allow is stale while its rationale naming the resolution consumer is accurate.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/schema.rs::ParsedData`, `crates/cobre-io/src/validation/schema.rs::ParsedData`
- **Evidence:** `data.penalties` has zero references workspace-wide, so the field at schema.rs:75 is dead in fact;
- **Fix-shape:** Split the two by fact. For `penalties`, decide the field's fate rather than annotating it: either delete the field and its construction at schema.rs:669 (the sentinel already carries every value any parser needs), or, if the global defaults genuinely belong in the bundle for a future check, register it as a reserved seam in the mirror with an owner and a consuming milestone, and rewrite the comment to name the reader that will land instead of asserting one that already exists.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-019 · Sev B (A-risk) · speculative-generality · effort M · confidence high**
The defect is the ORDER-unguarded positional zip between FILE_ENTRIES and manifest_fields_mut (structural.rs:376): the sole guard asserts equal length only (structural.rs:612-623, comment: length not order), so swapping any two same-required-ness optional rows silently misassigns presence flags exactly as the helper's doc warns (structural.rs:396-397) while all tests stay green; conceding the title's 'three redundant lists' framing, the named-bool struct itself earns its place via the type-checked manifest.<field> reads in schema.rs, so the residue is the unguarded order coupling, not the existence of named fields.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/structural.rs::FileManifest`, `crates/cobre-io/src/validation/structural.rs::FILE_ENTRIES`, `crates/cobre-io/src/validation/structural.rs::manifest_fields_mut`
- **Evidence:** Three lists describe the same 43 input files and must stay in the same order: 43 `pub bool` fields on `FileManifest`, 43 `FileEntry` rows in `FILE_ENTRIES` (the 44th match is the struct declaration itself), and 43 `&mut m.<field>` entries returned by `manifest_fields_mut`.
- **Fix-shape:** Collapse the three lists to one by making the file table the single declaration and deriving both the storage and the accessors from it. The shape that keeps the named, type-checked reads the 34 call sites in validation/schema.rs depend on is a single ordered table of files paired with an enum key, with the manifest holding one flag array indexed by that key and named accessor methods generated alongside it, so adding a file is one edit and a mis-order is a compile error rather than a silent misassignment.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Re-raise-of:** CD-031 (precedent citation only — this entry cites the boundary-context config-projection-sprawl calibration precedent, it does not re-raise CD-031)

**OD-020 · Sev C · speculative-generality · effort S · confidence high**
The removable defect is precisely the unread third parameter `_config: &Config` on write_dictionaries (dictionary.rs:71) and the module's sole `use crate::Config` (dictionary.rs:16) that exists only to name it; the caller write_training_results still legitimately holds &Config for its own fields, so only the writer's parameter and import can go, not the caller's signature.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/dictionary.rs::write_dictionaries`, `crates/cobre-io/src/output/dictionary.rs:71`, `crates/cobre-io/src/output/dictionary.rs:16`, `crates/cobre-io/src/output/results_writer.rs::write_training_results`
- **Evidence:** The parameter is spelled `_config`, the compiler-sanctioned marker for declared-and-never-read, and the file's sole `use crate::Config` exists only to name that unread parameter's type.
- **Fix-shape:** Drop the third parameter from `write_dictionaries`, drop the now-unneeded `use crate::Config` from the dictionary module, and drop the forwarded argument at the single call site in the training results writer. Then check whether that caller still needs its own `&Config` for anything else before narrowing its signature too.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-7 (cross-reference; verdict travels to Epic 9).

**OD-021 · Sev C · speculative-generality · effort S · confidence high**
The narrower defect is needless pub visibility: both functions back only a same-module serde default-path attribute where a private fn suffices; and contra the title only default_bounds is actually crate-root public API (re-exported at output/mod.rs:47, zero external callers), while default_upper_bound_kind is pub but never re-exported, so it is merely pub inside a private module.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/manifest.rs::default_bounds`, `crates/cobre-io/src/output/manifest.rs::default_upper_bound_kind`, `crates/cobre-io/src/output/mod.rs:47`
- **Evidence:** `grep -rn '\bdefault_bounds\b\|\bdefault_upper_bound_kind\b' crates/ --include='*.rs'` — Both functions exist solely to satisfy a `#[serde(default = "...")]` attribute on a field in their own file: `default_upper_bound_kind` for `MetadataBounds.final_upper_bound_kind` at manifest.rs:214, `default_bounds` for `TrainingMetadata.bounds` at manifest.rs:412.
- **Fix-shape:** Make both functions private to the manifest module and remove `default_bounds` from the output module's re-export list and from the crate-root re-export in lib.rs. Verify the `schema` feature's export path does not name either function before narrowing, since a schemars-visible helper would change the committed schemas and CI diffs them.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-022 · Sev C · speculative-generality · effort S · confidence high**
The defect is exactly the two fields IterationRecord.time_bwd_setup_ms (mod.rs:141) and time_fwd_setup_ms (mod.rs:149): they are the only two time_* fields the convergence-path conversion loop skips (slots 8 and 11, training_output.rs:517-530), so they are populated at training_output.rs:234,237 and read nowhere; the columns they doc-arrow to are real but fed solely from the per-worker WorkerPhaseTimings path, not from IterationRecord.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/mod.rs::IterationRecord`, `crates/cobre-io/src/output/mod.rs::IterationRecord`, `crates/cobre-io/src/output/training_writer.rs::build_iteration_timing_batch`
- **Evidence:** A workspace-wide grep for a field read of either name returns nothing (exit 1).
- **Fix-shape:** Confirm with the training owner whether the two setup timings were meant to travel on the per-iteration record as well as the per-worker record. If not, delete both fields from `IterationRecord`, delete the two producer assignments in the cobre-sddp training-output builder, and delete the two zero-initialisers each in the cobre-io training writer, results writer, convergence reader and output-module tests plus the cobre-cli summary fixture;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-023 · Sev B · speculative-generality · effort M · confidence high**
Conceding the struct is not dead (its three fields are read by every parquet writer) and that pinning the encoding values is itself justified by the binary-formats spec (parquet_config.rs:5) and output-byte comparability, the narrower defect is the caller-varied-config machinery: no path constructs a non-default value, exports.compression is a rejected input, and the &ParquetWriterConfig parameter is threaded through only four functions while ~13 production writers rebuild the default locally, so the variability has no present consumer and cannot be honored uniformly.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/parquet_config.rs::ParquetWriterConfig`, `crates/cobre-io/src/output/parquet_config.rs::Default`, `crates/cobre-io/src/output/atomic.rs::write_parquet_atomic`, `crates/cobre-io/src/output/dictionary.rs:76`, `crates/cobre-io/src/output/stochastic.rs:132`, `crates/cobre-io/src/output/hydro_models.rs:87`, `crates/cobre-io/src/config/exports.rs::ExportsConfig`
- **Evidence:** The struct-literal grep returns only the declaration and the `Default` impl: nowhere in the workspace is a `ParquetWriterConfig` built with non-default fields, and the only field assignment anywhere is `cloned.row_group_size = 50_000` inside the type's own clone-independence test at parquet_config.rs:90.
- **Fix-shape:** Decide first whether Parquet compression, row-group size and dictionary encoding are a supported knob for third-party library consumers of cobre-io or a frozen internal constant set; the retired `exports.compression` input key argues for frozen.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**OD-024 · Sev C · speculative-generality · effort S · confidence high**
Conceding the byte-parsing body itself is correct and harmless (a faithful mirror of read_f64_vector), the narrower defect is purely its retention as dead code: at baseline the symbol resolves only at codec.rs:575, carries the scope's sole #[allow(dead_code)] (codec.rs:574), and its comment names neither owner nor landing reader, so it qualifies as neither sanctioned #[allow] census class.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs::read_f32_vector_as_f64`
- **Evidence:** The symbol resolves exactly once in the whole workspace, at its own declaration: no production caller, no test caller, no re-export.
- **Fix-shape:** Delete the function and its `#[allow(dead_code)]`. The byte-level reading pattern it claims to preserve is not at risk of being lost: `read_f64_vector`, `read_u32_vector` and `read_u8_vector` sit immediately beside it and demonstrate the identical bounds-checked shape, so a future f32 field costs one obvious copy of a neighbour rather than a rediscovery.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-6 (cross-reference; verdict travels to Epic 9).

**OD-025 · Sev C · speculative-generality · effort S · confidence high**
Conceding run_pipeline_with_artifacts is the legitimate working function and the four public lib.rs entry points each justify a distinct return shape, the narrower defect is the two pub(crate) intermediates run_pipeline (pipeline.rs:47) and run_pipeline_with_report (pipeline.rs:57): each is a one-line .map projection with exactly one caller that is itself a one-line lib.rs adapter (lib.rs:235 and lib.rs:266), so each re-derives a shape the public entry points already own and can be folded into its caller with no loss of any contract or caller.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/pipeline.rs::run_pipeline`, `crates/cobre-io/src/pipeline.rs::run_pipeline_with_report`, `crates/cobre-io/src/pipeline.rs::run_pipeline_with_artifacts`
- **Evidence:** `run_pipeline_with_artifacts` is the only function in the module that does work;
- **Fix-shape:** Collapse the module to its one working function and let the four public entry points in lib.rs do their own projection directly: the system-only loader maps away both the artifacts and the report, the artifacts loader maps away the report, the report loader maps the loaded case to its system, and the full entry point forwards unchanged. Keep the module doc's pointer about which public entry point returns warnings, restated against the public names rather than the private ones.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

### Test-bloat findings

**TD-001 · Sev B · duplication · effort M · confidence high**
Narrower than 'every test': the derived-Clone/Debug non-empty assertions (`all_variants_clone`, `all_variants_debug_non_empty`, `stopping_rule_result_debug_non_empty`, and the `format!("{:?}")` non-empty checks) plus the runtime value-echo inside the `*_fields_accessible` bodies are tautological -- discharged by codegen; but the fully-destructured `*_fields_accessible` tests (no `..`) still act as a weak compile-time tripwire on a field-set change, and `all_variants_construct` is a fixture-length pin rather than an echo/Debug test, so those are not pure echoes.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/constraints/training_event.rs::make_all_variants`, `crates/cobre-core/src/constraints/training_event.rs::all_variants_construct`, `crates/cobre-core/src/constraints/training_event.rs::all_variants_clone`, `crates/cobre-core/src/constraints/training_event.rs::all_variants_debug_non_empty`, `crates/cobre-core/src/constraints/training_event.rs::forward_pass_complete_fields_accessible`, `crates/cobre-core/src/constraints/training_event.rs::stage_row_selection_record_fields_accessible`
- **Evidence:** The file declares zero `impl` blocks, so `TrainingEvent` and its payload structs have no behaviour to test;
- **Fix-shape:** Delete the field-echo and derived-trait tests outright; they assert nothing a compile does not already guarantee.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-002 · Sev B · duplication · effort M · confidence high**
Narrower than 'all-same-value ... one per test module': HydroPenalties has no shared fixture or Default, so five zero-valued fixtures plus one uniform-`v` (`penalties_all`) each re-spell all 16 fields and a field addition breaks all six at once; but they are not identical constants -- five sites pin inflow_nonnegativity_cost = 1000.0 while system/builder.rs:511 pins it to 0.0 and penalties_all sets it to 1000.0 with the rest = v, so the residue is divergent near-uniform duplication (the copies disagree), not an all-same-value fixture.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/entities/hydro.rs::HydroPenalties`, `crates/cobre-core/src/entities/hydro.rs::penalties_all`, `crates/cobre-core/src/system/builder.rs::zero_penalties`, `crates/cobre-core/src/system/mod.rs:636`, `crates/cobre-core/src/topology/cascade.rs:145`, `crates/cobre-core/src/topology/network.rs:235`, `crates/cobre-core/tests/integration.rs::zero_hydro_penalties`
- **Evidence:** Six fixture sites in the sub-station build the same uniform-value `HydroPenalties`.
- **Fix-shape:** Give the type one shareable fixture where the type lives. Two composable moves: derive or hand-write `Default` for `HydroPenalties` so a fixture can write only the fields a test actually cares about and spread the rest, and expose a single uniform-value constructor from cobre-core behind the `test-support` feature the crate already declares (it currently gates exactly one item, `Hydro::declare_mirror_unit_group`), so cobre-io's and cobre-sddp's test modules can reach it as a dev-dependency feature instead of re-declaring it.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-003 · Sev C · duplication · effort S · confidence high**
Narrower than 'all eight exercise only derives': the pure clone/eq/hash tests (`test_equality`, `test_hash_consistency`, `test_bus_equality`, `test_contract_type_equality`, `annual_component_partial_eq_clone`, `test_hydro_storage_clone`) assert only derive- or std-library-guaranteed behavior and are tautological; but `test_copy` also acts as a compile-tripwire for EntityId: Copy (the `let b = a; ...

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/entity_id.rs::test_equality`, `crates/cobre-core/src/entity_id.rs::test_copy`, `crates/cobre-core/src/entity_id.rs::test_hash_consistency`, `crates/cobre-core/src/entities/bus.rs::test_bus_equality`, `crates/cobre-core/src/entities/energy_contract.rs::test_contract_type_equality`, `crates/cobre-core/src/entities/pumping_station.rs::test_pumping_station_construction`, `crates/cobre-core/src/model/scenario.rs::annual_component_partial_eq_clone`, `crates/cobre-core/src/constraints/initial_conditions.rs::test_hydro_storage_clone`
- **Evidence:** I read each of these eight bodies.
- **Fix-shape:** Delete the tests whose assertions are discharged by the derive. Where a test module would then be empty, as in `entities/pumping_station.rs`, delete the module rather than inventing a replacement;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-004 · Sev B · duplication · effort M · confidence high**
Narrower than 'ad-hoc comparators ... where the yardstick calls for one shared comparator': the defect is not that they are bloat (they back a real indexing/stride guard via make_distinct_bounds_table) nor that they collapse to one comparator (per-struct field enumeration is irreducible) -- it is only that they are non-exhaustive by construction, written as to_bits() chains instead of a `..`-free destructure, so a field added to any bounds struct compiles and is silently dropped from the bit-exactness assertion; only the scalar/Option<f64> primitive (`opt_f64_bits_eq`), re-rolled again in cobre-solver/tests/clp_determinism.rs, is genuinely shareable.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/resolved/bounds.rs::opt_f64_bits_eq`, `crates/cobre-core/src/model/resolved/bounds.rs::hydro_stage_bounds_bits_eq`, `crates/cobre-core/src/model/resolved/bounds.rs::hydro_block_bounds_bits_eq`, `crates/cobre-core/src/model/resolved/bounds.rs::thermal_block_bounds_bits_eq`, `crates/cobre-core/src/model/resolved/bounds.rs::line_bounds_bits_eq`, `crates/cobre-core/src/model/resolved/bounds.rs::pumping_bounds_bits_eq`, `crates/cobre-core/src/model/resolved/bounds.rs::contract_bounds_bits_eq`
- **Evidence:** Six struct-specific bit comparators plus one Option helper are hand-written inside one inline test module, each an explicit `a.field.to_bits() == b.field.to_bits()` chain per field.
- **Fix-shape:** Make the comparators exhaustive by construction rather than by review: destructure both sides with a full field pattern that has no `..` rest, so adding a field to a bounds struct fails to compile until the comparator names it. Then hoist the generic pieces, the scalar and `Option<f64>` bit comparison, into cobre-core's `test-support` surface as the shared exact-equality comparator the testing yardstick asks for, leaving only the per-struct field lists local.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-005 · Sev B · duplication · effort M · confidence high**
Narrower than 'copy-pasted across five test sites': the plain zero-varying builders (make_bus/make_line/make_thermal/make_ncs/make_group/make_hydro) are structurally duplicated across topology/network.rs, system/mod.rs and tests/integration.rs, differing only in trivial axes (name string, a 100->200 capacity), so a field add to Line/Bus/etc. is O(sites); but system/builder.rs's bus/line/hydro are a deliberately date+name-parameterized variant for canonical-order tests, not plain copies, and the full eight-name family is not present at every one of the five sites.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/topology/network.rs::make_line`, `crates/cobre-core/src/system/mod.rs::make_line`, `crates/cobre-core/tests/integration.rs::make_line`, `crates/cobre-core/src/system/builder.rs::line`, `crates/cobre-core/src/topology/cascade.rs::make_hydro`
- **Evidence:** The same builder family is declared in `topology/network.rs`, `system/mod.rs`, tests/integration.rs, `topology/cascade.rs` (make_hydro only) and twice inside `system/builder.rs` under the names `bus`/`line`/`hydro`/`thermal`/`ncs`/`contract`/`pumping`.
- **Fix-shape:** Hoist one entity-builder family into cobre-core behind the existing `test-support` feature, next to the entities it constructs, parameterised on the axes the current copies actually vary (id, bus id, operational date, name) and defaulting the rest, so a field addition is an O(1) edit rather than an O(sites) one. Have the five in-crate sites and tests/integration.rs call it, and drop the local copies.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-006 · Sev C · duplication · effort S · confidence high**
The < 1024 bound is an undocumented magic literal asserted only against a single-bus System, giving it too much headroom to catch an encoding regression on a realistic payload; it is at most a coarse compactness canary, narrower than the title's contract-free/tier-less framing.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/broadcast.rs::test_serialized_size_reasonable`
- **Evidence:** The literal 1024 occurs exactly once in the file, inside the assertion itself.
- **Fix-shape:** Delete the test. The postcard round-trip and rejection tests in the same module already pin every contract broadcast.rs owns, and a bare inequality against an unowned literal cannot fail for any reason a maintainer would want to hear about.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-007 · Sev B · duplication · effort M · confidence high**
The input path has no shared test-support module while the validation path does, and the fully-verbatim, fully-verified duplication is the 16 write_parquet plus 5 make_global bodies; write_json is 17 copies (not 18) with 2 non-verbatim variants, so the '39 across 20 files' total is 38 with fewer than 18 verbatim write_json.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/constraints/bounds.rs::write_parquet`, `crates/cobre-io/src/system/buses.rs::make_global`, `crates/cobre-io/src/system/buses.rs::write_json`, `crates/cobre-io/src/validation/semantic/mod.rs::test_support`, `crates/cobre-io/tests/helpers/mod.rs::write_file`
- **Evidence:** Three fixture helpers are copied verbatim across the whole sub-station B surface with zero variation: every one of the 16 write_parquet bodies hashes to the same md5, and every one of the 5 make_global bodies hashes to the same md5 over a 36-line window.
- **Fix-shape:** Give the cobre-io input path the same shared test-support home the validation path already has: one #[cfg(test)] sibling module (the natural spelling is a crate-level src/test_support.rs declared once in lib.rs behind #[cfg(test)], pub(crate) so system/, scenarios/, extensions/, constraints/ and resolution/ can all reach it) owning write_json, write_parquet and a GlobalPenaltyDefaults builder. Build the penalty fixture the way cobre-sddp's tests/common/builders.rs does rather than as a full struct literal, so that a new penalty field costs one edit instead of five: give GlobalPenaltyDefaults and HydroPenalties a test-only Default and have the fixture spread `..Default::default()`, overriding only the fields a given test actually reads.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-008 · Sev C · duplication · effort S · confidence high**
The durable, threshold-independent residue is the intra-directory homing inconsistency -- scenarios/estimation.rs extracted to a sibling while scenarios/correlation.rs stays inline in the same directory, alongside seven multi-thousand-line inline modules -- not the violation of the unratified ~500-LOC number (and the sibling has 36, not 65, test fns).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/constraints/generic.rs::tests`, `crates/cobre-io/src/scenarios/estimation.rs::tests`, `crates/cobre-io/src/scenarios/correlation.rs::tests`, `crates/cobre-io/src/resolution/bounds.rs::tests`, `crates/cobre-io/src/system/hydros.rs::tests`, `crates/cobre-io/src/stages.rs::tests`
- **Evidence:** docs/design/testing-architecture.md section 5.1 asks for one deterministic homing rule -- inline below roughly 500 test-LOC or 40 test fns, extracted to a sibling tests.rs above it -- and section 3.2 item 4 records the inline-giant-versus-extracted-sibling asymmetry as a ranked sustainability problem, naming cobre-sddp anchors.
- **Fix-shape:** Pick the threshold once and apply it mechanically across the input path rather than per author. Adopt the section 5.1 numbers as written (roughly 500 test-LOC or 40 test fns), extract the seven over-threshold modules to sibling tests.rs files following the shape scenarios/estimation.rs already uses -- `#[cfg(test)] mod tests;` in the parent, the module body moved verbatim into <module>/tests.rs with the crate-inner allow attributes carried along as module-inner attributes -- and leave everything under the threshold inline.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-009 · Sev C · duplication · effort S · confidence high**
The genuinely redundant triplicated surface is make_batch plus the five non-determinism common cases (valid-sorted, negative-std, nan-mean, missing-column, empty); the three per-parser declaration_order_invariance tests are load-bearing determinism-hard-rule pins and are NOT bloat, and each parser's unit-specific error-message assertions must survive any consolidation.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/scenarios/inflow_stats.rs::tests`, `crates/cobre-io/src/scenarios/load_stats.rs::tests`, `crates/cobre-io/src/scenarios/non_controllable_stats.rs::tests`, `crates/cobre-io/src/scenarios/non_controllable_stats.rs::make_batch`, `crates/cobre-io/src/scenarios/load_stats.rs::make_batch`
- **Evidence:** Three parsers over the identical four-column (entity_id, stage_id, mean, std) Parquet shape carry the same test template: valid-4-rows-sorted, negative-std reject, NaN-mean reject, missing-mean-column reject, empty-Parquet-returns-empty, declaration-order invariance.
- **Fix-shape:** Lift the shared four-column stats fixture into the shared test-support module from the first candidate -- one make_stats_batch(id_column_name, ids, stage_ids, means, stds) that takes the id column's name as an argument -- and drive the six common cases from a single table-driven or macro-generated block parameterized by (parse fn, id column name, unit suffix), so a new stats parser inherits the whole template instead of copying it. Reconcile the drift while doing it: decide whether mean-out-of-range and zero-std-accepted belong to all three parsers or only to the ones whose units make them meaningful, and give the missing-column test one spelling.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-010 · Sev C · duplication · effort S · confidence high**
The six binaries do verbatim re-declare 'mod helpers' and re-link arrow/parquet + the rlib six times, but per section 5.1 this is an opportunistic-only, off-critical-path cleanup for a non-solver crate (no solver-link amplifier); the only clearly self-justifying merges are the two sub-200-LOC binaries (scalar_parameters 136, productivity_resolution 199).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/tests/integration.rs::helpers`, `crates/cobre-io/tests/invariance.rs::helpers`, `crates/cobre-io/tests/resolver_builder_index_alignment.rs::helpers`, `crates/cobre-io/tests/post_study_stages.rs::helpers`, `crates/cobre-io/tests/load_case_productivity_resolution.rs::helpers`, `crates/cobre-io/tests/load_case_scalar_parameters.rs::helpers`
- **Evidence:** Cargo compiles one executable per tests/*.rs file, so each of the six input-path binaries is its own crate: helpers/mod.rs is compiled six times and the cobre-io rlib plus arrow and parquet are linked six times, and Cargo runs the resulting integration binaries sequentially.
- **Fix-shape:** Consolidate the six input-path binaries into one domain binary using the #[path]-submodule mechanism section 5.1 specifies: a new tests/load_case.rs root that declares mod helpers once and then includes each current file as a #[path] submodule, with each file's own `mod helpers;` line removed and its bare `helpers::` references rewritten to `crate::helpers::`. Per section 5.1 the per-file inner allow attributes ride along unchanged inside the module body and free items with colliding leaf names stay namespaced under their submodule, so the edit is mechanical and does not touch a single test body.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-011 · Sev C · duplication · effort S · confidence high**
test_bus_ordering_invariance is redundant because its bus2.name assertion -- the one surface not textually present in full_case -- is still logically implied by the whole-System assert_eq!; removal is safe only if that named bus handle is folded into full_case or its loss accepted, and the finding does NOT extend to test_stage_ordering_invariance.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/tests/invariance.rs::test_bus_ordering_invariance`, `crates/cobre-io/tests/invariance.rs::test_full_case_ordering_invariance`, `crates/cobre-io/tests/invariance.rs::make_shuffled_multi_entity_case`
- **Evidence:** `sed -n '173,232p;226,313p' crates/cobre-io/tests/invariance.rs` — The two tests are built from the identical pair of fixture builders (helpers::make_multi_entity_case and the file-local make_shuffled_multi_entity_case at line 25) and each loads both directories, so their inputs are the same values.
- **Fix-shape:** Delete test_bus_ordering_invariance and keep test_full_case_ordering_invariance, which already makes the same System-equality assertion over the same inputs and additionally checks n_hydros and n_stages. Keep test_stage_ordering_invariance unchanged: its stages[0].id == 0 assertions pin canonical order absolutely, which the equality assertion does not.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-012 · Sev C · duplication · effort S · confidence high**
The two IoError tests (test_load_error_io_display @152 and test_load_error_io_helper @240) are near-duplicates -- both build via LoadError::io and assert the same path+source contains facts, differing only in literals plus one matches! -- so one is redundant; and test_load_error_schema_display @181 re-covers the contains("bus_id") fact the module doctest at error.rs:27 already asserts.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/error.rs::LoadError`, `crates/cobre-io/src/error.rs::test_load_error_io_display`, `crates/cobre-io/src/error.rs::test_load_error_io_helper`, `crates/cobre-io/src/error.rs::test_load_error_schema_display`, `crates/cobre-io/src/error.rs::test_load_error_is_std_error`
- **Evidence:** Every assertion in the module is of the form `display.contains(<a value the test just put into the variant>)`, which the `#[error("...
- **Fix-shape:** Decide what the module is actually contracting for and test that, rather than the derive's own substitution. If the diagnostic wording is a user-facing contract, pin the full rendered message for one or two representative variants so a reorder or rewrite is caught, and drop the per-variant contains walks.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-013 · Sev C · duplication · effort S · confidence high**
The >= 17 floor in both the doctest (schema.rs:84) and test_generate_schemas_returns_expected_count (166) is looser than the 18 schemas produced, so deleting one un-name-pinned export passes a count-named test; and the three structural walks (172/182/193) collapse to one, since is_object implies !is_null and any schema with properties satisfies the structural-key check.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/schema.rs::test_generate_schemas_returns_expected_count`, `crates/cobre-io/src/schema.rs::test_all_schema_filenames_and_values_non_empty`, `crates/cobre-io/src/schema.rs::test_all_schemas_are_objects`, `crates/cobre-io/src/schema.rs::test_all_schemas_have_structure_keys`, `crates/cobre-io/src/schema.rs::generate_schemas`
- **Evidence:** `generate_schemas` builds 18 entries and `schemas/` holds 18 committed files, yet both the rustdoc doctest at line 84 and the unit test at line 166 assert only `>= 17`, so deleting one export passes the assertion whose name promises an expected count.
- **Fix-shape:** Fold the four structural walks into one test that asserts the strongest of the three properties, since the weaker two are implied, and stop re-generating the whole schema set once per assertion. Replace the floor of 17 in both the doctest and the unit test with an exact expected count, or better, with an assertion that the produced name set equals the full expected name set rather than the current thirteen-of-eighteen subset in `test_all_expected_schema_filenames_present`.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-014 · Sev B · duplication · effort M · confidence high**
The minimal-case corpus + write_file are duplicated across two same-crate inline modules (validation/schema.rs and validation/referential.rs) that could already share one #[cfg(test)] fixture, and make_minimal_case is restated in both referential.rs and tests/helpers/mod.rs; the tests/helpers copy alone is barrier-forced and only removable under section-5.2's unimplemented test-support convention.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/schema.rs::VALID_CONFIG_JSON`, `crates/cobre-io/src/validation/schema.rs::write_file`, `crates/cobre-io/src/validation/referential.rs::make_minimal_case`, `crates/cobre-io/src/validation/referential.rs::VALID_CONFIG_JSON`, `crates/cobre-io/tests/helpers/mod.rs::make_minimal_case`, `crates/cobre-io/tests/helpers/mod.rs::VALID_CONFIG_JSON`
- **Evidence:** After stripping leading/trailing whitespace, comments and the `pub` qualifier, the eight `VALID_*_JSON` constants in the `validation/schema.rs` inline test module are byte-identical (57 lines) to the eight in tests/helpers/mod.rs, and the four in the `validation/referential.rs` inline test module are byte-identical to the first four of that same corpus (52 lines).
- **Fix-shape:** Make the minimal-case corpus have exactly one owner. Hoist the eight JSON constants, `write_file` and `make_minimal_case` into a single crate-internal fixture module gated by a test-support cfg (the convention docs/design/testing-architecture.md section 5.2 prescribes: helpers live with the type they build, exposed through a `test-support` feature rather than a dedicated crate), then have both inline test modules and tests/helpers/mod.rs re-export from it instead of restating it.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-015 · Sev C · duplication · effort S · confidence high**
test_filling_guard_no_exit_no_error (hydro.rs:1795) is identical to test_filling_guard_entry_below_horizon_no_error (1692) except its assertion message, and because make_filling_hydro never sets exit_stage_id it cannot exercise the no-exit condition it is named for -- making it a coverage-free duplicate (the guard's rejection path is covered separately at 1771).

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/hydro.rs::test_filling_guard_no_exit_no_error`, `crates/cobre-io/src/validation/semantic/hydro.rs::test_filling_guard_entry_below_horizon_no_error`, `crates/cobre-io/src/validation/semantic/hydro.rs::make_filling_hydro`, `crates/cobre-io/src/validation/semantic/hydro.rs::test_filling_guard_exit_on_filling_errors`
- **Evidence:** The two seventeen-line test bodies differ on exactly one line, the assertion's failure message;
- **Fix-shape:** Delete the clone at 1795 and let the existing test at 1692 stand for the accepted case, since the two assert the identical fact about the identical fixture. If a well-formed counterpart to guard 5 is wanted for symmetry with the other guards, it has to actually vary the guard's input, which means building the hydro with `exit_stage_id` explicitly set to none at the call site rather than relying on a builder that cannot set it, so a future change to `make_filling_hydro`'s defaults cannot silently turn the test into a tautology again.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-016 · Sev B · duplication · effort M · confidence high**
Because validation/semantic/test_support is pub(super)-scoped, four validation/-level phase modules hand-roll the five proven-identical families -- the 1.0 penalty builder (dimensional/productivity_resolution/referential), zero_hydro_penalties = penalties_all(0.0), make_unit_group/make_pumping, and the full ParsedData skeleton restated in dimensional.rs and productivity_resolution.rs -- all removable by homing the fixture one level up.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/mod.rs::test_support`, `crates/cobre-io/src/validation/semantic/test_support.rs::penalties_all`, `crates/cobre-io/src/validation/semantic/test_support.rs::base_parsed_data`, `crates/cobre-io/src/validation/dimensional.rs::penalties_default`, `crates/cobre-io/src/validation/dimensional.rs::base_parsed_data`, `crates/cobre-io/src/validation/productivity_resolution.rs::penalties_default`, `crates/cobre-io/src/validation/productivity_resolution.rs::base_parsed_data`, `crates/cobre-io/src/validation/referential.rs::hydro_penalties`, `crates/cobre-io/src/validation/referential.rs::make_unit_group`, `crates/cobre-io/src/validation/scalar_parameters.rs::zero_hydro_penalties`
- **Evidence:** `mod test_support;` is private inside `validation/semantic/mod.rs` and all 31 of its helpers are `pub(super)`, so the module is reachable only from `validation::semantic`.
- **Fix-shape:** Move the fixture module up one level so its scope matches its audience: home it at `validation/` rather than `validation/semantic/`, widen the helper visibility from `pub(super)` to the crate-internal test surface, and delete the hand-rolled clones in the four phase modules in favour of it. Parameterize where the copies legitimately differ rather than forking: the penalty builders differ only in one scalar, which the existing `penalties_all(v)` signature already takes, and the stage builders differ only in block count and branching factor, which the existing `make_stage_with_blocks` shape already covers.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-017 · Sev C · duplication · effort S · confidence high**
Three boundary_tests mirror earlier tests with identical fixture arguments, and the stage_id=5 mirror (thermal.rs:3742) is strictly weaker than its original (3598) which also pins the diagnostic text; the in-file "do not delete" comment (3710-3714) has no register/design-doc backing (zero markdown references), so the duplication is not a sanctioned seam.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/thermal.rs::boundary_tests`, `crates/cobre-io/src/validation/semantic/thermal.rs::override_at_t_minus_1_acceptance_boundary`, `crates/cobre-io/src/validation/semantic/thermal.rs::test_thermal_bounds_override_stage_within_horizon_accepted`, `crates/cobre-io/src/validation/semantic/thermal.rs::test_thermal_bounds_override_stage_equals_n_rejected`, `crates/cobre-io/src/validation/semantic/thermal.rs::test_thermal_bounds_override_multiple_offending_rows`
- **Evidence:** Three of the four tests in `boundary_tests` build their fixture with arguments identical to a test earlier in the same file: (5, row(1,4)) at 3580 and 3721, (5, row(1,5)) at 3601 and 3742, (5, row(1,-1)) at 3637 and 3790.
- **Fix-shape:** Pick one home for the half-open-interval guard's boundary coverage and keep only that. The natural survivor is the earlier family, because it already asserts the diagnostic text and not just the violation count, and because `test_thermal_bounds_override_multiple_offending_rows` already carries the past-the-boundary case.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-018 · Sev B · duplication · effort M · confidence high**
make_config (43 lines) and make_system (6 lines) are byte-identical between output/convergence_reader.rs and output/results_writer.rs; make_output_context is identical except one DistributionInfo import line, so 49 lines are byte-identical, not the full 73 the title claims.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/convergence_reader.rs::make_config`, `crates/cobre-io/src/output/results_writer.rs::make_config`, `crates/cobre-io/src/output/convergence_reader.rs::make_output_context`, `crates/cobre-io/src/output/results_writer.rs::make_output_context`, `crates/cobre-io/src/output/convergence_reader.rs::make_system`, `crates/cobre-io/src/output/results_writer.rs::make_system`
- **Evidence:** The two `make_config()` bodies (43 lines each) hash identically at the pinned baseline, so they are byte-for-byte the same fixture.
- **Fix-shape:** Hoist the three fixtures to a single owner shared by the `output/` inline test modules and have both files call it. The natural home is a `#[cfg(test)]` fixture module under `output/` (a sibling `output/test_fixtures.rs` declared once from `output/mod.rs`, following the `validation/semantic/test_support.rs` precedent already established in this crate), or, if the fixtures are wanted by the crate's integration binaries too, a `test-support`-gated surface on cobre-io mirroring the `test-support` feature cobre-core already exposes.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-019 · Sev C · duplication · effort S · confidence high**
every_hydros_schema_column_has_description (3226) and every_hydro_bus_generation_schema_column_has_description (3239) are pure strict subsets of the exhaustive sweep with zero residual; new_energy_columns_have_descriptions (3193) is redundant only because the retained new_energy_columns_have_units already pins those same five column names.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/dictionary.rs::every_listed_schema_column_has_a_nonempty_description`, `crates/cobre-io/src/output/dictionary.rs::new_energy_columns_have_descriptions`, `crates/cobre-io/src/output/dictionary.rs::every_hydros_schema_column_has_description`, `crates/cobre-io/src/output/dictionary.rs::every_hydro_bus_generation_schema_column_has_description`, `crates/cobre-io/src/output/dictionary.rs::variables_csv_schemas`
- **Evidence:** `every_listed_schema_column_has_a_nonempty_description` iterates `variables_csv_schemas()` and asserts `!description_for(file, field).is_empty()` for every column of every listed schema.
- **Fix-shape:** Delete the three subsumed tests and rely on the exhaustive sweep, which is the stronger gate because a new column added to any listed schema fails it automatically while a per-schema test has to be remembered and written. Keep `bounds_hydro_id_column_has_a_description` and `new_energy_columns_have_units`: the first covers a table the sweep does not list, and the second asserts specific unit strings rather than mere non-emptiness, which is a different property.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-020 · Sev C · duplication · effort S · confidence high**
The defensible core is the three named helpers — fixed_delivery::read_batch and generic_constraints_echo::read_batch are byte-identical, solver_stats_writer::read_parquet is the same body under a different name — plus the ~34-38 per-function ParquetRecordBatchReaderBuilder re-imports; collapsing every one of the ~44 remaining four-line inline read-back sites into one helper is optional homing cleanup, not all defect.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/fixed_delivery.rs::read_batch`, `crates/cobre-io/src/output/generic_constraints_echo.rs::read_batch`, `crates/cobre-io/src/output/solver_stats_writer.rs::read_parquet`, `crates/cobre-io/src/output/simulation_writer.rs:2653`, `crates/cobre-io/src/output/stochastic.rs:851`
- **Evidence:** 49 total sites;
- **Fix-shape:** Give the `output/` test surface one read-back helper with a single agreed name, alongside the fixture module the first candidate calls for, and have the nine modules import it instead of re-deriving open-file / build-reader / take-first-batch per test. Fold the 34 in-function imports up to the test module's own import block while doing so.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-021 · Sev B · duplication · effort M · confidence high**
The 21 codec-only and 15 checkpoint-only tests are homed in policy/mod.rs (production ends line 28) while codec.rs carries 1 inline test and checkpoint.rs carries 0; the defect narrows to those two submodules, since records.rs already owns its 3 tests in-place, not the 'three submodules' the title states.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/policy/mod.rs::tests`, `crates/cobre-io/src/output/policy/codec.rs::tests`, `crates/cobre-io/src/output/policy/checkpoint.rs::read_policy_checkpoint`, `crates/cobre-io/src/output/policy/records.rs::tests`
- **Evidence:** The production body of `policy/mod.rs` ends at line 28 (module doc, three `pub mod` lines, four `pub use` blocks);
- **Fix-shape:** Move each test to the submodule it exercises: the 21 serialize/deserialize tests become an extracted sibling `codec` test module, the 15 write/read-checkpoint tests become a `checkpoint` test module, and the shared record fixtures (`make_cut_record`, `chain_manifest`, `make_metadata`, `make_stage_cuts_payload`, `make_basis_record`, `sample_manifest`) go to one fixture module both import. Leave in `policy/mod.rs` only tests that genuinely assert the composed round trip across all three submodules, if any survive that classification.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-6 (cross-reference; verdict travels to Epic 9).

**TD-022 · Sev C · duplication · effort S · confidence high**
The redundant surface is the count-only per-schema tests (thermals_schema_field_count and its count-only siblings, whose whole body is a length assert already in the umbrella's expected table) plus the bare length line inside the name-vector tests; the name-vector assertions themselves are NOT redundant (they pin the wire contract against renames) and must survive.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/schemas.rs::all_schema_functions_return_valid_schemas`, `crates/cobre-io/src/output/schemas.rs::thermals_schema_field_count`, `crates/cobre-io/src/output/schemas.rs::costs_schema_field_count_and_names`, `crates/cobre-io/src/output/schemas.rs::rank_timing_schema_field_count`
- **Evidence:** `all_schema_functions_return_valid_schemas` (line 1282) holds a twenty-row `expected: &[(&str, usize)]` table asserting the field count of every output schema.
- **Fix-shape:** Pick one registry for field counts. The umbrella table is the better owner because it is exhaustive and a new schema cannot be added without appearing there, so the seven count-only per-schema tests collapse into it, and `transit_seed` gains the umbrella row it is currently missing.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

**TD-023 · Sev B · duplication · effort M · confidence high**
parquet_helpers.rs has zero #[cfg(test)] module (166 lines, all non-test), so the six extractors' missing-column and wrong-type SchemaError message contract is pinned by no owner-level test; the narrower residue drops 'all twelve reachable transitively' — consumer paths like scenarios/inflow_history.rs intercept the missing-column case as a legacy-layout error before the helper's arm, so transitive coverage is partial, not uniform.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/parquet_helpers.rs::extract_required_int32`, `crates/cobre-io/src/parquet_helpers.rs::extract_required_date32`, `crates/cobre-io/src/parquet_helpers.rs::extract_optional_float64`
- **Evidence:** Zero `#[cfg(test)]` modules, which matches the inventory's `linesRaw: 166` equalling `linesNonTest: 166` for this file.
- **Fix-shape:** Add one inline test module to `parquet_helpers.rs` that builds a small in-memory `RecordBatch` and asserts, per extractor, the happy path, the missing-column message and the wrong-type message, so the parse-error contract is pinned once at its owner instead of being re-asserted incidentally in sixteen consumer test modules. This is the counterpart to the duplication findings: the same consolidation that removes redundant assertions elsewhere depends on the shared helper carrying its own contract test.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)

### Positives (recorded so the report is not a defect-only list)

- `crates/cobre-core/src/model/resolved/penalties.rs` — Ratified as consumed, not reserved, so no unwired-seam candidate is raised against these penalties. (sanctioned by plans/architecture-debt-audit/stations/core-io/prior-register.md — 'Hydro `storage_violation_below_cost` / `filling_target_violation_cost` penalties — KEEP (consumed, not reserved)', citing docs/design/reserved-seams-and-deferred-debt.md section 'Verified NOT reserved')
- `crates/cobre-core/src/model/scenario.rs` — Ratified as consumed by the window and historical samplers; (sanctioned by plans/architecture-debt-audit/stations/core-io/prior-register.md — '`historical_years` on `ScenarioSource` — KEEP (consumed, not reserved)', citing docs/design/reserved-seams-and-deferred-debt.md section 'Verified NOT reserved')
- `crates/cobre-io/src/stages.rs` — `graph_type: RawPolicyGraphType` accepts only the finite-horizon value and is a deletion candidate under the node-native engine. (sanctioned by plans/architecture-debt-audit/stations/core-io/prior-register.md — `Horizon-type config field (graph_type in stages.rs)`, register id `mirror:Horizon-type config field long-term fate`)
- `crates/cobre-io/src/config/training.rs` — Ratified reserved seam, not dead config: LipschitzConfig and its enclosing UpperBoundEvaluationConfig at :509 are loaded, schema-exported and unconsumed by design, reserving the vertex-based inner-approximation seam. (sanctioned by docs/design/reserved-seams-and-deferred-debt.md — reserved-seam register, `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`; prior-register.md sanctioned list)
- `crates/cobre-io/src/config/policy.rs` — Stage-addressed boundary source with no node selector is already on the deferred-debt register with a named trigger, so this station cites it rather than re-recording it. (sanctioned by docs/design/reserved-seams-and-deferred-debt.md — deferred-debt register, Boundary-policy source-node; prior-register.md `## Registered and open`)
- `crates/cobre-io/src/output/atomic.rs` — The tmp-write / explicit-flush / rename contract is stated once, in one module, and its doc explains why the flush cannot be left to Drop — Drop::drop cannot return an error, so a drop-flush swallows ENOSPC on the buffered tail and the rename installs a truncated file. (sanctioned by in-tree module doc, crates/cobre-io/src/output/atomic.rs:1-9)
- `crates/cobre-io/src/parquet_helpers.rs` — The read-side dedup precedent the sub-station D brief names: six typed extract helpers (required and optional int32/float64, uint32, date32) each producing the same SchemaError shape with the column name and the observed Arrow type. (sanctioned by cited as the dedup precedent in the sub-station D probe list)
- `crates/cobre-io/src/output/simulation_writer.rs` — CD-026 is genuinely resolved: one helper owns the create-dir / write / push tail that all fourteen entity families share, and its doc says so. (sanctioned by prior-register.md — 'CD-026 · write_scenario 12x write-partition repeat — RETIRE')
- `crates/cobre-io/src/output/policy/records.rs` — CD-010 is genuinely resolved: the typed enum mirrors EntityType in schemas/policy.fbs, the reader goes through family() rather than a raw byte, and the duplicated const in checkpoint.rs is gone. (sanctioned by prior-register.md — 'CD-010 · untyped entity_type: u8 state-family dictionary — RETIRE')
- `crates/cobre-io/src/output/manifest.rs` — A name that shadows atomic.rs::write_json_atomic but is not a duplicate: it serializes locally only to keep the ManifestError variant its callers match on, then delegates the crash-safe write to write_bytes_atomic, and an inline comment states exactly that division with the byte-identity guarantee. (sanctioned by in-tree rationale comment, crates/cobre-io/src/output/manifest.rs:528-530)
- `crates/cobre-core/src/constraints/training_event.rs` — Recorded as already-registered rather than re-discovered: the training-event vocabulary in an L0 crate and the stochastic/risk configuration on System and Stage are open Milestone-1 purification targets with live anchors, so this pass adds no new over-engineering finding on them. (sanctioned by prior-register.md 'Phase-1 purification targets' (Milestone 1 — training_event out of cobre-core; Milestone 1 — stochastic off System / Stage); partI-handoff.json dispositions I.3-1 (sharpen) and I.3-2 (keep))
- `crates/cobre-io/src/validation/referential.rs` — The reference layer builds its entity id sets once into `LookupSets` (crates/cobre-io/src/validation/referential.rs:73) and then makes exactly one pass per scenario table, testing membership with a hash lookup instead of a nested scan. (sanctioned by observed at the baseline: crates/cobre-io/src/validation/referential.rs:360-520)
- `crates/cobre-io/src/validation/semantic/scenarios.rs` — The stationarity gate builds its seasonal-standard-deviation map, its annual-parameter map and its per-hydro coefficient tree once, before the per-hydro loop, and the loop then does hash lookups only. (sanctioned by observed at the baseline: crates/cobre-io/src/validation/semantic/scenarios.rs:325-365)
- `crates/cobre-io/src/validation/semantic/correlation.rs` — The matrix rules are quadratic only because a correlation matrix is quadratic; (sanctioned by observed at the baseline: crates/cobre-io/src/validation/semantic/correlation.rs:12-80)
- `crates/cobre-core/src/system/builder.rs` — The order-invariance hard rule is guarded by a real property test, not by examples: nine entity collections are independently `prop_shuffle`d through `SystemBuilder::build()` and the canonical projection is compared. (sanctioned by docs/design/testing-architecture.md §5.9 ("Expand proptest ... to cover the declaration-order-invariance and reduction-order invariants directly (permute → assert identical bits)") and §3.1 item 3 ("a gate proves it exercises the condition it guards"))
- `crates/cobre-core/src/constraints/generic_constraint.rs` — Not bloat, and not a duplicate of a const. (sanctioned by CLAUDE.md hard rule "Never delete or weaken a load-bearing correctness contract"; the register's analogous "Retired input spellings — the reject tests are load-bearing, not bloat" entry in plans/architecture-debt-audit/stations/core-io/prior-register.md)
- `crates/cobre-core/tests/infra_genericity.rs` — A 41-line integration binary that shells out to `scripts/ci/check-infra-genericity.sh` and asserts a clean exit, so the L0 paradigm-neutrality rule is enforced by `cargo test` and not only by a CI job. (sanctioned by plans/architecture-debt-audit/stations/core-io/prior-register.md — "Infra-genericity gate scans output/policy/ ... Disposition: RETIRE the exemption claim; KEEP the two in-crate oracles as positives")
- `crates/cobre-core/src/entities/hydro.rs` — The mechanism the three fixture-duplication candidates need already exists and is wired correctly: the feature is declared, documented as "Must NOT be enabled in production builds", gated with the `any(test, feature = ...)` idiom, and enabled for the crate's own tests through the self-referential dev-dependency at Cargo.toml:39. (sanctioned by docs/design/testing-architecture.md §5.2 ("Keep the `test-support` cargo-feature mechanism the repo already uses ... do not introduce a dedicated test crate"))
- `crates/cobre-io/src/post_study_stages.rs` — A real unit/integration split rather than the same assertions run twice. (sanctioned by docs/design/testing-architecture.md section 2.1 -- the healthy-pyramid invariant (unit tests outnumber integration tests by roughly an order of magnitude))
- `crates/cobre-io/src/config/mod.rs` — A mechanical closure rather than a hand-listed case set: it walks every JSON object node of several maximal valid configs and asserts an injected unknown key is rejected at each pointer, which is exactly the property tier docs/design/testing-architecture.md section 5.3 prescribes for a claim that quantifies over all inputs. (sanctioned by docs/design/testing-architecture.md section 5.3 — the Property tier and its decision rule)
- `crates/cobre-io/src/validation/semantic/test_support.rs` — The module states and follows the right homing rule (single-use helpers stay in their own module's test block to keep the blast radius small), and its derived builders compose rather than fork: `make_stage_with_blocks` calls `make_stage` and mutates one field, with a doc comment saying why, so the two cannot drift. (sanctioned by docs/design/testing-architecture.md section 5.2 — helpers live with the type they build, exposed through a `test-support` feature)
- `crates/cobre-io/tests/clean_break_no_deprecated_fallbacks.rs` — A lexical clean-break gate that names, in its own doc comment, the per-site behavioural tests it complements rather than replaces (`retired_scheduler_spellings_are_deserialize_error`, `test_num_scenarios_removed_field_rejected`, and the FlatBuffers conformance check). (sanctioned by plans/architecture-debt-audit/stations/core-io/prior-register.md — "Executable oracles (the gate is the register entry)", KEEP the two in-crate oracles as positives; docs/design/testing-architecture.md section 5.1, non-solver crates are out of scope for binary consolidation)
- `crates/cobre-io/tests/metadata_back_compat.rs` — The legacy fixtures are hand-frozen JSON literals with the reason recorded in the file header — a struct-serialized fixture carries every field and so could never catch a field accidentally made required — and two further tests guard the fixtures themselves against acquiring the new keys. (sanctioned by docs/design/testing-architecture.md §3.1 item 2 (contract-pinning) — the fixture is the pin, and the `*_omits_new_keys` tests are its power self-check)
- `crates/cobre-io/tests/flatbuffers_schema_conformance.rs` — The gate is not dormant. (sanctioned by prior-register.md — "Retired input spellings" KEEP entry names this file as part of the load-bearing guard set)
- `crates/cobre-core/src/system/mod.rs` — The deserialize-only mirror struct is exactly the hand-maintained duplicate this station usually raises, and it is documented as a silent-corruption hazard ('postcard is non-self-describing, so a reorder silently decodes into the wrong fields').
- `crates/cobre-core/src/commissioning.rs` — One module owns the entire commissioning and filling-lifecycle predicate family for every equipment type: three total functions with no panic path, each documented as the single owner, keyed explicitly on the stage id rather than the stage index with the reason for that choice stated, and backed by a truth-table test that pins each branch and names the forbidden alternative it rules out.
- `crates/cobre-core/src/model/temporal/stage_key.rs` — Three zero-cost newtypes turn a whole class of positional bugs into compile errors: domain stage id versus study-horizon position versus calendar month, each previously a bare integer that a mismatched call site would have silently keyed by the wrong convention.
- `crates/cobre-core/src/model/resolved/generic.rs` — A hand-written serializer that sorts its composite keys before emitting, with the determinism reason stated inline.
- `crates/cobre-core/src/model/resolved/bounds.rs` — Each carries a `compile_fail` doctest asserting that the stage-level accessor a reader would reach for does not exist, paired with a compiling sibling showing the correct call.
- `crates/cobre-io/src/resolution/group_bounds.rs` — The single resolver that states the ordering precondition completely and correctly: it names `(operational_start_date, id)`, attributes it to `SystemBuilder::build`, warns explicitly that the parser's id-only sort coincides with it only when every entity shares one operational start date, separates the unit-group axis as having no equivalent divergence, and names the regression test that falsifies a resolve-before-resort regression.
- `crates/cobre-io/tests/resolver_builder_index_alignment.rs` — An eleven-test end-to-end suite that drives the real `load_case` pipeline against decks whose operational-start-date order is the reverse of their id order, covering hydro, bus penalty, line, thermal, pumping, contract, NCS and hydro unit group overrides, plus a direct assertion that the pipeline's presort key equals the builder's canonical key and that referential error order survives the post-validation sort.
- `crates/cobre-io/src/constraints/bounds.rs` — The block-eligibility documentation refuses to be the source of truth about itself: it states that a column is block-eligible exactly when its family's `<Family>BlockOverride` struct in `cobre_core::resolved` carries a field for it, and that the struct's field set is the check rather than the table.

### ↩︎ Cleared (dismissed — do not re-raise)

No over-engineering candidate targeted a ratified reserved seam: the E02-3 attackers pre-filed the sanctioned seams (`LipschitzConfig.mode`, the hydro storage/filling penalties, `historical_years`) under Positives with their mirror citations, so none re-entered the candidate set. The four dismissals below are merit-based (not sanctioned seams); each is cleared so the re-raise checker treats it as retired:

- **B-perf-00** — At baseline `tokenize` (generic.rs:629) is reached only via `parse_relation` <- `convert`'s per-constraint load-time loop (line 489) — the constraint-file-to-struct conversion, never the SDDP training/simulation hot paths enumerated in architecture-rules.md. The `Vec<char>` collect (line 631) is deliberate: it gives O(1) indexed two-char lookahead (`chars.get(i+1)` for `==`/`<=`/`>=`) and lets every error report a char-offset position (`at position {i}`) a human counting characters can locate;
- **B-perf-01** — The evidence resolves at baseline: `inline` (named_expression_inline.rs:57) builds its `HashMap<&str,&ParsedExpression>` index per top-level call, and `resolve_split_side` (generic.rs:1254-1255) does a linear `table.iter().any(...)` membership scan then calls `inline(&vec![...],table)`, rebuilding the index and allocating a one-element Vec for a single reference. Both call sites run inside `convert`'s per-constraint load-time loop (line 489), not a hot path.
- **B-perf-03** — Mechanism confirmed at baseline: `parse_inflow_history` (inflow_history.rs:144-152) maps every parsed `InflowHistoryRow` into a parallel `Vec<WindowedRecord>` consumed only by `validate_windowed_records` on the next line, then dropped. But this is a single O(rows) linear pass of a 4-field `Copy` struct (memcpy-able), dominated by O(rows) work already unconditionally present on the same rows: the parquet decode loop that builds `rows` from the record batches (rows.push at line 128) and the subsequent `rows.sort_by` (line 137, an O(rows log rows) pass).
- **C-over-engineering-00** — At the baseline all three enums derive serde::Serialize plus schemars::JsonSchema under #[serde(rename_all="snake_case")] (estimation.rs:15-18, scenario_source.rs:70-73, training.rs:106-109), so each exports a string-only JSON enum schema. serde's DERIVED Deserialize for a unit-variant enum accepts both the bare string ("pacf") and serde's externally-tagged single-key-map form ({"pacf":null});

### Part-I cross-references (items 1, 2, 3, 4, 6, 7)

- **CD-044** (A-architecture-04) → Part-I I.3-2, routed to Epic 9.
- **CD-045** (A-architecture-05) → Part-I I.3-1, routed to Epic 9.
- **OD-013** (A-over-engineering-03) → Part-I I.3-1, routed to Epic 9.
- **CD-050** (B-architecture-03) → Part-I I.3-2, routed to Epic 9.
- **OD-016** (B-over-engineering-02) → Part-I I.3-1, routed to Epic 9.
- **CD-051** (C-architecture-00) → Part-I I.3-7, routed to Epic 9.
- **CD-061** (D-architecture-03) → Part-I I.3-6, routed to Epic 9.
- **PD-015** (D-perf-00) → Part-I I.3-6, routed to Epic 9.
- **OD-020** (D-over-engineering-00) → Part-I I.3-7, routed to Epic 9.
- **OD-024** (D-over-engineering-04) → Part-I I.3-6, routed to Epic 9.
- **TD-021** (D-test-bloat-03) → Part-I I.3-6, routed to Epic 9.

The two out-of-station Part-I item-7 anchors (`crates/cobre-sddp/src/setup/params.rs::from_config`, `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig`) were handed to the cobre-sddp and cobre-cli stations via `partI-handoff.json`, not raised here.

### Owner gate — decisions

**Ratified 2026-09-08 (baseline `a136840d`).** Decisions taken in the main session over
the station digest stations/core-io/gate.md. Severity shows as
`new (reviewer: original)` wherever the owner downgraded.

**Gate: RETURNED 2026-09-08 · baseline `a136840d4f2ea137f685f0af6dac04254b983b60` · accepted 76 / downgraded 1 / rejected 0 / deferred 0 / overridden 0.**

| ID                                 | Decision  | Severity        | Alignment   | Rationale (owner)                                                                                                                                                                                                                                                                     | Trigger / override |
| ---------------------------------- | --------- | --------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| CD-046                             | downgrade | C (reviewer: B) | neutral     | Byte-level wire-payload reproducibility is a non-goal: Cobre determinism is result-level (bit-for-bit results, run-to-run, declaration-order invariance), so the six unguarded HashMaps are not a latent bug and the residue is a doc-accuracy nit on the System serde(skip) comment. | —                  |
| _(the other 76 confirmed entries)_ | accept    | as calibrated   | as recorded | Ratified as the attacker→defender→calibration pipeline recorded them; no conflicts holds, no rejects, no defers.                                                                                                                                                                      | —                  |

**Needs-human resolved (from the E02-4 defender pass):** is byte-level wire-payload
reproducibility an actual contract? → **Non-goal** — CD-046 downgraded B→C accordingly;
no other entry affected.

No entry was rejected, so the Cleared list above is unchanged; no entry was tagged
`conflicts`, so nothing was held or overridden.

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — stochastic

### Station 3 — cobre-stochastic (2026-09, baseline a136840d)

**Ratified 2026-09-08** — owner gate; baseline `a136840d`; accepted 35, downgraded 0, rejected 0, deferred 0, overridden 0.

**Station.** cobre-stochastic — the L1 uncertainty store. **Method.** Four lenses (architecture, performance, over-engineering, test-bloat) over four sub-stations: `par/` (PAR fitting + evaluation), `sampling/` (realized-value libraries + ForwardSampler), `tree-noise` (`tree/`+`noise/`+`normal/`+`correlation/`), and the outward seam (`context.rs`, `seeds.rs`, `season_cast/`, `provenance.rs`, `lib.rs`).

**Baseline.** `a136840d4f2ea137f685f0af6dac04254b983b60` (pinned). Read-only station: no tracked file was modified. 35 attacker candidates, 35 defended, 35 confirmed, 0 dismissed; every anchor resolves at the baseline.

#### Architecture

**CD-064 · Sev B · leaky-boundary · effort S · confidence high**
Narrowed to the SDDP two-pass phase phrasing that genuinely encodes a single-engine assumption: 'forward/backward passes' (precompute.rs:11), 'forward pass' (lag_transition.rs:60), 'training/simulation lag transitions' (lag_transition.rs:137) and 'forward-pass hot path' (lag_transition.rs:160). I concede 'LP RHS patching' and 'consumed read-only during optimization' (precompute.rs:1,4; PrecomputedPar:78) are generic LP-solver phrasing any deterministic ED/OPF consumer would also use, not engine paradigm nouns, so those two phrases are over-reach.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/precompute.rs:78`, `crates/cobre-stochastic/src/par/precompute.rs:11`, `crates/cobre-stochastic/src/par/lag_transition.rs:137`, `crates/cobre-stochastic/src/par/lag_transition.rs:160`, `crates/cobre-stochastic/src/par/lag_transition.rs:60`
- **Evidence:** cobre-stochastic is an L1 shared kernel whose declared scope is 'uncertainty store + scenario generation' and which the target layering forbids from carrying engine concepts; the crate exists to be reused by a second, non-decomposition engine (cobre-direct: ED/OPF/deterministic UC). These production doc comments describe the module's own data in terms of the SDDP engine's phase structure — 'forward/backward passes',… Re-derive: `for f in par/precompute.rs par/lag_transition.rs; do echo "----- $f -----"; git show a136840d4f2ea137f685f0af6dac04254b983b60:cra…`
- **Fix-shape:** Reword the flagged doc comments to consumer-agnostic language that describes the layout/behaviour intrinsically rather than by the SDDP engine's phases. 'contiguous for the forward/backward passes' becomes 'contiguous for sequential per-stage traversal'; 'consumed read-only on the forward-pass hot path' becomes 'consumed read-only on the per-stage evaluation hot path'; 'training/simulation lag transitions' becomes 'both evaluation-phase lag-transition call sites' (or names the two functions); 'LP RHS patching' / 'during optimization' becomes 'downstream solver right-hand-side construction' or simply 'consumed read-only by the value-generation path'. No type, signature, or byte of behaviour changes. Do NOT introduce any SDDP paradigm noun (cut, cost-to-go, state space, ring, Benders) in the replacement text, and do not name an engine crate — this stays a pure L1 reword.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**CD-065 · Sev B · missing-seam · effort M · confidence high**
The structural residue only: the four realized-value Option<&Library> fields share one flat Copy bundle with the &StochasticContext store handle and no type-level separation of the two concerns/lifetimes, so a reader of the seam cannot tell store from per-run input without reading build_forward_sampler's body — the store/param split the alignment epic should reshape (the shape that survives the Part IV/V Switchable<T> unification), not a runtime defect.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs:263`, `crates/cobre-stochastic/src/sampling/mod.rs:403`
- **Evidence:** The single #[derive(Copy)] seam type carries, in one flat bundle, a persistent store handle (`ctx: &StochasticContext`, whose own doc says it provides `tree, seeds, correlation, and entity order` — all generation-side artifacts: the fitted-PAR opening tree, the base/forward seeds, the pre-decomposed correlation) alongside four realized-value `Option<&…Library>` references plus per-run `class_schemes`/`stages`/`dims`… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/sampling/mod.rs | sed -n '262,289p'; echo ---; git…`
- **Fix-shape:** Split the one flat bundle into two clearly-named halves at the seam: a persistent store handle (today `&StochasticContext`; the generation-side tree/seeds/correlation/entity-order the sampler consumes read-only) and a transient per-run realized-value source selection (the four library references plus the per-class schemes that pick and require them). The source-selection half is where the four-way validity coupling lives and should own a constructor/validation that makes the `scheme -> required library` pairing a type-level fact rather than a runtime check in `build_class_sampler`; leave the coupling-collapse mechanics themselves to the over-engineering sibling lens. This keeps the store as a store and the source-selection as a bag, so a reader of the seam sees two concerns, not eight peer fields. Roadmap note only (not executed here): under Part IV/V the realized-value libraries and th…
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part V.2 — Phase 1 Switchable<T> uncertainty store (stochastic off System/Stage))
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).
- **Queued to:** alignment

**CD-066 · Sev B (A-risk) · bad-abstraction · effort M · confidence high**
Only the single `class_name != "inflow"` Historical gate at mod.rs:436 is a correctness rule (not a diagnostic) riding on the &str class identity, contradicting the "used only in error messages" doc comment; the two format! diagnostics and the correlation-side entity_type matching are legitimate and not indicted.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs:360`, `crates/cobre-stochastic/src/sampling/mod.rs:297`
- **Evidence:** `ClassSchemes` already carries the class identity as three typed fields (`inflow`/`load`/`ncs`, confirmed in context.rs:17-23), and `ClassDimensions` mirrors it. But `build_forward_sampler` flattens that typed identity back to bare string literals `"inflow"`/`"load"`/`"ncs"` passed via `ClassSamplerParams.class_name: &str`, and `build_class_sampler` then gates a real correctness rule — Historical replay is inflow-on… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/sampling/mod.rs | sed -n '296,297p;359,362p;436,437…`
- **Fix-shape:** Introduce a small closed class discriminant (e.g. an entity-class enum with `Inflow|Load|Ncs`) carried on `ClassSamplerParams` in place of the `&str`, so the Historical-only-for-inflow gate becomes a match/`==` on a typed value the compiler checks, and the three build sites pass the variant instead of a literal. Keep the human-readable label for diagnostics as a `Display`/`as_str` off that enum so error messages are unchanged. Purely local type-safety; advances no roadmap phase and introduces no engine vocabulary.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Reviewer rating:** C — recalibrated to B (A-risk) because raised from C per the silent-divergence seam precedent: the untyped &str class gate silently mis-gates on a call-site literal typo.

**CD-067 · Sev B (A-risk) · bad-abstraction · effort M · confidence high**
The in-station residue is the in-crate GroupFactor.entity_type String field plus the two `!=` dispatch sites (resolve.rs:286, 352) that a boundary-parse to an in-crate EntityClass{Inflow,Load,Ncs} enum at build() would make compiler-checked; the source-of-truth String is cobre-core's CorrelationEntity.entity_type (scenario.rs:594) which is OUT of station (a cross-station note, not changed here), so the confirmed defect is the stochastic-side stringly-typed carrier + comparisons, not the cobre-core field.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:28`, `crates/cobre-stochastic/src/correlation/resolve.rs:338`, `crates/cobre-stochastic/src/tree/generate.rs:332`
- **Evidence:** The entity class is a closed 3-value set (inflow/load/ncs) but is carried as a String on GroupFactor and dispatched by `!=`/`==` at two sites in resolve.rs (apply_correlation_for_class L352, resolve_class_positions L286). The producers are bare string literals at generate.rs:332-334 and sampling/mod.rs:227/236/245. Nothing links the literal producer to the stored value at compile time: a typo (`"inlfow"`) or a vocab… Re-derive: `git show a136840d:crates/cobre-stochastic/src/correlation/resolve.rs | sed -n '27,28p;351,353p' ; git show a136840d:crates/cobre-…`
- **Fix-shape:** Introduce an in-crate closed enum EntityClass { Inflow, Load, Ncs }. Parse the cobre-core-sourced CorrelationEntity.entity_type String exactly once, at DecomposedCorrelation::build, into the enum stored on GroupFactor (replacing the String field). Type the apply_correlation_for_class and resolve_class_positions `entity_type` parameter and the generate.rs / sampling/mod.rs call-site literals on the enum, so every class comparison becomes an exhaustive compiler-checked match with a single String->enum boundary at build. The source String remains owned by cobre-core (a cross-station note, not a change here); the boundary-parse keeps this fix entirely inside cobre-stochastic. No paradigm noun and >=2 consumers, so it does not trip the L1 purity guardrail.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**CD-068 · Sev B · duplication · effort M · confidence high**
The narrow residue is that the full-vector applier trio (apply_correlation + resolve_positions + GroupFactor.positions) is pub production surface with NO production caller (kept alive only as the per-class path's differential-test oracle, test_per_class_tree_matches_full_vector_*) and the per-class precompute (resolve_class_positions + GroupFactor.class_positions) is wholly unwired (zero callers, class_positions never populated) -- so exactly one applier (apply_correlation_for_class) and zero precompute run in production; this is duplicated-but-tested surface plus a dead precompute, NOT literal unreferenced dead code, and its removal-vs-wiring is the open question candidate perf-00 pulls the other way on.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:31`, `crates/cobre-stochastic/src/correlation/resolve.rs:253`, `crates/cobre-stochastic/src/correlation/resolve.rs:277`, `crates/cobre-stochastic/src/correlation/resolve.rs:307`
- **Evidence:** Production correlation runs exclusively through apply_correlation_for_class on the linear-scan fallback: the per-class precompute resolve_class_positions is never called in any build, so GroupFactor.class_positions stays None and the apply_group_precomputed branch it feeds is unreachable. The entire full-vector twin (apply_correlation + resolve_positions + GroupFactor.positions) has no non-test workspace consumer —… Re-derive: `git grep -nE 'apply_correlation\b|\.resolve_positions|resolve_class_positions' a136840d -- crates/cobre-stochastic/src crates/cob…`
- **Fix-shape:** Collapse to a single correlation applier. Preferred: since production uses only apply_correlation_for_class on the scan path, remove the full-vector twin (apply_correlation, resolve_positions, GroupFactor.positions) that only tests exercise and the never-called resolve_class_positions + GroupFactor.class_positions, leaving one applier over the shared apply_group_scan and one position-cache concept. Alternative, if the position-precompute is genuinely wanted for the forward-sampler hot loop: wire resolve_class_positions once at ForwardSampler/opening-tree setup and delete the redundant full-vector method rather than keeping both. Either way keep the spectral transform byte-identical and preserve the BTreeMap deterministic iteration order noted in-code; retire, not re-point, the segment-relative-vs-full-vector position-base distinction so the trap cannot be mis-merged.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**CD-069 · Sev C · duplication · effort S · confidence high**
The residue is strictly the missing shared parameter-carrier: the three structs are byte-identical in field set and types (differing only in name and one Lhs doc line), so the defect is purely a triplicated bag-of-parameters seam (Sev C); each struct is a live, exercised type (not a speculative one-consumer abstraction) and the three point generators legitimately keep distinct algorithms -- only the parameter carrier is duplicated, nothing about the generators themselves.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/lhs.rs:89`, `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs:236`, `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs:236`
- **Evidence:** The three point-wise generators (sample_lhs_point, scrambled_halton_point, scrambled_sobol_point/_precomputed) each take a distinct spec struct whose field set is byte-for-byte the same six fields with the same types. out_of_sample.rs constructs all three from the same FreshNoiseSpec. There is no shared type, so the point-sampling parameter contract is expressed in triplicate and any new field (or a rename) is an O(… Re-derive: `git show a136840d:crates/cobre-stochastic/src/tree/lhs.rs | sed -n '88,102p' ; git show a136840d:crates/cobre-stochastic/src/tree…`
- **Fix-shape:** Introduce one shared spec type (e.g. QmcPointSpec / PointSampleSpec) in a common noise or tree module carrying sampling_seed/iteration/scenario/stage_id/total_scenarios/dim, and have sample_lhs_point, scrambled_halton_point, scrambled_sobol_point and scrambled_sobol_point_precomputed all accept it. The three generators keep their distinct algorithms; only the parameter carrier is unified, so a future field is a one-line change. Three present consumers make this a real de-duplication, not a speculative one-consumer abstraction, so it does not trip the L1 purity guardrail.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**CD-070 · Sev B (A-risk) · duplication · effort L · confidence high**
The confirmed defect is confined to (1) the second parallel copy of the declared-id->study-index resolver (stage_id_to_index at context.rs:406 duplicating cobre-io's StageIdResolver) as the sole drift risk, and (2) the placement observation that the L1 store constructor owns realized-value external-scenario adaptation; it is NOT a dispute of the ratified External-authoritative behavior or the sigma=0 rejection, and the sample-moment derivation itself is correctly single-owned (not duplicated).

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/context.rs:406`, `crates/cobre-stochastic/src/context.rs:419`, `crates/cobre-stochastic/src/context.rs:444`, `crates/cobre-stochastic/src/context.rs:492`
- **Evidence:** context.rs is the L1 generation store's constructor module. Under the External scheme (build_stochastic_context branches at lines 675-681, 717-740, 747-766) it reaches into System's realized-value external-scenario tables (external_scenarios / external_load_scenarios / external_ncs_scenarios) and synthesizes fitted InflowModel/LoadModel from their sample moments, and it reimplements cobre-io's declared-id->study-ind… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/context.rs | sed -n '397,412p;475,505p'`
- **Fix-shape:** Roadmap-aligned: under Part-1 purification the external-scenario tables leave `System` for the cobre-stochastic uncertainty store; the External-ingestion adaptation (sample-moment -> fitted-model synthesis) then moves WITH those tables into the store rather than living in the context constructor, and the paradigm-neutral declared-id->study-index resolver homes once at L0 (cobre-core temporal, the same destination the mirror already registers for the season_cast relocation) so this crate (L1) and cobre-io (L2) consume one resolver instead of two parallel implementations. No engine noun and no L1->engine dependency is introduced. As a bounded interim step short of purification, extract the four External-ingestion helpers (stage_id_to_index, resolve_row_stages, external_derived_load_models, external_ar0_inflow_models) into their own submodule so the store constructor is not the owner of re…
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part V.2 — Phase 1 Switchable<T> uncertainty store (stochastic off System/Stage))
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).
- **Queued to:** alignment

**CD-071 · Sev C · leaky-boundary · effort M · confidence high**
The defect narrows to just the 2 pass-through seed fields (base_seed, forward_seed) being RunParams-shaped config echoed through the store's getters rather than computed payload; the struct is correctly shaped and correctly a many-reader store today, so the residue is only that these two fields mark the exact seam to unpick when the RunParams split lands, not that the store is malformed or over-abstracted now.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/context.rs:258`, `crates/cobre-stochastic/src/context.rs:317`
- **Evidence:** The 13 private fields split into four roles: 5 precomputed components (par_lp, correlation, opening_tree, normal_lp, ncs_normal) that ARE the store; 5 derived-layout values (ncs_entity_ids, entity_order, dim, n_load_buses, n_stochastic_ncs) that are addressing metadata recomputable from noise_entity_order; 2 pass-through config values (base_seed, forward_seed) echoed straight from the builder args; and 1 metadata fi… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/context.rs | sed -n '257,275p'`
- **Fix-shape:** Phase-1 destination note, NOT executed by this spec (lens Q4 is alignmentHint-only). When Part IV/V introduces the Switchable<T> uncertainty store, the precomputed-component fields become the store payload and the entity_order/dim/n_* fields its addressing metadata, while base_seed/forward_seed migrate to the RunParams half of the Model/RunParams/PolicyState split rather than being echoed through the store. No restructuring here; recorded so the alignment epic knows this struct is the carve target and the config-echo/store fusion is the seam to unpick.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part V.2 — Phase 1 Switchable<T> uncertainty store (stochastic off System/Stage))
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).
- **Queued to:** alignment

#### Performance

**PD-020 · Sev B · asymmetry · effort M · confidence high**
Narrowed: the parallelizable region is specifically the INITIAL per-hydro estimation loop of estimate_ar_with_pacf_annual (the `for &hydro_id in hydro_ids { ... estimates.push(...) }` block starting estimation.rs:397), NOT the subsequent annual reduction (apply_annual_prepass_reductions / reduce_entity_orders_annual), which mutates `estimates` across hydro index sets and must remain serial in this change. It is a fitting/setup-time asymmetry (one fit per solve), and correctness requires the flat_map_iter/collect canonical hydro_ids-order reassembly (estimation.rs:947-951) so the result stays declaration-order-invariant.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/estimation.rs:397`, `crates/cobre-stochastic/src/par/fitting/estimation.rs:952`
- **Evidence:** estimate_all_hydro_ar_coefficients (classical PACF path) fits every hydro under hydro_ids.par_iter().flat_map_iter(...).collect(); estimate_ar_with_pacf_annual (the PAR(p)-A path) does the identical per-hydro work — conditional_facp_partitioned + estimate_periodic_ar_annual_coefficients per season, pushed to `estimates` — inside a plain serial `for &hydro_id in hydro_ids`. Each annual iteration reads only immutable… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/par/fitting/estimation.rs | sed -n '395,427p'; echo…`
- **Fix-shape:** Lift the initial per-hydro block of estimate_ar_with_pacf_annual (the `for &hydro_id in hydro_ids { ... estimates.push(...) }` region) into the same hydro_ids.par_iter().flat_map_iter(|&hydro_id| { build this hydro's Vec<ArCoefficientEstimate> }).collect() shape the classical estimate_all_hydro_ar_coefficients already uses, so each hydro's seasons are produced in its closure and collect reassembles them. The in-code determinism rationale MUST be preserved byte-for-byte: the classical site states 'flat_map_iter/collect reassembles the per-hydro blocks in canonical hydro_ids order, and the inner per-season PACF -> Yule-Walker solve is bit-identical to a single-threaded pass -- thread scheduling cannot change the output. flat_map_iter (not flat_map): each hydro's Vec is small (n_seasons), so nesting work-stealing over it would gain nothing.' The annual closure has the identical property (i…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-021 · Sev B · duplication · effort M · confidence high**
Narrowed: the double date-set intersection holds only for seasons that pass the MIN_CORRELATION_PAIRS gate — a rejected season intersects once (in min_pairs) then `continue`s (correlation.rs:418-420), so it pays the walk once, not twice. And this is fitting/setup-time work (one correlation build per solve), so the confirmable cost is bounded setup latency and allocation traffic scaling with n_hydros^2 * observation-length, NOT a per-scenario/per-stage runtime-hot-path regression. Any fix must keep the two documented determinism contracts (canonical hydro_ids-order collect; NaiveDate-ordered pair accumulation) so declaration-order invariance is preserved.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/correlation.rs:391`, `crates/cobre-stochastic/src/par/fitting/correlation.rs:438`, `crates/cobre-stochastic/src/par/fitting/correlation.rs:199`
- **Evidence:** compute_seasonal_matrices, per season, (a) clones every hydro's date-keyed residual map into season_residuals via `.cloned()`, (b) computes min_pairs by walking all O(n_hydros^2) pairs and intersecting the two date sets with contains_key hashes, then (c) if kept, calls compute_pearson_correlation_matrix which walks the SAME O(n_hydros^2) pairs and re-intersects the same date sets via r_j.get(&date). Residuals are pr… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/par/fitting/correlation.rs | sed -n '396,427p'; ech…`
- **Fix-shape:** Two independent, fitting-local reductions with no cross-crate blast radius. (i) Compute each pair's overlap count once: have compute_pearson_correlation_matrix (or a shared helper it calls) return the min pair count alongside the matrix, and drop compute_seasonal_matrices' separate min_pairs pre-walk -- or gate the season on the count the matrix build already derives. (ii) Store per-(hydro, season) residuals as a date-sorted Vec<(NaiveDate, f64)> (or one flat sorted buffer) instead of HashMap<NaiveDate,f64>, so the pairwise intersection is a linear merge-walk of two sorted date lists rather than n contains_key/get hashes; this also removes compute_pearson_correlation_matrix's per-pair sort_unstable since the inputs arrive sorted, and lets compute_seasonal_matrices borrow rather than `.cloned()` each hydro's season map. Both the parallel producer and the matrix accumulator carry determin…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-022 · Sev C · duplication · effort S · confidence high**
Narrowed: the borrow-view substitution is valid only where the consumer never mutates the owned vector — verified true at the two reduction loops and the PAR-A initial estimate, which build obs_refs: Vec<&[f64]> and only read (e.g. the classical closure's `obs_by_season[season].len() < 2` guard reads, never writes). It is setup-time memory-traffic debt (one fit per solve), so the confirmable defect is bounded allocation/copy volume (2x classical / up to 4x PAR-A of raw history), NOT a per-iteration hot-path regression; and the classical par_iter closure's canonical hydro_ids-order collect (estimation.rs:947-951) is unaffected by swapping clone_from for a shared-immutable borrow.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/estimation.rs:959`, `crates/cobre-stochastic/src/par/fitting/estimation.rs:1067`, `crates/cobre-stochastic/src/par/fitting/estimation.rs:689`, `crates/cobre-stochastic/src/par/fitting/estimation.rs:407`
- **Evidence:** group_obs is a HashMap<(EntityId,usize),Vec<f64>> already owning each hydro-season's observation vector. Every consumer -- the parallel estimate_all_hydro_ar_coefficients closure, the serial reduce_entity_orders, and in the PAR-A path estimate_ar_with_pacf_annual plus reduce_entity_orders_annual -- rebuilds a local obs_by_season: Vec<Vec<f64>> via clone_from(obs), a full deep copy. The reduction and annual-estimate… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/par/fitting/estimation.rs | sed -n '1063,1073p'`
- **Fix-shape:** Where the consumer only reads observations (the two reduction loops and the PAR-A initial estimate all build obs_refs: Vec<&[f64]> and pass borrows into the FACP/YW primitives), construct that borrow view directly from group_obs -- e.g. (0..n_seasons).map(|s| group_obs.get(&(hydro_id, s)).map_or(&[][..], Vec::as_slice)).collect() -- instead of clone_from into an owned Vec<Vec<f64>>. group_obs outlives every pass within the fitter, so the borrows are valid. This keeps the classical parallel closure's canonical-order determinism unchanged (it still returns per-hydro results in hydro_ids order) and does not alter any numeric result; it only removes the intermediate owned copies. No paradigm noun, engine dependency, or new abstraction.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path) — recorded as deferred debt, below the Sev-A/B performance-sweep threshold.

**PD-023 · Sev B · duplication · effort M · confidence high**
Scoped to the OutOfSample QmcSobol forward path only (SAA/InSample/Historical/External are allocation-free per the positives): the direction matrix + scramble params are recomputed and heap-allocated per (iteration, scenario, stage) though invariant across the scenario axis, and the existing SobolPrecomputed/sobol_ctx seam already amortizes this bit-identically — so the residue is wiring an existing dormant hoist whose sole production caller hard-codes None, not new infrastructure and not a determinism risk.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/class_sampler.rs:311`, `crates/cobre-stochastic/src/sampling/out_of_sample.rs:103`, `crates/cobre-stochastic/src/sampling/out_of_sample.rs:84`
- **Evidence:** The composite ForwardSampler::sample -> ClassSampler::fill is documented as called for every (iteration, scenario, stage) triple. On the OutOfSample QmcSobol path each such call reaches fill_uncorrelated with sobol_ctx == None, so scrambled_sobol_point rebuilds the direction matrix (dim x [u32;32]) and the scramble-params vector every draw. Direction matrix and scramble params depend only on (forward_seed, iteration… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/class_sampler.rs | sed -n '287,313p' ; git show a136840d:crates/cobre-stoc…`
- **Fix-shape:** Give the OutOfSample class sampler (or the caller that owns the per-(iteration,stage) loop) a per-stage SobolPrecomputed cache keyed on (forward_seed, iteration, noise_group_id, dim), built once when the stage/iteration first appears and reused across every scenario draw, then pass Some(&ctx) into the already-present `sobol_ctx` parameter of fill_uncorrelated so the precomputed branch (currently dead) is taken. No change to the generator body is needed (SobolPrecomputed / scrambled_sobol_point_precomputed already exist in the tree-noise crate). Determinism is preserved bit-for-bit because SobolPrecomputed::new derives its seed from the same derive_opening_seed(sampling_seed, iteration, stage_id) the point function uses, and the per-scenario XOR/scramble math is identical. The cache lifetime/ownership decision is a sampler-side concern in this sub-station; the tree-noise crate ships the…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-024 · Sev B · missing-seam · effort M · confidence high**
Scoped to the OutOfSample QmcHalton forward path: the per-scenario fresh prime-sieve + nested Vec<Vec<Vec<u32>>> scramble-table allocation and recomputation is scenario-invariant and should be hoisted per (iteration, stage); because NO HaltonPrecomputed seam exists (unlike Sobol's dormant one), the precompute primitive is a cross-station dependency on the tree-noise cell and only the sampler-side wiring is this sub-station's part; determinism is preserved (tables are a pure function of the existing seed tuple).

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/out_of_sample.rs:125`, `crates/cobre-stochastic/src/sampling/class_sampler.rs:311`
- **Evidence:** Same per-(iteration,scenario,stage) call site as candidate 1 (fill_uncorrelated with sobol_ctx None; the Halton arm has no ctx parameter at all). Every forward draw re-runs a prime sieve and rebuilds a three-level nested Vec of scramble permutation tables whose outer extent scales with dim and whose inner permutations are sized by the prime bases; the digit depth is derived from spec.total_scenarios. These tables de… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/out_of_sample.rs | sed -n '125,135p' ; git show a136840d:crates/cobre-stoc…`
- **Fix-shape:** Mirror the Sobol hoist: introduce a per-(iteration,stage) precomputed Halton context (a HaltonPrecomputed holding the prime list and the scramble tables) built once per (forward_seed, iteration, noise_group_id, dim, total_scenarios) and reused across every scenario draw. This requires a new precompute type + `_precomputed` point entry in the tree-noise crate (qmc_halton), consumed here via a `halton_ctx: Option<&HaltonPrecomputed>` parameter on fill_uncorrelated symmetric to the existing sobol_ctx — flag as a cross-station dependency on the tree-noise sub-station for the primitive; the sampler-side wiring (cache ownership + threading) is this sub-station's part. Determinism preserved: the tables depend only on the seed tuple already used, so the precomputed and per-call results are bit-identical.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-025 · Sev B · duplication · effort M · confidence high**
Scoped to the OutOfSample LHS forward path, and a redundant-COMPUTE finding (not allocation — perm_scratch is caller-owned): the scenario-invariant set of `dim` Fisher-Yates permutations of `total_scenarios` strata is reshuffled once per scenario, giving the quadratic-in-scenario O(dim*total_scenarios^2) per stage that a per-(iteration,stage) permutation cache (or the existing generate_lhs batch primitive) collapses to O(dim*total_scenarios), determinism preserved.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/out_of_sample.rs:92`, `crates/cobre-stochastic/src/sampling/class_sampler.rs:311`
- **Evidence:** For a single (iteration, scenario, stage) point of `dim` entities, sample_lhs_point performs `dim` complete Fisher-Yates shuffles over `n = total_scenarios` elements — O(dim * total_scenarios) per point — but extracts only one stratum index per dimension. Called once per scenario across the whole forward set, the per-stage cost is O(dim * total_scenarios^2). The permutation for a given dimension is identical for eve… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/out_of_sample.rs | sed -n '92,102p' ; git show a136840d:crates/cobre-stoch…`
- **Fix-shape:** Hoist the per-(iteration,stage) LHS permutation set across the scenario axis: build the `dim` stratification permutations once per (forward_seed, iteration, noise_group_id, total_scenarios) into a per-stage cache (a stride-dim*n usize buffer, or a batch already produced by the tree-noise `generate_lhs` batch path), then each scenario reads its stratum column with no reshuffle. The draw RNG stays per-(scenario,dimension) so stratified values remain distinct. Owning the permutation cache is a sampler-side concern in this sub-station; the batch permutation primitive lives in tree-noise (cross-reference the existing generate_lhs batch generator). Determinism preserved: perm_rng seeding is unchanged, so the same permutation and the same stratum-per-scenario are produced bit-for-bit.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-026 · Sev B · duplication · effort M · confidence high**
Narrowed: the confirmed defect is a pure eliminable-work perf issue -- the stage-invariant per-entity position resolution (entity_order.iter().position at resolve.rs:428) recomputed n_openings x n_groups times per stage because resolve_class_positions has zero callers so class_positions stays None; I concede the scan output is numerically correct and bit-identical to the precomputed/full-vector path (proven by test_per_class_tree_matches_full_vector_*), so this is not a correctness defect, and the fix requires either &mut DecomposedCorrelation plumbing into generate_opening_tree or build-time resolution.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/generate.rs:327`, `crates/cobre-stochastic/src/correlation/resolve.rs:390`, `crates/cobre-stochastic/src/correlation/resolve.rs:277`
- **Evidence:** resolve_class_positions is the ONLY writer of GroupFactor.class_positions, and git grep finds nothing but its definition and two doc-comment references crate-wide -- no call site. So in production gf.class_positions is always None, and apply_correlation_for_class (resolve.rs:338), called three times per opening from generate_opening_tree's inner loop (generate.rs:332-334, inside `for opening_idx in 0..n_openings` at… Re-derive: `git grep -n 'resolve_class_positions' a136840d4f2ea137f685f0af6dac04254b983b60 -- '*.rs'`
- **Fix-shape:** Hoist entity-position resolution to once per tree. Either give generate_opening_tree a &mut DecomposedCorrelation and call resolve_class_positions once per class (inflow_order/load_order/ncs_order) before the stage loop so every opening lands on apply_group_precomputed; or resolve positions at correlation/context build time so the DecomposedCorrelation arrives pre-resolved. Preserve the existing per-class reassembly and declaration order exactly -- the output must stay bit-for-bit identical. Do not name any engine paradigm noun; the seam stays a generic position cache on the correlation type.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-027 · Sev B · missing-seam · effort M · confidence high**
Narrowed to the live path: at baseline only apply_group_scan's n>64 branch (resolve.rs:426/436-437, three transient Vecs) is actually reached from the tree loop -- apply_group_precomputed is unreachable in production because class_positions stays None (per perf-00), so its n>64 allocation (resolve.rs:380-381) is latent, not currently on any production path; the confirmed residue is the per-opening transient scratch allocation in apply_group_scan for any correlation group exceeding MAX_STACK_DIM=64.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:379`, `crates/cobre-stochastic/src/correlation/resolve.rs:425`, `crates/cobre-stochastic/src/tree/generate.rs:332`
- **Evidence:** MAX_STACK_DIM is 64 (resolve.rs:18). Both application paths take a stack fast-path only for n <= 64; for any group whose dimension exceeds 64 they fall to a heap branch that allocates a fresh `gathered: Vec<f64>` plus `correlated = vec![0.0; n]` (apply_group_precomputed:380-381), and the scan path additionally allocates a `positions: Vec<usize>` (apply_group_scan:426,436-437). These functions are called once per ope… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/correlation/resolve.rs | sed -n '379,387p;424,438p'`
- **Fix-shape:** Give the apply functions caller-owned reusable scratch: allocate gathered/correlated (and, for the scan path, positions) once, sized to the maximum group dimension, in generate_opening_tree before the stage/opening loops and thread them in, or hold them on a per-tree workspace. Keep the small-group stack fast path unchanged. Numerics are untouched -- only the buffer lifetime changes.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path; deferred pending a profile) — queued to the performance sweep (see `perf-queue.json`).
- **Queued to:** performance-sweep

**PD-028 · Sev C · duplication · effort S · confidence high**
The eliminable cost is precisely the stage-constant profile+slice resolution (the profile_for_stage HashMap probe plus the factors.get BTreeMap<String> string-keyed walk) repeated 3 x n_openings per stage; I concede the per-group entity_type filter and the spectral transform must stay per-call/per-group -- only the profile-name and group-factor-slice lookup is hoistable to once per stage, so the defect is repeated map traversal, not the per-opening application itself.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:345`, `crates/cobre-stochastic/src/tree/generate.rs:332`
- **Evidence:** profile_for_stage does a HashMap<i32,String>::get (resolve.rs:246), then factors.get(profile_name) is a BTreeMap<String,_>::get that walks the tree doing String comparisons. generate_opening_tree calls apply_correlation_for_class three times per opening (generate.rs:332-334, inflow/load/ncs) inside `for opening_idx in 0..n_openings`, with a stage_id that is constant for the whole inner loop. So the same profile + gr… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/correlation/resolve.rs | sed -n '345,351p'`
- **Fix-shape:** Resolve the &[GroupFactor] slice for the stage once, before the opening loop in generate_opening_tree (e.g. a per-stage accessor on DecomposedCorrelation that returns the group-factor slice for a stage_id), then iterate openings against that borrowed slice, filtering by entity_type in the loop. Folds naturally into candidate 1's once-per-stage/tree resolution.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path) — recorded as deferred debt, below the Sev-A/B performance-sweep threshold.

**PD-029 · Sev C · duplication · effort M · confidence high**
The quadratic is real but bounded to a single setup-time construction (StudySetup::new, once per study) and to l_state = max PAR lag order (a small bounded model parameter, not n_scenarios/n_stages/n_iterations), so the confirmed residue is a low-magnitude O(n_hydros*l_state^2*S) setup-time redundancy fixable by walking the occurrence chain once per hydro, never a hot-path or per-iteration cost.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/seeds.rs:104`, `crates/cobre-stochastic/src/season_cast/mod.rs:615`, `crates/cobre-stochastic/src/season_cast/mod.rs:302`
- **Evidence:** derive_inflow_seeds calls season_occurrence(k) for k = 0..=l_state (seeds.rs:95 and the k-loop at seeds.rs:104). Each season_occurrence(k) recomputes the in_progress anchor and then calls nth_previous_occurrence(anchor, k), whose body (season_cast/mod.rs:313) walks the backward occurrence chain k steps FROM THE ANCHOR every time. So step 1..k-1 are re-walked once for every k: total walk steps are Sum(k, k=1..l_state… Re-derive: `git show a136840d:crates/cobre-stochastic/src/seeds.rs | sed -n '104,111p'; git show a136840d:crates/cobre-stochastic/src/season_…`
- **Fix-shape:** Walk the backward season-occurrence chain once per hydro instead of restarting per k. Add a StageCalendar method (or a season_cast free helper) that, given the anchor and a max depth, yields the sequence of SeasonPeriodWindow occurrences 0..=l_state by advancing previous_season_period_window incrementally, so each occurrence and its O(S) season lookup is computed exactly once; derive_inflow_seeds then casts each returned window onto merged. This stays engine-neutral (no cut/state/Benders noun), introduces no one-consumer abstraction beyond the existing single caller, and preserves the exact walk semantics (same previous_season_period_window / season_for_date sequence) so seed values stay bit-identical.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path) — recorded as deferred debt, below the Sev-A/B performance-sweep threshold.

**PD-030 · Sev C · duplication · effort S · confidence high**
Real O(n_hydros*(|record|+|conditioning|)) re-scan, but confined to the one-per-study StudySetup::new construction (never a per-scenario/stage/iteration path); the confirmed residue is a setup-time overscan fixable by a single group-by-hydro_id pass, and any fix must preserve the returned canonical-order positional layout bit-for-bit.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/seeds.rs:73`
- **Evidence:** record (the whole-study inflow_history) and conditioning (recent_observations) hold rows for ALL hydros. The hydro loop (seeds.rs:73) filters each full slice by row.hydro_id == hydro.id (seeds.rs:76 and 85) for every hydro, so each of the two slices is scanned n_hydros times. This is O(n_hydros * (|record| + |conditioning|)); a single grouping pass keyed by hydro_id built before the loop would be O(|record| + |condi… Re-derive: `git show a136840d:crates/cobre-stochastic/src/seeds.rs | sed -n '73,92p'`
- **Fix-shape:** Before the hydro loop, do one pass over record and one over conditioning to bucket rows by hydro_id (e.g. a HashMap<EntityId, Vec<RealizedWindow>>), then index each hydro's bucket inside the loop. Keep the existing hydros-canonical-order iteration for the returned position, and preserve the row order within each bucket so merge_layered_windows sees the same window sequence and seeds stay bit-identical. Engine-neutral, no new cross-crate dependency, single existing consumer.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path) — recorded as deferred debt, below the Sev-A/B performance-sweep threshold.

#### Over-engineering

**OD-026 · Sev C · speculative-generality · effort S · confidence high**
Narrowed: the fatal check inside validate_par_parameters (validation.rs:120, the zero-std/ar_order>0 error propagated by `?`) is live and load-bearing at context.rs:616 and must stay. The confirmable residue is only the WARNING side — the single-variant ParWarning enum plus the warnings: Vec<ParWarning> field of ParValidationReport — which is computed at validation.rs:138 then discarded by the `let _report =` bind at the sole production caller, with no production reader. Any surfacing of the warning would trigger the Python-parity rule (an owner call).

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/validation.rs:54`, `crates/cobre-stochastic/src/par/validation.rs:39`, `crates/cobre-stochastic/src/par/validation.rs:117`
- **Evidence:** `ParWarning` has exactly one variant (`LowResidualVariance`) and `ParValidationReport` is a one-field wrapper `{ warnings: Vec<ParWarning> }` — a structured warning taxonomy with a single inhabitant, the textbook wrapper-enum-with-one-inhabitant shape. The apparatus is production-unread: `git grep -nE '\bvalidate_par_parameters\b' a136840d -- 'crates/**/*.rs'` shows the sole production caller is `crates/cobre-stocha… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/par/validation.rs | sed -n '38,66p'`
- **Fix-shape:** Collapse the single-inhabitant taxonomy: either (a) if the low-residual-variance diagnostic has no intended reader, drop the `warnings` accumulation and return the fatal-check outcome directly (the sole caller already discards it), retiring `ParWarning`, `ParValidationReport`, and the two re-exports; or (b) if the warning is meant to surface, replace the one-variant enum with the bare `LowResidualVariance { hydro_id, stage_id, explained_variance }` struct and actually consume it at context.rs (which, under the Python-parity rule, would require mirroring the surfaced warning in cobre-python — an owner call, not a drive-by). Keep the enum form only if a second warning variant is genuinely imminent, in which case record it as an intended extension seam rather than leaving it as silent debt. No L1-purity conflict: the fix introduces no engine noun, no engine-crate dependency, and removes su…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**OD-027 · Sev C · redundant-wrapper · effort S · confidence high**
Narrowed: the confirmable residue is that all three bare forms have zero production callers (only crate tests + doctests exercise them) and estimate_ar_coefficients/estimate_seasonal_stats are additionally re-exported at the crate root (lib.rs:46-47) on top of the par::fitting:: path. I concede a single documented no-season-map convenience overload is a defensible public-API ergonomics choice for an L1 reusable kernel, so the defect is the unmarked, production-unconsumed redundancy of the trio (and its intent should be marked if kept), not that the functions are dead weight to delete outright.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/ar_coefficients.rs:88`, `crates/cobre-stochastic/src/par/fitting/correlation.rs:115`, `crates/cobre-stochastic/src/par/fitting/seasonal_stats.rs:221`
- **Evidence:** Each bare form is a pure forwarding shim: `estimate_ar_coefficients` (ar_coefficients.rs:88) and `estimate_seasonal_stats` (seasonal_stats.rs:221 -> `estimate_seasonal_stats_with_season_map(.., None)`) and `estimate_correlation` (correlation.rs:115 -> `estimate_correlation_with_season_map(.., None)`) do nothing but call their `_with_season_map` twin with `None`. Two of the three are re-exported at the crate root (li… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/par/fitting/ar_coefficients.rs | sed -n '88,101p'`
- **Fix-shape:** Remove the ergonomic overload asymmetry: since `_with_season_map` already takes `Option<&SeasonMap>` and is the sole production entry, either demote the three bare forwarders to `pub(crate)` (or delete them and have the inline tests/doctests call the `_with_season_map` form with `None` directly), and stop re-exporting `estimate_ar_coefficients`/`estimate_seasonal_stats` at the crate root. If the no-season-map convenience form is deliberately part of the crate's public API, mark that intent explicitly and keep exactly one such surface rather than a crate-root re-export plus a `par::fitting::` re-export of a test-only shim. Behaviour-neutral (the shims already delegate); blast radius is validation-free (the shim bodies and their test/doctest call sites).
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**OD-028 · Sev B · over-parameterization · effort M · confidence high**
Representability only: the config permits exactly two invalid/redundant states — (a) a class scheme selects External/Historical but the paired Option is None (caught only at runtime), and (b) a library is Some while the class scheme never reads it (silent dead input) — and the hard-wired historical_library:None for load/ncs shows the four Options are not independent; folding each class's scheme+library into a per-class source enum makes both states unrepresentable. No claim of a runtime bug: the MissingScenarioSource check is correct today.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs:263`, `crates/cobre-stochastic/src/sampling/mod.rs:317`
- **Evidence:** The four Options are not four independent knobs. build_forward_sampler pairs class_schemes.inflow with EITHER historical_library OR external_inflow_library (mutually exclusive by inflow scheme), and hard-wires historical_library: None for the load and ncs classes so they read only external_load_library / external_ncs_library. Validity is a (3 class x scheme) matrix flattened into 4 loosely-typed Options: each field… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/mod.rs | sed -n '263,289p;359,388p'`
- **Fix-shape:** Fold each class's scheme selector and its required library into one per-class source enum whose data-bearing variants carry the borrow: e.g. InflowSource { InSample, OutOfSample, Historical(&HistoricalScenarioLibrary), External(&ExternalScenarioLibrary) } and a LoadSource / NcsSource with only { InSample, OutOfSample, External(&ExternalScenarioLibrary) } (load and ncs never take a historical library, so the hard-wired historical_library: None disappears). build_forward_sampler then matches per-class variants instead of pairing a scheme with a maybe-present Option, so the scheme-to-library requirement becomes a compile-time consequence, the InSample-with-library silent-ignore state stops being representable, and the two mutually-exclusive inflow libraries can no longer both be Some. The MissingScenarioSource diagnostics move up to the config-construction boundary (where a library is or i…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**OD-029 · Sev B (A-risk) · over-parameterization · effort L · confidence high**
The rationale's "no natural sub-grouping exists" clause is false — the four are a documented atomic seed reset together at each outer boundary and re-threaded verbatim through >=6 signatures; the defensible residue is the silent same-typed transposition hazard between derived_accum and derived_weight (both &[f64], both length n_hydros, adjacent positional args) plus the reset-together invariant restated across >=3 doc blocks, which a by-ref DerivedSeed aggregate removes. The #[allow] itself stays (all three functions remain >7 args), so this is not an eliminate-the-lint finding.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/external.rs:262`, `crates/cobre-stochastic/src/sampling/eta_inversion.rs:37`, `crates/cobre-stochastic/src/sampling/historical.rs:296`
- **Evidence:** The rationale asserts the four seed inputs (derived_lag_values, l_state, derived_accum, derived_weight) have 'no natural sub-grouping'. The code contradicts that: the same 4-tuple is re-declared verbatim as positional parameters in run_eta_inversion (eta_inversion.rs:37, passed straight through) and in standardize_historical_windows (historical.rs:296, whose own comment at line 292 says it 'mirrors standardize_exter… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/external.rs | sed -n '256,273p'`
- **Fix-shape:** Introduce a by-reference aggregate for the stage-0 derived seed, e.g. DerivedSeed<'a> { lag_values: &'a [f64], l_state: usize, accum: &'a [f64], weight: &'a [f64] }, constructed once where the seed is computed and passed as a single argument through standardize_external_inflow, run_eta_inversion, standardize_historical_windows, and the caller build_external_inflow_library. The shared per-hydro canonical-position ordering and 'empty accum means reset-to-zero' invariant, currently restated in three near-identical doc-comment blocks, attaches to the type once. This shrinks the 11-arg / 14-arg / 12-arg signatures and removes the same-typed &[f64] transposition hazard. This does NOT dispute the mirror's blanket sanction of allow-with-rationale as a load-bearing lint class; it disputes only the factual claim of this specific rationale. Once the seed is a struct the #[allow(too_many_arguments)…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**OD-030 · Sev C · speculative-generality · effort S · confidence high**
Narrowed: the defensible residue is only that SweepDirection::Ascending and its comparator arm (opening_tree.rs:168) have ZERO non-test constructors while every production path passes Descending -- an unwired second variant exercised solely by opening_tree.rs's own tests. I concede this does NOT establish the enum must be deleted: SweepDirection is a minimal, engine-neutral two-variant sort-direction API (not a speculative multi-variant fan-out), so keeping a two-way ordering knob on a generic L1 primitive is defensible; the finding is the unwired variant, not that the `direction` parameter is itself over-abstraction.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/opening_tree.rs:14`, `crates/cobre-stochastic/src/tree/opening_tree.rs:120`
- **Evidence:** SweepDirection has two variants but only Descending is ever selected outside tests. The two Ascending constructions (opening_tree.rs:662,676) sit above the sole #[cfg(test)] boundary at line 438, so they are unit-test-only. The single production caller in cobre-sddp/src/setup/mod.rs:469 passes Descending, and the only other production reference (cobre-sddp/tests/parity.rs) is also test code passing Descending. The A… Re-derive: `git grep -n 'SweepDirection' a136840d4f2ea137f685f0af6dac04254b983b60 -- crates/ | grep -vE 'lib.rs|tree/mod.rs|context.rs:60'; e…`
- **Fix-shape:** Collapse the speculative generality: since every production caller sorts largest-key-first, drop the SweepDirection enum and the direction parameter on set_solve_order, making the primitive sort descending unconditionally, and remove the now-dead SweepDirection::Descending argument from the one production caller (cobre-sddp/src/setup/mod.rs:469) plus its test mirrors (cobre-sddp/tests/parity.rs). The Ascending-only unit tests then either delete or fold into a private test helper that exercises the comparator directly, so no production surface exists solely to be tested. If instead the ascending ordering is a deliberately reserved seam for a planned selection policy, keep it but add an owner-signed reservation note (the CLAUDE.md 'unwired config is reserved, not dead' pattern) at the enum so the unwired variant reads as reserved rather than as silent speculative generality. Either way ke…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

**OD-031 · Sev C · speculative-generality · effort S · confidence high**
Confirmed as three pub StochasticError variants with zero production construction site (not compiler-dead: each is Display/Debug-exercised by inline tests and exhaustively matched); the sharper residue is that SpectralDecompositionFailed is additionally a live doc lie at context.rs:593 (promising a return spectral.rs never makes, since it maps failures to InvalidCorrelation), whereas SeedDerivationError and UnsupportedSamplingScheme are merely never-constructed variants removable with no match-arm edit.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/error.rs:5`, `crates/cobre-stochastic/src/error.rs:20`, `crates/cobre-stochastic/src/error.rs:45`, `crates/cobre-stochastic/src/error.rs:74`, `crates/cobre-stochastic/src/context.rs:593`
- **Evidence:** Of the 9 StochasticError variants, three have zero production construction sites workspace-wide. SpectralDecompositionFailed is never returned — every spectral-decomposition failure in correlation/spectral.rs returns InvalidCorrelation (spectral.rs:83, spectral.rs:99), yet context.rs:593 documents build_stochastic_context as returning SpectralDecompositionFailed, a promise no code path keeps. SeedDerivationError is… Re-derive: `for v in SpectralDecompositionFailed SeedDerivationError UnsupportedSamplingScheme; do echo "== $v =="; git grep -n "$v" a136840d…`
- **Fix-shape:** Two roads, owner's pick. (a) Delete the three never-produced variants from the pub StochasticError enum, delete the context.rs:593 doc bullet that promises SpectralDecompositionFailed, and retarget the cobre-sddp from_stochastic_error test (cobre-sddp/src/error.rs:241, out of station — cross-reference only) to a live variant such as InsufficientData. (b) If a producer is genuinely planned for any of them, enter that variant in the reserved-seam register with an owner and a consuming milestone, which the register's own contract requires and which none of the three currently has. Blast radius is bounded: error.rs enum + one context.rs doc line + one cobre-sddp test. Removing variants is a breaking change to the pub StochasticError (re-exported at lib.rs `pub use error::StochasticError`), so it rides the same 'licensed public-API break' gate the mirror applies to the superseded cut-sync pu…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)

#### Test bloat

**TD-024 · Sev B · duplication · effort M · confidence high**
Narrowed: the confirmable byte-identical duplication is make_bus (forward_sampler.rs:61 == conformance.rs:23) and make_hydro (forward_sampler.rs:74 == conformance.rs:61) across the two in-scope binaries, with no tests/common module and no test-support feature to home them. I concede identity_correlation (forward_sampler.rs:161 vs conformance.rs:130) is a near-duplicate differing by the `ids` vs `entity_ids` param name, not byte-identical; and the wider halton/sobol/lhs/reproducibility/saa_golden_value prelude spread and the crate-wide integration-binary total are out of this sub-station (owned by tree-noise and the test-corpus station per the scope walls and the prior-register Oracle cross-reference), so the surviving claim is scoped to the forward_sampler<->conformance make_bus/make_hydro pair.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/tests/forward_sampler.rs:74`, `crates/cobre-stochastic/tests/forward_sampler.rs:61`, `crates/cobre-stochastic/tests/forward_sampler.rs:161`, `crates/cobre-stochastic/tests/conformance.rs:61`, `crates/cobre-stochastic/tests/conformance.rs:23`, `crates/cobre-stochastic/tests/conformance.rs:130`
- **Evidence:** The two par integration binaries in scope re-declare the same entity-construction prelude verbatim; make_stage is the only per-family divergence (forward_sampler `(index,id,bf,method)` vs conformance `(index,id,branching_factor)`), matching the yardstick's observation that only make_stage_* differs per family. There is no crate-local tests/common and no test-support feature, so the prelude is copy-pasted per file. A… Re-derive: `diff <(git show a136840d:crates/cobre-stochastic/tests/forward_sampler.rs | sed -n '74,122p') <(git show a136840d:crates/cobre-st…`
- **Fix-shape:** Introduce a single shared fixture surface for the crate's cobre-core entity builders and mod-declare it once per binary, so make_bus/make_hydro/make_inflow_model/identity_correlation are authored once. Roadmap-consistent options: (a) low-risk interim — a crate-local `tests/common/` module (the yardstick §5.1 interim state before test-support collapse), each binary `mod common;` then `use crate::common::*`; (b) the yardstick's target — hoist the cobre-core entity builders behind a `test-support` cargo feature so the builders live with the type they build (§5.2). Because these construct cobre-core types (Hydro/Bus/InflowModel/CorrelationModel), cobre-core's test-support surface is the durable home; the crate-local tests/common is the safe first step. Keep the per-family make_stage variants as thin overrides. Neutral to the layering (test infra only); introduces no new production abstracti…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-025 · Sev B · duplication · effort S · confidence high**
Narrowed: the provable byte-identical duplication is specifically the make_model pair (precompute.rs:653 == evaluate.rs:515, empty diff) and the make_stage pair (precompute.rs:628 vs evaluate.rs:490) differing only by the dummy_date wrapper. The broader '~9 par test sites' fragmentation claim is softer: the validation.rs:160 make_model hardcodes mean_m3s:100.0 and aggregate.rs:169 make_stage uses duration 720.0/branching_factor 1 vs precompute/evaluate's 744.0/10, so those copies are structurally-similar-but-divergent (different default literals), not byte-identical dupes — the field-add-fragmentation cost is real but the strictly-identical residue is the precompute<->evaluate make_model/make_stage pair.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/precompute.rs:653`, `crates/cobre-stochastic/src/par/evaluate.rs:515`, `crates/cobre-stochastic/src/par/precompute.rs:628`, `crates/cobre-stochastic/src/par/evaluate.rs:490`, `crates/cobre-stochastic/src/par/validation.rs:160`, `crates/cobre-stochastic/src/par/aggregate.rs:169`
- **Evidence:** Every par unit-test module rebuilds the same two cobre-core structs from scratch. precompute and evaluate carry a genuinely identical make_model; the make_stage copies differ only in default constants, not shape. There is no shared O(1)-field-add Stage/InflowModel builder in this crate (it exists only in cobre-sddp/tests/common), so a field added to Stage or InflowModel must be edited in every copy — the fixture-fra… Re-derive: `diff <(git show a136840d:crates/cobre-stochastic/src/par/precompute.rs | sed -n '653,669p') <(git show a136840d:crates/cobre-stoc…`
- **Fix-shape:** Provide one parameterized builder each for Stage and InflowModel (the O(1) field-add builder pattern the yardstick §3.1.4 credits to cobre-sddp/tests/common/builders.rs), reachable from both the inline unit tests and the integration binaries via the crate test-support surface proposed in the companion finding; each existing make_stage/make_model becomes a thin wrapper that overrides only the defaults it cares about (duration, branching_factor, mean). Delete the byte-identical make_model copy pair outright. Keeps every test body and assertion unchanged (a refactor, not a coverage change); neutral to layering; no new production abstraction.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-026 · Sev C · duplication · effort S · confidence high**
Narrowed: the confirmable residue is the maintenance/drift hazard of a duplicated 35-line deterministic PAR(2) generator across two sibling files (tests.rs:2032 and estimation/tests.rs:149), whose fix hoists one copy to a shared #[cfg(test)] par/fitting helper. I concede the divergence is currently latent, not actual: at the baseline the two bodies are byte-identical modulo the cosmetic lcg/lcg_state rename, so no golden or assertion discrepancy exists today — the surviving claim is the drift risk, not a present numeric bug.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/tests.rs:2032`, `crates/cobre-stochastic/src/par/fitting/estimation/tests.rs:149`
- **Evidence:** A non-trivial deterministic PAR(2) generator (the seed source for the roundtrip and PACF analytical tests in both files) is maintained in two places that differ only by a cosmetic rename; a change to the noise generator in one silently diverges from the other, weakening the shared-seed contract both rely on. Re-derive: `diff <(git show a136840d:crates/cobre-stochastic/src/par/fitting/tests.rs | sed -n '2032,2066p') <(git show a136840d:crates/cobre…`
- **Fix-shape:** Hoist the one simulator into a single shared par/fitting test helper that both sibling files import (a `#[cfg(test)]` helper module under par/fitting/, or the crate test-support surface if the companion findings introduce it), and delete the copy. Behavior-neutral; the generated series is identical by construction so no golden/assertion changes. Neutral to layering.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-027 · Sev C · duplication · effort S · confidence high**
Narrowed: the confirmable residue is the byte-identical 11-line population mean/std helper duplicated within a single file — pop_mean_std (tests.rs:1157) and pop_mean_std_ann (tests.rs:2354) — both inside the crate's own inline #[cfg(test)] module. It is a minor (severity-C) in-file test-hygiene dedup with zero behavior divergence, no coverage change, and no production or public-API impact; the fix is a trivial single-file edit repointing the `_ann` call sites at pop_mean_std.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/tests.rs:1157`, `crates/cobre-stochastic/src/par/fitting/tests.rs:2354`
- **Evidence:** Two identically-bodied helpers coexist in one file, separated only by an `_ann` suffix on the name; the `_ann` copy adds no annual-specific behavior. Pure duplication with zero divergence — one helper serves both call sites. Re-derive: `diff <(git show a136840d:crates/cobre-stochastic/src/par/fitting/tests.rs | sed -n '1157,1168p') <(git show a136840d:crates/cobre…`
- **Fix-shape:** Delete pop_mean_std_ann and point its call sites at the single pop_mean_std (or, if these move under the consolidated par/fitting test helper, keep one copy there). Behavior-neutral; identical output. Neutral to layering.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-028 · Sev B · duplication · effort M · confidence high**
Narrowed to the verified mechanical duplications — the 3x byte-identical uniform_tree, the 2x one-cosmetic-line make_hydro (external/historical) and the 2x identical dated_stage — as the consolidation-worthy core; mod.rs's make_hydro is a divergent variant (2024 + mirror-unit-group) and the ~9 Stage builders take different parameter shapes, so the byte-identical claim holds only for the enumerated set and the broader hand-rolled-Stage boilerplate is a weaker structural-similarity claim.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/external.rs:2513`, `crates/cobre-stochastic/src/sampling/historical.rs:1902`, `crates/cobre-stochastic/src/sampling/mod.rs:670`, `crates/cobre-stochastic/src/sampling/class_sampler.rs:446`, `crates/cobre-stochastic/src/sampling/mod.rs:791`, `crates/cobre-stochastic/src/sampling/insample.rs:56`
- **Evidence:** Three genuinely-shared fixtures are re-authored per file inside one sub-station. uniform_tree (builds a cobre_stochastic OpeningTree) is copied verbatim 3x; monthly_season_map (builds a cobre_core SeasonMap) 3x with 2 byte-identical; make_hydro (a 46-line cobre_core Hydro literal that spells out all 15 HydroPenalties fields) 3x differing only cosmetically; dated_stage 2x verbatim. Beyond these named helpers, the Sta… Re-derive: `for f in external historical class_sampler mod window insample; do git show a136840d:crates/cobre-stochastic/src/sampling/$f.rs |…`
- **Fix-shape:** Hoist each shared builder to the crate that owns the type it constructs, behind a test-support feature, per testing-architecture.md 5.2 (helpers live with the type they build). The cobre_core entities - make_hydro (Hydro/HydroPenalties), the Stage builders and dated_stage (Stage/StageStateConfig/ScenarioSourceConfig), monthly_season_map (SeasonMap) - belong in cobre-core's test-support surface (the universal base dependency), parameterized so a caller overrides only the field it cares about (an O(1)-field-add builder), which collapses the ~9 Stage literals and the 46-line Hydro literal to one definition each. uniform_tree builds a cobre_stochastic OpeningTree and has three in-crate consumers, so it homes in cobre-stochastic's own test-support surface (three consumers, so not the single-consumer abstraction the L1 purity guardrail forbids). No SDDP paradigm noun and no engine-crate depen…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-029 · Sev C · duplication · effort M · confidence high**
Narrowed to the single verified byte-identical duplication — saa_golden_value.rs's 28-line identity_correlation copy of the QMC binaries' fixture; 're-declares the integration-binary fixture prelude' over-reaches because saa carries only this one of the ~8 shared prelude helpers and its make_stage is a distinct 3-arg builder, so identity_correlation is the only byte-identical copy.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/tests/saa_golden_value.rs:53`, `crates/cobre-stochastic/tests/saa_golden_value.rs:32`
- **Evidence:** saa_golden_value.rs (my thematically-sampling integration binary, 139 lines) carries a verbatim copy of the same identity_correlation fixture the tree-noise QMC integration binaries (halton/sobol/lhs_integration.rs) each declare, plus its own Stage builder. This is the integration-binary layer of the same fixture-fragmentation smell as candidate 1: the near-verbatim prelude the station test-bloat lens names for halt… Re-derive: `for f in saa_golden_value halton_integration sobol_integration lhs_integration; do echo --- $f; git show a136840d:crates/cobre-st…`
- **Fix-shape:** When the shared integration prelude is consolidated (testing-architecture.md 5.2/5.1 - a per-crate test-support surface plus the single integration binary), saa_golden_value.rs drops its private identity_correlation and make_stage and pulls both from cobre-stochastic's test-support fixtures alongside the QMC binaries. saa's own contribution is small (one 28-line duplicate + one Stage builder); the bulk of the integration-prelude debt is the tree-noise QMC-prelude headline, so this candidate is scoped to the saa binary and defers the cross-binary consolidation to that cell's finding. No behavior change: the six pinned golden constants and their assertions are untouched.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-030 · Sev B · duplication · effort M · confidence high**
Narrowed: the confirmed residue is strictly the byte-identical eight-helper fixture prelude physically copied across the QMC trio (halton/sobol/lhs), with make_hydro also copied into reproducibility (verified) and no tests/common/ home, already drifting into two identity_correlation return types; I concede the ~50 genuinely per-family lines (make_stage_<family> and build_<family>_context, differing only in NoiseMethod variant / block presence / panic-string) are legitimate and must survive any consolidation as thin per-family wrappers -- the finding is the shared prelude only, not the whole file, and it is distinct from (cites, does not restate) the cobre-sddp Oracle-harness mirror entry.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/tests/halton_integration.rs:207`, `crates/cobre-stochastic/tests/sobol_integration.rs:207`, `crates/cobre-stochastic/tests/lhs_integration.rs:206`, `crates/cobre-stochastic/tests/halton_integration.rs:43`, `crates/cobre-stochastic/tests/halton_integration.rs:105`
- **Evidence:** The two diffs return empty (exit 0), so the eight fixture helpers -- approx_erf (h/s/lhs L43-51), norm_cdf (L53-55), identity_correlation (h/s L105-133, lhs L104-132), correlated_correlation (h/s L135-163, lhs L134-162), identity_correlation_model (h/s L165-192, lhs L164-191), make_bus (h/s L194-205, lhs L193-204), make_hydro (h/s L207-255, lhs L206-254; a single 49-line copy), make_inflow_model (h/s L257-267, lhs L… Re-derive: `bash -c 'diff <(git show a136840d:crates/cobre-stochastic/tests/halton_integration.rs | sed -n "41,55p;105,267p") <(git show a136…`
- **Fix-shape:** Hoist the shared fixture prelude out of the per-binary copies into one home. The crate already dev-depends on cobre-core with its `test-support` feature but exposes NO `test-support` feature of its own and has NO tests/common/. Two roadmap-consistent shapes: (a) interim -- add crates/cobre-stochastic/tests/common/mod.rs holding make_bus/make_hydro/make_inflow_model/identity_correlation/correlated_correlation/identity_correlation_model plus the numeric approx_erf/norm_cdf, and `mod common;` it once from each integration binary (mirrors cobre-sddp/tests/common/); (b) end-state per the yardstick §5.2 -- expose the entity/InflowModel/correlation builders behind a cobre-stochastic `test-support` feature gated `#[cfg(any(test, feature = "test-support"))]`, and push the deck-agnostic numeric helpers (approx_erf/norm_cdf) into cobre-core's `test-support` surface (the yardstick's rule that gener…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-031 · Sev C · asymmetry · effort M · confidence high**
Narrowed: no RATIFIED homing threshold exists at baseline -- testing-architecture.md section 5 is a 'Proposed standard' and section 5.1's ~500-test-LOC/~40-fn line is explicitly a 'proposal' -- so the surviving claim asserts only (a) the verified intra-crate inconsistency (inline vs extracted sibling coexist with no deciding rule) and (b) the extreme outlier tree/generate.rs at ~2030 inline test-LOC versus the already-extracted par/fitting form; it is a Sev C uniformity/navigability smell resolvable by relocation, NOT evidence that any specific numeric threshold is the correct cut nor a coverage/bloat defect.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/generate.rs:341`, `crates/cobre-stochastic/src/correlation/resolve.rs:446`, `crates/cobre-stochastic/src/normal/precompute.rs:299`, `crates/cobre-stochastic/src/tree/opening_tree.rs:438`, `crates/cobre-stochastic/src/correlation/spectral.rs:286`
- **Evidence:** Within this one crate the unit-test homing rule is a coin-flip -- exactly the yardstick's §3.2.4 (inline-giant vs extracted-sibling asymmetry) and §5.1 (a single deterministic homing threshold, proposal ~500 test-LOC / ~40 test fns). tree/generate.rs keeps ~2030 test-LOC INLINE (4x over the proposed threshold), correlation/resolve.rs ~531 and normal/precompute.rs ~400 inline, while par/fitting extracts its tests to… Re-derive: `bash -c 'for f in tree/generate.rs tree/opening_tree.rs correlation/resolve.rs normal/precompute.rs correlation/spectral.rs noise…`
- **Fix-shape:** Adopt the yardstick §5.1 deterministic homing rule as a crate-wide lint: unit tests stay inline below a fixed threshold (the proposal is ~500 test-LOC or ~40 test fns), extracted to a sibling `tests.rs` above it. Under that rule tree/generate.rs (~2030), correlation/resolve.rs (~531) move to `<module>/tests.rs` siblings (matching the already-extracted par/fitting form), while small modules (noise/rng.rs ~40, noise/seed.rs) stay inline. Pure relocation, no test body/assertion/tier/gate change, count-neutral -- resolves the intra-crate coin-flip without touching coverage.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-032 · Sev C · duplication · effort M · confidence high**
The duplication narrows to the byte-identical make_bus/make_stage/make_hydro/identity_correlation across context.rs, provenance.rs and tests/reproducibility.rs plus the fourth date-differing make_hydro in seeds.rs; it does NOT extend to make_inflow_model (a legitimate parameterized-superset vs fixed-value split) nor to seeds.rs's make_stage/season-map builders (distinct signatures), so the defensible residue is one shared entity-builder surface owed for exactly those four builders, coverage-neutral with no test deleted.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/context.rs:852`, `crates/cobre-stochastic/src/provenance.rs:96`, `crates/cobre-stochastic/tests/reproducibility.rs:68`, `crates/cobre-stochastic/src/seeds.rs:198`
- **Evidence:** Five entity builders recur across three seam test regions. Bodies were read in full at baseline (context.rs:793-960, provenance.rs:35-200, tests/reproducibility.rs:1-160): make_stage (context 814 / provenance 58 / reproducibility 43), make_bus (context 839 / provenance 83 / reproducibility 30), make_hydro (context 852 / provenance 96 / reproducibility 68) and identity_correlation (context 914 / provenance 158 / repr… Re-derive: `for f in src/context.rs src/provenance.rs src/seeds.rs tests/reproducibility.rs; do echo "== $f"; git show a136840d:crates/cobre-…`
- **Fix-shape:** Coverage-neutral de-duplication (no test deleted, per yardstick §7): hoist the shared engine-neutral entity builders into a single test-support surface and have all three seam test regions consume it. Yardstick-canonical home (§5.2 'helpers live with the type they build') is cobre-core's `test-support` feature, since make_bus/make_stage/make_hydro/make_inflow_model/identity_correlation construct cobre-core entities (Bus/Stage/Hydro/InflowModel/CorrelationModel) — the two inline `#[cfg(test)]` modules use them via `#[cfg(test)]` and the integration binary via a `test-support` dev-dependency feature. A lighter, L1-safe alternative that stays inside this crate is a crate-internal `#[cfg(any(test, feature = "test-support"))] mod test_support` in cobre-stochastic exposing the builders so tests/reproducibility.rs can reach them; it has 3+ consumers (both inline modules plus the integration bi…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

**TD-033 · Sev C · duplication · effort S · confidence high**
The test adds zero behavioral coverage beyond the existing helper-level unit tests (season_period_window/nth_previous_occurrence/cast each already tested at 687-986); its only residual function is to pin season_occurrence's delegation to those three helpers, an implementation-detail change-detector rather than a contract, so the residue is remove-or-reduce (not that the test is wholly inert: it would fire if the wrapper stopped delegating).

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/season_cast/mod.rs:1271`, `crates/cobre-stochastic/src/season_cast/mod.rs:615`
- **Evidence:** season_occurrence (615) IS the composition season_period_window -> nth_previous_occurrence -> cast over self.stages.first(). The test recomputes that exact same composition over the same inputs (expected_in_progress via season_period_window on stages[0]; the same &windows) and asserts it equals season_occurrence(...). Both sides invoke the identical delegating code path, so the assertion is f(x)==f(x); the test's ow… Re-derive: `git show a136840d:crates/cobre-stochastic/src/season_cast/mod.rs | sed -n '615,626p;1290,1308p'`
- **Fix-shape:** Remove the test (not a coverage reduction per yardstick §7: it asserts nothing the helper-level unit tests do not already prove, because the wrapper is a straight delegation). If the intent is to guard that the wrapper keeps delegating rather than growing an independent implementation, replace the redundant equivalence assertion with a one-line comment/pointer at season_occurrence stating that the helper-level tests are authoritative, or fold a single k-value smoke assertion into an existing calendar test. Do not keep a full 0..=3 loop whose two operands are the same call path.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus

#### Positives (recorded so the report is not a defect-only list)

- `scripts/ci/check-infra-genericity.sh` — cobre-stochastic is inside the infrastructure-genericity gate, so any SDDP/paradigm noun leaking into this L1 crate is a gate failure caught by `cargo test`, not a lens finding. (sanctioned by the CI gate membership; the one architecture substring hit, `cut_points` at `season_cast/mod.rs:439`, is a cut-set naming coincidence, not engine vocabulary — recorded, not raised)
- `crates/cobre-stochastic/src/season_cast/mod.rs` — `cobre-io` (L2) consumes `season_cast` / `StageCalendar` from this crate (L1) at four sites, a consistent higher-to-lower L2→L1 edge; the recorded stage-calendar relocation is therefore a core-vs-stochastic homing question, not a layering inversion. (sanctioned by plans/architecture-debt-audit/stations/stochastic/prior-register.md — `## Layering evidence`)
- Q1 paradigm-leakage grep over crates/cobre-stochastic/src/par (git grep -niE '(sddp|benders|cut|cost.to.go|state.space)' at baseline) — No genericity GATE failure. Every literal hit is a false positive: the 'cut' hits are the substring inside 'conseCUTive' (estimation.rs:97/107/112/114, lag_transition.rs:308), and the only real \bcut\b token ('pre-cut record row', lag_tran… (sanctioned by scripts/ci/check-infra-genericity.sh PATTERN (line 79) + tail-test-module truncation (awk '/^#[cfg(test)]/{exit}'))
- Dependency direction of the par sub-station — par imports only cobre_core (L0 — EntityId, InflowModel, Stage, SeasonMap, StageLagTransition, AnnualComponent), chrono, and rayon; no cobre-sddp / engine crate and no LP/solver crate appears in any production 'use'. The L1->L0 edge is lay… (sanctioned by target-layering-brief.md section 1 (L1 shared kernels depend downward on L0))
- Q4 Phase-1 (Switchable<T> uncertainty store) hint for par — The fitted-process representation carried by PrecomputedPar (built from cobre_core::scenario::InflowModel parameters) is the piece that would migrate into the Part-IV/V Switchable<T> uncertainty store when the stochastic model is purified… (sanctioned by target-layering-brief.md Part-I item 1 (I.3-1) and section 2 phase 1)
- crates/cobre-stochastic/src/par/fitting/{yw_matrices.rs,periodic_ar.rs,partitioned_covariance.rs} — The Yule-Walker / FACP primitives are already allocation-conscious: build_periodic_yw_matrix_into writes into caller buffers reused across the increasing-order solves of estimate_periodic_ar_coefficients (matrix_buf/rhs_buf), and condition… (sanctioned by in-code allocation-reuse rationale in periodic_ar.rs and partitioned_covariance.rs)
- crates/cobre-stochastic/src/par/fitting/estimation.rs:952 and correlation.rs:210 (the crate's only two rayon sites) — Both rayon regions carry a correct, load-bearing determinism rationale (canonical hydro_ids reassembly via flat_map_iter/collect and map/collect; inner per-season solve / ar_sum accumulation bit-identical to single-threaded). These are Voi… (sanctioned by in-code Determinism rationale (estimation.rs 947-951, correlation.rs 205-208))
- crates/cobre-stochastic/src/par/lag_kernel.rs — `LagIndex` trait with `LagMajor` and `EntityMajor` implementors driving the generic `advance_lag_chain<L: LagIndex>` — Not speculative generality: both implementors have real production consumers, so the trait is a genuine two-consumer abstraction (satisfies the target-layering 'second consumer precedes the abstraction' rule). `LagMajor` is used in product… (sanctioned by target-layering-brief.md §3.4 (a two-consumer abstraction is architecture, not a one-consumer smell))
- crates/cobre-stochastic/src/par/fitting/estimation.rs — private `#[allow(clippy::too_many_arguments)]` / `too_many_lines` fns `estimate_ar_with_pacf_annual` (line 300), `apply_annual_prepass_reductions` (line 523), `reduce_entity_orders_an… — Each is a private fn (not a builder, not public surface) carrying a D4 `// Rationale:` clause explaining that its arguments are independently-sourced lookup/stat tables spanned by no context struct and that bundling them would just displac… (sanctioned by CLAUDE.md comment rule D4 (rationale-above-suppression) + reserved-seams-and-deferred-debt.md '#[allow(...)] census' Load-bearing class)
- crates/cobre-stochastic/src/par/mod.rs:27-31 — the `#[allow(deprecated)]` re-export block over `evaluate.rs`'s `evaluate_par*` / `solve_par_noise*` surface (3 items carry `#[deprecated(since = 0.1.2)]` at evaluate.rs:109,240,452) — Cleared, not raised: the deprecated-with-fallback surface and its `#[allow(deprecated)]` sites are explicitly declared out of scope for this sweep, with a dedicated clean-break ticket owning the disposition. Recorded here as a sanctioned o… (sanctioned by reserved-seams-and-deferred-debt.md 'Audit-evidence -> #[allow(...)] census' (the `#[allow(deprecated)]` sites in cobre-stochastic `{lib.rs,par/mod.rs,par/evaluate.rs}` are a narrower class; verifyin…)
- crates/cobre-stochastic/src/par/fitting/tests.rs — analytical (MMS-style) tier: *_hand_computed, *_analytical_verification, YW-matrix residual/roundtrip and rhs-matches-extended checks (e.g. build_periodic_yw_matrix_forward_prediction_two_… — These prove correctness against a hand-derived truth, not 'same as last time' — exactly the analytical/Method-of-Manufactured-Solutions tier the yardstick §3.1.1 says to preserve and name. Not bloat; the depth is the suite's best asset and… (sanctioned by docs/design/testing-architecture.md §3.1.1 (analytical-derivation tier))
- crates/cobre-stochastic/src/par/lag_kernel.rs test module — independent-oracle differential tests: oracle_shift_lag_major (line offset 30 in test mod), oracle_shift_entity_major, oracle_scalar_weight_finalize, lag_kernel_downstream_ring_re… — The kernel is checked against a separately-written reference implementation (an oracle) across layouts and weightings — property/differential testing, the yardstick's ideal for sort/reduction/canonicalization invariants (§3.2.7). Legitimat… (sanctioned by docs/design/testing-architecture.md §3.1.5 / §3.2.7 (property + oracle testing))
- crates/cobre-stochastic/src/par/fitting/{estimation.rs:1283, mod.rs, estimation/tests.rs, tests.rs} — extracted sibling `tests.rs` form via `#[cfg(test)] mod tests;` — par/fitting already uses the extracted-sibling homing the yardstick §5.1 prescribes (rather than a multi-thousand-line inline module), so it is on the correct side of the inline-giant-vs-extracted-sibling asymmetry the yardstick §3.2.4 fla… (sanctioned by docs/design/testing-architecture.md §5.1 (extracted-sibling homing))
- Cross-reference — wider fixture-prelude spread and oracle-harness duplication (NOT restated here) — The same make_bus/make_hydro/identity_correlation prelude also appears in halton_integration/sobol_integration/lhs_integration (tree-noise substation headline) and reproducibility/saa_golden_value; the crate-wide integration-binary total (… (sanctioned by prior-register.md 'Oracle test-harness duplication' (cross-reference); attacker-prompt scope walls (crate-wide totals -> test-corpus station))
- sampling/mod.rs, external.rs, historical.rs, class_sampler.rs, window.rs, out_of_sample.rs, insample.rs, eta_inversion.rs (paradigm-leakage grep, architecture Q1) — `git grep -niE '(sddp|benders|cut|cost.to.go|state.space)'` over the whole sampling manifest at the baseline returns exactly one hit — the substring `cut` inside `consecutive` in a test assertion string at insample.rs:116 (`expected at lea… (sanctioned by architecture lens Q1 / scripts/ci/check-infra-genericity.sh (L1 purity))
- crates/cobre-stochastic/src/sampling/class_sampler.rs::ClassSampler (enum, line 63) and its fill/apply_initial_state — The per-class noise source is a closed-variant `enum` (InSample/OutOfSample/Historical/External) with `match` dispatch — exactly the CLAUDE.md-mandated enum dispatch over `Box<dyn Trait>` for a closed set. The correlation contract (only Ou… (sanctioned by CLAUDE.md hard rule (never Box<dyn Trait>; enum dispatch for closed variant sets))
- crates/cobre-stochastic/src/sampling/external.rs::ExternalScenarioLibrary (line 57) and historical.rs::HistoricalScenarioLibrary (line 92) — Both are cleanly-classified realized-value data containers ('pure data container', 'no sampling logic') with documented stage-major / window-major eta layouts and bounds-checked accessors. They hold standardized eta only; selection and cor… (sanctioned by architecture lens Q2 classification)
- crates/cobre-stochastic/src/sampling/external.rs::standardize_external_simple (line 399) — The `η = (value - mean)/std` standardization shared by non-PAR external classes is factored into one generic body that 'carries no entity vocabulary' — per-class field access is injected via two closures, and `standardize_external_load`/`_… (sanctioned by in-code rationale at external.rs:258-261, 697-700 and historical.rs:292-294 (arity justified); over-engineering lens owns the too_many_arguments / four-Option items)
- crates/cobre-stochastic/src/context.rs::StochasticContext / build_stochastic_context (seam sub-station) and the cobre-sddp ForwardSamplerConfig construction sites (simulation/state.rs:552, training/forward/sampler.rs:23, training/forward_p… — Cross-reference only. Candidate 1 cites `StochasticContext` as the store handle but does NOT anchor there — the store type and its 7-arg `build_stochastic_context` constructor belong to the `seam` sub-station (and CD-001 is RESOLVED per pr… (sanctioned by prior-register.md CD-001 (resolved); scope walls (cobre-sddp anchors are other stations))
- crates/cobre-stochastic/src/sampling/eta_inversion.rs (run_eta_inversion) and the named standardization chain (external.rs standardize_external_inflow:262, standardize_external_simple:399, standardize_external_load:597, standardize_externa… — The perf hot spot the dispatch pointed at is clean w.r.t. per-scenario/per-stage allocation. run_eta_inversion pre-allocates every scratch buffer (past_lag_buf, lag_state, lag_buf, lag_accum, lag_weight_accum, raw_rate_buf, incoming_scratc… (sanctioned by in-code rationale in eta_inversion.rs run_eta_inversion doc comment ('this driver owns only the scratch buffers and the loop nest') and out_of_sample.rs fill_uncorrelated doc comment ('No heap alloca…)
- crates/cobre-stochastic/src/sampling/mod.rs ForwardSampler::sample (line 213) and class_sampler.rs ClassSampler::fill (InSample/Historical/External/SAA arms) — The composite forward draw is allocation-free on the hot path for every scheme except the QMC/LHS OutOfSample arms flagged above: sample() only split_at_mut()s the caller-owned noise_buf and hands per-class segments down; InSample/Historic… (sanctioned by in-code rationale on ForwardSampler ('reused across all (iteration, scenario, stage) calls without per-call allocation'))
- crates/cobre-stochastic/src/par/fitting/estimation.rs and correlation.rs rayon sites (out of the sampling sub-station manifest) — The crate's only two rayon sites carry a canonical-hydro_ids-order reassembly determinism rationale that any fix-shape must preserve byte-for-byte. None of the three candidates above touch par/fitting or any rayon site — all fixes are conf… (sanctioned by cross-reference to the par sub-station manifest (par/fitting/estimation.rs, correlation.rs); out of scope for this cell)
- docs/design/reserved-seams-and-deferred-debt.md audit-evidence #[allow(...)] census vs the sampling too_many_arguments sites (external.rs:261, historical.rs:295, eta_inversion.rs:36) — Each of the three #[allow(clippy::too_many_arguments)] sites in the sampling manifest carries a // Rationale: comment and falls in the mirror-sanctioned 'Load-bearing / refactor-decision lint' class, so none is plan-dead or accidental. My… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md audit-evidence section, '#[allow(...)] census', Load-bearing class (too_many_arguments / too_many_lines with a // Rationale:))
- crates/cobre-stochastic/tests/saa_golden_value.rs GOLDEN_S0_* constants (lines 25-30) and their assert_eq guards in saa_golden_value_regression — Six f64 constants pinning the base_seed=42 SAA opening-tree output, asserted bit-exact - a legitimate golden bit-exact regression guard, not a wire-byte pin duplicating a const. This is the only const-pin in the sampling test/integration s… (sanctioned by docs/design/testing-architecture.md 3.1 (golden bit-exact tier) and 5.10 (keep SHA/bit-exact for the small deliberate golden set))
- crates/cobre-stochastic/src/sampling/external.rs mod sample_moment_reduction_proptests (line 3147; derive_external_sample_moments_is_declaration_order_invariant at 3157) — A declaration-order-invariance property test over derive_external_sample_moments - exactly the proptest expansion the yardstick asks the suite to grow toward for its sort/reduction invariants, so it is worth protecting, not trimming. (sanctioned by docs/design/testing-architecture.md 5.9 (expand proptest to cover declaration-order/reduction-order invariants))
- The extensive-form / branching-value oracle harness under crates/cobre-sddp/tests (mirror entry) — Cross-reference only: the integration-prelude duplication in candidate 2 is the cobre-stochastic analogue of the recorded oracle-harness duplication; cited here rather than restated, and my candidate 2 is anchored in cobre-stochastic/tests… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md L285 (Oracle test-harness duplication) / prior-register.md Oracle entry, disposition cross-reference)
- All 15 non-generated tree-noise files (tree/{mod,generate,lhs,opening_tree,qmc_halton/mod,qmc_sobol/mod}.rs, noise/*, normal/*, correlation/*) — Paradigm-leakage clean: git grep -niE '(sddp|benders|cut|cost.to.go|state.space)' over the full manifest yields a single hit, generate.rs:46, which is the substring 'cut' inside 'conseCUTive' — not an engine noun. No decomposition-engine t… (sanctioned by scripts/ci/check-infra-genericity.sh (L1 purity guardrail))
- crates/cobre-stochastic/src/normal/precompute.rs PrecomputedNormal::build typed on cobre_core::scenario::LoadModel — The 'normal noise' precompute binds to LoadModel specifically; this is roadmap-consistent rather than a smell — the target layering has LoadModel leaving System for the cobre-stochastic uncertainty store (target-layering-brief Part-I item… (sanctioned by prior-register.md (LoadModel conflation = not-ours, cobre-core))
- tree/qmc_sobol/sobol_directions.rs (21229 lines, generated Joe-Kuo table) — Excluded from any density / god-module / LOC-share claim over tree-noise per the inventory correction (60.3% of the crate's non-test lines, 88.3% of the tree-noise sweep). No architecture finding is raised against it; its size is the gener… (sanctioned by inventory.json corrections[1])
- Phase-1 (Switchable<T> uncertainty store) alignment note — OpeningTree (tree/opening_tree.rs), DecomposedCorrelation (correlation/resolve.rs), PrecomputedNormal (normal/precompute.rs) — These read-only generation-side caches are the tree-noise pieces that would migrate into the Part IV/V Switchable<T> uncertainty store when the stochastic representation is lifted off System/Stage (target-layering-brief Part-I item 1). Rec… (sanctioned by target-layering-brief.md Part-I item 1 (advances-1))
- crates/cobre-stochastic/src/tree/generate.rs generate_stage_raw_noise + generate_saa + HistoricalResiduals branch — Batch noise generation is dispatched once per stage (generate_stage_raw_noise), not per opening; the batch methods fill the whole stage slice in a single call. The per-opening fresh Pcg64 in generate_saa (rng_from_seed at generate.rs:217)… (sanctioned by CLAUDE.md determinism contract (reproducibility + declaration-order invariance); noise/seed.rs and noise/rng.rs module docs (absolute-seed derivation, resume invariant))
- Enum census over crates/cobre-stochastic/src (8 pub enum) — Re-run at baseline returns exactly the expected 8 pub enum and none is literally single-variant, so there is no dead one-valued-enum smell across the crate; the only effectively-one-valued case is SweepDirection, raised as the sole candida… (sanctioned by attacker-prompt over-engineering checklist item (1) — enum census expectation of 8, none one-valued)
- crates/cobre-stochastic/src/tree/qmc_halton/mod.rs radical_inverse (#[allow(dead_code)], line ~93) — A pub(crate) module primitive marked #[allow(dead_code)] with an explicit in-code rationale that it is reached only from tests until a downstream caller wires it in. Under the reserved-is-not-dead precondition this is a sanctioned reserved… (sanctioned by in-code rationale at qmc_halton/mod.rs:91-93 ('Module primitive ... reached only from tests until a downstream caller wires it in') + CLAUDE.md 'unwired config/seam is reserved, not dead')
- Out-of-cell over-engineering checklist items (2) ForwardSamplerConfig four parallel Option<&…Library> (sampling/mod.rs), (3) #[allow(deprecated)] PAR re-export block (lib.rs), (4) 11-arg standardize_external_inflow (sampling/external.rs) — Cross-reference only: checklist items 2 and 4 anchor in the sampling sub-station and item 3 in the seam sub-station (lib.rs/context.rs), all outside the tree-noise manifest. Left to the sto-oe-sampling / sto-oe-seam sibling workers; not ra… (sanctioned by scope walls — candidate anchors must sit in the tree-noise manifest; these anchor in sampling/ and lib.rs (seam))
- Cross-reference: mirror 'Oracle test-harness duplication' (docs/design/reserved-seams-and-deferred-debt.md L285; prior-register disposition = cross-reference) — The QMC fixture-prelude duplication (candidate 1) is the same class of integration test-harness duplication the mirror already records for the SDDP oracle harness, but this instance is freshly anchored inside crates/cobre-stochastic/tests… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md 'Oracle test-harness duplication' (L285); prior-register.md cross-reference disposition)
- crates/cobre-stochastic/src/noise/{rng,quantile,seed}.rs inline #[cfg(test)] modules — Examined for the homing asymmetry (candidate 2) and found correctly homed: rng.rs (~40 inline test-LOC), quantile.rs (~100), seed.rs (~173) all sit at or below the yardstick §5.1 proposed threshold, so keeping them inline is the correct fo… (sanctioned by docs/design/testing-architecture.md §5.1 (homing threshold))
- crates/cobre-stochastic/src/lib.rs (public surface; #[allow(deprecated)] PAR re-export block at line 44) — Clean, engine-neutral crate surface (no paradigm noun). The ~14-symbol #[allow(deprecated)] PAR re-export block is the over-engineering lens's checklist item #3, not an architecture finding; cross-referenced here, not raised. (sanctioned by over-engineering lens checklist item 3 (attacker-prompt.md))
- crates/cobre-stochastic/src/season_cast/mod.rs (StageCalendar, season_period_window, cast, merge_layered_windows) — Crate-home relocation is DO-NOT-RAISE. The recorded L2->L1 evidence (cobre-io -> cobre-stochastic at 4 sites: estimation.rs:70, inflow_seeding.rs:22, thermal.rs:17, scenarios.rs:12) is already captured in the inventory/prior-register and i… (sanctioned by mirror reserved-seams L449 'Stage-calendar crate home'; prior-register 'Stage-calendar crate home' (keep); inventory layeringEvidence)
- Q1 paradigm-leakage sweep over all six seam files — Zero engine/paradigm nouns. The unanchored lens grep's only hit is `cut_points` (season_cast/mod.rs:439, a day-boundary variable for window merging). The authoritative CI gate scripts/ci/check-infra-genericity.sh matches `\bcut\b` (word-bo… (sanctioned by scripts/ci/check-infra-genericity.sh word-boundary pattern)
- crates/cobre-stochastic/src/context.rs (StochasticContext as a many-reader store) — StochasticContext is consumed by ~30 files across cobre-sddp and cobre-cli (git grep StochasticContext), so it is a genuine established store, NOT a one-consumer abstraction; the L1-purity guardrail #4 (one-consumer abstraction) is clear f… (sanctioned by L1 purity guardrail #4 (attacker-prompt.md))
- crates/cobre-stochastic/src/lib.rs — the #[allow(deprecated)] re-export block (checklist item 3) — It is a compatibility shim, and it is already scheduled for removal under a sanctioned owner — so it is a positive here, not a candidate. Only three of the block's symbols are genuinely #[deprecated]: par/evaluate.rs's evaluate_par_inflow… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md ~L1097-1100: the #[allow(deprecated)] sites at {lib.rs, par/mod.rs, par/evaluate.rs} are a fourth, narrower class tied to the deprecated-with-fallback…)
- crates/cobre-stochastic/src/season_cast/mod.rs — public calendar surface — No one-consumer or forwarding abstraction: post_study_calendar_stages (cobre-sddp lp/builder/template, policy_export, setup/mod.rs:1191), covers_exactly (cobre-io validation/semantic/thermal.rs:665), hour_window_shares (cobre-sddp setup/mo… (sanctioned by plans/architecture-debt-audit/stations/stochastic/prior-register.md — 'Stage-calendar crate home' (disposition: keep; relocation deferred, L2->L1 edge is consistent).)
- crates/cobre-stochastic/tests/reproducibility.rs — deterministic_reproducibility (200), declaration_order_invariance (285), seed_sensitivity (328), infrastructure_genericity_no_sddp_references (346), and the local build_fixture helper (159) — These are the crate's core determinism/order-invariance gates and are exactly the correctness-depth the yardstick says to preserve, not prune (§3.1 determinism discipline). infrastructure_genericity_no_sddp_references is the in-test mirror… (sanctioned by docs/design/testing-architecture.md §3.1 (determinism discipline at HPC grade; O(1) fixture builder))
- crates/cobre-stochastic/src/season_cast/mod.rs — cast_proptests and merge_proptests (proptest! blocks at the tail of the test module) — Property tests over generated segment/window inputs (cast partition == day-weighted mean; merge of disjoint windows == concatenated cast). These are precisely the invariant-over-generated-inputs coverage the yardstick asks the suite to EXP… (sanctioned by docs/design/testing-architecture.md §5.9)
- CROSS-REFERENCE — mirror 'Oracle test-harness duplication' (docs/design/reserved-seams-and-deferred-debt.md L285; prior-register.md) — Candidate 1 is the SAME fixture-fragmentation class the yardstick §5.2 resolves, but a DISTINCT instance: it is anchored inside crates/cobre-stochastic (context.rs, provenance.rs, seeds.rs, tests/reproducibility.rs), not in cobre-sddp/test… (sanctioned by prior-register.md 'Oracle test-harness duplication' (disposition: cross-reference))

#### ↩︎ Cleared (dismissed — do not re-raise)

None. All 35 confirmed candidates were defended and recorded; no candidate was dismissed and no over-engineering candidate targeted a ratified reserved seam (the attackers pre-filed the sanctioned seams — the generated `sobol_directions.rs` table, the `#[allow(clippy::too_many_arguments)]` rationale on `standardize_external_inflow`, the two rayon determinism rationales — under Positives with their citations).

#### Prior-register dispositions

- **Stage-calendar crate home** — keep.
- **External-noise take/fill glue duplication** — not-ours.
- **Deterministic (σ = 0) AR(p > 0) external inflow stays rejected** — keep.
- **`LoadModel` conflates physical load with its stochastic model** — not-ours.
- **Cross-path static-RHS contract not yet in `.claude/rules/sddp.md`** — not-ours.
- **Oracle test-harness duplication** — cross-reference.
- **CD-001 — setup config-projection sprawl / CLI non-root reconstruction** — resolved.

CD-001 is resolved at the baseline (the CLI calls the single shared `build_stochastic_context_for_study`, definition `crates/cobre-sddp/src/setup/stochastic_pipeline.rs:424`, sole caller `crates/cobre-cli/src/commands/run/setup.rs:409`); its only open question — whether an engine-neutral context constructor belongs at L1 beside `crates/cobre-stochastic/src/context.rs:601` — leaves the station as an alignment hint, not a new duplication finding.

#### Phase-1 uncertainty-store notes

Alignment hints only (Epic 9 adjudicates). The seam findings tagged `advances-1` mark where the Phase-1 `Switchable<T>` uncertainty store would form: `StochasticContext` (`context.rs`) is today both a computed store (five precomputed components) and a config/layout echo, and `ForwardSamplerConfig` (`sampling/mod.rs`) straddles the generation-side and realized-value halves. The fitted-process code (PAR estimation in `par/`, opening-tree generation in `tree/`, seed derivation) stays generation-only; the standardized realized libraries and the sampler are the store side. No fix is executed here.

#### Handoff queues

- **performance-sweep** (7): every Sev-A/B PD entry, each UNMEASURED and carrying a claim type, a 4t layout and a profiled symbol — see `perf-queue.json`. The Sev-C perf findings stay recorded as deferred debt below the sweep threshold and are not queued. No timing is asserted by this station.
- **test-corpus** (10): every TD entry with its integration binaries and line ranges, cross-referencing the mirror's `Oracle test-harness duplication` section — see `td-queue.json`.
- **alignment** (3): the `advances-1` / Part-I entries with their Part IV/V citation and the CD-001 L1-homing question — see `alignment-queue.json`.

#### Owner gate — decisions

Ratified 2026-09-08 in the main session over plans/architecture-debt-audit/stations/stochastic/gate.md. All 35 confirmed entries accepted as recorded; no downgrade, reject, defer or override; the 7 prior-register dispositions stand as recorded; the `cut_points` needs-human item is recorded not-a-finding. Severity is shown as `new (reviewer: original)` on any downgrade (none here). No timing number is asserted.

| ID | Decision | Severity | Alignment | Rationale (owner) | Trigger / override | Queue |
| -- | -------- | -------- | --------- | ----------------- | ------------------ | ----- |
| CD-064 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | none |
| PD-021 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-020 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| TD-025 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| TD-024 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| CD-066 | accept | B (A-risk) | neutral | accepted as recorded (Sev-B batch) | — | none |
| CD-065 | accept | B | advances-1 | accepted as recorded (Sev-B batch) | — | alignment |
| OD-029 | accept | B (A-risk) | neutral | accepted as recorded (Sev-B batch) | — | none |
| OD-028 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | none |
| PD-023 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-025 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-024 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| TD-028 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| CD-070 | accept | B (A-risk) | advances-1 | accepted as recorded (Sev-B batch) | — | alignment |
| CD-068 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | none |
| CD-067 | accept | B (A-risk) | neutral | accepted as recorded (Sev-B batch) | — | none |
| PD-027 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| PD-026 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | perf |
| TD-030 | accept | B | neutral | accepted as recorded (Sev-B batch) | — | test-debt |
| OD-027 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| OD-026 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-022 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| TD-027 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| TD-026 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| TD-029 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| CD-071 | accept | C | advances-1 | accepted as recorded (Sev-C batch) | — | alignment |
| OD-031 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-029 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-030 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| TD-032 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| TD-033 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |
| CD-069 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| OD-030 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| PD-028 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | none |
| TD-031 | accept | C | neutral | accepted as recorded (Sev-C batch) | — | test-debt |

**Prior-register dispositions ratified (as recorded):** Stage-calendar crate home → keep; External-noise take/fill glue duplication → not-ours; Deterministic (σ = 0) AR(p > 0) external inflow stays rejected → keep; `LoadModel` conflates physical load with its stochastic model → not-ours; Cross-path static-RHS contract not yet in `.claude/rules/sddp.md` → not-ours; Oracle test-harness duplication → cross-reference; CD-001 — setup config-projection sprawl / CLI non-root reconstruction → resolved.

**Cleared by this gate (do not re-raise):** none.

**Gate: RETURNED 2026-09-08** — baseline `a136840d`; accepted 35, downgraded 0, rejected 0, deferred 0, overridden 0.

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
