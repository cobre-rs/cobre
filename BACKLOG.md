# Cobre Architecture Debt Backlog

A living backlog of architectural smells and technical debt, built by walking the
full lifecycle of a `cobre run` command from `main()` to the last output write.

- **Method**: station-by-station descent through the real call tree (not a fan-out).
- **Scope**: architecture-only — god modules/functions/structs, bad/missing
  abstractions, asymmetric ownership, leaky crate boundaries, coupling,
  duplication. **Out of scope** (by hard rule): reserved/unwired config, load-bearing
  contract comments, determinism contracts, perf micro-optimization.
- **Status**: read-only investigation. No repo mutation outside this gitignored file.

Baseline: 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c (pinned 2026-09-17)
Previous baselines: a136840d (pinned 2026-09-05, superseded 2026-09-17)
Ledger: this file is the sole home of finding IDs (`CD-` / `PD-` / `OD-` / `TD-`).
Workers return JSON only; the main session is the sole writer. The 2026-09 quality
evaluation writes exactly two tracked surfaces: the ID-free mirror
`docs/design/reserved-seams-and-deferred-debt.md` and this directory
(`plans/architecture-debt-audit/`, tracked on the evaluation branch so its work
sessions carry git evidence); no crate, docs, script, CI or schema file is touched
(amends the Status bullet above).
Protocol bound: 228.319 s (median of 3 timed runs after one warm-up, layout `4t`,
deck `~/git/cobre-bridge/example/cobre_reduzido`; measured 2026-09-17 at the
pinned baseline on the profiling-profile binary — see `measurements/CAL/`. The deck
was re-sanctioned 2026-09-17 by owner decision after the original `cobre_reduzido_2`
was lost; the superseded bound at `a136840d` on that deck was 234.781 s).
Protocol bound (enumerated): 30.434 s (layout `2t`, deck
`~/git/cobre-bridge/example/cobre-mar-26-rv2-reduced`, owner-limited to two workers;
measured 2026-09-17 at the pinned baseline — see `measurements/CAL-ENUM/`; the
superseded bound at `a136840d` was 32.589 s).
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
- **Note (2026-09-17):** still OPEN, structure unchanged, risk narrowed. The CLI/Python plan merged at
  `develop` `24e76cdb` collapsed the Python side to ONE internal owner (`cobre.run.run` is a thin wrapper over
  `Study::{new,train,load_policy,simulate}_native`; the `_if_any` helpers are called from `Study::train_native`
  only), but the cross-crate hand-mirror is intact: `write_training_outputs` (`cobre-cli/.../run/outputs.rs:58`)
  re-inlines the fpha/evaporation writes (`:95`, `:107`) and the census `scenario_summary` reshape is still
  copied (`outputs.rs:221` ↔ `cobre-python/src/run.rs:799`). What changed is the DETECTION: a golden test
  (`cobre-python/tests/test_cli_python_determinism_parity.py`) now runs `examples/1dtoy` through both entry
  points and compares the whole output tree value-for-value under two literal wall-clock masks, an
  import-resolving parity gate (`scripts/ci/check_python_parity.py`, 18 shared writers) replaced the grep, and
  the two content divergences the test would have exposed were fixed rather than masked (CLI simulation
  `solver_version`; the Python `setup` timings section). Drift is now caught in CI; the owner is still two.
  Fix-shape unchanged (hoist to a crate both depend on); Wave 5 of the 2026-08-22 table.

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
- **Status:** partial (2026-09-17) — the Python-parity half is FIXED: `cobre.io.validate` runs the boundary
  reconciliation as its phase 11 (`cobre-python/src/io.rs:167`, via the shared `reconcile_boundary_policy`),
  so a boundary configuration `cobre validate` rejects is rejected by the binding too (CLI/Python plan
  ticket, `develop` `fc81427a`). The abstraction half is OPEN: `PrepPhase` still has three variants
  (`cobre-sddp/src/validate_phases.rs:33`) under a doc that says "four SDDP preparation steps" (`:20`),
  and both front ends still run the boundary check outside `PrepPhase`/`prep_phase_metadata`. Remaining
  fix: fold the boundary check into `PrepPhase` or correct the doc to three.

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
- **Status:** fixed (2026-09-14) — `HydroStagePenalties` is deleted outright (owner decision on 2026-09-13; not the alias PRIORITIES §Tier 3 suggested): `PenaltiesDefaults.hydro` and the resolved per-(hydro, stage) table store `cobre_core::HydroPenalties`; the sixteen-field copy in `cobre-io/src/resolution/penalties.rs` is a move of `hydro.penalties`; the FPHA clause moved onto `HydroPenalties.turbined_cost` (softened at the boundary review to "should exceed `spillage_cost`; not enforced by validation" because `check_fpha_penalty_rule` only checks `>= 0` and `test_5b_fpha_penalty_equal_spillage_valid` asserts equality is valid). 224 occurrences in 34 files renamed, BREAKING CHANGELOG entry. `plans/quality-tier3-footguns` ticket-001, `25603faf`, merged to `develop` at `3e90024f`.

**CD-041 · Sev B · asymmetry · effort M · confidence high**
DisconnectedBus (error.rs:65) and InvalidPenalty (error.rs:72) are emitted by no production/validation path and both carry a false 'Emitted by cobre-io validation' doc line while cobre-io imports ValidationError zero times; DisconnectedBus additionally has a discarded buses builder parameter (network.rs:90-91) behind a TODO. Narrower than 'constructed nowhere', because DisconnectedBus is constructed in the test_error_trait test at error.rs:196.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/error.rs::ValidationError`, `crates/cobre-core/src/error.rs:72`, `crates/cobre-core/src/topology/network.rs::build`
- **Evidence:** Outside its own declaration and Display arm, `DisconnectedBus` appears only in a TODO comment and `InvalidPenalty` appears nowhere at all — neither variant is ever constructed in production or test code.
- **Fix-shape:** Decide whether the disconnected-bus rule is wanted, then make the code say so. If it is wanted, implement it where the topology is already being built — that is the one place with every entity family in hand — and drop the parameter's discard;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — the two variants deleted as under OD-010 and `NetworkTopology::build` lost its unread `buses` parameter (7 → 6 args, 12 call sites, nine dead `buses` bindings deleted, `BusSpec`/`make_bus` pruned) (ticket-018). The disconnected-bus rule was decided against, not implemented — the fix-shape's other road. `feat/quality-tier45-closeout` dd03c153 + 8a72c69d (pending merge).

**CD-042 · Sev C · asymmetry · effort S · confidence high**
The uniform-stride flat-index arithmetic is repeated at 14 sites across hydro/line/pumping/contract with no shared helper and, unlike thermal, no stride debug_assert: a DRY/symmetry and guard-asymmetry gap only. It is not a correctness bug (all four share the bounds-checked n_stages stride) and the thermal_cell_index helper is justified by thermal's genuinely distinct padded stride.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/resolved/bounds.rs::thermal_cell_index`, `crates/cobre-core/src/model/resolved/bounds.rs::ResolvedBounds`
- **Evidence:** The flat cell-index invariant is owned once for the thermal table by `thermal_cell_index`, which also carries a debug assertion protecting its stride.
- **Fix-shape:** Give the four remaining families the same treatment the thermal table already has: one private index helper per family, or a single shared helper taking the stride, so the layout rule is stated once and each accessor reads a named call rather than repeating the multiply-add. This is a mechanical, behaviour-preserving change confined to one file, and it makes a future stride change for any family a one-line edit instead of a fourteen-site sweep.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — `ResolvedBounds::cell_index(entity_index, stage_index)` (`crates/cobre-core/src/model/resolved/bounds.rs`) is the one owner of the uniform stride, with the same `debug_assert!` shape as `thermal_cell_index`; all 14 open-coded hydro/line/pumping/contract sites route through it and the module, struct and `n_stages` docs name the helper once. Behaviour-preserving (cobre-core 336 unit tests unchanged). `develop` @ `eb0b82ef`.

**CD-043 · Sev B · asymmetry · effort M · confidence high**
SystemBuilder::build re-sorts stages by id (builder.rs:364) but neither reassigns nor validates Stage.index (only cobre-io stages.rs:811-815 writes it; validate.rs has no check), so a non-cobre-io producer can silently supply an index disagreeing with the post-sort slot: a latent footgun, not a live wrong result on the cobre-io path (stable sort of already-indexed input).

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/temporal.rs::Stage`, `crates/cobre-core/src/system/builder.rs::build`, `crates/cobre-core/src/model/temporal/stage_key.rs::StudyPos`
- **Evidence:** `Stage.index` is documented as the position in the canonical-ordered stage vector.
- **Fix-shape:** Move the assignment to the sort. The builder already establishes the canonical stage order, so it should reassign each stage's position immediately after sorting, making the field a derived value that cannot disagree with its slot and letting the cobre-io parser drop its own loop.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-14) — `SystemBuilder::build` assigns `Stage.index = position` immediately after its stage sort; the `cobre-io/src/stages.rs` loop is deleted and `Stage.index`'s doc names the builder as the sole writer for a stage reachable via `System::stages()`. The epic-boundary review then found the live pre-build reader the ticket's survey missed: `validate_inflow_seeding` runs `precompute_stage_lag_transitions` on the parser's `StagesData`, and `compute_period_transition` read `.skip(stage.index + 1)` — every pre-build index was 0. `compute_period_transition` now takes the slice position (`par/lag_transition.rs`), which also corrects a latent over-skip on `stochastic_pipeline.rs`'s `id >= 0`-filtered `study_stages` slice when pre-study stages exist (d26/d30/d43 goldens unchanged). Regression test through the real parser: `stages::tests::test_prebuild_lag_transition_uses_position_not_index`. ticket-002 + `b8a69b57`.

**CD-044 · Sev B · asymmetry · effort M · confidence high**
StageLagTransition (temporal.rs:175) has zero cobre-core consumers (all 4 occurrences are its own declaration/comments) and its field doc references the L3 pub(crate) symbol accumulate_and_shift_lag_state (noise.rs:232) that no cobre-core reader can resolve; the strictly-defensible residue is this zero-consumer plus broken-L0-doc-contract pair, leaving physical relocation to Epic-9 layering adjudication.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/temporal.rs::StageLagTransition`, `crates/cobre-core/src/model/temporal.rs:202`
- **Evidence:** All four cobre-core occurrences are the declaration itself and its own doc/section comment — the type has no cobre-core consumer.
- **Fix-shape:** Relocate the type to the crate that owns the uncertainty representation, cobre-stochastic, alongside the PAR machinery that is its only L1 consumer, and re-express its field docs against that crate's own kernel rather than against an engine-private function. Nothing in cobre-core reads it, so the move is a pure re-home plus an import change in the two consuming crates.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-2 (cross-reference; verdict travels to Epic 9).
- **Status:** fixed (2026-09-15) — the two comments in `crates/cobre-core/src/model/temporal.rs` that cited the engine-private `accumulate_and_shift_lag_state` now describe the lag-state update in terms of the fields and "the lag-state accumulation kernel"; no private symbol of another crate and no engine phase name remains, `cargo doc -D warnings` clean. The re-home of the type is NOT a defect fix but a layering decision: it has eight `cobre-stochastic` and six `cobre-sddp` consumers, so its crate home goes to the W10 alignment adjudication, not to a sweep. `develop` @ `eb0b82ef`.

**CD-045 · Sev B · asymmetry · effort M · confidence high**
The three *_models accessors (mod.rs:432/438/466) promise canonical order that no write path enforces (setters silent at builder.rs:229/237/244, build never sorts them) and with_scenario_models (mod.rs:571) replaces inflow_models post-construction with no sort/validation, while L1's PrecomputedPar::build depends on that order; the four 'raw' tables carry an explicit setter precondition and are honest delegation, so they fall outside the defect.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/system/builder.rs::build`, `crates/cobre-core/src/system/builder.rs::inflow_models`, `crates/cobre-core/src/system/mod.rs::inflow_models`, `crates/cobre-core/src/system/mod.rs::with_scenario_models`
- **Evidence:** `build` sorts nine collections (seven operational families via `sort_canonical`, plus stages and generic constraints).
- **Fix-shape:** Make the L0 owner own the invariant it advertises. Either the builder sorts these seven tables into their documented canonical key the way it already sorts the other nine, or it validates them as sorted and returns a validation error otherwise — the second is cheaper and preserves the current cobre-io behaviour byte-for-byte, since cobre-io already emits them sorted.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).
- **Correction (2026-09-11):** `PrecomputedPar::build` does not depend on model order (hash-keyed). Three tables, not "seven".
- **Status:** fixed (2026-09-14) — validate, not sort: new `ValidationError::UnsortedModelTable { table, position }`; `build` checks the three tables after the duplicate checks through one `check_canonical_order` helper (strict decrease only; duplicate keys accepted); `System::with_scenario_models` is fallible (`Result<Self, ValidationError>`, `EstimationError::Validation` `#[from]` variant, one arm in `cobre-sddp/src/error.rs`), CHANGELOG entry. Exposed two pre-existing contract violations: 31 test fixtures in cobre-sddp and cobre-stochastic built model tables stage-major (reordered, values unchanged), and `run_partial_estimation` appended pre-study rows unsorted (now sorted by `(hydro_id, stage_id)` like `seasonal_stats_to_rows`; regression test `test_partial_estimation_partial_year_study_orders_canonically`). ticket-003 + `b8a69b57`.

**CD-046 · Sev C · asymmetry · effort M · confidence high**
The wire-reproducibility rationale in the System serde(skip) comment (mod.rs:64) is enforced by three unrelated bespoke mechanisms yet six HashMap fields on HorizonGraph/CascadeTopology/NetworkTopology serialize unguarded as non-skipped SystemRepr fields (mod.rs:157/158/160), an inconsistency with no single owner and no guard test; explicitly NOT a live wrong result today (single-serialize-then-broadcast plus a value-equality round-trip guard).

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/system/mod.rs::System`, `crates/cobre-core/src/model/horizon.rs::HorizonGraph`, `crates/cobre-core/src/topology/cascade.rs::CascadeTopology`, `crates/cobre-core/src/topology/network.rs::NetworkTopology`
- **Evidence:** cobre-core states the same rule three times and enforces it three different ways: `serde(skip)` on the seven index maps with an explicit wire-reproducibility rationale, a hand-written `Serialize` that sorts its composite keys, and a `BTreeMap` chosen over `HashMap` for the correlation profiles.
- **Fix-shape:** Give the rule one owner instead of three restatements. State once, in the crate root or the system module, that anything reachable from the System payload must serialize in a content-determined order, and satisfy it uniformly — the cheapest route is switching these six fields to an ordered map, since all six are keyed by an already-`Ord` entity id or stage id and none is on a hot path where the lookup cost would matter.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** narrowed by the Tier-4/5 wave, not closed: the `topology/network.rs::NetworkTopology` anchor is gone with the type (`dd03c153`), and the `CascadeTopology` half is now `#[serde(skip)]` and rebuilt on deserialize with a bit-equality round-trip guard (`postcard_roundtrip_rebuilds_cascade_topology_bit_equal`), so only the `HorizonGraph` map fields remain unguarded. Re-anchor or re-scope the entry before the doc-drift sweep (W7); do not close it on this evidence.
- **Status:** fixed (2026-09-15) — after the Tier-4/5 narrowing the only unordered map on the wire was `HorizonGraph::stage_discount_rate_overrides`; it is a `BTreeMap<i32, f64>` (public field type change, Rust-API line in the CHANGELOG), the rule is stated once on the `System` struct doc (derived maps are `serde(skip)` and rebuilt unconditionally; anything that stays on the wire is key-ordered) with the per-field skip comment reduced to a pointer that keeps the `rebuild_indices` contract, and `postcard_wire_bytes_are_identical_regardless_of_discount_override_insertion_order` (`system/mod.rs`, `serde`-gated, runs under the workspace gate) pins byte identity across insertion orders. `develop` @ `eb0b82ef`.

**CD-047 · Sev C · asymmetry · effort S · confidence high**
Six numeric extractors in extensions/{hydro_geometry,hydro_energy_productivity,tailrace_curves}.rs re-implement parquet_helpers' extract_required_int32/float64 (byte-identical to each other, three-line delta vs the shared exports), and the File::open->try_new->build->rows prologue recurs 28x across 19 files uncovered by parquet_helpers; the evaporation_models.rs:158 string extractor is NOT part of the defect (a genuine gap -- parquet_helpers offers no Utf8 extractor, documented at :157).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/extensions/hydro_geometry.rs::extract_int32_column`, `crates/cobre-io/src/extensions/hydro_energy_productivity.rs::extract_int32_column`, `crates/cobre-io/src/extensions/tailrace_curves.rs::extract_int32_column`, `crates/cobre-io/src/constraints/bounds.rs::parse_line_bounds`, `crates/cobre-io/src/scenarios/inflow_stats.rs::parse_inflow_seasonal_stats`, `crates/cobre-io/src/parquet_helpers.rs::extract_required_int32`
- **Evidence:** The md5 line is over the 23-line bodies of `extract_int32_column` sliced from the three extension files with sed;
- **Fix-shape:** Extend `parquet_helpers.rs` past column extraction to cover the reader itself: one helper that takes a path and returns the batch reader with the three error mappings applied, so each parser opens with a single call and keeps only its own column reads and row loop. Delete the six copied extractors in `extensions/` in favour of the shared pair, accepting the one-word change in the missing-column message or reconciling the two spellings first.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** 77 `try_new` prologue sites in 29 files (tests included), not 28 in 19.
- **Correction (2026-09-15):** there are 30 production reader prologues in 20 files, of which 28 in 19 share the `LoadError` mapping shape and adopted the helper; the two in `output/convergence_reader.rs` (since deleted with the report subcommand) map to `OutputError` / `Option` and were excluded. The heading's `28× across 19 files` is therefore the adopting count and the 2026-09-11 correction's `77 in 29` is the with-tests total; both stand.
- **Status:** fixed (2026-09-15) — `pub(crate) open_record_batch_reader(path)` in `parquet_helpers.rs` (+3 tests) owns the open → build prologue with the `LoadError` mappings; 28 prologues in 19 files fold onto it and the `constraints/mod.rs`/`extensions/mod.rs` parser recipes collapse to four steps (ticket-032); the six copied extension extractors deleted onto `extract_required_int32`/`extract_required_float64` (ticket-033, recorded under CD-063). The two `convergence_reader.rs` prologues were deliberately excluded — they map to `OutputError` / `Option`, not `LoadError`. `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**CD-048 · Sev B · asymmetry · effort M · confidence high**
bounds.rs:24 and penalties.rs:23 state `sorted by ID` for entity families that all carry operational_start_date, contradicting the enforced `(operational_start_date, id)` canonical order (builder.rs sort_canonical, pipeline.rs:289) on the publicly reachable resolve_bounds/resolve_penalties surface -- a doc-contract-vs-enforced-contract drift, not a live miscompute; generic_bounds.rs:15's `sorted by ID` is correct (no date axis) and the three `must be sorted` spellings (ncs_bounds/load_factors/ncs_factors) are underspecified rather than wrong, so `two wrong statements` is the exact residue.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/resolution/bounds.rs::BoundsEntitySlices`, `crates/cobre-io/src/resolution/penalties.rs::PenaltiesEntitySlices`, `crates/cobre-io/src/resolution/ncs_bounds.rs::resolve_ncs_bounds`, `crates/cobre-io/src/resolution/load_factors.rs::resolve_load_factors`, `crates/cobre-io/src/resolution/ncs_factors.rs::resolve_ncs_factors`, `crates/cobre-io/src/resolution/group_bounds.rs::resolve_hydro_unit_group_bounds`, `crates/cobre-io/src/lib.rs:130`
- **Evidence:** Seven resolvers share one precondition — slice position becomes the table's entity index — but state it four different ways.
- **Fix-shape:** Give the canonical key one owner and make every resolver doc point at it instead of restating it. cobre-core already owns the ordering in `sort_canonical`;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-14) — `SystemBuilder::build`'s doc is the single owner (key + "slice position is the entity's canonical index"); the two wrong statements, the three underspecified ones, `resolve_hydro_unit_group_bounds`, `CaseArtifacts` and `sort_into_canonical_order` are one-clause pointers at it (the comparator warning "not `(id, date)` or `id` alone" survived the Deletion Test). Scope widened with owner approval to seven parser docs (`cobre-io/src/system/*.rs`) that restated the key as the builder's tiebreak; `CaseArtifacts`'s pointer was corrected at the boundary review (its fields are row tables with their own key columns, not entity-index-aligned slices). `generic_bounds.rs` untouched. ticket-005.

**CD-049 · Sev C · asymmetry · effort S · confidence high**
The two resolvers share an identical algorithm skeleton and five parallel tests collapsible to one generic routine parameterised over the id accessor and destination table; residual differences are confined to the entity/entry/output-table types, local names, and one incidental `usize::try_from` spelling (load_factors.rs:59 vs ncs_factors.rs:61). `Same function twice` over-reaches (they are two monomorphizations over distinct output-table types, not literal copies), but the duplication is real and has two live consumers.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/resolution/load_factors.rs::resolve_load_factors`, `crates/cobre-io/src/resolution/ncs_factors.rs::resolve_ncs_factors`
- **Evidence:** A full `diff -u` of the two files shows no structural divergence anywhere.
- **Fix-shape:** Collapse the two into one resolution routine parameterised over the entity id accessor and the destination table, keeping `resolve_load_factors` and `resolve_ncs_factors` as thin named entry points so the call sites in the pipeline stay readable and the two output types remain distinct. Two present consumers make this a fold of existing duplication, not a speculative seam, so it does not trip the one-consumer-abstraction rule.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — a private `FactorKind` trait with two zero-sized markers and one generic `resolve_factors` in `resolution/factors.rs`; `resolve_load_factors`/`resolve_ncs_factors` are byte-identical public wrappers; `load_factors.rs`/`ncs_factors.rs` deleted and their ten tests re-homed as `resolution::factors` tests (ticket-035). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**CD-050 · Sev B · asymmetry · effort M · confidence high**
Exactly two semantic branches -- block-duration>0 (semantic/stages.rs:88-103) and CVaR alpha/lambda range (:104-131), rules 4-5 in the mod.rs:83-84 table -- are unreachable for any pipeline-loaded deck because parse-layer validate_block_hours (stages.rs:623) and the CVaR-range half of validate_risk_measure (:661) reject such decks first and validate_schema bails before pipeline.rs:89; this is a duplicated-rule/doc-table drift against the retired-rule-42 convention, NOT a correctness gap (the rules ARE enforced at parse time), and the parser's unrecognized-risk-measure-string check (stages.rs:665) has no semantic counterpart and is excluded from the redundancy.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/stages.rs::validate_block_hours`, `crates/cobre-io/src/stages.rs::validate_risk_measure`, `crates/cobre-io/src/stages.rs::validate_raw_stages`, `crates/cobre-io/src/stages.rs::convert_stages`, `crates/cobre-io/src/validation/semantic/stages.rs::check_stage_structure`
- **Evidence:** The Layer-5b rule table in `validation/semantic/mod.rs:83-84` claims rules 4 and 5 — block duration positive, CVaR alpha in (0,1] and lambda in [0,1] — as the semantic layer's own, sourced from stages.json.
- **Fix-shape:** Decide which layer owns each of the two rules and delete the other copy, following the retired-rule-42 precedent already recorded in the semantic module doc. If the parse layer keeps them, strike rules 4 and 5 from the Layer-5b table with a note that stages.rs owns them, delete the two unreachable branches from `check_stage_structure`, and delete or relabel the two unit tests so they stop reading as coverage of a live rule.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-2 (cross-reference; verdict travels to Epic 9).
- **Status:** fixed (2026-09-15) — the parse layer owns both rules (`stages.rs`'s `validate_block_hours` / `validate_risk_measure`, pinned by `test_error_block_hours_zero`, `test_error_cvar_alpha_zero`, `test_error_cvar_lambda_out_of_range`); the two unreachable branches left `check_stage_structure` with their four dead-branch tests, and Layer 5b rows 4 and 5 are retired with the parse-layer pointer in the wording of the other retired rows. `develop` @ `eb0b82ef`.

**CD-051 · Sev B · asymmetry · effort M · confidence high**
The scenario-source admission rules are enforced only lazily inside the accessor and cobre-io's own load pipeline (`run_pipeline`) never rejects an invalid config, the nine let-else/`.ok()` swallows resting on a false premise comment at scenarios.rs:459 that `validate_config` does not establish; but rejection is not lost end-to-end because each consumer (validate.rs:382, setup/mod.rs:385, cobre-python) re-propagates, so the defect is a mislocated/duplicated admission gate, not a production silent-accept.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/config/mod.rs::validate_scenario_source_cfg`, `crates/cobre-io/src/config/mod.rs::validate_openings_cfg`, `crates/cobre-io/src/config/mod.rs::validate_config`, `crates/cobre-io/src/validation/semantic/scenarios.rs::check_external_scheme_has_files`
- **Evidence:** Five config admission rules — historical-scheme restricted to the inflow class, seed required once any class leaves in-sample, `openings` only under `training`, historical year range ordered, `historical_years` only with a historical class — live in `validate_scenario_source_cfg` / `validate_openings_cfg`, whose only caller is `convert_scenario_source_config` at :227-228, itself reachable only from the two `Config::*_scenario_source` accessors.
- **Fix-shape:** Move the admission decision to the point where the config is admitted, not to whoever happens to read it. Resolve both scenario sources once inside the Layer-2 config gate, report their failures into the validation context alongside every other layer's findings, and hand the already-resolved values to the semantic rules so those rules take a resolved value rather than a fallible accessor.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-7 (cross-reference; verdict travels to Epic 9).
- **Status:** fixed (2026-09-11) — `validate_config` resolves both scenario sources, so `parse_config` is the single admission gate; the training-only pre-checks in `cli validate.rs` and `cobre-python/src/io.rs` are gone; `cli_validate.rs` and `test_io.py` pin validate/run parity for an invalid simulation source. `fix/quality-tier1` a729a259 (merged to develop 2026-09-11).

**CD-052 · Sev B · asymmetry · effort M · confidence high**
Confirmed narrowly: `LoadError::CrossReferenceError` is a dead variant (zero production producers, findings routed through `ConstraintError`) retaining two cobre-python consumer arms, and the overlapping line/hydro-filling predicate pairs can drift undetected because only cobre-io's copy fires in the production pipeline, not that cobre-core's builder validation is itself redundant (it legitimately guards cobre-core's public builder for direct/test constructors).

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/referential.rs::validate_referential_integrity`, `crates/cobre-io/src/validation/referential.rs::check_line_references`, `crates/cobre-io/src/validation/semantic/hydro.rs::check_filling_guards`, `crates/cobre-io/src/error.rs::LoadError`
- **Evidence:** Six entity cross-reference rules exist in matched pairs, one per crate, and the predicates coincide: cobre-core `validate_line_refs` checks `line.source_bus_id`/`target_bus_id` against the bus registry, cobre-io `check_line_references` checks the same two fields against the same set.
- **Fix-shape:** Name one owner per invariant class. The cross-reference and filling predicates are paradigm-neutral entity invariants, so either they live beside the entity in cobre-core and cobre-io's Layer 3 delegates to them, or cobre-io stays the sole owner and the cobre-core builder copies are deleted as unreachable — but not both, and the choice should be recorded once rather than settled per rule.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** the two `cobre-python` consumer arms the entry names are exhaustive matches (the kind-name table and the exception mapping) and had to be edited in the same ticket as the variant, together with a `crates/cobre-io/README.md` row the entry does not name.
- **Status:** fixed (2026-09-15) — `LoadError::CrossReferenceError` deleted (6 → 5 variants); both `cobre-python` exhaustive matches narrowed without a wildcard (manifest clippy the gate) and the README row deleted; the `cobre-io` and `cobre-cli` tests retargeted onto `PolicyIncompatible` Display and the CLI `ParseError` mapping; pytest green (ticket-029). The predicate-ownership half (one owner per invariant class, whether the duplicated line and hydro predicates collapse) was deliberately NOT done per D11 and stays for the generalization station. `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**CD-053 · Sev B · asymmetry · effort M · confidence high**
Confirmed as-scoped to two concrete divergences: the block-id-range defect is `BusinessRuleViolation` in block_bounds.rs but `InvalidValue` in referential.rs, and the duplicate-row rule keys per-column in Layer 5a versus per-row in Layer 3 so a disjoint-column duplicate is legal for six families and rejected for generic constraint bounds; plus the NCS negative-value check sits in the reference-only Layer 3 module.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/referential.rs::check_generic_constraint_bounds_validity`, `crates/cobre-io/src/validation/semantic/block_bounds.rs::check_bound_block_id_range`, `crates/cobre-io/src/validation/semantic/block_bounds.rs::check_duplicate_bound_rows`, `crates/cobre-io/src/validation/referential.rs::check_ncs_bounds_and_factors`
- **Evidence:** Two rules of the bound-override family are implemented twice.
- **Fix-shape:** Give the bound-override family one home and one rule set. The natural landing zone is the Layer 5a module that already models the family generically: extend its per-family descriptor table to cover generic constraint bounds and NCS bounds, and delete the Layer 3 copies, leaving Layer 3 with the dangling-id checks that its own header claims.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — Layer 5a wins (owner, 2026-09-15): `GENERIC_CONSTRAINT` is the seventh `FamilyMeta` row in `semantic/block_bounds.rs`, so generic-constraint bounds get the per-column duplicate rule and `BusinessRuleViolation` for an out-of-range `block_id`; the duplicate loops left `referential.rs::check_generic_constraint_bounds_validity`, which keeps only its non-family-shaped checks (endpoint presence, static-fold inversion, reference without activation). Deck-visible (CHANGELOG Changed entry), pinned by `test_generic_constraint_bounds_block_id_range_is_business_rule_violation` and `test_generic_constraint_bounds_disjoint_column_duplicate_now_accepted`. Deviation on the NCS half: `NcsBoundsRow` has no `block_id`, and the two NCS negative-value branches were unreachable (`constraints/ncs_bounds.rs::parse_ncs_bounds` rejects negative and non-finite `available_generation_mw`; `scenarios/non_controllable_factors.rs::validate_block_factors` rejects `factor <= 0.0`, both pinned by parser tests), so they were deleted with their two tests rather than relocated — guardian-verified, same precedent as the rules 4/5 retirement. `develop` @ `eb0b82ef`.

**CD-054 · Sev B · asymmetry · effort M · confidence high**
Confirmed narrowly: the uniform flat 'field is in id-set' reference blocks and the 20 copies of the `s.id >= 0` study-stage predicate are open-coded despite in-crate precedents (`FamilyMeta` table, `StageIdResolver`), and the header comment stands in for one message-template definition, conceding that Option-valued, nested, and per-plant-scoped references are not mechanically table-collapsible, so the residue is redundant idiom restatement rather than a single table replacing all 48 blocks.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/referential.rs::check_scenario_references`, `crates/cobre-io/src/validation/referential.rs::check_bounds_references`, `crates/cobre-io/src/validation/semantic/block_bounds.rs::FamilyMeta`
- **Evidence:** The check-and-emit shape for a dangling reference — index the rows, test membership in an id set, push an `InvalidReference` entry whose message reads "<RowType>[i] references non-existent <Entity> <id> via field '<field>'" — is written out 48 times in one file, spread over 13 functions;
- **Fix-shape:** Align the referential module to the pattern its sibling already proves rather than propagating the open-coded one. A descriptor carrying the row label, source file, target entity name and field name, plus one emit helper taking a descriptor and an id set, collapses the bulk of the 48 blocks and makes the message template a single definition instead of a header comment describing 48 copies.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** `s.id >= 0` occurs 34× crate-wide (2 in `referential.rs`), not "20 copies".
- **Status:** fixed (2026-09-15) — `DanglingRefDescriptor { file, target_entity, field }` with `emit_dangling_ref` (indexed rows) delegating to `emit_dangling_ref_at` (entity-named locations) is the single owner of the `references non-existent … via field` message; 38 named descriptors plus one runtime-built one (the correlation entity-type dispatch) carry 39 of the 48 sites, and the 9 left open-coded have a different shape (the seven `validate_variable_ref_entity` sites with no field clause, the unit-group-bounds site with its extra clause). Messages byte-identical: the 58 surviving referential tests and the two integration tests pinning message order pass unchanged. The second `s.id >= 0` copy in this file left with the deleted Layer-3 loop; the one remaining lives in `collect_study_stage_ids`, already shared by its two callers — the other crate-wide copies are out of this entry's scope. `develop` @ `eb0b82ef`.

**CD-055 · Sev C · asymmetry · effort S · confidence high**
Confirmed narrowly on the concrete drift: registry row 26 asserts a live semantic rule for `simulation.sampling_scheme.type`, a field `deny_unknown_fields` now rejects at parse-time with a different ErrorKind, while curated retirements elsewhere prove the table is maintained, so the demonstrated defect is this one drifted row, the unbound prose registry being the mechanism rather than a second proven drift.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/mod.rs:105`, `crates/cobre-io/src/validation/semantic/mod.rs::validate_semantic_stages_penalties_scenarios`, `crates/cobre-io/src/config/simulation.rs::SimulationConfig`
- **Evidence:** The only index of the numbered semantic rules is the doc table in `semantic/mod.rs`, and nothing links a table row to the function that implements it: rules live in twelve sibling modules, are wired through two hand-maintained dispatch lists, and are described in a third place.
- **Fix-shape:** Bind rule identity to code so the registry cannot drift silently. The cheapest version keeps the table but attaches each rule number to its implementing function as a doc anchor, so a removed rule leaves a dangling reference the doc build notices.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — row 26 retired (the field no longer exists under that name; `simulation.selection.method` is rejected at parse time by serde's `deny_unknown_fields` tag); both Layer 5a/5b tables audited at dispatch-list granularity against the ~49 dispatched functions: seven checks that ran but had no row are tabulated (5a rows 25b, 26a, 26b, 34a, 50; 5b rows 16a, 44a), three drifted rows reworded (5a rows 14 and 25, 5b row 31), one local inconsistency fixed (`travel_time.rs`'s `check_horizon_inertness` said Row 3b, its module table said 4), and every dispatched function carries a `Rule N` tag on its first doc line so a row greps to its implementation — the cheap binding the fix-shape asked for; no registry type. `develop` @ `eb0b82ef`.

**CD-056 · Sev B · asymmetry · effort M · confidence high**
Confirmed narrowly: the stage-axis out-of-horizon validator is missing for the five block-eligible non-thermal bound families (hydro/line/pumping/contract/hydro_unit_group), whose out-of-horizon rows are silently dropped by resolve_bounds, whereas thermal (5a) and NCS (Layer 3, different ErrorKind) reject hard, so the fix is to add the five missing checks; thermal's guard is legitimately family-specific (padded resolution region), making this a coverage/consistency gap rather than a claim that the two existing rules are misplaced.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/semantic/thermal.rs::check_thermal_bounds_override_stage_range`, `crates/cobre-io/src/validation/semantic/block_bounds.rs:98`, `crates/cobre-io/src/validation/referential.rs::check_ncs_bounds_and_factors`
- **Evidence:** A stage-axis rule for bound-override rows exists for exactly two of the seven families: `thermal_bounds` (Layer 5a, `BusinessRuleViolation`) and `ncs_bounds` (Layer 3, `InvalidReference`).
- **Fix-shape:** Give the stage axis the same table-driven treatment the block axis already has, so one rule covers every bound family instead of two families having bespoke rules and five having none. It belongs beside the block-axis rule in the Layer 5a family module, keyed off the same per-family descriptor and the same study-stage set, and it should reuse the crate's stage resolver rather than a fresh id-non-negative scan.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** the "thermal's guard is legitimately family-specific (padded resolution region)" clause is wrong — the padded cells `[n_stages, n_stages + k_max)` receive base values only; `resolve_bounds` keys thermal override rows through the same `stage_index` map as every other family. Thermal's `[0, n_stages)` position test was therefore a latent defect for gapped or 1-based study id sets (which `StageIdResolver` explicitly admits): an undeclared id inside the range was silently dropped, the last declared id was rejected.
- **Status:** fixed (2026-09-11) — rule 49 `check_bound_stage_id_range` (`semantic/block_bounds.rs`) admits by declared-id set membership across all six bound families; rule 16 retired. The CD-053 half (absorbing the NCS Layer-3 stage check) was deliberately NOT done — NCS keeps its Layer-3 check; CD-053 stays open. `fix/quality-tier1` a729a259 + 0d4c8c22 (merged) + `fix/quality-tier1-followups` 19521701 (merged to develop 2026-09-12).

**CD-057 · Sev B (A-risk) · asymmetry · effort M · confidence high**
Confirmed narrowly: the positional `FILE_ENTRIES` to `manifest_fields_mut` zip is guarded only by an equal-length assertion that cannot detect a same-arity reordering, so swapping two entries silently misassigns presence flags; the `ParsedData`/schema.rs list is a third parallel restatement of the file set but keyed by name (a DRY/fan-out concern), not part of the positional join.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/structural.rs::manifest_fields_mut`, `crates/cobre-io/src/validation/structural.rs::FileManifest`, `crates/cobre-io/src/validation/structural.rs::validate_structure`, `crates/cobre-io/src/validation/schema.rs::ParsedData`
- **Evidence:** One input file is spelled in four places that must agree by hand: its path string in `FILE_ENTRIES` (43 entries, plus the one `struct FileEntry` declaration the count of 44 includes), its `bool` field on `FileManifest` (43 fields), its slot in the fixed-length 43-element mutable-reference array, and its path string again plus its field on `ParsedData` in `validation/schema.rs`.
- **Fix-shape:** Collapse the parallel lists to one keyed registry so a file is declared once and looked up by name rather than by ordinal. The manifest becomes a lookup keyed on the registry's own entry identity instead of a 43-field struct plus a 43-slot array, which removes the positional join and the hand-kept field order along with it.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Re-raise-of:** CD-031 (precedent citation only — this entry cites the boundary-context config-projection-sprawl calibration precedent, it does not re-raise CD-031)
- **Status:** fixed (2026-09-14) — one keyed registry: `pub enum InputFile` (43 variants, no `Default`), a private `INPUT_FILES: &[FileEntry { key, relative, required }]` table, and `FileManifest { flags: [bool; INPUT_FILE_COUNT] }` with `present(InputFile) -> bool` as the only public read and a `pub(crate)` setter; `manifest_fields_mut`, `FILE_ENTRIES` and the 43 `pub bool` fields are gone; `test_input_files_registry_invariants` pins unique paths, variant order == table order, and the eight-file required set. `ParsedData` in `validation/schema.rs` stays name-keyed (DRY residue, not the positional join). ticket-004.

**CD-058 · Sev B · asymmetry · effort M · confidence high**
The crash-safety hole is specifically the in-place (O_TRUNC) overwrites of manifest.bin at checkpoint.rs:244 and of entities.csv/variables.csv in dictionary.rs, which can leave a truncated file replacing the previous good one; the cuts/basis/states .bin payload writes are covered by the manifest-last commit-signal design and are not an independent crash-safety hole.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/atomic.rs::write_bytes_atomic`, `crates/cobre-io/src/output/policy/checkpoint.rs::write_policy_checkpoint`, `crates/cobre-io/src/output/policy/checkpoint.rs:242`, `crates/cobre-io/src/output/dictionary.rs::write_entities_csv`, `crates/cobre-io/src/output/dictionary.rs::write_variables_csv`
- **Evidence:** atomic.rs's module doc opens 'Single owner of the write-side crash-safety contract: write to {path}.tmp, flush explicitly (never via Drop), then rename', and twelve of the thirteen writer modules import it.
- **Fix-shape:** Route every remaining output write through output/atomic.rs. For the policy artifact, serialize each payload to bytes as it already does and hand the buffer to write_bytes_atomic instead of std::fs::write, keeping manifest.bin last so the commit-signal ordering is unchanged;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** payload writes were NOT covered by manifest-last on the resume path — the same directory is rewritten in place and the old `manifest.bin` was never removed, so a crash mid-rewrite paired the old commit signal with new payloads. Also: `read_policy_checkpoint` lists `cuts/`, `basis/`, `states/` (no inventory) and the pool count against the manifest is only a `debug_assert_eq!` (`cobre-sddp/src/cut/fcf.rs::from_deserialized`), so a rewrite with fewer pools or with states export off left stale payloads that a release build read silently (the stale last pool becoming the terminal witness in `cobre-cli/src/commands/run/policy.rs`).
- **Status:** fixed (2026-09-11) — every checkpoint payload, the manifest and both dictionary CSVs go through `write_bytes_atomic`; a rewrite removes `manifest.bin`, then every previous `.bin` (and `states/` when none is written), before writing; manifest stays last. `fix/quality-tier1` 8375e63b (merged) + `fix/quality-tier1-followups` 3b363161 (merged to develop 2026-09-12).

**CD-059 · Sev B · asymmetry · effort M · confidence high**
The confirmed defect is narrowly the false module-doc contract at output/mod.rs:6-8 (write_results does not 'write all output artifacts' and does not mirror load_case) and the resulting undocumented CLI/Python hand-mirror; it does not establish that write_results must be expanded to orchestrate every artifact - that consolidation is the 0a design choice, not part of the present defect.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/mod.rs:6`, `crates/cobre-io/src/output/results_writer.rs::write_results`, `crates/cobre-io/src/output/results_writer.rs::write_training_results`
- **Evidence:** write_results at results_writer.rs:166 calls write_training_results and write_simulation_results only;
- **Fix-shape:** This is the cobre-io-side owner shape for CD-025 rather than a new duplication finding, and it should attach to that entry's fix shape. Two things belong in cobre-io: first, correct the contract statement now — either write_results genuinely orchestrates the full artifact set, or the mod.rs doc and the function name stop claiming it does, because a false ownership claim in the module doc is what lets the CLI/Python twin drift unnoticed.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — the `output` module doc, `write_results`'s own doc and the crate doc in `crates/cobre-io/src/lib.rs` state the true contract — the training result tables, the training dictionaries and the training/simulation completion metadata, with the scenario Parquet data, policy checkpoint, provenance, hydro-model exports, stochastic echoes and solver-stats sidecars written by the callers through the individual writer modules under the parity gate. The orchestration consolidation stays with its own entry (CD-025). `develop` @ `2a14fe56`.

**CD-060 · Sev B · asymmetry · effort M · confidence high**
The load-bearing, concretely-defective residue is the over-broad self-description of the two enumerations versus their partial hand-maintained lists - chiefly the axis-spelling gate's 'a later file cannot reintroduce a variant without failing this one test' asserted over a 21-of-34 subset; the wider 'no single owner' framing is the shape of the fix, not itself the proven defect.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/schemas.rs::costs_schema`, `crates/cobre-io/src/output/dictionary.rs::variables_csv_schemas`, `crates/cobre-io/src/output/schemas.rs::one_spelling_per_axis_across_every_output_schema`, `crates/cobre-io/src/output/stochastic.rs::noise_openings_schema`, `crates/cobre-io/src/output/hydro_models.rs::fpha_hyperplanes_schema`, `crates/cobre-io/src/output/dictionary.rs::bounds_schema`
- **Evidence:** `python3 - <<'EOF'` — schemas.rs opens with '//! Arrow schema definitions for all Parquet output files' yet 9 of the 34 output schemas are declared privately in three sibling writers.
- **Fix-shape:** Give the output-schema family one owner and derive both consumers from it. Move the nine sibling-declared schemas into output/schemas.rs alongside the twenty-five already there, then replace the three hand-maintained lists with a single crate-internal table that pairs each output file's relative path with its schema function.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** 10 of 35 schemas are declared outside `schemas.rs`; the gate list covers 22 of 35 (not 9/34 and 21/34).
- **Status:** fixed (2026-09-15) — live count was 25 in `schemas.rs` and 9 outside (not 26/10); all nine moved into `schemas.rs` as `pub(crate)`, and `OUTPUT_SCHEMAS: &[SchemaRegistryEntry]` (34 rows: schema fn plus optional `variables.csv` label) now drives `one_spelling_per_axis_across_every_output_schema` — its "a later file cannot reintroduce a variant" claim is true for the whole family — and `dictionary.rs::variables_csv_schemas()` (same 22 rows, same order, same labels; `variables_csv_total_columns` unchanged), with `output_schema_registry_has_no_duplicate_or_missing_rows` pinning the count and rejecting duplicate fns. A `path` column was tried and dropped at guardian review: its only production reader would have been a keep-alive assert, and each schema fn's doc already names its path. `develop` @ `2a14fe56`.

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
- **Status:** fixed (2026-09-15) — `SIMULATION_FAMILIES: &[SimulationFamily]` (14 rows: subpath, `fn(&System) -> bool` declared-predicate, batch adapter with one common signature over the unchanged per-family builders) drives both `SimulationParquetWriter::new` and `write_scenario`; the rule is stated once on `SimulationFamily` (a declared family's directory exists from construction; a partition is written only for a non-empty payload), `write_scenario` shrank from ~220 lines to a loop and lost its `too_many_lines` allow, the module's directory-tree doc gained the missing `transit_seed/` family, and `simulation_family_table_has_no_duplicate_or_missing_rows` pins the count against the payload's family fields. Output bytes unchanged (30 writer tests, parity script 0 mismatches). `develop` @ `2a14fe56`.

**CD-063 · Sev C · asymmetry · effort S · confidence high**
The concrete defensible defect is the cross-domain mis-homing of ensure_parent_dir inside stochastic.rs (imported by three unrelated writers) together with its two verbatim open-coded copies in scaling_report.rs:24 and provenance.rs:26; the 'ten identical prologues' is the broader repetition a fix would fold, not itself the load-bearing residue.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/parquet_helpers.rs::extract_required_date32`, `crates/cobre-io/src/output/stochastic.rs::ensure_parent_dir`, `crates/cobre-io/src/output/fixed_delivery.rs::write_fixed_delivery`, `crates/cobre-io/src/output/scaling_report.rs::write_scaling_report`, `crates/cobre-io/src/output/provenance.rs::write_provenance_report`
- **Evidence:** parquet_helpers.rs is the read side's answer to per-parser open-coding: six typed extract helpers, crate-internal, imported by every Parquet parser.
- **Fix-shape:** Add a write-side counterpart to parquet_helpers.rs beside output/atomic.rs, so the directory has one owner for the mechanics of getting a batch onto disk. Move ensure_parent_dir there out of the stochastic domain writer, fold the two open-coded copies in scaling_report.rs and provenance.rs onto it, and add one helper that takes a target path and a RecordBatch and performs the ensure-parent / default-config / atomic-write sequence, so the ten prologues collapse to one call each and the domain writers keep only their batch builders.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** the six copied extractors were byte-identical to one another with no cross-file wording difference; the one-word difference is against the shared pair, which says `missing required column "{name}"`, and that is the spelling the fold adopted, so three suites' assertions were reworded. `crates/cobre-io/src/output/solver_stats_writer.rs`'s `create_dir_all` on the output directory is a third directory-creation site the entry does not name, excluded and still open.
- **Status:** fixed (2026-09-15) — `ensure_parent_dir` homed in `output/atomic.rs` beside the atomic writers, the `scaling_report.rs`/`provenance.rs` inline copies folded onto it, and `write_batch_atomic(path, batch)` added (+1 round-trip test) and adopted by ten writers (ticket-034); the six copied extractors deleted onto the shared pair with three suites' missing-column assertions reworded to `missing required column` (ticket-033). Nine writers deliberately excluded — the dictionary writer threads its config, the training/simulation writers use `self.config` or create no parent — and `crates/cobre-io/src/output/solver_stats_writer.rs`'s `create_dir_all` on the output directory was excluded from the `ensure_parent_dir` home and stays open. `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

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
- **Correction (2026-09-11):** O(N · window depth), not O(N²): the overlap walk breaks past the window end.

**PD-008 · Sev B · allocation · effort M · confidence high**
Confirmed, narrowed to a one-time study-setup MPI broadcast payload (not any per-iteration hot-path cost) and with the title's count corrected: the skipped-and-rebuilt siblings are the seven entity index maps plus stage_index (eight, not 'three fields above'). The defensible residue is that cascade+network's five HashMaps are transmitted on the wire despite being pure, content-determined derivations of the seven entity slices already serialized ahead of them in the same struct, and thus locally reconstructible in rebuild_indices.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/system/mod.rs::System`, `crates/cobre-core/src/system/mod.rs::SystemRepr`, `crates/cobre-core/src/system/mod.rs::rebuild_indices`, `crates/cobre-core/src/topology/cascade.rs::CascadeTopology`, `crates/cobre-core/src/topology/network.rs::NetworkTopology`
- **Evidence:** The eight `HashMap` index fields carry `serde(skip)` and are rebuilt by `rebuild_indices` from `From<SystemRepr>` at :221;
- **Fix-shape:** Give `cascade` and `network` the same treatment their sibling index maps already have: mark both fields skipped on the wire, drop them from the deserialize shadow struct, and extend the existing rebuild step that `From<SystemRepr>` already calls so it reconstructs both topologies from the deserialized entity slices alongside the seven entity indices and the stage index. The rebuild is safe to make unconditional because both builders are content-determined: the cascade's topological order is drawn from a min-heap keyed on the raw entity id (so it does not inherit the surrounding map's iteration order), and every upstream and per-bus list is explicitly sorted by id before the builder returns.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** the narrowing (one-time broadcast, eight skipped maps) is accurate as written and the payload shape was unchanged by it; the shape change came from the `NetworkTopology` deletion in the same ticket, not from skipping the cascade.
- **Status:** fixed (2026-09-15) — `System.cascade` carries `serde(skip)` (nine skipped fields) and `rebuild_indices` reconstructs it from the entity slices on deserialize; `CascadeTopology` derives `Default`; `postcard_roundtrip_rebuilds_cascade_topology_bit_equal` pins the rebuild bit-for-bit; the network half left the wire with the type; MPI/broadcast gate passed (ticket-020). `feat/quality-tier45-closeout` dd03c153 + 8a72c69d (pending merge).

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
- **Correction (2026-09-15):** the `allgatherv` site the diagnosis abbreviates as `simulation.rs:305-357` is `crates/cobre-cli/src/commands/run/simulation.rs`, not `src/commands/simulation.rs`.
- **Status:** fixed (2026-09-15) — `SimulationOutput.partitions_written` deleted (7 → 6 pub fields); `SimulationParquetWriter` loses the inventory field and `write_partition` takes `&self` (14 call sites unchanged); `merge_simulation_metadata` keeps its two byte-identical `allreduce` calls and drops the `allgatherv` pair with its cast allow; MPI gate passed (ticket-028). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

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
- **Status:** fixed (2026-09-15) — `ValidationError::{DisconnectedBus, InvalidPenalty}` deleted with their Display arms and the false `Emitted by cobre-io validation` doc line (8 → 6 variants, Display exhaustive); `test_error_trait` retargeted onto `MissingUnitGroups`; the `network.rs` TODO gone with the file (ticket-018). `feat/quality-tier45-closeout` dd03c153 + 8a72c69d (pending merge).

**OD-011 · Sev B · speculative-generality · effort M · confidence high**
The single missing wire is at columns.rs:1136: the NCS curtailment objective is sourced from the stage-invariant entity field ncs.curtailment_cost instead of ResolvedPenalties::ncs_penalties(ncs_idx, stage_idx), making the NCS resolved axis write-only in production while its three siblings are read at columns.rs:347/687/722; the resolution and write path (incl. the penalty_overrides_ncs override at resolution/penalties.rs:387) is fully functional -- only the LP read is absent.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/model/resolved/penalties.rs::ncs_penalties`, `crates/cobre-core/src/model/resolved/penalties.rs::ncs_penalties_mut`, `crates/cobre-core/src/model/resolved/penalties.rs::NcsStagePenalties`, `crates/cobre-core/src/model/resolved/penalties.rs::ResolvedPenalties`
- **Evidence:** Every remaining call of the read accessor `ncs_penalties` sits past the `#[cfg(test)]` marker of its file (cobre-core line 416 is past 328;
- **Fix-shape:** Decide the axis one way and make the code say so. Either wire it: have the NCS column builder take its objective coefficient from the resolved per-(ncs, stage) cell the way the hydro, line and bus column builders already take theirs, which makes the declared stage override effective and puts all four penalty families on one read path.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Reviewer rating:** A — downgraded here because defender narrowed the claim to a single missing LP read at `columns.rs:1136`; the resolve and write path (including the `penalty_overrides_ncs` override) is functional, so the blast radius is one call site, not a spreading structural gap.
- **Status:** fixed (2026-09-11) — `fill_ncs_columns` prices curtailment from `ctx.resolved.penalties.ncs_penalties(ncs_sys_idx, stage_idx)`, on the same read path as the hydro, line and bus fills; `ncs_objective_tests` pins a stage override changing the coefficient. `fix/quality-tier1` a729a259 (merged to develop 2026-09-11).

**OD-012 · Sev C · speculative-generality · effort S · confidence high**
The genuinely unconsumed public surface is the population-statistics arm: population ci_95_half_width and the count accessor have only test callers, and the population variance/std_dev are public entry points whose sole non-test use is internal delegation within a population branch no production path reaches (the one consumer uses the sample arm exclusively); variance and sample_variance are conceded to be live internal delegates, not deletable outright.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/stats/welford.rs::WelfordAccumulator`, `crates/cobre-core/src/stats/welford.rs::variance`, `crates/cobre-core/src/stats/welford.rs::std_dev`, `crates/cobre-core/src/stats/welford.rs::sample_variance`, `crates/cobre-core/src/stats/welford.rs::ci_95_half_width`, `crates/cobre-core/src/stats/welford.rs::count`
- **Evidence:** The only production consumer of the accumulator is the forward-pass statistics aggregator, and it calls new, update, mean, sample_std_dev and sample_ci_95_half_width.
- **Fix-shape:** Keep the arm the single consumer uses and drop the mirrored one. Remove the population variance, population standard deviation, population confidence half-width and the sample variance accessor, keeping sample standard deviation and sample confidence half-width as the surface the aggregator reads, and fold their internal delegation into the two survivors.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — `WelfordAccumulator` exposes `new`/`update`/`mean`/`sample_std_dev`/`sample_ci_95_half_width` (plus `Default`); `count`, `variance`, `std_dev`, `sample_variance` and `ci_95_half_width` deleted with `sample_variance` folded into `sample_std_dev`; the `progress.rs` test retargeted onto the sample arm (divisor 5 → 4), `welford_count_tracks_updates` deleted, the README `stats::welford` row reworded (ticket-019). `feat/quality-tier45-closeout` dd03c153 + 8a72c69d (pending merge).

**OD-013 · Sev B · speculative-generality · effort M · confidence high**
The load-bearing residue is the build-time plus broadcast-wire cost of a reader-less structure: the System.network field lacks the serde(skip) its seven sibling index fields carry, so NetworkTopology is serialized into every postcard System payload (mirrored in SystemRepr) despite zero production reader -- the only callers are three sites in cobre-core/tests/integration.rs and the unit test at system/mod.rs:970.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/topology/network.rs::NetworkTopology`, `crates/cobre-core/src/topology/network.rs::build`, `crates/cobre-core/src/topology/network.rs:91`, `crates/cobre-core/src/system/mod.rs:87`, `crates/cobre-core/src/system/mod.rs::network`
- **Evidence:** The first count is zero: no file outside crates/cobre-core mentions `NetworkTopology`, `BusGenerators`, `BusLineConnection`, `BusLoads`, or calls `System::network()` anywhere in the workspace, cobre-python included.
- **Fix-shape:** Give it an owner and a consuming milestone or delete it, which is the register's own admission rule for an inert surface. The plausible owner is the reserved power-flow vertical, since bus-to-line, bus-to-generator and bus-to-load adjacency is exactly what a network formulation would pull;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).
- **Correction (2026-09-15):** the crate-root re-export of `NetworkTopology` is not among the entry's anchors and sat at `crates/cobre-core/src/lib.rs:109` at the entry baseline and at `3e90024f`; the `test_support` module declaration added in epic-01 (`7c28de14`) moved it to `:111`, where the deletion found it. Retired with the type.
- **Status:** fixed (2026-09-15) — `topology/network.rs` deleted with `System::network`, the `BusGenerators`/`BusLineConnection`/`BusLoads` adjacency types and their crate-root exports; `SystemRepr` 26 fields; the `cobre-core` `test-support` network tests and the two integration callers deleted with it (ticket-020). `feat/quality-tier45-closeout` dd03c153 + 8a72c69d (pending merge).

**OD-014 · Sev C · speculative-generality · effort S · confidence high**
The four helper FUNCTIONS (not the Broadcast* mirror types, which setup.rs:288 genuinely consumes) are absent from the production MPI path; serialize_system/deserialize_system retain a single integration round-trip assertion (integration.rs:989-995) as their only non-file consumer and serialize_parameters/deserialize_parameters have none beyond their own doctests, so the defensible residue is that the four functions duplicate the encoding cli/broadcast.rs:419/:456 open-codes, not that the whole broadcast module is unused.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/broadcast.rs::serialize_system`, `crates/cobre-io/src/broadcast.rs::deserialize_system`, `crates/cobre-io/src/broadcast.rs::serialize_parameters`, `crates/cobre-io/src/broadcast.rs::deserialize_parameters`
- **Evidence:** All four are `pub` and crate-root re-exported.
- **Fix-shape:** Delete the four helper functions and their crate-root re-exports, keeping the `Broadcast*` mirror types and their `From` conversions, which the CLI genuinely consumes. Rewrite the module doc so the seam it describes is the one that exists - the mirror types plus the generic value broadcaster in the CLI - rather than a usage example built on the deleted helpers, and keep the integration round-trip assertion by expressing it as a direct postcard round-trip over `System`, so the guarantee that `System`'s `Deserialize` rebuilds its lookup indices stays pinned by a test.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** besides the single integration round-trip assertion the entry names, twelve in-file `#[cfg(test)]` callers had to be re-expressed, not one; the three `Broadcast*` mirror types stay (reached from `crates/cobre-cli/src/commands/run/setup.rs`); and the `cobre-cli` path the diagnosis abbreviates as `cli/broadcast.rs` is `src/commands/broadcast.rs`, not `src/broadcast.rs`.
- **Status:** fixed (2026-09-15) — the four free `broadcast.rs` functions and their crate-root re-exports deleted; the three `Broadcast*` mirror types kept byte-identical under a rewritten module doc; the seven in-file tests and the integration `test_postcard_round_trip` re-expressed as direct postcard round-trips (two through the mirrors), so the index-rebuild guarantee stays pinned; MPI subset passed (ticket-021). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-015 · Sev C · speculative-generality · effort S · confidence high**
Both are pub speculative surfaces with no non-test consumer (only crate-root re-exports plus their own #[cfg(test)] assertions and doctests) - the title zero-consumer claim holds strictly for non-test callers; load_scalar_parameters_json specifically is the OD-002 shape (schema.rs:544 open-codes the constraints/generic_parameters.json join it wraps), while build_season_stage_map is NOT a duplicate of the production season-map owner but a distinct raw stage_id->season_id builder that nothing calls (resolve_stage_seasons at residual_derivation.rs:219 produces dense ordinals instead).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/extensions/scalar_parameters.rs::load_scalar_parameters_json`, `crates/cobre-io/src/stages.rs::build_season_stage_map`
- **Evidence:** Every hit outside the two defining files is a `pub use` re-export;
- **Fix-shape:** Delete both functions together with their crate-root and module re-exports. If a case-relative scalar-parameter entry point is wanted, re-add it beside the first real caller and route the schema-validation site through it, so a single owner holds the `constraints/generic_parameters.json` path literal - the same collapse already executed for the boundary checkpoint path accessor.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** `build_season_stage_map` lived in `crates/cobre-io/src/stages.rs` (`:509` at `3e90024f`), as the anchor says and not in `extensions/scalar_parameters.rs` as the plan's spec placed it; its three in-file tests went with it.
- **Status:** fixed (2026-09-15) — `load_scalar_parameters_json` and `build_season_stage_map` deleted with their module and crate-root re-exports and the latter's three in-file tests; `validation/schema.rs` keeps open-coding the join through `parse_scalar_parameters_json` (OD-002's business) (ticket-022). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-016 · Sev B · speculative-generality · effort M · confidence high**
The zero-non-test-consumer claim holds exactly (only lib.rs:134/137 re-exports; three in-file tests plus a no_run doctest exercise it); the load-bearing residue is the by-construction divergence at mod.rs:397-403 (residual derivation gated on manifest.stages_json) versus the unconditional pipeline.rs:224 call, so the second public carrier can leave residual_std_ratio = 1.0 unresolved - but the eleven per-file load_* helpers it wraps are NOT dead (schema.rs consumes them), so the confirmed defect is the aggregate orchestrator plus ScenarioData only.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/scenarios/mod.rs::load_scenarios`, `crates/cobre-io/src/scenarios/mod.rs::ScenarioData`, `crates/cobre-io/src/pipeline.rs:225`
- **Evidence:** Outside its own file the only two hits for either symbol are the crate-root `pub use` lines: no production caller, no integration-test caller and no Python-binding caller exists anywhere in the workspace.
- **Fix-shape:** Delete `load_scenarios` and `ScenarioData` together with their crate-root re-exports, and re-home the three inline tests that exercise them onto the assembly surface production actually uses. If an aggregate entry point is wanted as a supported library convenience rather than deleted, it must stop being a second assembly implementation: have it consume the same parsed artifacts the pipeline consumes so the residual-ratio derivation cannot be manifest-gated in one path and unconditional in the other, and drop the gate entirely.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-1 (cross-reference; verdict travels to Epic 9).
- **Correction (2026-09-15):** the entry's `lib.rs:134`/`:137` re-export lines held at its baseline and at `3e90024f`; the `test_support` module declaration added in epic-02 (`ac2e608a`) moved them to `:136`/`:139`, where the deletion found them. Of the three inline tests, two were deleted as subsumed or vacuous and one was retargeted into an existing integration test, rather than all three being re-homed as the fix-shape proposes.
- **Status:** fixed (2026-09-15) — `ScenarioData` and `load_scenarios` deleted with their `lib.rs` re-exports; the `residual_std_ratio` derivation property retargeted into `crates/cobre-io/tests/integration.rs::test_inflow_history_wired_into_system` (non-vacuity guard plus the `(0, 1)` bound; inversion proved to fail at `sqrt(1 - 0.3²)`); the README sentence restated without the type (ticket-023). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-017 · Sev C · speculative-generality · effort S · confidence high**
default_severity (validation/mod.rs:95) has no caller outside its own unit test AND its BusinessRuleViolation->Error classification contradicts the sole BusinessRuleViolation emission (add_warning at season.rs:175-177); the defensible residue is an uncalled, already-divergent parallel severity table, conceding the title's 'purports to describe the call sites' framing since the method's doc only claims a per-kind default, never a mirror of the emission sites.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/mod.rs::default_severity`, `crates/cobre-io/src/validation/semantic/season.rs:177`
- **Evidence:** The only references to `default_severity` anywhere in crates/ are its own declaration and the four assertions of its own unit test at validation/mod.rs:419-429;
- **Fix-shape:** Delete the method and its tautological unit test, leaving `add_error` / `add_warning` as the single owner of severity. If a per-kind default is actually wanted, invert the direction instead of deleting: make the table the one that decides, by routing every diagnostic through a single `add` entry point that consults the kind, and turn the `season.rs` warning into a deliberate documented override rather than a silent divergence.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** `BusinessRuleViolation` has 58 `add_error` emissions and 1 `add_warning`; there is no "sole emission".
- **Status:** fixed (2026-09-15) — the `impl ErrorKind` block (`default_severity`) and its tautological unit test deleted whole, leaving `add_error`/`add_warning` the single owner of severity; all fifteen `ErrorKind` variants intact and `season.rs` untouched; a pure deletion (ticket-024). The inverted road (table-driven `add`) deliberately NOT taken. `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-018 · Sev B · speculative-generality · effort M · confidence high**
ParsedData.penalties (schema.rs:75) has zero data.penalties reads workspace-wide and its rationale's saving is false (Layer-5 reads hydro.penalties at scenarios.rs:179, not the bundle); the scalar_parameters #[allow(dead_code)] (schema.rs:113) is merely REDUNDANT because the field is read at pipeline.rs:96 and moved at :244, so — narrowing the title — only its allow is stale while its rationale naming the resolution consumer is accurate.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/schema.rs::ParsedData`, `crates/cobre-io/src/validation/schema.rs::ParsedData`
- **Evidence:** `data.penalties` has zero references workspace-wide, so the field at schema.rs:75 is dead in fact;
- **Fix-shape:** Split the two by fact. For `penalties`, decide the field's fate rather than annotating it: either delete the field and its construction at schema.rs:669 (the sentinel already carries every value any parser needs), or, if the global defaults genuinely belong in the bundle for a future check, register it as a reserved seam in the mirror with an owner and a consuming milestone, and rewrite the comment to name the reader that will land instead of asserting one that already exists.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** at `3e90024f` the two allows sat at `crates/cobre-io/src/validation/schema.rs:77` (`penalties`, field `:78`) and `:116` (`scalar_parameters`, field `:117`) — the entry's `:75`/`:113` are its own-baseline values — and the redundant allow is in `validation/schema.rs`, not `extensions/scalar_parameters.rs:116` as the plan's spec placed it; the `:77` allow guarded a genuinely dead field.
- **Status:** fixed (2026-09-15) — `ParsedData.penalties` deleted with its construction, its `#[allow(dead_code)]` and the false rationale; the redundant allow on `scalar_parameters` removed (field kept, clippy clean); `minimal_global_penalties` deleted from `test_support`; `allow(dead_code)` in `cobre-io/src` 5 → 3 (ticket-022). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-019 · Sev B (A-risk) · speculative-generality · effort M · confidence high**
The defect is the ORDER-unguarded positional zip between FILE_ENTRIES and manifest_fields_mut (structural.rs:376): the sole guard asserts equal length only (structural.rs:612-623, comment: length not order), so swapping any two same-required-ness optional rows silently misassigns presence flags exactly as the helper's doc warns (structural.rs:396-397) while all tests stay green; conceding the title's 'three redundant lists' framing, the named-bool struct itself earns its place via the type-checked manifest.<field> reads in schema.rs, so the residue is the unguarded order coupling, not the existence of named fields.

- **Station:** cobre-core + cobre-io (sub-station C)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/validation/structural.rs::FileManifest`, `crates/cobre-io/src/validation/structural.rs::FILE_ENTRIES`, `crates/cobre-io/src/validation/structural.rs::manifest_fields_mut`
- **Evidence:** Three lists describe the same 43 input files and must stay in the same order: 43 `pub bool` fields on `FileManifest`, 43 `FileEntry` rows in `FILE_ENTRIES` (the 44th match is the struct declaration itself), and 43 `&mut m.<field>` entries returned by `manifest_fields_mut`.
- **Fix-shape:** Collapse the three lists to one by making the file table the single declaration and deriving both the storage and the accessors from it. The shape that keeps the named, type-checked reads the 34 call sites in validation/schema.rs depend on is a single ordered table of files paired with an enum key, with the manifest holding one flag array indexed by that key and named accessor methods generated alongside it, so adding a file is one edit and a mis-order is a compile error rather than a silent misassignment.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Re-raise-of:** CD-031 (precedent citation only — this entry cites the boundary-context config-projection-sprawl calibration precedent, it does not re-raise CD-031)
- **Status:** fixed (2026-09-14) — see CD-057: the enum's discriminant is the ordinal, so a table/enum reorder fails the registry test and a wrong key is a compile error; no named per-file accessor was generated (seven of the 43 flags have no reader). ticket-004.

**OD-020 · Sev C · speculative-generality · effort S · confidence high**
The removable defect is precisely the unread third parameter `_config: &Config` on write_dictionaries (dictionary.rs:71) and the module's sole `use crate::Config` (dictionary.rs:16) that exists only to name it; the caller write_training_results still legitimately holds &Config for its own fields, so only the writer's parameter and import can go, not the caller's signature.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/dictionary.rs::write_dictionaries`, `crates/cobre-io/src/output/dictionary.rs:71`, `crates/cobre-io/src/output/dictionary.rs:16`, `crates/cobre-io/src/output/results_writer.rs::write_training_results`
- **Evidence:** The parameter is spelled `_config`, the compiler-sanctioned marker for declared-and-never-read, and the file's sole `use crate::Config` exists only to name that unread parameter's type.
- **Fix-shape:** Drop the third parameter from `write_dictionaries`, drop the now-unneeded `use crate::Config` from the dictionary module, and drop the forwarded argument at the single call site in the training results writer. Then check whether that caller still needs its own `&Config` for anything else before narrowing its signature too.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Part-I:** I.3-7 (cross-reference; verdict travels to Epic 9).
- **Correction (2026-09-15):** the crate-root re-export of `write_dictionaries` sat at `crates/cobre-io/src/lib.rs:120` at the entry baseline and at `3e90024f`, and at `:122` from epic-02 (`ac2e608a`) onward; both the `output/mod.rs` and the crate-root sites re-export by name, so the signature change needed no export edit.
- **Status:** fixed (2026-09-15) — `write_dictionaries(path, system)`: the unread `_config` parameter, the `use crate::Config` import and the forwarded argument at the single `results_writer.rs` call site dropped; the 24 dictionary tests unedited (ticket-025). The caller `write_training_results` keeps its own `&Config` (still read for its own fields). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-021 · Sev C · speculative-generality · effort S · confidence high**
The narrower defect is needless pub visibility: both functions back only a same-module serde default-path attribute where a private fn suffices; and contra the title only default_bounds is actually crate-root public API (re-exported at output/mod.rs:47, zero external callers), while default_upper_bound_kind is pub but never re-exported, so it is merely pub inside a private module.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/manifest.rs::default_bounds`, `crates/cobre-io/src/output/manifest.rs::default_upper_bound_kind`, `crates/cobre-io/src/output/mod.rs:47`
- **Evidence:** `grep -rn '\bdefault_bounds\b\|\bdefault_upper_bound_kind\b' crates/ --include='*.rs'` — Both functions exist solely to satisfy a `#[serde(default = "...")]` attribute on a field in their own file: `default_upper_bound_kind` for `MetadataBounds.final_upper_bound_kind` at manifest.rs:214, `default_bounds` for `TrainingMetadata.bounds` at manifest.rs:412.
- **Fix-shape:** Make both functions private to the manifest module and remove `default_bounds` from the output module's re-export list and from the crate-root re-export in lib.rs. Verify the `schema` feature's export path does not name either function before narrowing, since a schemars-visible helper would change the committed schemas and CI diffs them.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** `default_bounds` is public at `cobre_io::output::default_bounds`, not at the crate root; `manifest.rs` carries no `JsonSchema` derive.
- **Status:** fixed (2026-09-15) — `default_bounds`/`default_upper_bound_kind` are private functions without `#[must_use]`, `default_bounds` dropped from the `output/mod.rs` re-export; the `#[serde(default = "…")]` attribute strings byte-identical and the seven legacy-JSON back-compat tests are the proof; no schemars surface involved (`cobre-io` has no `serde` feature) (ticket-026). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-022 · Sev C · speculative-generality · effort S · confidence high**
The defect is exactly the two fields IterationRecord.time_bwd_setup_ms (mod.rs:141) and time_fwd_setup_ms (mod.rs:149): they are the only two time_* fields the convergence-path conversion loop skips (slots 8 and 11, training_output.rs:517-530), so they are populated at training_output.rs:234,237 and read nowhere; the columns they doc-arrow to are real but fed solely from the per-worker WorkerPhaseTimings path, not from IterationRecord.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/output/mod.rs::IterationRecord`, `crates/cobre-io/src/output/mod.rs::IterationRecord`, `crates/cobre-io/src/output/training_writer.rs::build_iteration_timing_batch`
- **Evidence:** A workspace-wide grep for a field read of either name returns nothing (exit 1).
- **Fix-shape:** Confirm with the training owner whether the two setup timings were meant to travel on the per-iteration record as well as the per-worker record. If not, delete both fields from `IterationRecord`, delete the two producer assignments in the cobre-sddp training-output builder, and delete the two zero-initialisers each in the cobre-io training writer, results writer, convergence reader and output-module tests plus the cobre-cli summary fixture;
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — `IterationRecord.time_bwd_setup_ms`/`time_fwd_setup_ms` deleted (28 pub fields, 16 `time_*`) together with the `PartialRecord` fields, the two producer assignments in the `cobre-sddp` event arms and the zero-initialisers across six files — 31 pure deletions; no Parquet schema or column touched, the per-worker setup columns come from `WorkerPhaseTimings` (D9(a)) (ticket-027). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

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
- **Status:** fixed (2026-09-15) — `read_f32_vector_as_f64` deleted with its `#[allow(dead_code)]` and rationale (`codec.rs` allow count 5 → 4, `dead_code` allows 0); private, so no CHANGELOG line (ticket-030). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

**OD-025 · Sev C · speculative-generality · effort S · confidence high**
Conceding run_pipeline_with_artifacts is the legitimate working function and the four public lib.rs entry points each justify a distinct return shape, the narrower defect is the two pub(crate) intermediates run_pipeline (pipeline.rs:47) and run_pipeline_with_report (pipeline.rs:57): each is a one-line .map projection with exactly one caller that is itself a one-line lib.rs adapter (lib.rs:235 and lib.rs:266), so each re-derives a shape the public entry points already own and can be folded into its caller with no loss of any contract or caller.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/pipeline.rs::run_pipeline`, `crates/cobre-io/src/pipeline.rs::run_pipeline_with_report`, `crates/cobre-io/src/pipeline.rs::run_pipeline_with_artifacts`
- **Evidence:** `run_pipeline_with_artifacts` is the only function in the module that does work;
- **Fix-shape:** Collapse the module to its one working function and let the four public entry points in lib.rs do their own projection directly: the system-only loader maps away both the artifacts and the report, the artifacts loader maps away the report, the report loader maps the loaded case to its system, and the full entry point forwards unchanged. Keep the module doc's pointer about which public entry point returns warnings, restated against the public names rather than the private ones.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — `run_pipeline`/`run_pipeline_with_report` folded into `run_pipeline_with_artifacts`; `load_case` and `validate_case` are `.map` projections in `lib.rs`; not a public-API change (ticket-031). `feat/quality-tier45-closeout` 09381faa + 883c022e (pending merge).

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
- **Correction (2026-09-15):** the divergence is wider than the one field the entry describes: `zero_penalties` (`system/builder.rs`) was uniform `0.0` across all sixteen fields, every other zero-ish fixture was `0.0` on fifteen fields with `1000.0` on `inflow_nonnegativity_cost`, and `penalties_all(v)` hard-coded `1000.0` for that field instead of parameterising it.
- **Status:** fixed (2026-09-15) — `HydroPenalties::uniform(v)` (`entities/hydro.rs`) is the one uniform-value constructor, gated `#[cfg(any(test, feature = "test-support"))]` in `cobre-core`; the per-site sixteen-field hydro literals in `cobre-core` call it and a fixture that differs on one field overrides it with a functional update (ticket-001). A sibling `GlobalPenaltyDefaults::uniform(v)` was added by the same ticket but no fixture is uniform across the global type's fields, so it had zero callers and was deleted at plan completion (plan-level review finding). `cobre-io`'s own `penalties_all`/`make_global` were homed once in its fixture module by ticket-007/008 (TD-007), not folded onto these. `feat/quality-tier45-closeout` 7c28de14 + 5e06c30e + 16ca7c66 (pending merge).

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
- **Correction (2026-09-11):** comparators are exhaustive today; `clp_determinism.rs` has no `opt_f64_bits_eq` (it uses `SolveBits`). Latent.
- **Correction (2026-09-15):** a bare `f64_bits_eq` did not exist anywhere in the workspace at `3e90024f` (`git grep -n 'fn f64_bits_eq' 3e90024f -- crates` is empty); the fix introduced it in `cobre-core`'s `test_support` and the comparators became its callers.
- **Status:** fixed (2026-09-15) — the six struct comparators in `model/resolved/bounds.rs` destructure both sides with a `..`-free field pattern, so a field added to a bounds struct fails to compile until the comparator names it; `f64_bits_eq`/`opt_f64_bits_eq` hoisted to `cobre_core::test_support` as the shared bit comparators and `test_hydro_stage_bounds_bits_eq_distinguishes_signed_zero_and_equates_nan` pins the semantics (ticket-003). `clp_determinism.rs` untouched (it uses `SolveBits`, per the 2026-09-11 correction). `feat/quality-tier45-closeout` 7c28de14 + 5e06c30e + 16ca7c66 (pending merge).

**TD-005 · Sev B · duplication · effort M · confidence high**
Narrower than 'copy-pasted across five test sites': the plain zero-varying builders (make_bus/make_line/make_thermal/make_ncs/make_group/make_hydro) are structurally duplicated across topology/network.rs, system/mod.rs and tests/integration.rs, differing only in trivial axes (name string, a 100->200 capacity), so a field add to Line/Bus/etc. is O(sites); but system/builder.rs's bus/line/hydro are a deliberately date+name-parameterized variant for canonical-order tests, not plain copies, and the full eight-name family is not present at every one of the five sites.

- **Station:** cobre-core + cobre-io (sub-station A)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-core/src/topology/network.rs::make_line`, `crates/cobre-core/src/system/mod.rs::make_line`, `crates/cobre-core/tests/integration.rs::make_line`, `crates/cobre-core/src/system/builder.rs::line`, `crates/cobre-core/src/topology/cascade.rs::make_hydro`
- **Evidence:** The same builder family is declared in `topology/network.rs`, `system/mod.rs`, tests/integration.rs, `topology/cascade.rs` (make_hydro only) and twice inside `system/builder.rs` under the names `bus`/`line`/`hydro`/`thermal`/`ncs`/`contract`/`pumping`.
- **Fix-shape:** Hoist one entity-builder family into cobre-core behind the existing `test-support` feature, next to the entities it constructs, parameterised on the axes the current copies actually vary (id, bus id, operational date, name) and defaulting the rest, so a field addition is an O(1) edit rather than an O(sites) one. Have the five in-crate sites and tests/integration.rs call it, and drop the local copies.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** the axis list is incomplete: `downstream_id` and the three `max` bounds are live axes too, the mirror-unit-group state space is three-valued (on a bus / on `EntityId(0)` / absent), and no `2020`-dated no-group `Hydro` preset occurs in `cobre-core` at all — every `cobre-core` copy is `2024-01-01` and the `2020` variant lives in `cobre-stochastic`.
- **Status:** fixed (2026-09-15) — one entity-builder family in `crates/cobre-core/src/test_support.rs` — `BusSpec`/`LineSpec`/`HydroSpec`/`ThermalSpec`/`NcsSpec`/`ContractSpec`/`PumpingSpec`/`UnitGroupSpec`/`StageSpec`, each `Default` plus a `make_*` constructor parameterised on the axes the copies varied (id, bus, downstream, dates, name, the `max` bounds, the mirror-unit-group state) — replaces the `topology/network.rs`, `system/mod.rs`, `topology/cascade.rs`, `system/builder.rs` and `crates/cobre-core/tests/integration.rs` copies (ticket-002); `test_support::tests::make_hydro_mirror_unit_group_has_three_states` pins the three-valued axis. `feat/quality-tier45-closeout` 7c28de14 + 5e06c30e + 16ca7c66 (pending merge).

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
- **Correction (2026-09-15):** `write_json` was 18 definitions (15 byte-identical plus 3 variants), not 17 + 2; two byte-identical `write_parquet_batches` copies were undeclared; and `make_global` is not expressible through a uniform constructor, so only the all-`1.0` fixture was homed. The four rustdoc `make_global` stubs with `unimplemented!()` bodies are not definitions.
- **Status:** fixed (2026-09-15) — `write_parquet`, `write_parquet_batches`, `write_json` and `make_global` have one definition each in `crates/cobre-io/src/test_support.rs`; every byte-identical copy deleted and its callers pointed at the module (ticket-007 the Parquet writers, ticket-008 the JSON writer and the global-penalties fixture). The non-uniform `make_global` variants were deliberately NOT folded onto a parameterised constructor — they are not expressible through one; only the all-`1.0` fixture was homed. `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

**TD-008 · Sev C · duplication · effort S · confidence high**
The durable, threshold-independent residue is the intra-directory homing inconsistency -- scenarios/estimation.rs extracted to a sibling while scenarios/correlation.rs stays inline in the same directory, alongside seven multi-thousand-line inline modules -- not the violation of the unratified ~500-LOC number (and the sibling has 36, not 65, test fns).

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/constraints/generic.rs::tests`, `crates/cobre-io/src/scenarios/estimation.rs::tests`, `crates/cobre-io/src/scenarios/correlation.rs::tests`, `crates/cobre-io/src/resolution/bounds.rs::tests`, `crates/cobre-io/src/system/hydros.rs::tests`, `crates/cobre-io/src/stages.rs::tests`
- **Evidence:** docs/design/testing-architecture.md section 5.1 asks for one deterministic homing rule -- inline below roughly 500 test-LOC or 40 test fns, extracted to a sibling tests.rs above it -- and section 3.2 item 4 records the inline-giant-versus-extracted-sibling asymmetry as a ranked sustainability problem, naming cobre-sddp anchors.
- **Fix-shape:** Pick the threshold once and apply it mechanically across the input path rather than per author. Adopt the section 5.1 numbers as written (roughly 500 test-LOC or 40 test fns), extract the seven over-threshold modules to sibling tests.rs files following the shape scenarios/estimation.rs already uses -- `#[cfg(test)] mod tests;` in the parent, the module body moved verbatim into <module>/tests.rs with the crate-inner allow attributes carried along as module-inner attributes -- and leave everything under the threshold inline.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-11):** mechanical application hits ~15 input-path modules, not seven.

**TD-009 · Sev C · duplication · effort S · confidence high**
The genuinely redundant triplicated surface is make_batch plus the five non-determinism common cases (valid-sorted, negative-std, nan-mean, missing-column, empty); the three per-parser declaration_order_invariance tests are load-bearing determinism-hard-rule pins and are NOT bloat, and each parser's unit-specific error-message assertions must survive any consolidation.

- **Station:** cobre-core + cobre-io (sub-station B)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/scenarios/inflow_stats.rs::tests`, `crates/cobre-io/src/scenarios/load_stats.rs::tests`, `crates/cobre-io/src/scenarios/non_controllable_stats.rs::tests`, `crates/cobre-io/src/scenarios/non_controllable_stats.rs::make_batch`, `crates/cobre-io/src/scenarios/load_stats.rs::make_batch`
- **Evidence:** Three parsers over the identical four-column (entity_id, stage_id, mean, std) Parquet shape carry the same test template: valid-4-rows-sorted, negative-std reject, NaN-mean reject, missing-mean-column reject, empty-Parquet-returns-empty, declaration-order invariance.
- **Fix-shape:** Lift the shared four-column stats fixture into the shared test-support module from the first candidate -- one make_stats_batch(id_column_name, ids, stage_ids, means, stds) that takes the id column's name as an argument -- and drive the six common cases from a single table-driven or macro-generated block parameterized by (parse fn, id column name, unit suffix), so a new stats parser inherits the whole template instead of copying it. Reconcile the drift while doing it: decide whether mean-out-of-range and zero-std-accepted belong to all three parsers or only to the ones whose units make them meaningful, and give the missing-column test one spelling.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Correction (2026-09-15):** the out-of-range case is NCS-specific: it pins the availability-factor domain `[0, 1]`, a rule the inflow (m³/s) and load (MW) parsers do not have and must not acquire. The gap was closed by recording that the rule does not apply, plus exactly one new test — zero standard deviation accepted for the inflow-stats parser, this plan's single declared count-parity exception.
- **Status:** fixed (2026-09-15) — `make_stats_batch(id_column, …)` plus the `assert_stats_{happy_path,missing_column,negative_std,nan_mean,empty_file}` template in `crates/cobre-io/src/test_support.rs` drive the three stats parsers' common cases; the per-parser `declaration_order_invariance` tests and unit-specific message assertions kept (ticket-012). The out-of-range case was deliberately NOT propagated (NCS-specific, see correction); the one declared addition is `scenarios::inflow_stats::tests::test_zero_std_m3s_is_accepted`. `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

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
- **Status:** fixed (2026-09-15) — the eight `VALID_*_JSON` constants, `write_file` and `make_minimal_case` have one owner in `crates/cobre-io/src/test_support.rs`; the `validation/schema.rs`, `validation/referential.rs` and `crates/cobre-io/tests/helpers/mod.rs` copies were deleted outright (no re-export shim; `r#"` corpus count in the three files 0) (ticket-009). `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

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
- **Correction (2026-09-15):** 30 `pub(super)` functions, not 31 (`base_parsed_data` was private); the three `ParsedData` skeletons diverged on the bus vector and on the `Config` construction mechanism (parse versus struct literal), not on one scalar, so the fold proved the `Config` values equal before collapsing and parameterised the bus vector; `extensions/scalar_parameters.rs:210` was a false lead.
- **Status:** fixed (2026-09-15) — `validation/semantic/test_support.rs` lifted to the crate root as `crates/cobre-io/src/test_support.rs` behind `#[cfg(any(test, feature = "test-support"))]` (`tempfile` an optional dependency under the feature; the `ParsedData` builders `#[cfg(test)]`) (ticket-005); the four phase modules' hand-rolled penalty builders, `make_unit_group` and `ParsedData` skeletons fold onto it, the two validation-phase `Config` literals proved equal (`test_support::tests::test_minimal_config_equals_validation_phase_struct_literal`) and deleted, `base_parsed_data` parameterised on the bus vector (ticket-006). `make_pumping` kept as a distinct builder. `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

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
- **Status:** fixed (2026-09-15) — `make_system`/`make_config`/`make_output_context` homed once in `test_support::output`; the `convergence_reader.rs`/`results_writer.rs` copies deleted; the 40 output tests keep their names (ticket-010). `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

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
- **Status:** fixed (2026-09-15) — `test_support::output::read_first_batch(path)` is the one read-back helper; `fixed_delivery::read_batch`, `generic_constraints_echo::read_batch` and `solver_stats_writer::read_parquet` deleted onto it (ticket-010), and 27 of the 44 inline four-statement sites converted with their in-function `ParquetRecordBatchReaderBuilder` imports deleted rather than hoisted (ticket-011). The 17 sites that read past the first batch, keep the reader for a second `next()` or inspect the builder schema were deliberately left inline; `hydro_models.rs` and `convergence_reader.rs` untouched. `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

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
- **Correction (2026-09-11):** 11 count-only tests (10 with an umbrella row), not seven.

**TD-023 · Sev B · duplication · effort M · confidence high**
parquet_helpers.rs has zero #[cfg(test)] module (166 lines, all non-test), so the six extractors' missing-column and wrong-type SchemaError message contract is pinned by no owner-level test; the narrower residue drops 'all twelve reachable transitively' — consumer paths like scenarios/inflow_history.rs intercept the missing-column case as a legacy-layout error before the helper's arm, so transitive coverage is partial, not uniform.

- **Station:** cobre-core + cobre-io (sub-station D)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-io/src/parquet_helpers.rs::extract_required_int32`, `crates/cobre-io/src/parquet_helpers.rs::extract_required_date32`, `crates/cobre-io/src/parquet_helpers.rs::extract_optional_float64`
- **Evidence:** Zero `#[cfg(test)]` modules, which matches the inventory's `linesRaw: 166` equalling `linesNonTest: 166` for this file.
- **Fix-shape:** Add one inline test module to `parquet_helpers.rs` that builds a small in-memory `RecordBatch` and asserts, per extractor, the happy path, the missing-column message and the wrong-type message, so the parse-error contract is pinned once at its owner instead of being re-asserted incidentally in sixteen consumer test modules. This is the counterpart to the duplication findings: the same consolidation that removes redundant assertions elsewhere depends on the shared helper carrying its own contract test.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target-layering brief)
- **Status:** fixed (2026-09-15) — `parquet_helpers.rs` gained its inline contract module — per extractor the happy path, the missing-column message and the wrong-type message (+18 tests) — and it landed before any consolidation relied on the extractors (ticket-004). `feat/quality-tier45-closeout` ac2e608a + bd29eb5e + 9c5bbf5a (pending merge).

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
- **Status:** fixed (2026-09-15) — the six phrases reworded in `crates/cobre-stochastic/src/par/{precompute,lag_transition}.rs` (sequential per-stage traversal; per-stage evaluation path / hot path; lag-accumulation call site; the two evaluation-phase lag-transition call sites; `[ForwardSampler]` as the crate's own type where the prose said "forward sampler"); the two conceded generic LP phrases stay; `check-infra-genericity.sh` green. `develop` @ `eb0b82ef`.

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
- **Correction (2026-09-11):** the gate is at `sampling/mod.rs::` the entity-type match (`sampling/mod.rs:360` at `1baeeadb`); a typo fails loudly, so this is not a silent-accept.
- **Status:** fixed (2026-09-14) — `ClassSamplerParams` carried `class: EntityClass` (ticket-006: `EntityClass::as_str` added beside `from_wire`, the historical gate compares `EntityClass::Inflow`, `derive_class_forward_seed(u64, EntityClass)` hashes `class.as_str().as_bytes()` so both pinned constants are unchanged); ticket-007 then folded the class into `resolve_class_source`, which rejects `Historical` for load and NCS with the same message text. No `Display` impl. `plans/quality-tier3-footguns` tickets 006/007, `662bb3cc`.

**CD-067 · Sev B (A-risk) · bad-abstraction · effort M · confidence high**
The in-station residue is the in-crate GroupFactor.entity_type String field plus the two `!=` dispatch sites (resolve.rs:286, 352) that a boundary-parse to an in-crate EntityClass{Inflow,Load,Ncs} enum at build() would make compiler-checked; the source-of-truth String is cobre-core's CorrelationEntity.entity_type (scenario.rs:594) which is OUT of station (a cross-station note, not changed here), so the confirmed defect is the stochastic-side stringly-typed carrier + comparisons, not the cobre-core field.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:28`, `crates/cobre-stochastic/src/correlation/resolve.rs:338`, `crates/cobre-stochastic/src/tree/generate.rs:332`
- **Evidence:** The entity class is a closed 3-value set (inflow/load/ncs) but is carried as a String on GroupFactor and dispatched by `!=`/`==` at two sites in resolve.rs (apply_correlation_for_class L352, resolve_class_positions L286). The producers are bare string literals at generate.rs:332-334 and sampling/mod.rs:227/236/245. Nothing links the literal producer to the stored value at compile time: a typo (`"inlfow"`) or a vocab… Re-derive: `git show a136840d:crates/cobre-stochastic/src/correlation/resolve.rs | sed -n '27,28p;351,353p' ; git show a136840d:crates/cobre-…`
- **Fix-shape:** Introduce an in-crate closed enum EntityClass { Inflow, Load, Ncs }. Parse the cobre-core-sourced CorrelationEntity.entity_type String exactly once, at DecomposedCorrelation::build, into the enum stored on GroupFactor (replacing the String field). Type the apply_correlation_for_class and resolve_class_positions `entity_type` parameter and the generate.rs / sampling/mod.rs call-site literals on the enum, so every class comparison becomes an exhaustive compiler-checked match with a single String->enum boundary at build. The source String remains owned by cobre-core (a cross-station note, not a change here); the boundary-parse keeps this fix entirely inside cobre-stochastic. No paradigm noun and >=2 consumers, so it does not trip the L1 purity guardrail.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Correction (2026-09-11):** producers are the three `sampling/mod.rs` tag sites (`:231/:240/:249` at `1baeeadb`); cobre-io rejects an unknown `entity_type` in `referential.rs` (`:447–461`), so the finding is not user-visible.
- **Status:** fixed (2026-09-12) — in-crate `EntityClass { Inflow, Load, Ncs }` with `EntityClass::from_wire` parsed once in `DecomposedCorrelation::build` (an unknown tag is `InvalidCorrelation`); `GroupFactor.entity_type: EntityClass`; the applier and both producers (`ForwardSampler::sample`, `generate_opening_tree`) pass variants. ebba0508. CD-066's sampler-side gate is NOT covered (see its status).

**CD-068 · Sev B · duplication · effort M · confidence high**
The narrow residue is that the full-vector applier trio (apply_correlation + resolve_positions + GroupFactor.positions) is pub production surface with NO production caller (kept alive only as the per-class path's differential-test oracle, test_per_class_tree_matches_full_vector_*) and the per-class precompute (resolve_class_positions + GroupFactor.class_positions) is wholly unwired (zero callers, class_positions never populated) -- so exactly one applier (apply_correlation_for_class) and zero precompute run in production; this is duplicated-but-tested surface plus a dead precompute, NOT literal unreferenced dead code, and its removal-vs-wiring is the open question candidate perf-00 pulls the other way on.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:31`, `crates/cobre-stochastic/src/correlation/resolve.rs:253`, `crates/cobre-stochastic/src/correlation/resolve.rs:277`, `crates/cobre-stochastic/src/correlation/resolve.rs:307`
- **Evidence:** Production correlation runs exclusively through apply_correlation_for_class on the linear-scan fallback: the per-class precompute resolve_class_positions is never called in any build, so GroupFactor.class_positions stays None and the apply_group_precomputed branch it feeds is unreachable. The entire full-vector twin (apply_correlation + resolve_positions + GroupFactor.positions) has no non-test workspace consumer —… Re-derive: `git grep -nE 'apply_correlation\b|\.resolve_positions|resolve_class_positions' a136840d -- crates/cobre-stochastic/src crates/cob…`
- **Fix-shape:** Collapse to a single correlation applier. Preferred: since production uses only apply_correlation_for_class on the scan path, remove the full-vector twin (apply_correlation, resolve_positions, GroupFactor.positions) that only tests exercise and the never-called resolve_class_positions + GroupFactor.class_positions, leaving one applier over the shared apply_group_scan and one position-cache concept. Alternative, if the position-precompute is genuinely wanted for the forward-sampler hot loop: wire resolve_class_positions once at ForwardSampler/opening-tree setup and delete the redundant full-vector method rather than keeping both. Either way keep the spectral transform byte-identical and preserve the BTreeMap deterministic iteration order noted in-code; retire, not re-point, the segment-relative-vs-full-vector position-base distinction so the trap cannot be mis-merged.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Status:** fixed (2026-09-12) — `apply_correlation`, `resolve_positions`, `GroupFactor.positions`, `apply_group_scan`, `apply_group_precomputed`, `apply_correlation_for_class` and the four `test_per_class_tree_matches_full_vector_*` oracles are deleted; one applier (`apply_groups_for_class`) over positions resolved at build; the scheduled-profile composition is re-covered by `apply_groups_for_class_differs_between_scheduled_and_default_profile`. ebba0508, fe439fc0.

**CD-069 · Sev C · duplication · effort S · confidence high**
The residue is strictly the missing shared parameter-carrier: the three structs are byte-identical in field set and types (differing only in name and one Lhs doc line), so the defect is purely a triplicated bag-of-parameters seam (Sev C); each struct is a live, exercised type (not a speculative one-consumer abstraction) and the three point generators legitimately keep distinct algorithms -- only the parameter carrier is duplicated, nothing about the generators themselves.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/lhs.rs:89`, `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs:236`, `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs:236`
- **Evidence:** The three point-wise generators (sample_lhs_point, scrambled_halton_point, scrambled_sobol_point/_precomputed) each take a distinct spec struct whose field set is byte-for-byte the same six fields with the same types. out_of_sample.rs constructs all three from the same FreshNoiseSpec. There is no shared type, so the point-sampling parameter contract is expressed in triplicate and any new field (or a rename) is an O(… Re-derive: `git show a136840d:crates/cobre-stochastic/src/tree/lhs.rs | sed -n '88,102p' ; git show a136840d:crates/cobre-stochastic/src/tree…`
- **Fix-shape:** Introduce one shared spec type (e.g. QmcPointSpec / PointSampleSpec) in a common noise or tree module carrying sampling_seed/iteration/scenario/stage_id/total_scenarios/dim, and have sample_lhs_point, scrambled_halton_point, scrambled_sobol_point and scrambled_sobol_point_precomputed all accept it. The three generators keep their distinct algorithms; only the parameter carrier is unified, so a future field is a one-line change. Three present consumers make this a real de-duplication, not a speculative one-consumer abstraction, so it does not trip the L1 purity guardrail.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Status:** fixed (2026-09-12) — `NoisePointSpec` (`tree/point_spec.rs`, re-exported at the crate root) is the one carrier for the LHS, Halton and Sobol generators; the three structs are gone. 66b9b788. Residue recorded as CD-073 (its `stage_id` field carries the noise-group id).

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
- **Correction (2026-09-11):** path class is HOT (forward sampler, per iteration × scenario × stage under `scheme: out_of_sample` + `qmc_sobol`), not "setup/fitting-time"; the Measurement bullet's path-class clause is superseded.
- **Status:** fixed (2026-09-12) — `ForwardSampler::rebuild_noise_tables` builds one `SobolPrecomputed` per `(noise_group_id, noise_method)` and per class once per iteration (`ClassNoiseTables::refill`, `sampling/tables.rs`); `scrambled_sobol_point` reads it and is the only production generator, the direct one survives as `#[cfg(test)] scrambled_sobol_point_reference` (`sobol_point_matches_reference`). `crates/cobre-stochastic/tests/forward_sampler_golden.rs` pins the pre-change bits; `cobre-sddp/tests/forward_sampler_no_alloc.rs` asserts zero allocations across Sobol, Halton and LHS draws with a 70-entity group. `plans/quality-tier2-hotpath` 66b9b788 … fe439fc0 (merged to develop 2026-09-12).

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
- **Correction (2026-09-11):** path class is HOT (forward sampler under `scheme: out_of_sample` + `qmc_halton`), not "setup/fitting-time".
- **Status:** fixed (2026-09-12) — `HaltonPrecomputed` (primes + scramble tables, `tree/qmc_halton`) is built once per `(noise_group_id, noise_method)` table and read by `scrambled_halton_point`; the direct generator is `#[cfg(test)] scrambled_halton_point_reference` (`halton_point_matches_reference`). Same wave as PD-023.

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
- **Correction (2026-09-11):** path class is HOT (forward sampler under `scheme: out_of_sample` + `lhs`), not "setup/fitting-time".
- **Status:** fixed (2026-09-12) — `LhsPrecomputed` holds the `dim × total_scenarios` stratum table, built once per table with `perm_rng` consumed in the original order; `sample_lhs_point` reads one column per draw, O(dim · n) per group instead of O(dim · n²) (`lhs_point_matches_reference`, `lhs_precomputed_strata_rows_are_permutations`). Same wave as PD-023.

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
- **Status:** fixed (2026-09-12) — `DecomposedCorrelation::build(model, entity_order, dims)` resolves `GroupFactor.class_positions: Box<[usize]>` once at construction (`resolve_into`); the scan fallback is deleted (`test_class_positions_match_linear_scan_three_class_model`). Same wave as PD-023, ebba0508.

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
- **Status:** fixed (2026-09-12) — `DecomposedCorrelation::apply_groups_for_class(groups, class, class_noise, scratch)` correlates a group of any width from caller-owned scratch: `ScratchBuffers.corr_scratch` (sized `2 * noise_dim`, threaded through `SampleRequest.corr_scratch`) on the forward path, one `Vec` per tree in `generate_opening_tree`. The allocation guard in PD-023's status covers the wide-group path. ebba0508.

**PD-028 · Sev C · duplication · effort S · confidence high**
The eliminable cost is precisely the stage-constant profile+slice resolution (the profile_for_stage HashMap probe plus the factors.get BTreeMap<String> string-keyed walk) repeated 3 x n_openings per stage; I concede the per-group entity_type filter and the spectral transform must stay per-call/per-group -- only the profile-name and group-factor-slice lookup is hoistable to once per stage, so the defect is repeated map traversal, not the per-opening application itself.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs:345`, `crates/cobre-stochastic/src/tree/generate.rs:332`
- **Evidence:** profile_for_stage does a HashMap<i32,String>::get (resolve.rs:246), then factors.get(profile_name) is a BTreeMap<String,_>::get that walks the tree doing String comparisons. generate_opening_tree calls apply_correlation_for_class three times per opening (generate.rs:332-334, inflow/load/ncs) inside `for opening_idx in 0..n_openings`, with a stage_id that is constant for the whole inner loop. So the same profile + gr… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/correlation/resolve.rs | sed -n '345,351p'`
- **Fix-shape:** Resolve the &[GroupFactor] slice for the stage once, before the opening loop in generate_opening_tree (e.g. a per-stage accessor on DecomposedCorrelation that returns the group-factor slice for a stage_id), then iterate openings against that borrowed slice, filtering by entity_type in the loop. Folds naturally into candidate 1's once-per-stage/tree resolution.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Measurement:** UNMEASURED (setup/fitting-time path, not the training hot path) — recorded as deferred debt, below the Sev-A/B performance-sweep threshold.
- **Status:** fixed (2026-09-12) — `DecomposedCorrelation::groups_for_stage(stage_id)` resolves the profile and the group slice once per draw in `ForwardSampler::sample` and once per stage above the opening loop in `generate_opening_tree`. ebba0508.

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
- **Status:** fixed (2026-09-15) — road (a): `ParValidationReport` and `ParWarning` deleted with their re-exports; `validate_par_parameters` returns `Result<(), StochasticError>` with the fatal zero-standard-deviation guard byte-identical; five warning-property tests deleted, four renamed, the `ParValidationReport` doctest dropped (ticket-036). The warning was deliberately NOT surfaced (road (b)), so no Python-parity change. `feat/quality-tier45-closeout` 719ed408 + 858e3ad7 (pending merge).

**OD-027 · Sev C · redundant-wrapper · effort S · confidence high**
Narrowed: the confirmable residue is that all three bare forms have zero production callers (only crate tests + doctests exercise them) and estimate_ar_coefficients/estimate_seasonal_stats are additionally re-exported at the crate root (lib.rs:46-47) on top of the par::fitting:: path. I concede a single documented no-season-map convenience overload is a defensible public-API ergonomics choice for an L1 reusable kernel, so the defect is the unmarked, production-unconsumed redundancy of the trio (and its intent should be marked if kept), not that the functions are dead weight to delete outright.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/fitting/ar_coefficients.rs:88`, `crates/cobre-stochastic/src/par/fitting/correlation.rs:115`, `crates/cobre-stochastic/src/par/fitting/seasonal_stats.rs:221`
- **Evidence:** Each bare form is a pure forwarding shim: `estimate_ar_coefficients` (ar_coefficients.rs:88) and `estimate_seasonal_stats` (seasonal_stats.rs:221 -> `estimate_seasonal_stats_with_season_map(.., None)`) and `estimate_correlation` (correlation.rs:115 -> `estimate_correlation_with_season_map(.., None)`) do nothing but call their `_with_season_map` twin with `None`. Two of the three are re-exported at the crate root (li… Re-derive: `git show a136840d4f2ea137f685f0af6dac04254b983b60:crates/cobre-stochastic/src/par/fitting/ar_coefficients.rs | sed -n '88,101p'`
- **Fix-shape:** Remove the ergonomic overload asymmetry: since `_with_season_map` already takes `Option<&SeasonMap>` and is the sole production entry, either demote the three bare forwarders to `pub(crate)` (or delete them and have the inline tests/doctests call the `_with_season_map` form with `None` directly), and stop re-exporting `estimate_ar_coefficients`/`estimate_seasonal_stats` at the crate root. If the no-season-map convenience form is deliberately part of the crate's public API, mark that intent explicitly and keep exactly one such surface rather than a crate-root re-export plus a `par::fitting::` re-export of a test-only shim. Behaviour-neutral (the shims already delegate); blast radius is validation-free (the shim bodies and their test/doctest call sites).
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Status:** fixed (2026-09-15) — the three bare forwarders deleted with their docs hoisted onto the `_with_season_map` survivors (no rename, per OD-027(a)); 46 test calls converted, three doctests moved, four prose references in `cobre-sddp`'s `estimation_integration.rs` reworded to behaviour phrasing (ticket-037). `feat/quality-tier45-closeout` 719ed408 + 858e3ad7 (pending merge).

**OD-028 · Sev B · over-parameterization · effort M · confidence high**
Representability only: the config permits exactly two invalid/redundant states — (a) a class scheme selects External/Historical but the paired Option is None (caught only at runtime), and (b) a library is Some while the class scheme never reads it (silent dead input) — and the hard-wired historical_library:None for load/ncs shows the four Options are not independent; folding each class's scheme+library into a per-class source enum makes both states unrepresentable. No claim of a runtime bug: the MissingScenarioSource check is correct today.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs:263`, `crates/cobre-stochastic/src/sampling/mod.rs:317`
- **Evidence:** The four Options are not four independent knobs. build_forward_sampler pairs class_schemes.inflow with EITHER historical_library OR external_inflow_library (mutually exclusive by inflow scheme), and hard-wires historical_library: None for the load and ncs classes so they read only external_load_library / external_ncs_library. Validity is a (3 class x scheme) matrix flattened into 4 loosely-typed Options: each field… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/mod.rs | sed -n '263,289p;359,388p'`
- **Fix-shape:** Fold each class's scheme selector and its required library into one per-class source enum whose data-bearing variants carry the borrow: e.g. InflowSource { InSample, OutOfSample, Historical(&HistoricalScenarioLibrary), External(&ExternalScenarioLibrary) } and a LoadSource / NcsSource with only { InSample, OutOfSample, External(&ExternalScenarioLibrary) } (load and ncs never take a historical library, so the hard-wired historical_library: None disappears). build_forward_sampler then matches per-class variants instead of pairing a scheme with a maybe-present Option, so the scheme-to-library requirement becomes a compile-time consequence, the InSample-with-library silent-ignore state stops being representable, and the two mutually-exclusive inflow libraries can no longer both be Some. The MissingScenarioSource diagnostics move up to the config-construction boundary (where a library is or i…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Status:** fixed (2026-09-14) — two private enums inside the factory, `InflowSource<'a> { InSample, OutOfSample, Historical(&'a …), External(&'a …) }` and `ClassSource<'a>` without a `Historical` variant; `resolve_inflow_source` / `resolve_class_source` build them from the unchanged public `ForwardSamplerConfig`, `ClassSamplerParams.source: ClassSource` replaces `scheme` + two `Option`s, the hard-wired `historical_library: None` is gone, every `MissingScenarioSource` text and raise order is byte-identical; `test_build_historical_with_library_ignores_external_library` covers the both-libraries-`Some` state. The fix-shape's "diagnostics move to the config-construction boundary" was rejected by the owner (2026-09-13). ticket-007.

**OD-029 · Sev B (A-risk) · over-parameterization · effort L · confidence high**
The rationale's "no natural sub-grouping exists" clause is false — the four are a documented atomic seed reset together at each outer boundary and re-threaded verbatim through >=6 signatures; the defensible residue is the silent same-typed transposition hazard between derived_accum and derived_weight (both &[f64], both length n_hydros, adjacent positional args) plus the reset-together invariant restated across >=3 doc blocks, which a by-ref DerivedSeed aggregate removes. The #[allow] itself stays (all three functions remain >7 args), so this is not an eliminate-the-lint finding.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/sampling/external.rs:262`, `crates/cobre-stochastic/src/sampling/eta_inversion.rs:37`, `crates/cobre-stochastic/src/sampling/historical.rs:296`
- **Evidence:** The rationale asserts the four seed inputs (derived_lag_values, l_state, derived_accum, derived_weight) have 'no natural sub-grouping'. The code contradicts that: the same 4-tuple is re-declared verbatim as positional parameters in run_eta_inversion (eta_inversion.rs:37, passed straight through) and in standardize_historical_windows (historical.rs:296, whose own comment at line 292 says it 'mirrors standardize_exter… Re-derive: `git show a136840d:crates/cobre-stochastic/src/sampling/external.rs | sed -n '256,273p'`
- **Fix-shape:** Introduce a by-reference aggregate for the stage-0 derived seed, e.g. DerivedSeed<'a> { lag_values: &'a [f64], l_state: usize, accum: &'a [f64], weight: &'a [f64] }, constructed once where the seed is computed and passed as a single argument through standardize_external_inflow, run_eta_inversion, standardize_historical_windows, and the caller build_external_inflow_library. The shared per-hydro canonical-position ordering and 'empty accum means reset-to-zero' invariant, currently restated in three near-identical doc-comment blocks, attaches to the type once. This shrinks the 11-arg / 14-arg / 12-arg signatures and removes the same-typed &[f64] transposition hazard. This does NOT dispute the mirror's blanket sanction of allow-with-rationale as a load-bearing lint class; it disputes only the factual claim of this specific rationale. Once the seed is a struct the #[allow(too_many_arguments)…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Correction (2026-09-11):** arities are 11/13/13, not 11/14/12.
- **Status:** fixed (2026-09-14) — `DerivedSeed<'a> { lag_values, l_state, accum, weight }` (`Copy`) in `seeds.rs` with `DerivedInflowSeeds::as_seed(l_state)`, re-exported at the crate root; replaces the four positional parameters in all six signatures (`standardize_external_inflow` 11→8, `run_eta_inversion` 13→10, `standardize_historical_windows` 13→10, `build_external_inflow_library` 11→8, `build_historical_inflow_library` 13→10, `build_scenario_libraries` 13→10 — the entry's "three functions" undercounted). All six `#[allow(too_many_arguments)]` stay (still above 7); the three "no natural sub-grouping" rationales are rewritten and the two "mirrors <callee>" ones in `scenario_libraries.rs` name `DerivedSeed`. `seed_digest` hashes the same bytes. ticket-008 (+ `par_a_historical_replay.rs`, omitted from the ticket's frontmatter, added with owner approval).

**OD-030 · Sev C · speculative-generality · effort S · confidence high**
Narrowed: the defensible residue is only that SweepDirection::Ascending and its comparator arm (opening_tree.rs:168) have ZERO non-test constructors while every production path passes Descending -- an unwired second variant exercised solely by opening_tree.rs's own tests. I concede this does NOT establish the enum must be deleted: SweepDirection is a minimal, engine-neutral two-variant sort-direction API (not a speculative multi-variant fan-out), so keeping a two-way ordering knob on a generic L1 primitive is defensible; the finding is the unwired variant, not that the `direction` parameter is itself over-abstraction.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/opening_tree.rs:14`, `crates/cobre-stochastic/src/tree/opening_tree.rs:120`
- **Evidence:** SweepDirection has two variants but only Descending is ever selected outside tests. The two Ascending constructions (opening_tree.rs:662,676) sit above the sole #[cfg(test)] boundary at line 438, so they are unit-test-only. The single production caller in cobre-sddp/src/setup/mod.rs:469 passes Descending, and the only other production reference (cobre-sddp/tests/parity.rs) is also test code passing Descending. The A… Re-derive: `git grep -n 'SweepDirection' a136840d4f2ea137f685f0af6dac04254b983b60 -- crates/ | grep -vE 'lib.rs|tree/mod.rs|context.rs:60'; e…`
- **Fix-shape:** Collapse the speculative generality: since every production caller sorts largest-key-first, drop the SweepDirection enum and the direction parameter on set_solve_order, making the primitive sort descending unconditionally, and remove the now-dead SweepDirection::Descending argument from the one production caller (cobre-sddp/src/setup/mod.rs:469) plus its test mirrors (cobre-sddp/tests/parity.rs). The Ascending-only unit tests then either delete or fold into a private test helper that exercises the comparator directly, so no production surface exists solely to be tested. If instead the ascending ordering is a deliberately reserved seam for a planned selection policy, keep it but add an owner-signed reservation note (the CLAUDE.md 'unwired config is reserved, not dead' pattern) at the enum so the unwired variant reads as reserved rather than as silent speculative generality. Either way ke…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Correction (2026-09-15):** there was exactly one `Ascending`-only in-crate test (`solve_order_ascending_first_is_smallest_key`), not the two the plan's spec recorded nor the three its epic overview claimed: `solve_order_ties_broken_by_canonical_omega` is mixed and kept its descending half, and the third cited line sat inside `solve_order_is_idempotent_function_of_keys`, a `Descending`-only test. The removal took a public parameter out of `cobre-sddp`'s production surface (`setup/mod.rs`), not only `cobre-stochastic`'s.
- **Status:** fixed (2026-09-15) — `SweepDirection` deleted; `OpeningTree::set_solve_order(keys)` sorts descending by key as a documented contract, ties broken by ascending canonical order; the one `Ascending`-only test deleted; `noise_key.rs` untouched; every golden and determinism pin unedited and green (stochastic, `cobre-sddp` `parity`/`deterministic`/`conformance`, MPI subset) (ticket-038). The reserved-seam road (owner-signed reservation note) deliberately NOT taken. `feat/quality-tier45-closeout` 719ed408 + 858e3ad7 (pending merge).

**OD-031 · Sev C · speculative-generality · effort S · confidence high**
Confirmed as three pub StochasticError variants with zero production construction site (not compiler-dead: each is Display/Debug-exercised by inline tests and exhaustively matched); the sharper residue is that SpectralDecompositionFailed is additionally a live doc lie at context.rs:593 (promising a return spectral.rs never makes, since it maps failures to InvalidCorrelation), whereas SeedDerivationError and UnsupportedSamplingScheme are merely never-constructed variants removable with no match-arm edit.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/error.rs:5`, `crates/cobre-stochastic/src/error.rs:20`, `crates/cobre-stochastic/src/error.rs:45`, `crates/cobre-stochastic/src/error.rs:74`, `crates/cobre-stochastic/src/context.rs:593`
- **Evidence:** Of the 9 StochasticError variants, three have zero production construction sites workspace-wide. SpectralDecompositionFailed is never returned — every spectral-decomposition failure in correlation/spectral.rs returns InvalidCorrelation (spectral.rs:83, spectral.rs:99), yet context.rs:593 documents build_stochastic_context as returning SpectralDecompositionFailed, a promise no code path keeps. SeedDerivationError is… Re-derive: `for v in SpectralDecompositionFailed SeedDerivationError UnsupportedSamplingScheme; do echo "== $v =="; git grep -n "$v" a136840d…`
- **Fix-shape:** Two roads, owner's pick. (a) Delete the three never-produced variants from the pub StochasticError enum, delete the context.rs:593 doc bullet that promises SpectralDecompositionFailed, and retarget the cobre-sddp from_stochastic_error test (cobre-sddp/src/error.rs:241, out of station — cross-reference only) to a live variant such as InsufficientData. (b) If a producer is genuinely planned for any of them, enter that variant in the reserved-seam register with an owner and a consuming milestone, which the register's own contract requires and which none of the three currently has. Blast radius is bounded: error.rs enum + one context.rs doc line + one cobre-sddp test. Removing variants is a breaking change to the pub StochasticError (re-exported at lib.rs `pub use error::StochasticError`), so it rides the same 'licensed public-API break' gate the mirror applies to the superseded cut-sync pu…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Status:** fixed (2026-09-15) — road (a): `SpectralDecompositionFailed`, `SeedDerivationError` and `UnsupportedSamplingScheme` deleted (9 → 6 variants), the exhaustive Debug match kept without a wildcard, the `context.rs` doc bullet promising `SpectralDecompositionFailed` gone (`jacobi_eigen` is infallible and every fallible spectral path returns `InvalidCorrelation`, non-PD input clipped); `cobre-sddp`'s `from_stochastic_error` test retargeted onto `InsufficientData` (ticket-039). `feat/quality-tier45-closeout` 719ed408 + 858e3ad7 (pending merge).

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
- **Correction (2026-09-15):** the anchors are incomplete: `crates/cobre-stochastic/tests/forward_sampler_golden.rs` cloned the whole prelude too (`make_bus`/`make_hydro`/`identity_correlation` at `:192`/`:205`/`:292` at `3e90024f`) and the entry names only `crates/cobre-stochastic/tests/forward_sampler.rs` and `crates/cobre-stochastic/tests/conformance.rs`.
- **Status:** fixed (2026-09-15) — `forward_sampler.rs`, `conformance.rs` and `forward_sampler_golden.rs` take `make_bus`/`make_hydro`/`make_inflow_model`/`identity_correlation` from `tests/common` (the eleven-helper clone md5-proved identical before deletion; `common/mod.rs` gained `method_stage`, `make_sampler_config`, `build_test_system`, `build_test_ctx`, `stages_from_system`, `tables_for`); the forward-sampler golden arrays and scalars byte-identical; eight of nine binaries on the prelude (ticket-017). `feat/quality-tier45-closeout` 2bf30ece + 59c5c09f (pending merge).

**TD-025 · Sev B · duplication · effort S · confidence high**
Narrowed: the provable byte-identical duplication is specifically the make_model pair (precompute.rs:653 == evaluate.rs:515, empty diff) and the make_stage pair (precompute.rs:628 vs evaluate.rs:490) differing only by the dummy_date wrapper. The broader '~9 par test sites' fragmentation claim is softer: the validation.rs:160 make_model hardcodes mean_m3s:100.0 and aggregate.rs:169 make_stage uses duration 720.0/branching_factor 1 vs precompute/evaluate's 744.0/10, so those copies are structurally-similar-but-divergent (different default literals), not byte-identical dupes — the field-add-fragmentation cost is real but the strictly-identical residue is the precompute<->evaluate make_model/make_stage pair.

- **Station:** cobre-stochastic (sub-station par)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/par/precompute.rs:653`, `crates/cobre-stochastic/src/par/evaluate.rs:515`, `crates/cobre-stochastic/src/par/precompute.rs:628`, `crates/cobre-stochastic/src/par/evaluate.rs:490`, `crates/cobre-stochastic/src/par/validation.rs:160`, `crates/cobre-stochastic/src/par/aggregate.rs:169`
- **Evidence:** Every par unit-test module rebuilds the same two cobre-core structs from scratch. precompute and evaluate carry a genuinely identical make_model; the make_stage copies differ only in default constants, not shape. There is no shared O(1)-field-add Stage/InflowModel builder in this crate (it exists only in cobre-sddp/tests/common), so a field added to Stage or InflowModel must be edited in every copy — the fixture-fra… Re-derive: `diff <(git show a136840d:crates/cobre-stochastic/src/par/precompute.rs | sed -n '653,669p') <(git show a136840d:crates/cobre-stoc…`
- **Fix-shape:** Provide one parameterized builder each for Stage and InflowModel (the O(1) field-add builder pattern the yardstick §3.1.4 credits to cobre-sddp/tests/common/builders.rs), reachable from both the inline unit tests and the integration binaries via the crate test-support surface proposed in the companion finding; each existing make_stage/make_model becomes a thin wrapper that overrides only the defaults it cares about (duration, branching_factor, mean). Delete the byte-identical make_model copy pair outright. Keeps every test body and assertion unchanged (a refactor, not a coverage change); neutral to layering; no new production abstraction.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus
- **Status:** fixed (2026-09-15) — the `precompute.rs`/`evaluate.rs` `make_model`/`make_stage` pair and the other par copies route through `cobre_core::test_support::make_stage` and `cobre_stochastic::test_support::{InflowModelSpec, make_inflow_model}`, each former copy a thin override of the defaults it cares about; the byte-identical `make_model` pair collapsed (tickets 014, 015). `feat/quality-tier45-closeout` 2bf30ece + 59c5c09f (pending merge).

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
- **Correction (2026-09-11):** `sampling/mod.rs` `make_hydro` (`:670` at `1baeeadb`) is the canonical 9-copy variant, not a divergent outlier.
- **Correction (2026-09-15):** all five line anchors had drifted by this plan's baseline `3e90024f`: `sampling/mod.rs:670 → :921` (`make_hydro`), `sampling/mod.rs:791 → :1042` (`uniform_tree`), `sampling/class_sampler.rs:446 → :486` (`uniform_tree`), `sampling/external.rs:2513 → :2511` and `sampling/historical.rs:1902 → :1905` (`make_hydro`).
- **Status:** fixed (2026-09-15) — the sampling-module `make_hydro`/`make_bus`/`Stage` literals route through `cobre_core::test_support` (`HydroSpec`/`BusSpec`/`StageSpec`, the latter gaining `index`/`start_date`/`end_date` plus `single_block`; `dummy_date` deleted in `par/precompute.rs` and `par/evaluate.rs`) (ticket-014); `uniform_tree`, `MonthlyLabels` + `monthly_season_map`, `quarterly_season_map` and `weekly_season_map` homed once in `crates/cobre-stochastic/src/test_support.rs` behind the crate's new `test-support` feature (ticket-015). `feat/quality-tier45-closeout` 2bf30ece + 59c5c09f (pending merge).

**TD-029 · Sev C · duplication · effort M · confidence high**
Narrowed to the single verified byte-identical duplication — saa_golden_value.rs's 28-line identity_correlation copy of the QMC binaries' fixture; 're-declares the integration-binary fixture prelude' over-reaches because saa carries only this one of the ~8 shared prelude helpers and its make_stage is a distinct 3-arg builder, so identity_correlation is the only byte-identical copy.

- **Station:** cobre-stochastic (sub-station sampling)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/tests/saa_golden_value.rs:53`, `crates/cobre-stochastic/tests/saa_golden_value.rs:32`
- **Evidence:** saa_golden_value.rs (my thematically-sampling integration binary, 139 lines) carries a verbatim copy of the same identity_correlation fixture the tree-noise QMC integration binaries (halton/sobol/lhs_integration.rs) each declare, plus its own Stage builder. This is the integration-binary layer of the same fixture-fragmentation smell as candidate 1: the near-verbatim prelude the station test-bloat lens names for halt… Re-derive: `for f in saa_golden_value halton_integration sobol_integration lhs_integration; do echo --- $f; git show a136840d:crates/cobre-st…`
- **Fix-shape:** When the shared integration prelude is consolidated (testing-architecture.md 5.2/5.1 - a per-crate test-support surface plus the single integration binary), saa_golden_value.rs drops its private identity_correlation and make_stage and pulls both from cobre-stochastic's test-support fixtures alongside the QMC binaries. saa's own contribution is small (one 28-line duplicate + one Stage builder); the bulk of the integration-prelude debt is the tree-noise QMC-prelude headline, so this candidate is scoped to the saa binary and defers the cross-binary consolidation to that cell's finding. No behavior change: the six pinned golden constants and their assertions are untouched.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus
- **Correction (2026-09-11):** `identity_correlation` is at `saa_golden_value.rs:53`, not `:32`.
- **Correction (2026-09-15):** the QMC trio already imported the correlation fixture from `tests/common` (hoisted by `80e6e896`, before this plan); `saa_golden_value.rs` held the only remaining private copy.
- **Status:** fixed (2026-09-15) — `saa_golden_value.rs` drops its private `identity_correlation` for the prelude's (ticket-016) and keeps a same-name blockless `make_stage` wrapper over the prelude's builder (ticket-017); the six golden constants byte-identical. `feat/quality-tier45-closeout` 2bf30ece + 59c5c09f (pending merge).

**TD-030 · Sev B · duplication · effort M · confidence high**
Narrowed: the confirmed residue is strictly the byte-identical eight-helper fixture prelude physically copied across the QMC trio (halton/sobol/lhs), with make_hydro also copied into reproducibility (verified) and no tests/common/ home, already drifting into two identity_correlation return types; I concede the ~50 genuinely per-family lines (make_stage_<family> and build_<family>_context, differing only in NoiseMethod variant / block presence / panic-string) are legitimate and must survive any consolidation as thin per-family wrappers -- the finding is the shared prelude only, not the whole file, and it is distinct from (cites, does not restate) the cobre-sddp Oracle-harness mirror entry.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/tests/halton_integration.rs:207`, `crates/cobre-stochastic/tests/sobol_integration.rs:207`, `crates/cobre-stochastic/tests/lhs_integration.rs:206`, `crates/cobre-stochastic/tests/halton_integration.rs:43`, `crates/cobre-stochastic/tests/halton_integration.rs:105`
- **Evidence:** The two diffs return empty (exit 0), so the eight fixture helpers -- approx_erf (h/s/lhs L43-51), norm_cdf (L53-55), identity_correlation (h/s L105-133, lhs L104-132), correlated_correlation (h/s L135-163, lhs L134-162), identity_correlation_model (h/s L165-192, lhs L164-191), make_bus (h/s L194-205, lhs L193-204), make_hydro (h/s L207-255, lhs L206-254; a single 49-line copy), make_inflow_model (h/s L257-267, lhs L… Re-derive: `bash -c 'diff <(git show a136840d:crates/cobre-stochastic/tests/halton_integration.rs | sed -n "41,55p;105,267p") <(git show a136…`
- **Fix-shape:** Hoist the shared fixture prelude out of the per-binary copies into one home. The crate already dev-depends on cobre-core with its `test-support` feature but exposes NO `test-support` feature of its own and has NO tests/common/. Two roadmap-consistent shapes: (a) interim -- add crates/cobre-stochastic/tests/common/mod.rs holding make_bus/make_hydro/make_inflow_model/identity_correlation/correlated_correlation/identity_correlation_model plus the numeric approx_erf/norm_cdf, and `mod common;` it once from each integration binary (mirrors cobre-sddp/tests/common/); (b) end-state per the yardstick §5.2 -- expose the entity/InflowModel/correlation builders behind a cobre-stochastic `test-support` feature gated `#[cfg(any(test, feature = "test-support"))]`, and push the deck-agnostic numeric helpers (approx_erf/norm_cdf) into cobre-core's `test-support` surface (the yardstick's rule that gener…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus
- **Correction (2026-09-15):** the correlation builders (`identity_correlation`, `correlated_correlation`, `identity_correlation_model`) were already hoisted to `crates/cobre-stochastic/tests/common/mod.rs` by `80e6e896`, before this plan, so this plan's share was the rest of the prelude, not the correlation half.
- **Status:** fixed (2026-09-15) — `crates/cobre-stochastic/tests/common/mod.rs` is the prelude — named re-exports of the `cobre-core` and `cobre-stochastic` test-support surfaces, the presets `deficit_bus`/`sized_hydro`/`default_inflow_model`/`saa_stage` and the correlation builders (`correlation_model` renamed `correlated_correlation_model`), under `#![allow(dead_code, unused_imports)]` with a reason — and the QMC trio plus `reproducibility.rs` consume it (ticket-016). `approx_erf` deliberately NOT re-exported (no direct caller; unused re-exports deleted by rule). `feat/quality-tier45-closeout` 2bf30ece + 59c5c09f (pending merge).

**TD-031 · Sev C · asymmetry · effort M · confidence high**
Narrowed: no RATIFIED homing threshold exists at baseline -- testing-architecture.md section 5 is a 'Proposed standard' and section 5.1's ~500-test-LOC/~40-fn line is explicitly a 'proposal' -- so the surviving claim asserts only (a) the verified intra-crate inconsistency (inline vs extracted sibling coexist with no deciding rule) and (b) the extreme outlier tree/generate.rs at ~2030 inline test-LOC versus the already-extracted par/fitting form; it is a Sev C uniformity/navigability smell resolvable by relocation, NOT evidence that any specific numeric threshold is the correct cut nor a coverage/bloat defect.

- **Station:** cobre-stochastic (sub-station tree-noise)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/tree/generate.rs:341`, `crates/cobre-stochastic/src/correlation/resolve.rs:446`, `crates/cobre-stochastic/src/normal/precompute.rs:299`, `crates/cobre-stochastic/src/tree/opening_tree.rs:438`, `crates/cobre-stochastic/src/correlation/spectral.rs:286`
- **Evidence:** Within this one crate the unit-test homing rule is a coin-flip -- exactly the yardstick's §3.2.4 (inline-giant vs extracted-sibling asymmetry) and §5.1 (a single deterministic homing threshold, proposal ~500 test-LOC / ~40 test fns). tree/generate.rs keeps ~2030 test-LOC INLINE (4x over the proposed threshold), correlation/resolve.rs ~531 and normal/precompute.rs ~400 inline, while par/fitting extracts its tests to… Re-derive: `bash -c 'for f in tree/generate.rs tree/opening_tree.rs correlation/resolve.rs normal/precompute.rs correlation/spectral.rs noise…`
- **Fix-shape:** Adopt the yardstick §5.1 deterministic homing rule as a crate-wide lint: unit tests stay inline below a fixed threshold (the proposal is ~500 test-LOC or ~40 test fns), extracted to a sibling `tests.rs` above it. Under that rule tree/generate.rs (~2030), correlation/resolve.rs (~531) move to `<module>/tests.rs` siblings (matching the already-extracted par/fitting form), while small modules (noise/rng.rs ~40, noise/seed.rs) stay inline. Pure relocation, no test body/assertion/tier/gate change, count-neutral -- resolves the intra-crate coin-flip without touching coverage.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus
- **Correction (2026-09-11):** 16 modules exceed 500 inline test LOC (e.g. `sampling/external.rs` 2312); the anchor list is under-scoped.

**TD-032 · Sev C · duplication · effort M · confidence high**
The duplication narrows to the byte-identical make_bus/make_stage/make_hydro/identity_correlation across context.rs, provenance.rs and tests/reproducibility.rs plus the fourth date-differing make_hydro in seeds.rs; it does NOT extend to make_inflow_model (a legitimate parameterized-superset vs fixed-value split) nor to seeds.rs's make_stage/season-map builders (distinct signatures), so the defensible residue is one shared entity-builder surface owed for exactly those four builders, coverage-neutral with no test deleted.

- **Station:** cobre-stochastic (sub-station seam)
- **Baseline:** `a136840d4f2ea137f685f0af6dac04254b983b60`
- **Anchors:** `crates/cobre-stochastic/src/context.rs:852`, `crates/cobre-stochastic/src/provenance.rs:96`, `crates/cobre-stochastic/tests/reproducibility.rs:68`, `crates/cobre-stochastic/src/seeds.rs:198`
- **Evidence:** Five entity builders recur across three seam test regions. Bodies were read in full at baseline (context.rs:793-960, provenance.rs:35-200, tests/reproducibility.rs:1-160): make_stage (context 814 / provenance 58 / reproducibility 43), make_bus (context 839 / provenance 83 / reproducibility 30), make_hydro (context 852 / provenance 96 / reproducibility 68) and identity_correlation (context 914 / provenance 158 / repr… Re-derive: `for f in src/context.rs src/provenance.rs src/seeds.rs tests/reproducibility.rs; do echo "== $f"; git show a136840d:crates/cobre-…`
- **Fix-shape:** Coverage-neutral de-duplication (no test deleted, per yardstick §7): hoist the shared engine-neutral entity builders into a single test-support surface and have all three seam test regions consume it. Yardstick-canonical home (§5.2 'helpers live with the type they build') is cobre-core's `test-support` feature, since make_bus/make_stage/make_hydro/make_inflow_model/identity_correlation construct cobre-core entities (Bus/Stage/Hydro/InflowModel/CorrelationModel) — the two inline `#[cfg(test)]` modules use them via `#[cfg(test)]` and the integration binary via a `test-support` dev-dependency feature. A lighter, L1-safe alternative that stays inside this crate is a crate-internal `#[cfg(any(test, feature = "test-support"))] mod test_support` in cobre-stochastic exposing the builders so tests/reproducibility.rs can reach them; it has 3+ consumers (both inline modules plus the integration bi…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L1 cobre-stochastic)
- **Queued to:** test-corpus
- **Status:** fixed (2026-09-15) — `crates/cobre-stochastic/tests/reproducibility.rs` consumes the `tests/common` prelude (ticket-017) and the `context.rs`/`provenance.rs`/`seeds.rs` inline `make_bus`/`make_hydro`/`make_stage` are thin wrappers over `cobre_core::test_support` (`BusSpec`/`HydroSpec`/`StageSpec`) (ticket-014); no test deleted. The in-`src` `identity_correlation` copies were deliberately NOT folded (five remain; recorded in the Tier-4/5 reconciliation subsection). `feat/quality-tier45-closeout` 2bf30ece + 59c5c09f (pending merge).

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
- All 15 non-generated tree-noise files (tree/{mod,generate,lhs,opening_tree,qmc_halton/mod,qmc_sobol/mod}.rs, noise/_, normal/_, correlation/*) — Paradigm-leakage clean: git grep -niE '(sddp|benders|cut|cost.to.go|state.space)' over the full manifest yields a single hit, generate.rs:46, which is the substring 'cut' inside 'conseCUTive' — not an engine noun. No decomposition-engine t… (sanctioned by scripts/ci/check-infra-genericity.sh (L1 purity guardrail))
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

| ID     | Decision | Severity   | Alignment  | Rationale (owner)                  | Trigger / override | Queue     |
| ------ | -------- | ---------- | ---------- | ---------------------------------- | ------------------ | --------- |
| CD-064 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | none      |
| PD-021 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| PD-020 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| TD-025 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | test-debt |
| TD-024 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | test-debt |
| CD-066 | accept   | B (A-risk) | neutral    | accepted as recorded (Sev-B batch) | —                  | none      |
| CD-065 | accept   | B          | advances-1 | accepted as recorded (Sev-B batch) | —                  | alignment |
| OD-029 | accept   | B (A-risk) | neutral    | accepted as recorded (Sev-B batch) | —                  | none      |
| OD-028 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | none      |
| PD-023 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| PD-025 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| PD-024 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| TD-028 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | test-debt |
| CD-070 | accept   | B (A-risk) | advances-1 | accepted as recorded (Sev-B batch) | —                  | alignment |
| CD-068 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | none      |
| CD-067 | accept   | B (A-risk) | neutral    | accepted as recorded (Sev-B batch) | —                  | none      |
| PD-027 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| PD-026 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | perf      |
| TD-030 | accept   | B          | neutral    | accepted as recorded (Sev-B batch) | —                  | test-debt |
| OD-027 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| OD-026 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| PD-022 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| TD-027 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | test-debt |
| TD-026 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | test-debt |
| TD-029 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | test-debt |
| CD-071 | accept   | C          | advances-1 | accepted as recorded (Sev-C batch) | —                  | alignment |
| OD-031 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| PD-029 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| PD-030 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| TD-032 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | test-debt |
| TD-033 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | test-debt |
| CD-069 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| OD-030 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| PD-028 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | none      |
| TD-031 | accept   | C          | neutral    | accepted as recorded (Sev-C batch) | —                  | test-debt |

**Prior-register dispositions ratified (as recorded):** Stage-calendar crate home → keep; External-noise take/fill glue duplication → not-ours; Deterministic (σ = 0) AR(p > 0) external inflow stays rejected → keep; `LoadModel` conflates physical load with its stochastic model → not-ours; Cross-path static-RHS contract not yet in `.claude/rules/sddp.md` → not-ours; Oracle test-harness duplication → cross-reference; CD-001 — setup config-projection sprawl / CLI non-root reconstruction → resolved.

**Cleared by this gate (do not re-raise):** none.

**Gate: RETURNED 2026-09-08** — baseline `a136840d`; accepted 35, downgraded 0, rejected 0, deferred 0, overridden 0.

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — solver-comm

### Station 4 — cobre-solver + cobre-comm (2026-09, baseline 077dbe2c)

**Ratified 2026-09-18** — owner gate; baseline `077dbe2c`; accepted 20, downgraded 0, rejected 0, deferred 0, overridden 0; informational kept 1; 14 needs-human answered.

**Station.** cobre-solver + cobre-comm — the two L0 crates of the target layering (LP backend boundary behind `SolverInterface`; communicator trait/factory). **Method.** Four lenses (architecture, performance, over-engineering, test-bloat), one attacker per lens over BOTH crates, one read-only defender per surviving candidate; reserved seams applied as ingest FILTERS (the shared-memory communicator hierarchy, the superseded cut-sync methods) before any defender.

**Baseline.** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (pinned 2026-09-17; the E01 scaffold heading above keeps its minted `a136840d`). Read-only station: no tracked file under `crates/`, `scripts/` or `.github/` was modified. 24 attacker candidates: 21 defended, 20 confirmed, 1 dismissed, 2 dup-of, 1 informational; every anchor resolves at the baseline (`anchor-probe.md`, 109 checked / 0 failing + 1 `.c` anchor by direct grep). Calibration: 20 entries — 1 × B (A-risk), 7 × B, 12 × C; reviewer downgraded on 1 (recorded per entry). Alignment hints are provisional — Epic 9 adjudicates.

#### Architecture

**CD-075 · Sev C · duplication · effort S · confidence high**
Narrowed to the mechanism at this station only: because a crate-level `[lints]` table replaces rather than overlays `[workspace.lints]`, and no checker in scripts/ci/ compares the two, omitting any of the four allow-by-default entries (`unwrap_used`, `expect_used`, `panic`, `missing_docs`) from the hand copy in crates/cobre-solver/Cargo.toml:37-47 or crates/cobre-comm/Cargo.toml:35-45 disables that check with CI still green. Conceded: neither station replica has drifted at the baseline (6 of 6, `unsafe_code` alone flipped), and the silent failure mode does NOT extend to `too_many_arguments` -- the one key that drifted in the out-of-station python replica -- because `clippy::all` still reports it and `-D warnings` at .github/workflows/ci.yml:154 turns it red.

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-001, ingest architecture-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/Cargo.toml:41`, `crates/cobre-solver/Cargo.toml:37`, `crates/cobre-comm/Cargo.toml:39`, `crates/cobre-comm/Cargo.toml:35`
- **Evidence:** Both station replicas are faithful at this baseline — 6 of 6 clippy entries and both rust entries, with only `unsafe_code` flipped from `forbid` to `allow`, which FFI and the `unsafe impl Send/Sync` justify and which is not what this candidate is about. The third replica is the proof that the mechanism, not the current text, is the defect: crates/cobre-python/Cargo.toml carries 5 of 6 and is missing `too_many_arguments = "deny"`. That instance is the cli-python station's to raise; cited here only to show the hand-… Re-derive: `sed -n '35,45p;65,75p' Cargo.toml | grep -vE '^\s*$|^#' && sed -n '37,47p' crates/cobre-solver/Cargo.toml && sed -n '35,45p' crates/cobre-comm/Cargo.…`
- **Fix-shape:** The Cargo constraint is real — a manifest cannot combine `lints.workspace = true` with a per-lint override — so the fix is not deduplication but making the copies checkable, and there are two shapes with a genuine trade-off for the owner. First shape, three copies and one guard: add a CI check that parses the workspace `[workspace.lints.*]` tables and each replicating manifest and asserts set equality modulo one declared exception list (today: `unsafe_code`), failing on any key present in the workspace table and absent from a replica. This keeps `forbid` at the workspace root, costs one script, and turns a silent omission into a red build; its weakness is that the exception list is itself hand-maintained. Second shape, remove the reason the copies exist: downgrade the workspace `unsafe_code` from `forbid` to `deny`, let all three crates carry plain `lints.workspace = true`, and have each FFI crate re-admit unsafe at its own crate root with a scoped `expect` carrying the justification next to the code it licenses. That leaves exactly one table in the workspace and no drift surface at…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver / cobre-comm)
- **Calibration:** CD-009 precedent (a hand-copied table with a bounded, checker-shaped fix): Cargo forbids the `lints.workspace = true` overlay beside a per-lint override, so the copies are forced and the debt is manifest-level drift, not spreading structure.
- **Reviewer rating:** B — recalibrated to C because blast radius is two manifests and the fix is a checker or a shared include, not a structural change — CD-009 duplication precedent, Sev C.
- **Owner decision (2026-09-18, R9):** Keep forbid + checker — the workspace `unsafe_code = forbid` stays the unoverridable default (CLAUDE.md hard rule); a scripts/ci checker diffs each per-crate `[lints]` copy against the workspace tables so drift is red CI; the four FFI crates keep their audited overrides.

**CD-076 · Sev C · leaky-boundary · effort S · confidence high**
Three of the seven put above-L0 vocabulary in the normative sentence itself with no adjacent generic restatement: highs/solver.rs:477 ('the primary warm-start mechanism for the backward pass', an L3 traversal phase with no L0 referent), trait_def.rs:212 (the public determinism guarantee scoped to 'a scenario's result' and 'which scenarios a worker happened to process' instead of to the solver handle), and trait_def.rs:95 (the qualifier 'scenario' on solve's otherwise crate-owned 'patches' precondition). The other four restate the L0 fact in the same sentence, mark the caller boundary as an 'e.g.', or duplicate a trait-level anchor, and the gate's inability to see any of them is not part of the defect.

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-003, ingest architecture-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs:66`, `crates/cobre-solver/src/backends/clp/retry.rs:74`, `crates/cobre-solver/src/backends/highs/solver.rs:320`, `crates/cobre-solver/src/backends/highs/solver.rs:477`, `crates/cobre-solver/src/trait_def.rs:95`, `crates/cobre-solver/src/trait_def.rs:207`, `crates/cobre-solver/src/trait_def.rs:212`
- **Evidence:** All seven sites are in the production region of their file (the scan exits at the first column-0 `#[cfg(test)]`) and all seven describe the L0 solver in terms of the caller's traversal — scenarios, a scenario loop, a backward pass, deep stages. None of `scenario`, `stage` or `backward pass` appears in the gate PATTERN, so unlike the `cut_nz_per_col` case this vocabulary is not evaded by a word-boundary accident: the gate has no pattern for it and never could flag it. The same scan over crates/cobre-comm/src return… Re-derive: `for f in $(git ls-tree -r --name-only 077dbe2c -- crates/cobre-solver/src crates/cobre-comm/src | grep -v tests.rs); do git show 077dbe2c:$f | awk -v…`
- **Fix-shape:** Restate each of the seven in terms the solver itself owns, and be explicit about which sentences are load-bearing. The three determinism sentences (clp/interface.rs:66 and trait_def.rs:207,212) are correctness contracts and must survive the rewrite, restated as a property of the solver handle — a solve's result must not depend on which models the same handle solved before it — rather than of the caller's loop; that phrasing is strictly stronger, because it binds any caller and not just the present one. clp/retry.rs:74 and highs/solver.rs:320,477 lose nothing: a numerically delicate LP that is in fact feasible, a per-solve cost that makes an option unusable when one handle is reused for thousands of solves, and a warm-start mechanism whose value is that no solver-clear call is issued, are all statements about LPs and handles. trait_def.rs:95 should state its precondition against the trait's own methods rather than the caller's patch vocabulary. Record the boundary honestly for the owner gate: the project's hard genericity rule enumerates `sddp`/`SDDP`/`Benders`, none of which appears…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** doc-only vocabulary in an L0 crate with no type or behaviour change; CD-064 (stochastic phase-vocabulary reword) is the nearest recorded precedent and sits at B only because it spans a whole L1 kernel — here three sentences.
- **Owner decision (2026-09-18, R10):** Now-fix reword — same treatment as stochastic CD-064: a pure doc reword in the generic register, no type or behaviour change, independent of the phase-1 shed; CD-076 stays neutral, Sev C.

**CD-077 · Sev C · asymmetry · effort S · confidence high**
At `clp/mod.rs:25` the `pub(crate) use retry::LADDER_RUNGS` re-export has no non-test consumer -- `interface.rs:6` reads the constant through its owning path `super::retry`, which is equally reachable from the sibling `clp/tests.rs` -- so that facade line and the `not(test)` suppression above it exist only to give the test module a shallower import path; the claim does not extend to a HiGHS/CLP facade-rule divergence (HiGHS defines no module-level rung constant) nor to the suppression's form, which is minimum-scope, rationale-carrying and mirrored at `cobre-sddp/src/workspace/mod.rs:31`.

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-004, ingest architecture-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/clp/mod.rs:24`, `crates/cobre-solver/src/backends/clp/mod.rs:25`, `crates/cobre-solver/src/backends/clp/interface.rs:6`, `crates/cobre-solver/src/backends/highs/mod.rs:23`
- **Evidence:** The comment states the situation exactly and is accurate: the re-export's only consumer is the sibling test module, and non-test code already reaches the constant by its owning path (interface.rs:6 uses `super::retry::LADDER_RUNGS`). So a test consumer is shaping the production module facade, and a lint-suppression attribute is being carried in production source to silence the consequence. The HiGHS facade re-exports only what non-test code consumes and needs no such attribute, so the two backend facades follow di… Re-derive: `git grep -n 'LADDER_RUNGS' 077dbe2c -- crates/cobre-solver/src/backends/clp && git show 077dbe2c:crates/cobre-solver/src/backends/clp/mod.rs | awk 'N…`
- **Fix-shape:** Have the sibling test module import the constant by its owning path — `super::retry::LADDER_RUNGS`, exactly what interface.rs:6 already does — and delete both the re-export and the `cfg_attr` allow from the facade. That removes a lint suppression from production source, removes a production surface that exists only for tests, and restores symmetry with the HiGHS facade, which re-exports only what non-test code consumes. Blast radius is three lines: one import in the sibling tests module and the two facade lines. Note for the owner what this candidate is not: it is not a claim that the constant should be private or that the tests should stop asserting on it, only that a test's import path should not appear in the production facade when the owning path is already reachable and already used.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** a single facade line and its lint suppression that exist for a test import path; cosmetic module-shape asymmetry.

**CD-078 · Sev B · duplication + asymmetry · effort M · confidence high**
The ffi/clp.rs:27-31 rationale is stale at the baseline — get_basis and install_basis decode and re-encode the CLP codes through BasisStatus with folding rather than round-tripping them verbatim as raw i32 — and its 'no symbolic definitions' conclusion is contradicted by the two private constants at backends/clp/solver.rs:18 and :22, which reset_cold_basis uses as the single bypass of the canonical to_clp_code/from_clp_code owner, leaving interface.rs:600's CLP_BASIS_* reference resolvable to nothing. Conceded: the code space is NOT ownerless, and naming the six values would add a single definition site, not compile-time anchoring, since the HiGHS constants are themselves hand-written literals.

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-005, ingest architecture-05)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/ffi/clp.rs:29`, `crates/cobre-solver/src/basis_status.rs:67`, `crates/cobre-solver/src/basis_status.rs:82`, `crates/cobre-solver/src/backends/clp/solver.rs:18`, `crates/cobre-solver/src/backends/clp/solver.rs:22`, `crates/cobre-solver/src/backends/clp/interface.rs:600`
- **Evidence:** The same six-value CLP code space is written out in four places at this baseline: the C header contract (csrc/clp_wrapper.h:209-210), the ffi/clp.rs:27-31 comment, bare literals in both directions of the canonical mapping (basis_status.rs:69-74 and 84-89), and two named private constants covering two of the six values in a third module (backends/clp/solver.rs:18,22, used at :290 and :299). The comment's premise is false: the codes are not round-tripped verbatim — they are decoded into the canonical enum at backend… Re-derive: `git show 077dbe2c:crates/cobre-solver/src/basis_status.rs | awk 'NR>=9&&NR<=12||NR>=67&&NR<=75{printf "%d:%s\n",NR,$0}' && git show 077dbe2c:crates/c…`
- **Fix-shape:** Give the CLP code space one owner in the binding module, mirroring what the HiGHS side already does: declare all six values as named constants in ffi/clp.rs next to the header they mirror, have both directions of the canonical mapping match on those names instead of bare numerals, and have the two private constants in backends/clp/solver.rs use them rather than redeclare two of the six. Rewrite the ffi/clp.rs:27-31 rationale to say what is true — the codes are interpreted in both directions, so they get names — instead of the round-tripped-verbatim premise that argued against naming them. Then backends/clp/interface.rs:600's `CLP_BASIS_*` reference resolves to something real, and the numerals stop being duplicated across three modules. This is a two-way door with no public surface change: the mapping functions keep their signatures, so the blast radius is one binding module, one enum body, and two constants. It is worth doing above C severity because the numerals are a warm-start correctness contract — a value transposed in one of the three copies mis-installs a basis silently rathe…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** four spellings of one FFI code space with a stale rationale and a private bypass of the canonical mapping — a real smell confined to the CLP backend; the HiGHS half already has the named-constant owner.

**CD-079 · Sev B (A-risk) · leaky-boundary · effort M · confidence high**
Two additions survive, both scoped to the E9 collateral estimate and neither to crate behaviour: (a) the handoff clips exactly two of its three assertion spans — conformance.rs is 153-157 not 154-156 and types.rs is 779-783 not 779-782, while freeze.rs:320-324 is correct as recorded; (b) the definite 'four StageTemplate test fixtures' list omits a fifth in-src fixture at types.rs:749-753 plus three in backends/clp/tests.rs, four in backends/highs/tests.rs, five further in freeze.rs (:437,:475,:555,:755,:800), and names none of the eight in tests/conformance.rs. Stripped: 'seven further freeze.rs blocks' over-counts by two, since freeze.rs:162 is the production copy-forward named as the fix site and :324 is a named assertion span, leaving five; and '22 in-src mentions' is a matching-line count (24 occurrences), two of whose types.rs lines are the declaration and its doc comment rather th…

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-006, ingest architecture-06)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/types.rs:235`, `crates/cobre-solver/src/types.rs:270`, `crates/cobre-solver/src/types.rs:278`, `crates/cobre-solver/src/types.rs:287`, `crates/cobre-solver/src/types.rs:290`, `crates/cobre-solver/src/types.rs:297`, `crates/cobre-solver/src/freeze.rs:158`, `crates/cobre-solver/src/backends/clp/tests.rs:33`, `crates/cobre-solver/src/backends/highs/tests.rs:32`, `crates/cobre-solver/tests/conformance.rs:153`
- **Evidence:** The five declarations resolve at exactly the lines the handoff records, and the doc vocabulary is present verbatim in the L0 container: `N * L`, `AR lags`, `hydros`, `maximum PAR lag order`, `FPHA`, `uniform lag stride` (and `SIMD vectorization` on the following line of the same block). The write-only disposition holds field by field: the only production propagation is the copy-forward at freeze.rs:158-162 (`out.n_state = base.n_state;` through `out.max_par_order = base.max_par_order;`), and the two other sites th… Re-derive: `git show 077dbe2c:crates/cobre-solver/src/types.rs | grep -nE 'pub n_state|pub n_transfer|pub n_dual_relevant|pub n_hydro|pub max_par_order|N \* L|AR…`
- **Fix-shape:** Delete the five fields from the L0 container so it carries only the CSC arrays and the dimensions the LP itself needs; nothing in either station crate reads them, and the geometry they describe is already owned one layer up by `StateSpace` (crates/cobre-sddp/src/lp/indexer/state_space.rs:97,100,104) and `StageRowLayout` (crates/cobre-sddp/src/lp/builder/layout.rs:509), which the sole writer (crates/cobre-sddp/src/lp/builder/template.rs:445,459-463) already reads from. Because the production removal is mechanical, the whole cost of the shed is fixture collateral, so the ticket must size itself from a census rather than a hand-written list: 22 in-src mentions across six files plus 18 across the seven integration binaries, and the two clipped assertion spans widened to conformance.rs:153-157 and types.rs:779-783. Fold the doc drift into the same shed instead of carrying it forward: types.rs:282-286 still asserts `n_dual_relevant` equals `n_state`, while the only production writer hard-codes zero and says why (crates/cobre-sddp/src/lp/builder/layout.rs:1353-1355 — column bounds pin the…
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV.5 (shed line :1471) and the Part V purification row (:1372) — StageTemplate sheds n_state/n_transfer/n_dual_relevant/n_hydro/max_par_order)
- **Calibration:** CD-010 precedent (L0/L2 leaky boundary with a typed-shape fix, Sev B): the five multistage fields leak engine geometry into the L0 container, but the blast radius is three named propagation sites plus fixture collateral, not the whole crate — hence B (A-risk), not A.
- **Part-I:** I.3-8 (cross-reference; verdict travels to Epic 9 with the per-field dispositions in `partI-handoff.json`).
- **Queued to:** alignment

**CD-080 · Sev C · asymmetry · effort S · confidence high**
Narrowed off ffi/mod.rs and off any harm claim: the CLP-only build's carried HiGHS declarations are benign (no link demand per build.rs:30 and :97, no lint noise per ffi/highs.rs:5), and the un-gated `mod highs` is not a missing cfg. The defensible residue is (a) the expression asymmetry inside crates/cobre-solver/src/basis_status.rs, whose HiGHS half (:39-60) references named ffi::highs constants while its CLP half (:67-92) uses bare integer literals, which is what makes the module unconditional, and (b) crates/cobre-solver/README.md:106-107, which states that ffi::highs compiles only with the `highs` feature - a gating contract that is false at this baseline and is nowhere corrected in ffi/mod.rs.

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-007, ingest architecture-07)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/ffi/mod.rs:15`, `crates/cobre-solver/src/ffi/mod.rs:18`, `crates/cobre-solver/src/basis_status.rs:9`, `crates/cobre-solver/src/ffi/highs.rs:5`
- **Evidence:** The `clp` binding module carries `#[cfg(feature = "clp")]` and the flat HiGHS re-export carries `#[cfg(feature = "highs")]`, but the `highs` module declaration itself carries no cfg, so the whole HiGHS binding module is compiled in every configuration — including the CLP-only build that CI checks, clippy-lints, builds and runs tests against. The mechanism is not an oversight in ffi/mod.rs: basis_status.rs:9-12 imports five HiGHS constants un-gated, and it is that import which forces the un-gated declaration. The m… Re-derive: `git show 077dbe2c:crates/cobre-solver/src/ffi/mod.rs | grep -nE 'mod highs|mod clp|cfg' && git show 077dbe2c:crates/cobre-solver/src/ffi/highs.rs | s…`
- **Fix-shape:** Make the two backends symmetric at the boundary so a single-backend build compiles only its own bindings: once the CLP code space has named constants in its own binding module, each half of the canonical mapping references only its own backend's module and can carry that backend's cfg, letting ffi/mod.rs gate `highs` exactly as it already gates `clp`. The blast radius is smaller than it looks and worth stating, because it is what makes the gating safe: the only consumers of the HiGHS half outside the HiGHS backend module are in a `#[cfg(test)]` module of crates/cobre-sddp/src/policy/policy_load.rs, and production cobre-sddp uses only the declaration-order-independent discriminant codec, so gating touches one backend module plus one test module. If the owner prefers to keep both mappings un-gated in one place — a defensible call, since a single conversion site is easier to audit than two — then the un-gated `mod highs` should be made deliberate rather than incidental: state in ffi/mod.rs that the HiGHS binding module is unconditional because the neutral vocabulary type depends on it,…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** expression asymmetry between the two halves of basis_status.rs plus a README line; the defender conceded every harm claim.
- **Needs-human (owner gate):** Owner picks the residue's direction: keep the canonical mapping unconditional and correct README.md:106-107, or name the CLP codes in ffi::clp and gate each half - the latter also makes BasisStatus::to_highs_code/from_highs_code and the legacy-checkpoint agreement test (basis_status.rs:280-289) HiGHS-only.
- **Owner decision (2026-09-18, R5):** Keep mapping unconditional; fix README — the canonical BasisStatus mapping stays feature-independent; name the CLP codes in ffi::clp for symmetry and correct README.md:106-107 — smallest blast radius, no test becomes HiGHS-only.

**CD-081 · Sev C · leaky-boundary (naming) · effort S · confidence high**
The private, contract-unpinned FreezeScratch.cut_nz_per_col -- declaration at crates/cobre-solver/src/freeze.rs:22 plus its four production uses at 137, 138, 141 and 188 -- is an unsanctioned L0 vocabulary residue, narrowly because it carries no serialized key and therefore falls outside the deliberate key/type divergence that crates/cobre-io/src/config/training.rs documents for cut_selection and that output/mod.rs applies to cuts_active. The gate half of the title does not survive: the word-character evasion is already recorded in partI-handoff.json's genericityGateBlindSpot and owned by E7, and the #[cfg(test)] truncation is a documented intentional exclusion rather than a second blind spot.

- **Station:** cobre-solver (solver-comm, lens architecture; attacker SC-ARCH-008, ingest architecture-08)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/freeze.rs:22`, `crates/cobre-solver/src/freeze.rs:137`, `crates/cobre-solver/src/freeze.rs:141`, `crates/cobre-solver/src/freeze.rs:188`
- **Evidence:** Fresh gate run in this session, exit 0. Five occurrences of the engine noun live above the file's `#[cfg(test)]` boundary (line 244) — the declaration plus four production uses — and every one of them is invisible to the gate, because the alternation asks for `\bcut\b` and an underscore is a word character, so `cut_nz_per_col` contains no word boundary after `cut`. Two independent blind spots therefore hide the same word in this file: the `_`-joined identifier in the scanned region, and the gate's awk prefilter, w… Re-derive: `sed -n '79p' scripts/ci/check-infra-genericity.sh && git show 077dbe2c:crates/cobre-solver/src/freeze.rs | awk '/^#\[cfg\(test\)\]/{exit} /cut_nz_per…`
- **Fix-shape:** Rename the private field and its four production uses to name what the vector actually holds — a per-column nonzero census of the rows being appended — so the identifier describes the CSC bookkeeping rather than the caller's row semantics, which is the whole point of the L0 purity rule. The field is private and never crosses the crate boundary, so the blast radius is the four production sites plus the in-src test region below freeze.rs:244; no public API, no schema and no wire format moves. Queue an E7 cross-reference recording both blind spots the rename exposes, so the gate owner sees that an `_`-joined identifier defeats a `\b`-anchored alternation and that the awk prefilter leaves every in-src test region unscanned. No gate edit is proposed here: the gate belongs to the build-ci station, and the rename is worth doing on its own merits even if the gate never changes.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** CD-011 naming-asymmetry precedent: a private identifier carrying a caller-loop noun, no serialized key, no public surface; the gate blind spot travels to E7, the rename stands on its own.
- **Queued to:** build-ci

#### Performance

**PD-032 · Sev B · asymmetry · effort M · confidence high**
Narrowed to the allocation only: cobre_clp_chg_bounds (crates/cobre-solver/csrc/clp_wrapper.c:208) heap-allocates, fills and frees a full-dimension sentinel-translation buffer on every one of the four chg_* crossings, so each per-stage-solve bound patch, including the single-column theta pin in training/forward/stage_solve.rs and training/forward/enumerated.rs, pays a fresh malloc and free for a translation that ClpModel::chg{Row,Column}{Lower,Upper} then largely repeats internally. The transfer-width half of the title is stripped: Clp_C_Interface.h exposes no subset bound setter, so the whole-array push from the two Rust setters is the C surface the plain-C shim is bound to, not an unjustified divergence from the HiGHS by-set path.

- **Station:** cobre-solver (solver-comm, lens performance; attacker SC-PERF-002, ingest performance-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs:352`, `crates/cobre-solver/src/backends/clp/interface.rs:399`, `crates/cobre-solver/csrc/clp_wrapper.c:208`
- **Evidence:** Both CLP setters patch the retained mirror at the requested indices and then hand the WHOLE array pointer to the wrapper; the wrapper resolves the length from Clp_getNumRows / Clp_getNumCols, heap-allocates a scratch array of that full length to translate the sentinel bounds, and frees it, once per crossing. Two crossings happen per setter call. The HiGHS realisation of the same trait method crosses once with the caller's index subset into a retained i32 scratch and allocates nothing. The declared shim surface off… Re-derive: `S=077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c; git show $S:crates/cobre-solver/src/backends/clp/interface.rs | grep -n 'fn set_row_bounds\|fn set_col_bo…`
- **Fix-shape:** Give the CLP path a subset-scoped bound write that mirrors the shape the sibling backend already has. Add index-scoped entry points to the cobre-owned C shim that take a count, an index array and the two bound arrays, translate the sentinels into a caller-owned scratch buffer retained on the Rust side instead of a per-crossing malloc, and reach the underlying per-index or ranged setter. Keep the retained mirror updated exactly as today so the frozen reload path and reproducibility contract are untouched, and keep the whole-array form for the load path that genuinely replaces every bound. If the underlying library truly offers no index-scoped setter that preserves the factorization, the narrower fix is still available: hoist the sentinel translation out of the shim into a retained Rust-side buffer so the per-crossing allocate and free disappears even when the transfer stays dimension-wide.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** per-call heap allocation on the per-stage-solve bound-patch path in one backend where the sibling backend writes the index subset; structural calibration only — UNMEASURED.
- **Measurement:** UNMEASURED — layout `4t`, mechanism: work proportional to the model dimension rather than to the number of patched indices, on every call: two FFI crossings each transferring the full retained bound array, plus one heap allocation and free of a full-length translation buffer inside the wrapper per crossing. Per-solve, not setup-time: the engine patches a handful of bounds per stage solve.; exercising call sites (cited, not anchored): crates/cobre-sddp/src/training/stage_solve_prep.rs:157, crates/cobre-sddp/src/training/stage_solve_prep.rs:164, crates/cobre-sddp/src/training/stage_solve_prep.rs:246, crates/cobre-sddp/src/training/forward/stage_solve.rs:95, crates/cobre-sddp/src/training/forward/enumerated.rs:321, crates/cobre-sddp/src/lp/builder/entries.rs:5893. Queued to the performance sweep (see `perf-queue.json`); no number is asserted here.
- **Queued to:** performance-sweep

**PD-033 · Sev B · asymmetry · effort M · confidence high**
Rust side only, and only the CSC trio: on the per-iteration append path (load_backward_lp, append_new_cuts_to_lp) CLP add_rows heap-allocates two scratch vectors plus three merged_nz-sized replacements for col_starts / row_indices / values and rewrites the whole mirror, discarding the capacity load_model reused one call earlier, when a retained-and-resized buffer set would yield byte-identical contents in the same write order. Stripped from the candidate: the wrapper's two batch-sized mallocs (a documented C-side sentinel-translation ownership boundary shared with the other bound wrappers, not debt), and the reusable-buffer-contract framing, whose only false element is the word 'patch' applied to the CSC trio -- the struct doc's reuse promise is scoped to load_model-resized buffers reused across solves, and the four retained bound vectors already honour it.

- **Station:** cobre-solver (solver-comm, lens performance; attacker SC-PERF-003, ingest performance-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs:215`, `crates/cobre-solver/src/backends/clp/interface.rs:302`
- **Evidence:** Each CLP append allocates a per-column counting vector, a new column-start vector, a new row-index vector and a new value vector both sized to the merged nonzero count, and a write-cursor vector, then rebuilds the retained column-major mirror in full and assigns the three new vectors over the retained fields, discarding the capacity those fields already held. The wrapper adds two more heap allocations per append for the translated row bounds. The sibling backend's realisation of the same trait method allocates not… Re-derive: `S=077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c; git show $S:crates/cobre-solver/src/backends/clp/interface.rs | grep -n 'fn add_rows\|let mut new_col_sta…`
- **Fix-shape:** Move the append's working storage into caller-owned scratch on the solver struct, the way the freeze path and the sibling backend already do. Because a column-major mirror cannot absorb appended rows in place, keep two retained mirror buffer sets on the struct and swap the roles per append instead of allocating a new set and dropping the old one; keep the per-column counting vector and the write-cursor vector as retained buffers cleared and regrown without shrinking. Hoist the sentinel bound translation out of the wrapper into a retained buffer on the Rust side so the wrapper's two allocations per append disappear as well. The resulting mirror contents and the order in which nonzeros are written must be unchanged so reproducibility and order-invariance hold, and the struct doc's reusable-buffer statement then becomes true of this path too.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** five fresh allocations and three wholesale buffer replacements on the per-iteration append path, against the struct's own reusable-buffer contract and the never-allocate hot-path rule; structural calibration only — UNMEASURED.
- **Measurement:** UNMEASURED — layout `4t`, mechanism: allocation per call plus a full rebuild pass per call: five heap allocations on the Rust side, two more inside the wrapper, three retained buffers replaced rather than reused, and a rebuild of the retained column-major mirror proportional to the merged nonzero count for an append whose payload is only the new rows. Per-solve on the training path, not setup-time; this collides with the project hard rule against allocating on hot paths.; exercising call sites (cited, not anchored): crates/cobre-sddp/src/training/backward/lp_setup.rs:31, crates/cobre-sddp/src/cut/row.rs:257, crates/cobre-sddp/src/cut/row.rs:347, crates/cobre-sddp/src/training/lower_bound.rs:158. Queued to the performance sweep (see `perf-queue.json`); no number is asserted here.
- **Queued to:** performance-sweep

**PD-034 · Sev B · asymmetry · effort M · confidence high**
Only the read side qualifies: ClpSolver::get_basis crosses the FFI once per column and once per row on the per-solve capture path, where the HiGHS sibling crosses once into retained i32 buffers. The cold reset is conceded out of the finding — reset_cold_basis is reachable only from the escalate_solve failure ladder through escalate_run, never between successful solves, so its per-element write crossings are not a hot-path defect.

- **Station:** cobre-solver (solver-comm, lens performance; attacker SC-PERF-004, ingest performance-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/clp/interface.rs:606`, `crates/cobre-solver/src/backends/clp/solver.rs:280`
- **Evidence:** The CLP status accessors sit inside per-index loops over every column and then every row, so one capture pays num_cols plus num_rows separate crossings; the cold reset pays the same count on the write side. The sibling backend performs the identical trait operation with a single crossing into two retained i32 buffers. The doc above the CLP loop states the cause as a wrapper property (status is reported one element at a time, no bulk array in the wrapper), and the declared shim surface confirms it: ffi/clp.rs decla… Re-derive: `S=077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c; git show $S:crates/cobre-solver/src/backends/clp/interface.rs | grep -n 'fn get_basis\|cobre_clp_get_colu…`
- **Fix-shape:** Add bulk status transfer to the cobre-owned CLP shim so the trait method crosses once per capture instead of once per element: one shim function that fills a caller-provided i32 array for the columns and one for the rows by looping on the C or C++ side, and the symmetric writers for the cold reset. The Rust side keeps the retained i32 buffers the sibling backend already carries and converts into the shared status vocabulary in a plain loop with no crossing. The shim already reaches methods that exist only on the C++ simplex class, so the bulk loop has somewhere to live. The status mapping and the resulting basis contents must stay byte-identical so the determinism harness continues to pass unchanged; this fix shape changes only how many times the boundary is crossed, never which statuses are produced or how a rejected offer is reported.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** one FFI crossing per column and per row on the per-solve basis capture where the sibling crosses once into retained buffers; the cold reset is conceded out; structural calibration only — UNMEASURED.
- **Measurement:** UNMEASURED — layout `4t`, mechanism: one FFI crossing per matrix element instead of one per capture: the crossing count grows with num_cols plus num_rows on both the read and the cold-write path. Per-solve, not setup-time: the engine captures a basis on the forward and backward solve paths and resets to a cold basis between solves.; exercising call sites (cited, not anchored): crates/cobre-sddp/src/training/backward/outcome_aggregation.rs:154, crates/cobre-sddp/src/training/forward/stage_solve.rs:248, crates/cobre-sddp/src/training/forward/enumerated.rs:442. Queued to the performance sweep (see `perf-queue.json`); no number is asserted here.
- **Queued to:** performance-sweep

#### Over-engineering

**OD-032 · Sev C · speculative-generality · effort S · confidence high**
At the baseline ExecutionTopology::is_homogeneous (crates/cobre-comm/src/topology.rs:34) has no IN-WORKSPACE production reader and no reserved-seam entry, so it is the single item in cobre-comm's topology surface that still owes the delete-or-register disposition the mirror's dead-surface standard (:1210) requires of every shipped public item — a registration/decision gap in the crate that sweep never covered. It is NOT speculative structure: no trait, generic, indirection or knob, just a six-line query over already-public hosts/ranks. And because cobre-comm publishes to crates.io, the absence of consumers is established only inside this workspace, so the candidate's implied conclusion that the predicate and its four tests can simply be dropped does not follow from its own evidence.

- **Station:** cobre-comm (solver-comm, lens over-engineering; attacker SC-OE-001, ingest over-engineering-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-comm/src/topology.rs:34`
- **Evidence:** Workspace-wide at the baseline, the only references to is_homogeneous are its declaration and the four unit tests in the same file; no crate reads it. The judgement is on consumers, not shape: the sibling accessors on the same struct are genuinely consumed (num_hosts and leader_hostname from cobre-cli's run command and summary printer), and every MpiRuntimeInfo / SlurmJobInfo field reaches the run summary through topology.mpi / topology.slurm, so ExecutionTopology, HostInfo, MpiRuntimeInfo and SlurmJobInfo are NOT… Re-derive: `git grep -n 'is_homogeneous' 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c -- crates; git grep -n 'num_hosts()\|leader_hostname()' 077dbe2c287b92c2d0c6a12…`
- **Fix-shape:** Drop the predicate and its four unit tests and let the caller that eventually needs a heterogeneity decision express it where the policy lives; today the only consumer of layout information is cobre-cli, which formats the layout from num_hosts and leader_hostname and needs no boolean. If the owner intends a heterogeneous-topology guard, the missing artifact is a reserved-seam entry with an activating milestone and an owner, mirroring how the shared-memory hierarchy is registered, not an unregistered unconsumed accessor. Either way the heterogeneity policy stays out of L0: cobre-comm reports the topology it measured and does not decide what an uneven rank distribution means.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-comm)
- **Calibration:** one public predicate with zero in-workspace readers and no register entry — the delete-or-register disposition the mirror's dead-surface standard requires.
- **Owner decision (2026-09-18, R8):** Delete — leftover, not a planned heterogeneous-layout guard; drop the predicate and its four unit tests, the eventual caller re-adds it beside its use.

**OD-033 · Sev B · speculative-generality · effort M · confidence high**
Only the acquire half of the CLP hot-start lifecycle - the shim/extern pair cobre_clp_mark_hot_start plus cobre_clp_solve_from_hot_start and the two safe methods at crates/cobre-solver/src/backends/clp/solver.rs:350 and :394 - has no production caller and is absent from the reserved-seam register, so the defect is the missing register entry (owner plus consuming milestone) for that pair alone; the release half (unmark_hot_start, its three interface.rs call sites and Drop) is production-wired and the determinism harness is a contract-pinning exerciser, so neither is part of the unwired surface nor needs registering.

- **Station:** cobre-solver (solver-comm, lens over-engineering; attacker SC-OE-003, ingest over-engineering-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/clp/solver.rs:350`, `crates/cobre-solver/src/backends/clp/solver.rs:394`
- **Evidence:** Excluding the two test files, every remaining hit is either the C shim, the extern declaration, or the definition itself: no production path calls mark_hot_start or solve_from_hot_start. The release half of the same lifecycle IS production-wired -- unmark_hot_start is invoked from three call sites in clp/interface.rs and from Drop (solver.rs:509) -- so the asymmetry is real and not an artifact of my grep. The only exerciser is the determinism harness (crates/cobre-solver/tests/clp_determinism.rs, which drives the… Re-derive: `git grep -n 'mark_hot_start\|solve_from_hot_start' 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c -- crates | grep -v 'clp/tests.rs' | grep -v 'crates/cobr…`
- **Fix-shape:** Resolve the seam's status rather than leaving it implicit. One option is to wire the acquire/solve half on the CLP re-solve path that already invalidates it -- the three unmark_hot_start call sites in clp/interface.rs mark exactly where a snapshot would have to be taken and released, so the wiring is bounded to one backend's interface file. The other is to register the whole lifecycle (C++ shim, extern declarations, the two safe methods, the harness) in the reserved-seam register with an activating milestone and an owner, exactly as the shared-memory communicator hierarchy is registered. Removal is deliberately NOT proposed: the project rule is that an unwired capability is reserved until the owner rules otherwise, so the missing artifact is the registration, not the code. Nothing here transfers to HiGHS -- the snapshot token is a CLP concept and no equivalent surface should be invented to make the backends symmetric.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** an acquire/solve pair built through the C++ shim, the extern block and the safe wrapper with no production caller and no reserved-seam entry; bounded to the CLP backend but spanning three layers — Sev B pending the owner's milestone or retirement.
- **Needs-human (owner gate):** Owner must supply the activating milestone for the CLP hot-start acquire/solve pair, or rule it retired at the next licensed public-API break; the register admits an entry only with both an owner and a consuming milestone, and this station cannot invent one.
- **Owner decision (2026-09-18, R6):** Retire at the next licensed public-API break — OD-033 stays accepted at Sev B; the fix-shape becomes 'delete the acquire half (shim, extern, wrapper, harness) at the next licensed API break' — no reserved-seam entry, no wiring.

**OD-034 · Sev C · duplication · effort S · confidence high**
The 12-value agreement between `HighsProfile::default()` and the `default_options()` table (config.rs:58-77 against 168-238) is a load-bearing invariant for the delta-only dispatch in `ProfiledSolver::new`/`set_profile` (profiled.rs:38-58) that no test or compile-time assert pins; the defect is the missing guard alone, not the second surface, since the 17-entry table carries 5 non-profile options and remains the sole installer on the fresh-handle and retry-restore paths and so cannot be collapsed into the profile.

- **Station:** cobre-solver (solver-comm, lens over-engineering; attacker SC-OE-004, ingest over-engineering-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/highs/config.rs:58`, `crates/cobre-solver/src/backends/highs/config.rs:168`, `crates/cobre-solver/src/backends/highs/config.rs:10`
- **Evidence:** Every reference to default_options() is either a doc comment or one of the two apply loops in solver.rs; no test and no build step compares the table against Default for HighsProfile. Yet 12 of the 14 profile fields carry a literal twin in the 17-entry table with the same value: primal/dual_feasibility_tolerance 1e-9, simplex_scale_strategy 0, simplex_dual_edge_weight_strategy 1, presolve on, cost_perturbation 0.0 (dual_simplex_cost_perturbation_multiplier), simplex_price_strategy 1, refactor_error_tolerance 1e-6… Re-derive: `git grep -n 'default_options' 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c -- crates/cobre-solver`
- **Fix-shape:** Make one surface the owner of each default. Either derive HighsProfile::default() from the table by reading the typed entry for each field it names, or -- if the two types must stay separate -- state the invariant as a test that walks the table and asserts each named option's value against the corresponding profile field, so the doc comment's bit-for-bit claim is enforced instead of asserted. Nothing here should be generalized to CLP: CLP has no options table and adding one to match would be over-engineering.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** the defect is the missing guard on a 12-value agreement two tables already keep by hand; a compile-time or unit assertion closes it.

**OD-035 · Sev B · duplication · effort S · confidence high**
The profile-floored tolerance pair in highs/retry.rs (two f64::max bindings plus the two cobre_highs_set_double_option calls) has no single owner: it is written out verbatim four times at levels 3, 7, 10 and 11, with the level-3 and level-7 bodies byte-identical and only the floor literal (1e-8 versus 1e-7) separating the two groups. The residue is a floor-taking two-statement extraction with four call sites; the CLP rung-table comparison in the title is conceded and does not carry, because the twelve heterogeneous HiGHS levels are not tabulable the way CLP's five fixed-arity rungs are.

- **Station:** cobre-solver (solver-comm, lens over-engineering; attacker SC-OE-005, ingest over-engineering-05)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/highs/retry.rs:173`, `crates/cobre-solver/src/backends/highs/retry.rs:222`, `crates/cobre-solver/src/backends/highs/retry.rs:248`, `crates/cobre-solver/src/backends/highs/retry.rs:272`
- **Evidence:** The identical fragment -- two f64::max(floor, self.current_profile.{primal,dual}_feasibility_tolerance) bindings followed by two cobre_highs_set_double_option calls in the same order -- is written out four times (levels 3, 7, 10, 11), differing only in the floor literal (1e-8, 1e-8, 1e-7, 1e-7); the level-3 and level-7 bodies are byte-identical. Part of the HiGHS/CLP size asymmetry IS intrinsic: HiGHS exposes a string-keyed option API, which is what forces the OptionValue / DefaultOption dispatch in config.rs that… Re-derive: `git show 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c:crates/cobre-solver/src/backends/highs/retry.rs | grep -n 'f64::max\|cobre_highs_set_double_option'…`
- **Fix-shape:** Give the repeated fragment one owner: a private helper that takes the floor and applies both tolerance options in the current fixed order, and express the per-level deltas (floor, scaler ints, solver string) as a static rung table analogous to the CLP ladder's, so each level is data and the FFI call sequence is written once. The constraint that makes this bounded rather than risky is that the ladder is determinism-sensitive: the refactor must reproduce the exact per-level FFI call order and the exact set of options touched -- the escalation composition tests in the solver crate are what pin that, and a rung table is only legitimate if it preserves the order verbatim. Blast radius is the HiGHS retry escalation path alone (levels 3, 7, 10, 11 of one backend). Do not build a shared ladder abstraction across the two backends: HiGHS's string-keyed options and CLP's typed setters have no common shape worth a second layer, and a cross-backend ladder trait would be a one-consumer abstraction.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** one two-statement fragment written out four times in one retry ladder where the sibling backend is a declarative rung table; confined to retry.rs, so B not A.

#### Test bloat

**TD-035 · Sev C · duplication · effort S · confidence high**
Only crates/cobre-comm/src/factory.rs:419 is a verbatim vacuous restatement, asserting the same `CommBackend: Send + Sync` obligation that the production const fn at factory.rs:74 discharges under the identical `#[cfg(feature = "mpi")]` gate; of the two `FerrompiBackend` copies exactly one is redundant, because crates/cobre-comm/src/ferrompi.rs:488 and crates/cobre-comm/tests/factory_tests.rs:89 duplicate each other and the public-path reachability distinguishing the integration copy is already asserted by its siblings at factory_tests.rs:97 and :104. The title's claim that the const fn already performs the `FerrompiBackend` assertion does not hold: it entails it only transitively through the `Box<FerrompiBackend>` payload of `CommBackend::Mpi` at factory.rs:63, so that assertion is subsumed at this baseline rather than restated, and dropping both copies would unpin the `unsafe impl Sen…

- **Station:** cobre-comm (solver-comm, lens test-bloat; attacker SC-TB-001, ingest test-bloat-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-comm/src/factory.rs:74`, `crates/cobre-comm/src/factory.rs:419`, `crates/cobre-comm/src/ferrompi.rs:488`, `crates/cobre-comm/tests/factory_tests.rs:89`
- **Evidence:** factory.rs:74 is a production const fn whose recorded rationale is that its monomorphisation IS the compile-time check, so it fires on every cargo build. factory.rs:419 performs the identical monomorphisation inside #[cfg(test)], where it can only fail in a build that already failed to compile. ferrompi.rs:488 and factory_tests.rs:89 are the same body twice for FerrompiBackend, differing only in whether the type is named through crate:: or cobre_comm::. Four sites, two types, one mechanism that actually holds. Re-derive: `sed -n '70,77p' crates/cobre-comm/src/factory.rs; sed -n '417,422p' crates/cobre-comm/src/factory.rs; sed -n '487,491p' crates/cobre-comm/src/ferromp…`
- **Fix-shape:** Keep one mechanism per type and prefer the const-fn form, because it fires without cargo test and is the one already documented as load-bearing. Retire the three #[test] restatements; if FerrompiBackend deserves the same guarantee, give it a production const-fn assertion next to its unsafe impl Send/Sync rather than two test copies, so the check and the soundness argument sit together. This is not about the unsafe impl itself, which the manifest comment justifies.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-comm)
- **Calibration:** one vacuous #[test] restating a production const-fn obligation plus one duplicated Send+Sync probe.
- **Yardstick:** docs/design/testing-architecture.md §3.2 (sustainability: a test earns its place by catching a class of regression a cheaper check cannot) (see `td-queue.json`).
- **Queued to:** test-corpus

**TD-036 · Sev C · duplication · effort S · confidence high**
Only the src/local.rs to tests/local_conformance.rs overlap survives: the six body-identical LocalBackend Communicator assertions (allreduce identity sum, min and max, allreduce_buffer_mismatch, allgatherv_recv_too_small, broadcast_invalid_root) are duplicated with zero coverage residue beyond a single re-export smoke test, so the inline copies of those six are redundant. The factory.rs to tests/factory_tests.rs half of the title is dropped; neither of those two files is retirable, because the inline module needs the crate-private mpi_launch_detected() and the integration module asserts a CommBackend variant the inline module never checks.

- **Station:** cobre-comm (solver-comm, lens test-bloat; attacker SC-TB-002, ingest test-bloat-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-comm/src/local.rs:406`, `crates/cobre-comm/tests/local_conformance.rs:198`, `crates/cobre-comm/src/local.rs:360`, `crates/cobre-comm/tests/local_conformance.rs:244`, `crates/cobre-comm/src/local.rs:453`, `crates/cobre-comm/tests/local_conformance.rs:290`, `crates/cobre-comm/src/local.rs:379`, `crates/cobre-comm/tests/local_conformance.rs:44`, `crates/cobre-comm/src/local.rs:292`, `crates/cobre-comm/tests/local_conformance.rs:30`, `crates/cobre-comm/src/factory.rs:303`, `crates/cobre-comm/tests/factory_tests.rs:71`, `crates/cobre-comm/src/factory.rs:322`, `crates/cobre-comm/tests/factory_tests.rs:11`
- **Evidence:** The two versions send the same literals through the same public trait method and assert the same CommError variant with the same expected and actual values; only the assertion message differs (unexpected error against got). The inline module at local.rs:266-268 imports super::LocalBackend plus crate::{CommError, Communicator, ReduceOp}, and every one of those is publicly re-exported at lib.rs:57/59/60, so the inline copies reach no private surface that the integration binary lacks and the usual defence (the integr… Re-derive: `for n in test_local_allgatherv_recv_too_small test_local_broadcast_invalid_root; do echo "## $n"; sed -n "/fn $n(/,/^ }/p" crates/cobre-comm/src/loca…`
- **Fix-shape:** Name tests/local_conformance.rs the single owner of the public SS1.1-SS1.8 Communicator contract, which is the role its own module doc already claims, and retire the inline copies of contracts that are reachable through the public re-exports, keeping inline only assertions that need super:: privates. Keep test_local_collective_sequence and test_local_barrier_repeated, which have no inline counterpart, and keep one inline smoke assertion if the crate wants a test that survives with the integration binary removed. The lever here is single-owner maintenance, not link cost: testing-architecture.md section 5.1 explicitly puts the non-solver crates out of scope for binary consolidation because their binaries do not link the solver, so no consolidation of the two comm binaries is being proposed.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-comm)
- **Calibration:** six body-identical Communicator contract tests owned twice (inline and in the conformance binary) with zero coverage residue.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (one owner per public contract) (see `td-queue.json`).
- **Queued to:** test-corpus

**TD-037 · Sev B · duplication · effort M · confidence high**
Only the four in-src declarations are unjustified duplication: crates/cobre-solver/src/types.rs:736, src/freeze.rs:262, src/backends/clp/tests.rs:20 and src/backends/highs/tests.rs:19 are all #[cfg(test)] inside one crate and collapse to a single shared in-crate fixture module with no cargo feature and no coverage change - the clp_determinism.rs:13-14 reason does not apply to any of them, and freeze.rs:253 already names types.rs as the owner. The four integration-binary copies (conformance.rs:34, clp_determinism.rs:33, ffi_set_basis_non_alien_smoke.rs:29, clp_only_smoke.rs:24) are NOT confirmed under the proposed test_support hoist, because gating them on the non-default test-support feature empties three of them under CONTRIBUTING.md:71 and crates/cobre-solver/README.md:84 and voids the guarantee clp_only_smoke.rs:3-4 declares as its reason to exist.

- **Station:** cobre-solver (solver-comm, lens test-bloat; attacker SC-TB-003, ingest test-bloat-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/lib.rs:164`, `crates/cobre-solver/src/types.rs:736`, `crates/cobre-solver/src/freeze.rs:262`, `crates/cobre-solver/src/backends/clp/tests.rs:20`, `crates/cobre-solver/src/backends/highs/tests.rs:19`, `crates/cobre-solver/tests/conformance.rs:34`, `crates/cobre-solver/tests/clp_determinism.rs:33`, `crates/cobre-solver/tests/ffi_set_basis_non_alien_smoke.rs:29`, `crates/cobre-solver/tests/clp_only_smoke.rs:24`, `crates/cobre-solver/src/backends/clp/tests.rs:52`, `crates/cobre-solver/src/backends/highs/tests.rs:45`, `crates/cobre-solver/tests/conformance.rs:57`, `crates/cobre-solver/tests/clp_determinism.rs:58`
- **Evidence:** One whitespace-stripped hash for all eight declaration sites: the twenty-line builder is byte-identical everywhere, four times inside src and four times inside tests. Two further builders repeat the same way (make_fixture_row_batch four times at clp/tests.rs:52, highs/tests.rs:45, conformance.rs:57, clp_determinism.rs:58; make_empty_row_batch twice at freeze.rs:286 and sentinel_inf_row_probe.rs:76). The stated justification does not close the loop and has already rotted: clp_determinism.rs:13 says fixtures are re-… Re-derive: `for f in crates/cobre-solver/src/types.rs crates/cobre-solver/src/freeze.rs crates/cobre-solver/src/backends/clp/tests.rs crates/cobre-solver/src/bac…`
- **Fix-shape:** Add a fixtures submodule inside the already-shipped pub mod test_support at crates/cobre-solver/src/lib.rs:164, behind the existing test-support feature, and make it the single declaration of the SS1.1 stage template, the two-row batch and the empty batch. The four integration binaries import them exactly as conformance.rs:31-32 already imports test_support; the four in-src test modules reach them through crate::test_support. A binary that must also build without the feature keeps the whole-file gate that profile_retry_composition.rs:23 already demonstrates. Sequence this before the Phase-1 StageTemplate shed so the shed edits one builder instead of eight. Do not create a fixture crate: testing-architecture.md section 5.2 keeps the test-support cargo feature for exactly this purpose, and a separate crate would add a dev-dependency cycle, split the infra-genericity gate surface, and add no capability. Leave the genuinely distinct one-off builders alone (build_two_row_template and build_three_row_template in sentinel_inf_row_probe.rs, make_minimal_template in profile_retry_composition…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver; cross-references Part IV.5 (I.3-8))
- **Calibration:** the SS1.1 fixture is a StageTemplate builder re-declared in four in-src #[cfg(test)] modules (the four integration copies are the owner's call, see needs-human); every field change costs O(copies) edits — the ripple the yardstick's shared-fixture rule exists to prevent.
- **Part-I:** I.3-8 (cross-reference; verdict travels to Epic 9 with the per-field dispositions in `partI-handoff.json`).
- **Yardstick:** docs/design/testing-architecture.md §5.2 uniform test-support feature convention (not a dedicated test crate) (see `td-queue.json`).
- **Needs-human (owner gate):** Owner call on the four integration binaries: either give cobre-solver a tests/common shared-fixture module (the cobre-sddp idiom, no feature gate) or accept that a test-support-gated fixture makes the documented clp-only invocations (CONTRIBUTING.md:71, crates/cobre-solver/README.md:84) run zero cobre-solver integration tests.
- **Queued to:** test-corpus, alignment
- **Owner decision (2026-09-18, R7):** tests/common shared module — the four in-src copies collapse into one in-crate cfg(test) module; the four integration binaries share a tests/common fixture module (the cobre-sddp idiom, testing-architecture §5.1), no feature gate, so the documented clp-only invocations keep running the integration tests; test_support stays the seam for cross-crate consumers.

**TD-038 · Sev C · duplication · effort S · confidence high**
Narrower residue: test_research_probe_limit_status_on_ss11_lp (highs/tests.rs:846) asserts neither of the two model_status values it exists to observe (:857, :872 are printed only), so it cannot fail if a future HiGHS release lets SS1.1 reach the time or iteration limit — the exact premise the module comment's larger_lp justification rests on at :791-794. Dropped from the title: 'no behavioural assertion' (the two null-handle checks at :853/:867 and the helper's pass_lp == HIGHS_STATUS_OK at :833 do fail), and 'the fact is already prose' as a full substitute (the comment carries the conclusion, not the two observed status codes). The in-module fix precedent is the sibling test_research_partial_solution_availability (:1030), which asserts both statuses at :1044/:1065 while printing only the objective.

- **Station:** cobre-solver (solver-comm, lens test-bloat; attacker SC-TB-004, ingest test-bloat-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/src/backends/highs/tests.rs:846`, `crates/cobre-solver/src/backends/highs/tests.rs:796`, `crates/cobre-solver/src/backends/highs/tests.rs:787`
- **Evidence:** The only assertions in the test are two null-handle checks; both observed model statuses are printed and never asserted, so the test cannot fail for the property its own doc comment claims to document (it is labelled OBSERVATIONAL and says it documents that behavior). The conclusion it reaches is already recorded as durable prose at :791-794, where it justifies the existence of research_load_larger_lp, so the executable adds nothing the comment does not already carry, while costing two HiGHS create/run/destroy cyc… Re-derive: `sed -n '/fn test_research_probe_limit_status_on_ss11_lp/,/^ }/p' crates/cobre-solver/src/backends/highs/tests.rs | grep -n 'assert|eprintln|model_sta…`
- **Fix-shape:** Retire the #[test] and keep the module-comment sentence that records the finding, which is where the knowledge already lives. If the observed statuses must be held against a future HiGHS upgrade, give the test the assertion it lacks by pinning the two model-status codes instead of printing them; a printed value in a passing test is invisible under cargo test anyway. The other six test_research_* tests assert real properties at a layer the conformance binary cannot reach, because they set options not exposed through SolverInterface, and stay as they are.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** a research probe that prints the two statuses it exists to observe and asserts neither, restating the module comment's prose.
- **Yardstick:** docs/design/testing-architecture.md §3.2 (a test with no assertion is invisible and therefore dead) (see `td-queue.json`).
- **Queued to:** test-corpus

**TD-039 · Sev C · duplication · effort S · confidence high**
Narrowed to two anchors: the module-doc guarantee at clp_only_smoke.rs:3-4 is false under the very feature combination it names, because conformance.rs (six clp-gated tests) and clp_determinism.rs (crate-level clp gate at :16) already ship runnable integration tests that CI runs under --no-default-features --features clp, so this extra solver-linked binary does not earn its link cost. Conceded from the title: its single test is not a pure tolerance-only clone -- it asserts a strict SUBSET of the clp conformance coverage (no duals, no add_rows, no warm start) at a marginally TIGHTER tolerance (1e-9 against conformance 1e-8), so retiring it loses a slightly tighter objective/primal bound rather than nothing; and the redundancy claim holds against the clp-gated section of conformance.rs plus clp_determinism.rs jointly, not against conformance.rs alone.

- **Station:** cobre-solver (solver-comm, lens test-bloat; attacker SC-TB-005, ingest test-bloat-05)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/tests/clp_only_smoke.rs:48`, `crates/cobre-solver/tests/clp_only_smoke.rs:3`, `crates/cobre-solver/tests/conformance.rs:1450`, `crates/cobre-solver/tests/conformance.rs:26`
- **Evidence:** conformance.rs imports ClpSolver under #[cfg(feature = "clp")] at :26 and carries six clp-gated tests, so the clp-only build already ships runnable integration tests and the module doc claim at clp_only_smoke.rs:3-4 is false at this baseline. The assertion is also already there: test_solver_clp_load_model_and_solve (:1450) and clp_only_load_model_and_solve (:48) both build the same fixture, call load_model then solve(None), and assert objective 100.0 and primals 6.0/0.0/2.0; they differ only in the tolerance (1e-8… Re-derive: `grep -c 'cfg(feature = "clp")' crates/cobre-solver/tests/conformance.rs; grep -n 'fn test_solver_clp' crates/cobre-solver/tests/conformance.rs; sed -…`
- **Fix-shape:** Retire the binary and let the clp-gated section of conformance.rs be the clp-only guard it already is. If a standalone guard is genuinely wanted, the thing being guarded is a build-matrix property (that the clp-only feature combination still collects and links an integration binary), which belongs in the CI matrix rather than in a duplicated assertion; a surviving binary should then source its fixture from the test_support fixtures module and its module doc must be corrected, since the current wording states something conformance.rs already contradicts.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV crate table — L0 cobre-solver)
- **Calibration:** a whole solver-linked test binary whose module-doc guarantee is already met by the clp-gated conformance tests CI runs under the same feature set.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (integration binaries are expensive; group into one binary) (see `td-queue.json`).
- **Queued to:** test-corpus
- **Owner decision (2026-09-18, R11):** Retire the binary — clp_only_smoke.rs is retired; the clp-gated section of conformance.rs is the clp-only guard (testing-architecture §5.1: a solver-linked binary must earn its link cost).

**TD-040 · Sev C · duplication · effort S · confidence high**
Only test_fixture_stage_template_data (conformance.rs:137-158) is I.3-8 collateral - its assertions at :153-157 re-encode the five shed fields; test_fixture_row_batch_data (:160-170) carries none of them, so the title's second half holds for one of the two tests. What holds for both is narrower than the title: each asserts a same-file struct-literal builder against its own transcription, adds no relation the builder does not already contain, and invokes no SolverInterface method, while the objective and primal assertions at :72 and :176 already pin the same fixture against an LP-derived oracle.

- **Station:** cobre-solver (solver-comm, lens test-bloat; attacker SC-TB-006, ingest test-bloat-06)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-solver/tests/conformance.rs:138`, `crates/cobre-solver/tests/conformance.rs:161`, `crates/cobre-solver/tests/conformance.rs:153`
- **Evidence:** make_fixture_stage_template is declared at :34 in the same file, one hundred lines above the test, so every assertion compares a vec literal against a hand transcription of that same literal; the test exercises no production code and cannot detect a fixture edit, because an edit to the builder is exactly the change a maintainer would mirror into the assertions. Independently, :153-157 restate all five fields the Phase-1 shed removes, while partI-handoff.json records this collateral site as conformance.rs:154-156,… Re-derive: `sed -n '136,170p' crates/cobre-solver/tests/conformance.rs; grep -n 'fn make_fixture_stage_template' crates/cobre-solver/tests/conformance.rs`
- **Fix-shape:** If the fixtures move into the test_support fixtures module, one contract test belongs there beside the builder, asserting once that the shared SS1.1 LP is the LP the builders promise; the per-binary copy in conformance.rs is then redundant and goes with it. If the fixtures stay per-binary, delete both tests outright, because a constructor asserted against itself in the same file has no independent oracle. Either way the Phase-1 estimate for this site should carry five lines, not three.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV.5 (shed line :1471) — the fixture-contract test re-encodes the five shed fields)
- **Calibration:** two fixture-contract tests asserting a same-file struct literal against its own transcription; one of them re-encodes the five I.3-8 fields and goes with the shed.
- **Part-I:** I.3-8 (cross-reference; verdict travels to Epic 9 with the per-field dispositions in `partI-handoff.json`).
- **Yardstick:** docs/design/testing-architecture.md §3.2 (a tautological test restates another's coverage) (see `td-queue.json`).
- **Queued to:** test-corpus, alignment

#### Informational (recorded, no severity, no id)

- **LocalCommKind placement (owner gate R14, no id).** `LocalCommKind` (`crates/cobre-comm/src/traits.rs:275`) makes the trait-definition module import both concrete backends (`crates/cobre-comm/src/traits.rs:21`, `crates/cobre-comm/src/traits.rs:23`, behind the `shared-memory` feature) while `CommBackend` in `crates/cobre-comm/src/factory.rs:62` already does the identical enum dispatch in the module that owns concrete-backend enumeration. A placement question for whoever activates the shared-memory seam (mirror :79); the ratified seam itself is untouched.
- **SC-ARCH-009 / architecture-09 — solver capability / feature-query trait.** `SolverInterface` (`crates/cobre-solver/src/trait_def.rs:41`) carries no capability or feature query; the `compile_error!` pair (`crates/cobre-solver/src/lib.rs:44`, `crates/cobre-solver/src/lib.rs:50`) makes exactly one of `highs`/`clp` compile and `ActiveSolver`/`ActiveProfile` alias the winner, so the trait's only backend-varying surface is the opaque `Profile` associated type. Roadmap cross-reference: plans/generalizing/beyond-sddp-generalization.md § III.6 (per-feature capability traits arrive with their second consumer). **Alignment:** neutral. **Held as conflicts (L0 purity test):** the variant that builds the capability trait NOW has exactly one implementation — the one-consumer-abstraction condition — so it is tagged `conflicts`, HELD for owner override and excluded from the actionable set; roadmap-consistent alternative: keep this record, grow the trait when two backends must coexist in one binary (III.6). No such variant was raised at this baseline.

#### Positives (recorded so the report is not a defect-only list)

- crates/cobre-comm/src/traits.rs:372 SharedMemoryProvider hierarchy (with SharedRegion:320, LocalCommunicator:254, LocalCommKind:275, local.rs:155 HeapRegion, ferrompi.rs:264 split_local) — Ratified reserved seam with a named activating milestone; cleared at ingest, not a candidate, no id, no defender. (sanctioned by docs/design/reserved-seams-and-deferred-debt.md:79 — Shared-memory communicator trait hierarchy) _[architecture]_
- crates/cobre-solver/src/lib.rs:52 `pub(crate) mod ffi` with the single documented escape hatch at lib.rs:164 — The raw boundary is unnameable from outside the crate, and the one sanctioned exception is feature-gated, documented as verbatim pass-throughs, and named in the ffi module docs (ffi/mod.rs:10-13). An L0 crate that owns unsafe FFI and still exposes no raw handle to its consumers is the shape to prot… _[architecture]_
- crates/cobre-solver/src/lib.rs:44 and :50 — the both-backends and no-backend `compile_error!` pair — Backend selection is resolved at compile time, so L0 carries no run-time backend enum and no dispatch. This keeps the selection concern at the layer that owns it and is why the capability trait can wait for its second consumer instead of being built now. _[architecture]_
- crates/cobre-solver/src/basis_status.rs:101 to_discriminant_code / :116 from_discriminant_code — The one injective code space, with explicit arms so the encoding cannot move when the enum declaration order changes, and it is the only pair production cobre-sddp uses across the checkpoint and MPI wire paths (policy_export.rs:701,706; policy_load.rs:410,415; workspace.rs:85,92). A durable cross-b… _[architecture]_
- crates/cobre-comm/src/lib.rs:83 per_rank_counts / :94 prefix_displs — One owner for the allgatherv partition rule, in L0, with no engine vocabulary, serving several engine-side call sites. The single-owner shape is what lets the superseded cut-sync methods be deleted upstream without touching this crate. _[architecture]_
- crates/cobre-comm/src production doc comments — The multistage-vocabulary scan that finds seven sites in cobre-solver returns nothing here: cobre-comm describes itself only in terms of ranks, collectives and execution topology. The L0 genericity property already holds on this side and is worth keeping as the reference for the solver-side cleanup. _[architecture]_
- crates/cobre-solver/src/backends/highs/interface.rs:487 loud BasisInconsistent vs crates/cobre-solver/src/backends/clp/solver.rs:214-216 silent accept-and-repair — Intended backend behaviour, scoped by the trait contract itself (trait_def.rs:105 and :136-138 list BasisInconsistent as raised only where the backend validates, BasisRowCountMismatch as symmetric) and pinned by four conformance tests. Cited, never proposed for equalizing: the asymmetry is a proper… _[architecture]_
- crates/cobre-solver/src/backends/highs/interface.rs set_row_bounds:303, set_col_bounds:348, add_rows:253 and get_basis:500 — This backend is the target shape for the three CLP-side candidates in this envelope: subset-scoped bound writes that cross once per batch into a retained index scratch, an append that allocates nothing and only resizes retained buffers, and a single bulk basis transfer into retained buffers. Worth… _[performance]_
- crates/cobre-solver/src/freeze.rs FreezeScratch:21 and freeze_rows_into_template:57 — The freeze path is already allocation-free at steady state and the module doc says so for both the output template and the scratch: both are cleared and refilled per call without shrinking, and the caller is told to reuse them. The engine honours it, holding a scratch instance in the per-iteration… _[performance]_
- crates/cobre-solver/src/backends/profiled.rs set_profile:52 and solve:100 — The delta-only option dispatch is intact and nothing hot defeats it. The setter early-returns when the incoming profile equals the current one and solve forwards without re-applying by design. The unconditional trait-level forwarder that would defeat the guard resolves only in mock and test impleme… _[performance]_
- crates/cobre-comm/src/local.rs allgatherv:49, allreduce:97, broadcast:127 and barrier:134 — The single-process backend's collectives are argument validation plus one slice copy, with the two no-op collectives left as no-ops. No per-collective allocation and no partition recomputation on this path, so the single-process layout carries no collective overhead of its own. _[performance]_
- crates/cobre-comm/src/factory.rs — Backend construction and the topology probe run once per run, so their allocation and environment inspection are setup-time by the project rule and are explicitly not a finding in this envelope. _[performance]_
- crates/cobre-solver/src/backends/profiled.rs:30 ProfiledSolver and its delta gate at :52-58 — P9 verdict: the indirection is earned on consumers, not defended on shape. ProfiledSolver is the declared solver type of the engine workspace (crates/cobre-sddp/src/workspace/workspace.rs:723 and :772) and of the DCS solve path (cut/dcs.rs:590), and set_profile is driven from four distinct producti… (sanctioned by P9, judged by workspace-wide consumer count at the baseline) _[over-engineering]_
- crates/cobre-solver/src/lib.rs:164 pub mod test_support and its verbatim ffi pass-throughs — P12 verdict: the probe only licenses flagging a wrapper that adds NOTHING over ffi, and this one adds visibility. mod ffi is pub(crate) (lib.rs:52), so an integration test -- a separate crate -- cannot name crate::ffi at all; the feature-gated forwards are the only legal path from a test binary to… (sanctioned by P12, with lib.rs:52 pub(crate) mod ffi as the reason the forwards are not redundant) _[over-engineering]_
- crates/cobre-comm/src/ferrompi.rs:75-76 unsafe impl Send / Sync for FerrompiBackend — P12 verdict: the two impls are not a redundant wrapper but a documented soundness contract. The comment above them states the three-point argument (Mpi is constructed on the calling thread, single ownership bars any other thread from reaching MPI_Finalize via Mpi::drop, and ThreadLevel::Funneled re… (sanctioned by P12, with the RAII/ownership argument recorded inline at ferrompi.rs:64-74) _[over-engineering]_
- crates/cobre-solver/src/backends/clp/retry.rs:95 const RUNGS table (and the no-time-branching contract documented at :4-5) — The CLP ladder is the shape the HiGHS ladder should converge on: the escalation is data (one fixed-order const array of five rungs), and the module doc states the determinism property it buys -- fixed rung order, no randomness, no time-dependent branching, so results stay bit-for-bit identical acro… (sanctioned by Baseline reading of clp/retry.rs:4-5 and :95, used as the contrast case for the HiGHS candidate) _[over-engineering]_
- HiGHS-loud / CLP-silent basis validation (highs/interface.rs:487 SolverError::BasisInconsistent against clp/solver.rs:213-216) — Intended backend behaviour, cited and not raised: the divergence is a property of the two solver libraries (CLP's per-element setters accept an inconsistent offered basis and Clp_dual repairs it), the trait contract already scopes it, and four conformance tests pin both sides at this baseline. No f… (sanctioned by plans/architecture-debt-audit/stations/solver-comm/prior-register.md - Intended behaviour, never a finding) _[over-engineering]_
- crates/cobre-solver/tests/_q1_sign_convention_probe.rs — A decision record that asserts the row-equality against column-bound sign convention rather than merely observing it; its underscore prefix marks it as a probe kept deliberately, and its StageTemplate literals are a record of the convention, not fixture duplication. (sanctioned by plans/architecture-debt-audit/stations/solver-comm/prompts/sc-test-bloat.md — P14 Protected) _[test-bloat]_
- crates/cobre-solver/tests/_clp_sign_convention_probe.rs — Same protected class as the Q1 probe, and it already imports cobre_solver::test_support at :21, which is the second live proof that the feature-gated module is reachable from an integration binary. (sanctioned by plans/architecture-debt-audit/stations/solver-comm/prompts/sc-test-bloat.md — P14 Protected) _[test-bloat]_
- crates/cobre-solver/tests/conformance.rs:1601 and :1636 against crates/cobre-solver/src/backends/highs/tests.rs:548 and crates/cobre-solver/src/backends/clp/tests.rs:609 — Not duplication, and a later sweep should not collapse them. The conformance pair is the public-API pin the prior register names for the basis-validation contract; the sibling-module pair additionally asserts basis_consistency_failures and basis_offered, internal statistics an integration test cann… _[test-bloat]_
- crates/cobre-solver/tests/profile_retry_composition.rs:23 whole-file #[cfg(feature = "test-support")] gate — The idiom that makes the fixture hoist safe: an integration binary in this very directory already compiles to nothing without the test-support feature, so gating shared fixtures on that feature introduces no new build mode. _[test-bloat]_
- crates/cobre-solver/src/backends/highs/tests.rs:796 mod research_tests (the six asserting tests) — Real coverage below the wrapper: they set HiGHS options that SolverInterface does not expose and pin the non-optimal model-status mapping, which the conformance binary cannot reach through the public API. The module comment also records why the SS1.1 LP is too small for limit statuses to fire, whic… _[test-bloat]_

#### ↩︎ Cleared (dismissed — do not re-raise)

- **SC-PERF-001 / performance-01 — Partition helpers hand back freshly allocated owned vectors and the partition is recomputed from scratch at every collective** — dismissed on mechanism, not on a seam: The title's load-bearing clause -- the partition is recomputed at every collective -- does not hold at the baseline for the majority of the anchored consumers. cut_sync.rs:182 and :186 are not in the record-sync path the candidate attributes them to; both sit inside the body of CutSyncBuffers::with_distribution (declared at cut_sync.rs:173), the pre-allocating constructor, and its only production caller is training/session/mod.rs:252, executed once when the training session is built. The module doc at cut_sync.rs:… No `sanctionedBy` (no reserved seam involved); no `intendedBehaviour` (does not rest on the basis-validation asymmetry). Layout `2x2` recorded for the perf sweep's information only; not queued.
- **Shared-memory communicator trait hierarchy** (`SharedMemoryProvider`, `SharedRegion<T>`, `LocalCommunicator`, `LocalCommKind`, `HeapRegion<T>`, `FerrompiBackend::split_local`) — ratified reserved seam, mirror section *Shared-memory communicator trait hierarchy* (`docs/design/reserved-seams-and-deferred-debt.md:79` at this baseline; register OD-001 KEEP-RESERVED). Applied as an ingest filter: every lens filed it in Positives, no candidate proposed its removal, no defender saw it, no OD id exists for it.
- **Superseded cut-sync public methods** (`sync_cuts`, `pack_local_records`, `sync_packed_records`, superseded by `sync_level_records`; anchors in `crates/cobre-sddp/src/cut/cut_sync.rs:243`, `crates/cobre-sddp/src/cut/cut_sync.rs:400`, `crates/cobre-sddp/src/cut/cut_sync.rs:495`, `crates/cobre-sddp/src/cut/cut_sync.rs:581`) — raised by two lenses (SC-ARCH-002 / architecture-02, SC-OE-002 / over-engineering-02), recorded dup-of the mirror section *Superseded cut-sync public methods* (`docs/design/reserved-seams-and-deferred-debt.md:334` at this baseline; register CD-019) and handed to E5 with NO new id (`handoffs.json` records[]).
- **HiGHS-loud / CLP-silent basis-validation asymmetry** — intended backend behaviour pinned by `crates/cobre-solver/tests/conformance.rs:1601`, `crates/cobre-solver/tests/conformance.rs:1636`, `crates/cobre-solver/tests/conformance.rs:1672`, `crates/cobre-solver/tests/conformance.rs:1724`; never a finding. No defender dismissal rested on it (no `intendedBehaviour` was required).

#### Prior-register dispositions

- **Shared-memory communicator trait hierarchy (OD-001 KEEP-RESERVED)** — keep; ingest filter, see Cleared.
- **Superseded cut-sync methods (CD-019)** — dup-of → E5; anchors live in cobre-sddp, not in either station crate.
- **HiGHS-vs-CLP basis-validation divergence** — intended behaviour; cited by the attacker prompt, never raised. The `.claude/rules/sddp.md:446` sentence that calls it "unpinned by any test" is stale at this baseline (the four conformance tests above pin it) — doc drift handed to E7 (`handoffs.json` E7.observation).
- **Per-crate `[lints]` tables (seeded)** — delivered: CD-075 carries the fix-shape and severity the seed owed; the bare observation was never re-raised.
- **Five coarse re-raise hits** (architecture-01/05, over-engineering-05, test-bloat-02/05) — adjudicated distinct at ingest (`ingest-log.md`).

#### Part-I cross-references

**Item I.3-8 — StageTemplate carries multistage geometry (claim disposition: sharpen; proposed Alignment `advances-1`).** Per-field dispositions from `partI-handoff.json` (all write-only at the baseline; owner after the shed one layer up):

| field | anchor | disposition | owner after shed |
|---|---|---|---|
| `n_state` | `crates/cobre-solver/src/types.rs:270` | retire | crates/cobre-sddp/src/lp/indexer/state_space.rs:97 StateSpace.n_state |
| `n_transfer` | `crates/cobre-solver/src/types.rs:278` | retire | crates/cobre-sddp/src/lp/builder/template.rs:445 (derived from StateSpace.hydro_count and StateSpace.max_par_order — no separate stored owner is needed) |
| `n_dual_relevant` | `crates/cobre-solver/src/types.rs:287` | retire | crates/cobre-sddp/src/lp/builder/layout.rs:509 StageRowLayout.n_dual_relevant |
| `n_hydro` | `crates/cobre-solver/src/types.rs:290` | retire | crates/cobre-sddp/src/lp/indexer/state_space.rs:100 StateSpace.hydro_count |
| `max_par_order` | `crates/cobre-solver/src/types.rs:297` | retire | crates/cobre-sddp/src/lp/indexer/state_space.rs:104 StateSpace.max_par_order |

Propagation sites: `crates/cobre-solver/src/freeze.rs:158` (production, lines 158-162); `crates/cobre-solver/src/trait_def.rs:380` (test, lines 380-384); `crates/cobre-solver/src/backends/profiled.rs:252` (test, lines 252-256). Entries carrying the cross-reference: CD-079 (architecture-06), TD-037 (test-bloat-03), TD-040 (test-bloat-06). Renaming without shedding is a sharpen, not a retire (recorded in the handoff). The sixth field of the same struct is not in scope.

#### Handoff queues

- **E11 — reconciliation follow-up** (owner gate R12, no id): the HiGHS retry ladder branches on wall-clock time (`crates/cobre-solver/src/backends/highs/retry.rs:40`, `crates/cobre-solver/src/backends/highs/retry.rs:52`, `crates/cobre-solver/src/backends/highs/retry.rs:84`) while the CLP ladder documents no time-dependent branching (`crates/cobre-solver/src/backends/clp/retry.rs:4`); routed as a reproducibility follow-up for the HiGHS backend.
- **E7 — doc drift** (owner gate R16): `crates/cobre-comm/tests/local_conformance.rs:4` cites a `backend-testing.md` that does not exist at the baseline; the docs station owns the correction.
- **E9 — owner question** (owner gate R13): whether I.3-8 sheds into the engine now (geometry moves again at the 0b carve) or waits for the carve and sheds once — E9 decides; CD-079's dispositions are unchanged.
- **E10 — shaping questions** (owner gate R15): PD-032/PD-034 — does the CLP API expose an index-scoped bound writer and a bulk basis-status accessor that preserve the factorization; PD-033 — must the retained mirror stay fully merged; both answered by the fix ticket against the profile, not here.
- **E9 — alignment** (3): item I.3-8 per-field dispositions (five fields `retire`), the three propagation sites, proposed Alignment `advances-1`; entries CD-079, TD-037, TD-040 — `handoffs.json` E9.
- **E7 — build-ci gate blind spot** (1): `scripts/ci/check-infra-genericity.sh` reports clean on `cut_nz_per_col` (`crates/cobre-solver/src/freeze.rs:22`) because `_` is a word character and the pattern is `\bcut\b`; a second blind spot — the awk prefilter skips every in-src `#[cfg(test)]` region — and the `.claude/rules/sddp.md` doc drift travel with it. Entry CD-081. No script edit proposed here — `handoffs.json` E7.
- **E5 — cut-sync dup-of** (2 candidates, 0 ids): `sync_cuts` / `pack_local_records` / `sync_packed_records` at their cobre-sddp anchors, superseded by `sync_level_records`; `assignedId` stays null — `handoffs.json` E5 + records[].
- **E10 — performance sweep** (3): every Sev-A/B PD entry, each UNMEASURED with its layout and exercising call sites — PD-032 (`4t`, cobre_clp_chg_bounds), PD-033 (`4t`, ClpSolver::add_rows), PD-034 (`4t`, ClpSolver::get_basis). Layout contract: `4t` for solver-side / FFI-boundary claims, `2x2` for collective claims; no collective claim survived (performance-01 dismissed). See `perf-queue.json`.
- **E8 — test-corpus** (6): every TD entry with its binaries, anchors and the testing-architecture yardstick section it violates — TD-035, TD-036, TD-037, TD-038, TD-039, TD-040. See `td-queue.json`.

#### Owner gate — decisions

Ratified 2026-09-18 in the main session over the digest in `stations/solver-comm/gate.md` (gitignored plans tree, not an anchor) (16 AskUserQuestion rounds: Part-I first, two severity batches, the informational note, twelve needs-human items). All 20 calibrated entries accepted as recorded; no downgrade, reject, defer or override at the gate (CD-075's B→C is the calibration's house rating, accepted). Severity would read `new (reviewer: original)` on an owner downgrade; perf rows keep their layout and stay UNMEASURED until E10.

| ID | Decision | Severity | Alignment | Rationale (owner) | Trigger / override / handoff |
| -- | -- | -- | -- | -- | -- |
| OD-032 | accept | C | neutral | as recorded (R3) | direction: Delete |
| TD-035 | accept | C | neutral | as recorded (R3) | E8 |
| TD-036 | accept | C | neutral | as recorded (R3) | E8 |
| CD-075 | accept | C (reviewer: B) | neutral | as recorded (R3) | direction: Keep forbid + checker |
| CD-076 | accept | C | neutral | as recorded (R3) | direction: Now-fix reword |
| CD-077 | accept | C | neutral | as recorded (R3) | - |
| CD-078 | accept | B | neutral | as recorded (R2) | - |
| CD-079 | accept | B (A-risk) | advances-1 | as recorded (R1) | E9: dispositions unchanged |
| CD-080 | accept | C | neutral | as recorded (R3) | direction: Keep mapping unconditional; fix README |
| CD-081 | accept | C | neutral | as recorded (R3) | - |
| OD-033 | accept | B | neutral | as recorded (R2) | direction: Retire at the next licensed public-API break |
| OD-034 | accept | C | neutral | as recorded (R3) | - |
| OD-035 | accept | B | neutral | as recorded (R2) | - |
| PD-032 | accept | B | neutral | as recorded (R2) | E10: 4t, UNMEASURED |
| PD-033 | accept | B | neutral | as recorded (R2) | E10: 4t, UNMEASURED |
| PD-034 | accept | B | neutral | as recorded (R2) | E10: 4t, UNMEASURED |
| TD-037 | accept | B | neutral | as recorded (R2) | E9 cross-ref; E8; direction: tests/common shared module |
| TD-038 | accept | C | neutral | as recorded (R3) | E8 |
| TD-039 | accept | C | neutral | as recorded (R3) | E8; direction: Retire the binary |
| TD-040 | accept | C | advances-1 | as recorded (R3) | E9 cross-ref; E8 |

**Informational kept:** SC-ARCH-009 capability trait (R4) — no id, III.6 cross-reference, one-consumer hold stands; LocalCommKind placement (R14) recorded above.

**Cleared by this gate (do not re-raise):** none — no entry was rejected.

**Presented read-only, no decision taken:** the ratified shared-memory hierarchy (`SharedMemoryProvider`, `SharedRegion<T>`, `LocalCommunicator`, `LocalCommKind`, `HeapRegion<T>`, `FerrompiBackend::split_local`) stays sanctioned (mirror :79); the superseded cut-sync methods stay a dup-of handoff to E5 with no id (mirror :334, CD-019); the basis-validation asymmetry stays intended behaviour.

**Needs-human answers (14/14):** see `gate.md` §3.3 — CD-080 keep the mapping unconditional and fix the README; OD-033 retire the hot-start acquire half at the next licensed API break; TD-037 tests/common shared module for the integration binaries; OD-032 delete; CD-075 keep the workspace forbid and add a checker; CD-076 now-fix reword; TD-039 retire the binary; HiGHS wall-clock retry → E11 reproducibility follow-up; I.3-8 landing → E9 decides; LocalCommKind placement → informational; CLP API / retained-mirror questions → E10; stale `backend-testing.md` citation → E7.

**Gate: RETURNED 2026-09-18** — baseline `077dbe2c`; accepted 20, amended 0, downgraded 0, rejected 0, deferred 0, overridden 0.

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp

### STATION 5 — cobre-sddp (2026-09)

**Station.** cobre-sddp — the L3 SDDP engine of the target layering (setup carriers, the LP builder and indexer, the cut pool and selection, the forward/backward/simulation drivers, the per-worker workspaces, the policy writer/loader). **Method.** Four lenses (architecture, performance, over-engineering, test-bloat) × four sub-stations (5a setup + policy + stochastic + config, 5b lp/, 5c cut + training + solve + workspace, 5d simulation + production + support) = sixteen read-only attacker cells, a pre-ingest gate, five ingest screens (anchor, station scope, reserved seams, dup-of / re-raise, contract) and one read-only defender per surviving candidate. The reserved seams (`LipschitzConfig.mode`, the writer's second-family slot body, the anticipated `delivery_date` channel, the Legacy cost-scale branch, the `#[allow]` census, the superseded cut-sync methods) were applied as FILTERS: every lens filed them as positives, none reached a defender, none carries an id.

**Baseline.** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (pinned 2026-09-17; the E01 scaffold heading above keeps its minted `a136840d`). Read-only station: no tracked file under `crates/`, `scripts/`, `schemas/` or `.github/` was modified. 75 attacker candidates: 65 defended (49 confirmed, 16 dismissed, 0 null), 10 dup-of (5 into CD-007, 1 into OD-009, 4 cross-cell folds); every anchor resolves at the baseline (339 checked / 0 failing). Of the 49 confirmed, 5 sharpen a live prior id and are merged into it below (no new number); 44 mint ids — 2 × B (A-risk), 19 × B, 23 × C; reviewer downgraded on 12 and upgraded on 1 (both recorded per entry). Alignment hints are provisional — Epic 9 adjudicates. lp/ figures quoted here reconcile with measurements/lp-inventory.json at the same pin: 30 files, 44,682 lines, 11,655 non-test. Crate: 163 source files, 58,471 non-test lines; 40 integration binaries (72,356 lines) plus 5 tests/common files and 11 template_integration files.

#### Wave 4 / 6 / 7 dispositions (owned prior ids — reused, never re-minted)

One row per owned prior id from wave-dispositions.json (re-verified at `077dbe2c`), plus the existence-only row for PD-004. Retire rows carry the resolving commit; sharpen rows carry the superseded and the surviving claim; every anchor is symbol-level at this baseline (the Wave 4/6/7 line anchors predate the workspace/ directory-module split, the solve/ carve-out and the `ConstructionConfig` deletion). The `E05-5 delta` column is the defender-verified sharpening a new candidate added to the live entry — merged here, no new number. House severity follows the register's own precedents (retrofitted-variant class → B; CD-004 silent-divergence → B (A-risk); organisational asymmetry → C); where it differs from the register header, both are shown.

| ID | Wave | Disposition | Evidence anchor (path · symbol) | Superseded → surviving / resolved by | E05-5 delta (defended) | Sev (house / register) | Alignment |
|---|---|---|---|---|---|---|---|
| CD-001 | 4 | retire | `crates/cobre-sddp/src/setup/stochastic_pipeline.rs::build_stochastic_context_for_study` | resolved by `b051c410` | — | A | neutral |
| CD-003-construction-hop | 4 | retire | `crates/cobre-sddp/src/setup/params.rs::from_config` | resolved by `4075c4e8` — superseded: the Broadcast→Construction hop: a 21-field ConstructionConfig literal assembled by into_construction_config from StudyParams, the third near-isomorphic carrier; residue: Config → BroadcastConfig (cobre-cli commands/broadcast.rs) → StudyParams::from_config (setup/params.rs) is a two-carrier residue: the MPI arm still hand-projects Config into BroadcastConfig, then eve… | — | B | advances-0a |
| CD-006 | 4 | retire | `crates/cobre-sddp/src/setup/node_graph.rs::frontier_node` | resolved by `3f4c3db3` | — | C | neutral |
| CD-004 | 4 | sharpen | `crates/cobre-sddp/src/setup/params.rs::from_config` | superseded: three ~20-field near-isomorphic structs (StudyParams / ConstructionConfig / BroadcastConfig) and the params → ConstructionConfig hop — deleted by 4075c4e8 → surviving: the wire projection is untouched: Config → BroadcastConfig (crates/cobre-cli/src/commands/broadcast.rs, from_config at :143) and Config → StudyParams (setup/params.rs from_config at :186) are still two hand-kept field-for-field twins (both project export_stat… | 5a-architecture-01 confirmed: Verified delta over CD-004: the two projections are NOT independent Config readers -- BroadcastConfig::from_config calls StudyParams::from_config at broadcast.rs:144 and every wire field except training_enabled, policy_mode and the two scenario sources traces to that single params value -- so CD-00… | B (A-risk) | advances-0a |
| CD-005 | 4 | sharpen | `crates/cobre-sddp/src/setup/mod.rs::StudySetup` | superseded: a local struct split: 'half-finished NCS sub-struct extraction' plus a field-count census (35 fields, 2026-08-22) framed as god-struct cleanup → surviving: StudySetup conflates precomputed model state (stage_data, stochastic, fcf, hydro_models, scenario_libraries, node_graph, the ncs_* patch vectors) with resolved run-config (loop_params, simulation_config, policy_path, …); the setter wall in setup/accessors.rs… | 5a-architecture-00 confirmed: Delta over CD-005: of the seven setter-wall members, `StudySetup::set_budget` (setup/accessors.rs:62-65) is the only one with zero call sites anywhere in the repository at the pin AND the only one that is `pub` and ungated without a caller justifying the ungating (its three test-only siblings at ac… | B | advances-0a |
| CD-024-successor | 4 | sharpen | `crates/cobre-sddp/src/policy/resolved_parameters.rs::build_resolved_parameters` | superseded: the resolved-parameters broadcast pair CD-024 registered (serialize_/deserialize_resolved_parameters, RESOLVED_PARAMETERS_WIRE_VERSION) — deleted as dead code; CD-024 itself RESOL… → surviving: ResolvedParameters is still built per rank (policy/resolved_parameters.rs build_resolved_parameters; setup/mod.rs 'Per-rank, never broadcast'); the successor question is whether a rank-0-built resolved run config is broadcast — the seam activates against the… | — | C | conflicts — HELD (conflicts: one-consumer seam; alternative: activate-or-die against the single wire projection) |
| CD-015 | 6 | sharpen | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` | superseded: extract a standalone commit_by_scenario_cuts to mirror by_node_finish (a symmetry fix on its own) → surviving: compute_one_backward_node (backward_pass_state.rs :1727) still inlines by_scenario's post-dispatch aggregation while by_node's is extracted (training/backward/by_node.rs by_node_finish :416); the asymmetry is real and is resolved by the SAME extraction as CD-… | 5c-architecture-01 confirmed: Delta over CD-015: the live entry mislabels the inline residue. By-scenario's risk aggregation is NOT in compute_one_backward_node — aggregate_cut_into runs inside process_by_scenario_backward (by_scenario.rs:525) and the fn returns a finished StagedCut. The asymmetry that survives is narrower: exa… | B | neutral |
| CD-022 | 6 | sharpen | `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward` | superseded: a ~66-line duplicated span framed as its own dedup → surviving: the successor-outcome reification (successors loop → build_delta_cut_row_batch_into in training/forward/delta_cut_batch.rs → SuccessorEntry → SuccessorOutcomes in training/backward/mod.rs) is still inlined twice — compute_one_backward_node (:1727) and run_enu… | — | B | neutral |
| CD-018 | 6 | keep | `crates/cobre-sddp/src/cut/cut_selection.rs::CutSelectionStrategy` | keep: `CutSelectionStrategy` conflates two selection paradigms — periodic value-based (Level1/Lml1/Dominated share `select_for_stage`) and lazy-solve DCS (`Dynamic`), whose variant early-returns empty and whose real logic lives in `dcs.rs::lazy_solve_preloaded` | — | C | neutral |
| CD-035 | 6 | keep | `crates/cobre-sddp/src/training/backward/outcome_aggregation.rs::write_opening_outcome` | keep: `write_opening_outcome`/`accumulate_opening_outcome` take a bare `&CutStateProjection` where two role-distinct projections are live (`child_cut_layout` vs the cut-generating parent's `succ_spec.cut_state`) — passing the wrong one compiles | — | B | neutral |
| CD-034 | 6 | keep | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_operational_violation_entries` | keep: The min/max outflow entry blocks in `fill_operational_violation_entries` are two hand-mirrored `for blk` loops while the sibling rows file emits the same row pair from a two-element descriptor | — | B | advances-0b |
| CD-023 | 6 | sharpen | `crates/cobre-sddp/src/training/stage_solve_prep.rs::StageSolvePrep` | superseded: 're-home StageSolvePrep out of training/' as an unstarted move → surviving: the carve-out is half done: the neutral solve/ module exists (mod.rs, partition.rs, solver_phase.rs, stage_solve.rs) and owns the solve ENTRY (run_stage_solve), but training/stage_solve_prep.rs and training/stage_solve_prep/tests.rs still live under training/… | — | C | neutral |
| CD-021 | 7 | sharpen | `crates/cobre-sddp/src/workspace/workspace.rs::SolverWorkspace` | superseded: 'a flat workspace.rs mega-file asymmetric with the directory-module convention' — the organizational half is discharged: workspace/ is a directory module (mod.rs 32, context.rs 27… → surviving: all nine structs (CapturedBasis, WorkspaceSizing, BackwardAccumulators, ByNodeScratch, ScratchBuffers, SolverWorkspace, WorkspacePool, BasisStore, BasisStoreSliceMut) still share workspace/workspace.rs; the split-by-concern direction is preserved against the… | — | C | neutral |
| CD-007 | 7 | sharpen | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_load_balance_entries` | superseded: sizes as recorded 2026-08-22: entries.rs 10,079 and columns.rs 9,026 lines (the register already says GREW) → surviving: re-measured at the pin: entries.rs 10,093 lines (first #[cfg(test)] at L1680) and columns.rs 9,319 (L1305) — inline test modules dominate both while sibling builder submodules use extracted tests.rs (template/tests.rs 5,356, layout/tests.rs 3,565); two conven… | dup-of folded: 5a-test-bloat-00, 5b-test-bloat-00, 5b-test-bloat-01, 5c-test-bloat-04, 5d-test-bloat-05 | C | advances-0b |
| CD-012 | 7 | keep | `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management` | keep: `run_cut_management` (~250 lines, `too_many_lines`-allowed) inlines cut-selection + budget enforcement; its rationale defends against FREE-fn extraction, a category error since the natural refactor is `&mut self` methods | — | C | neutral |
| CD-014-remnant | 7 | keep | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` | keep: `compute_one_backward_node` (~290 lines, biggest fn in the crate) — the god-fn residue that remains after the shared successor-reification extraction (CD-022) and the by_scenario commit extraction (CD-015) relieve it | — | C | neutral |
| CD-016 | 7 | keep | `crates/cobre-sddp/src/training/backward_pass_state.rs::process_stage_backward` | keep: Both backward scheduler workers duplicate ~25 lines of per-worker `backward_accum` buffer pre-allocation (`process_stage_backward` vs `process_stage_backward_by_node`) | 5c-architecture-02 confirmed: Beyond the pre-allocation duplication CD-016 records, the same two backward drivers carry a second copied block -- the slot_increments into metadata_sync_contribution fold at by_scenario.rs:559-566 and again at by_node.rs:386-392 -- and the two copies have ALREADY diverged in their fold bound: by-s… | C | neutral |
| CD-028 | 7 | keep | `crates/cobre-sddp/src/training/forward/enumerated.rs::run_enumerated_forward` | keep: Enumerated forward and enumerated simulation reimplement the mid-level claim/scatter orchestration skeleton independently (`run_sweep`/`enumerated_sim_stage_worker` vs `run_enumerated_forward`/`enumerated_stage_worker`); only the low-level primitives are shar… | 5c-architecture-04 confirmed: Delta over CD-028: the shared owner claim_scatter.rs already exists but its module doc's consumer list (:1-4) names only by_node and the enumerated forward while simulation/enumerated.rs:26 is a third importer of both primitives, so the boundary statement a future reader trusts is false at the pin;… | B | neutral |
| CD-030 | 7 | keep | `crates/cobre-sddp/src/simulation/enumerated.rs::mark_own_paths` | keep: `mark_own_paths` (simulation/enumerated.rs) reimplements the ~10-line path-marking loop that is inlined in `training/forward/enumerated.rs`; folds with the CD-028 dedup | — | C | neutral |
| CD-037 | 7 | keep | `crates/cobre-sddp/src/setup/mod.rs::admission_gate` | keep: The gap-rule rejecters each re-evaluate `rule_is_gap` + `training_enumerated`, and which rejection message a sampled-forwards study sees depends on call order inside `admission_gate` (documented as deliberate) | — | C | neutral |
| CD-038 | 7 | keep | `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs::dot_trial_state` | keep: `CutStateProjection` exposes only the fused `dot_trial_state`; the cut-selection value sweep in `run_cut_management` open-codes the gather-only loop (via `global_state_index`) — loop-shape duplication, not a hazard | — | C | neutral |
| OD-009 | 7 | keep | `crates/cobre-sddp/src/training/session/runtime.rs::RuntimeHandles` | keep: Production items ship as `#[allow(dead_code)]` 'for symmetry, exercised only by tests' (`RuntimeHandles.export_states`, `RankDistribution.my_rank`, `ExchangeBuffers::new`) — the sanctioned Symmetry-or-test-retention class; `#[cfg(test)]`-gating is the cleaner… | dup-of folded: 5c-over-engineering-00 | C | neutral |
| PD-004 | — | existence-only (deferred, do-not-touch) | `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward` | exists at the baseline (721); do-not-touch: DEFERRED pending a profile (register L1316; do-not-touch list L1942) | — | register Sev (unchanged) | neutral |

Calibration notes on the reused rows: **CD-004** carries Sev B (A-risk) on its own precedent — the in-process and wire projections must agree field-for-field or MPI diverges silently from local; the A-risk marker is carried by the SILENCE of the divergence, not by blast radius — and Alignment `advances-0a` citing Part V §V.1 (Phase 0: the study-block admission-gate carrier) and Part IV §IV.4 (the engine-selection seam); the E05-5 delta narrows the divergence surface to one site (`export_states` at cobre-cli broadcast.rs:214). **CD-005** restates as the engine-tagged setup stages (advances-0a, Part V §V.1); its fix-shape cannot claim byte-neutrality (the NCS split feeds hot-path StageContext borrows and the patch-identity contract) and is marked for a re-baseline decision at the owner gate. **CD-024-successor** is the standing `conflicts` row: a one-consumer seam kept alive against a hypothetical consumer — HELD for owner override, excluded from the actionable set; roadmap-consistent alternative: activate-or-die against the single wire projection (Part V §V.0 'pull, don't push'). **CD-015** stays Sev B by the owner-elevated retrofitted-variant asymmetry class (live instances CD-013, CD-015, CD-027); its delta re-labels the inline residue as the cross-worker merge-and-commit tail. **CD-016**'s delta adds a second copied fold block that has already diverged in its bound (the CD-016 pre-allocation duplication stays Sev C). **CD-028**'s delta corrects the shared owner's consumer list and folds CD-030. **CD-021** is re-anchored under the workspace/ directory module (all nine structs still share workspace/workspace.rs) at Sev C. **CD-007** absorbs five per-manifest homing candidates as dup-of evidence at Sev C. **OD-009** absorbs one dup-of. Retire rows: CD-001 by `b051c410`, the CD-003 Construction hop by `4075c4e8`, CD-006 by `3f4c3db3`.

#### Part-I cross-references — item 7

**Item I.3-7 — the config type is SDDP-shaped (claim disposition: sharpen; proposed Alignment `advances-0a`, proposed phase 0a).** cobre-sddp-side disposition recorded here on `crates/cobre-sddp/src/setup/params.rs::from_config` (`StudyParams::from_config`): narrowed to the wire projection: Config → BroadcastConfig → StudyParams::from_config is still two hand-kept SDDP-shaped twins with no generic seam between the config layer and the engine. Changed since v0.12: ConstructionConfig and into_construction_config deleted (4075c4e8); StudyParams::from_config now captures export_states (setup/params.rs :339, from config.exports.states); BroadcastConfig survives unchanged in crates/cobre-cli/src/commands/broadcast.rs (:87; its own from_config at :143 projects the same fields, export_states at :214); the Phase enum stayed in cobre-sddp (solve/solver_phase.rs) and cobre-solver staye…. The other two sub-items are owned elsewhere — `Config.training` by the core+io station (E02, ratified) and `BroadcastConfig` by the cli+python station (E06). Entries carrying the cross-reference: CD-004 (reused, sharpened by 5a-architecture-01) and CD-082 (5a-architecture-02, the scalar-parameters placeholder). Handed to the generalization-alignment epic (E9) with partI-handoff.json.

**Item I.3-8 (StageTemplate multistage geometry — owned by the solver+comm station)** reaches this station through TD-045 (5a-test-bloat-05, `state_layout_for` clones in the integration binaries), which travels to E9 with the same reference.

#### Positives (recorded so the report is not a defect-only list)

**5a**

- crates/cobre-sddp/src/policy/policy_load.rs::validate_policy_load and PolicyLoadProof — Lens question 4 asked whether the policy round-trip contract is stated once or restated per file. It is stated once and then enforced by TYPE, which is stronger than either option: validate_policy_load is the only way to obtain a PolicyLoadProof<K>, the proof… (sanctioned by .claude/rules/sddp.md section 'Policy-load compatibility validation is mandatory') _[architecture]_
- crates/cobre-sddp/src/policy/policy_load.rs::checkpoint_terminal_cost_scale_factor — Examined and clean, not a finding. Reading only the terminal pool's cost_scale_factor and applying it to every pool looks like a missing homogeneity check, but the pinned contract explicitly specifies the terminal pool's own value as the FullFcf source scale,… (sanctioned by .claude/rules/sddp.md section 'Policy-load compatibility validation is mandatory') _[architecture]_
- crates/cobre-sddp/src/policy/policy_load.rs::rescale_cut_records_for_load legacy branch and LEGACY_COST_SCALE_FACTOR — Reserved seam, cited not raised, per the lens instruction. The None source-scale branch and the 1e6 legacy constant exist to keep pre-self-describing checkpoints loadable, and the equality fast path makes the still-default case a bit-exact no-op rather than a… (sanctioned by attacker-prompt.md lens question 4 (reserved seam) and docs/design/reserved-seams-and-deferred-debt.md) _[architecture]_
- crates/cobre-sddp/src/setup/bucket_topology.rs::build_transit_bucket_topology — Examined for the CD-074 shape and clean. There is a single derivation site, the resulting TransitBucketTopology is threaded into production rather than re-derived per call, and the only call site that passes different arguments sits inside the test-support wr… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md) _[architecture]_
- crates/cobre-sddp/src/policy/reconcile.rs::build_reconciliation_report, classify_op and FamilyTally — Result of the lens question 5 class sweep, reported as a positive because no new instance was found. Every variant of cobre_io::StateFamily -- HydroStorage, HydroInflowLag, AnticipatedThermalState, HydroTransitBucket -- has an explicit per-family tally and an… (sanctioned by .claude/rules/sddp.md section 'State pinning uses column bounds, not equality rows') _[architecture]_
- crates/cobre-sddp/src/policy/policy_load.rs::rescale_cut_records_for_load — the Legacy (None) cost-scale branch and LEGACY_COST_SCALE_FACTOR — Reserved seam, cited not raised. Its early return when the loading factor equals the legacy factor is documented as an exact bit-identical no-op and stated to be a correctness requirement rather than an optimization, so the branch that a perf sweep would be t… (sanctioned by register BACKLOG L2272; mirror docs/design/reserved-seams-and-deferred-debt.md:30 section Reserved-seam register (mirror entry owed by E11)) _[performance]_
- crates/cobre-sddp/src/stochastic/noise.rs — Examined in full and clean, and the buffer discipline here is worth protecting. Every working buffer is a ScratchBuffers field cleared and resized rather than reallocated; the copy helper clears and extends from slice so capacity is retained; the accumulator… _[performance]_
- crates/cobre-sddp/src/setup/stochastic_pipeline.rs — Examined and clean for this lens. The NCS factor entry builder does re-fetch the resolved factor table inside its innermost block loop, and two helpers deep-clone the study stage vector, but every production call site resolves to study setup, run once, with n… _[performance]_
- crates/cobre-sddp/src/setup/scenario_libraries.rs — Examined and clean. The per-stage scenario count helper rebuilds a stage id resolver on each of its class calls and the normal-model path reifies a dense per-entity-per-stage model set, but all passes are single and linear and all of it runs once at setup. _[performance]_
- crates/cobre-sddp/src/policy/policy_load.rs (validate_policy_load, compare_graph_manifest_identity, compare_manifest_slot_identity, select_boundary_pool, check_season_compatibility, check_topology_su… — Examined and clean, and the sharing is deliberate. The load path already hoists its indices: one identity index is built once and shared between the topology-subset check and the rebind, with an in-code note that the source is hashed once and not per consumer… _[performance]_
- crates/cobre-sddp/src/policy/reconcile.rs — Examined and clean. The identity index and the source interval index are built once by the caller and passed in; the rebind, the per-cut rebind, the dropped-position scan and the report build are single linear passes over aligned vectors. The one hash-map ite… _[performance]_
- crates/cobre-sddp/src/policy/policy_export.rs::build_stage_entity_manifest and crates/cobre-sddp/src/policy/orchestration.rs::write_checkpoint — Examined; a real re-derivation with no reach, so a positive rather than a candidate. The manifest builder re-derives its anticipated thermal list, its study stage list, two positional scans, the post-study delivery calendar and the extended delivery stages on… _[performance]_
- crates/cobre-sddp/src/setup/accessors.rs — Examined and clean. The setters, the terminal pool stage lookup, the terminal manifest and window builders and the graph manifest builder are one-shot; the three context builders are pure borrow-only struct literals with no allocation, which is what keeps the… _[performance]_
- crates/cobre-sddp/src/setup/bucket_topology.rs — Examined and deliberately not raised. The three sibling arc builders each independently recompute the study stage durations, the delivery stage durations, the study stage filter and the resolution extension, and the arrival density is quadratic in stages per… _[performance]_
- crates/cobre-sddp/src/setup/node_graph.rs (assemble_outcome_weights, successor_outcome_count, max_successor_outcome_count, advance_sampled_node, node_parent) — Examined and clean, and the shape is worth protecting. The outcome weight assembler clears and reserves into the caller's buffer rather than returning a fresh vector, and is documented as the single owner shared by the backward pass and the lower-bound root e… _[performance]_
- crates/cobre-sddp/src/stochastic/noise_key.rs — the total_cmp ordering discipline — Worth protecting explicitly, because it is the thing a perf rewrite of this file would break. Every comparison in the nearest-neighbour tie-break, the reversal accept test and the winning-tour test uses total_cmp, with in-code notes that partial_cmp, bare les… _[performance]_
- crates/cobre-sddp/src/stochastic/inflow_method.rs, crates/cobre-sddp/src/horizon_mode.rs, crates/cobre-sddp/src/stochastic/mod.rs, crates/cobre-sddp/src/policy/mod.rs — Examined and clean. Enums with match dispatch and module declarations; no allocation, no loops, nothing for a perf sweep to profile. _[performance]_
- crates/cobre-sddp/src/config.rs, crates/cobre-sddp/src/validate_phases.rs, crates/cobre-sddp/src/setup/params.rs, crates/cobre-sddp/src/setup/orchestration.rs, crates/cobre-sddp/src/setup/scenario_li… — Examined and clean for this lens: the config-time and report-time tier. These are default impls, plain struct declarations, a static-string mapper, a pure config projection and report builders. Every production call site resolves to study setup or to output a… _[performance]_
- crates/cobre-sddp/src/setup/mod.rs — the NCS slot-to-column bridge and the anticipated commitment history walk — Examined and clean. Both use a linear position scan inside a loop, and both carry a written rationale: the NCS bridge maps by entity id rather than by index so the map stays correct when only a subset is stochastic or the orders diverge, and the commitment wa… _[performance]_
- crates/cobre-sddp/src/policy/policy_export.rs::splice_reserved_state_block and ::reserve_boundary_inflow_lag_slots — the policy writer's second-family slot body — Reached during the sweep and deliberately not raised. The writer reserves the second boundary-state family's slot body with no reader at the pin; that is the reserved seam, not dead code, and CLAUDE.md's 'unwired config is reserved, not dead' rule governs it.… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § Reserved-seam register (mirror :871) / CD-039) _[over-engineering]_
- crates/cobre-sddp/src/policy/policy_load.rs::rescale_cut_records_for_load and ::LEGACY_COST_SCALE_FACTOR — the Legacy cost-scale load branch — Reached and not raised. The branch rescales cut records written under the legacy cost scale so an older checkpoint still loads. Recorded honestly: at this pin the sanction is the register BACKLOG entry plus the code's own rustdoc at policy_load.rs:52-60; ther… (sanctioned by plans/architecture-debt-audit/BACKLOG.md (register BACKLOG L2272) plus crates/cobre-sddp/src/policy/policy_load.rs rustdoc :52-60; mirror entry owed by E11) _[over-engineering]_
- LipschitzConfig.mode and the vertex-based UpperBoundEvaluationConfig it belongs to — cited from cobre-io, never anchored here — Loaded, validated and schema-exported by cobre-io and never read anywhere in the 5a manifest. A one-valued enum with no LP consumer is exactly the shape this lens would otherwise flag, and it is exactly what the hard rule protects: under CLAUDE.md, unwired co… (sanctioned by CLAUDE.md hard rule 'Unwired config is reserved, not dead' / docs/design/reserved-seams-and-deferred-debt.md § Reserved-seam register (mirror :54)) _[over-engineering]_
- crates/cobre-sddp/src/setup/mod.rs StudySetup.transit_bucket_topology and crates/cobre-sddp/src/setup/bucket_topology.rs::TransitBucketTopology — The stored field has no post-construction reader at the pin and carries '#[allow(dead_code)]'. It is not a candidate: the code states the condition for the allow to refire, and every field is consumed via the constructor's threaded local before the stored fie… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § Reserved-seam register (mirror :32)) _[over-engineering]_
- crates/cobre-sddp/src/horizon_mode.rs::HorizonMode — the only single-variant enum in the 5a manifest — The enum census over the manifest returns 11 enums and exactly one with a single variant, and that one is already documented as a reserved seam. It also earns its module on its own terms: three methods (is_terminal, validate, num_stages), a validate that reje… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § Post-lifecycle-walk findings (mirror :743, :749) — 'a documented reserved seam') _[over-engineering]_
- 5a enum census — 11 enums, variant counts recorded — HorizonMode 1; OpeningSource 2; Traversal 2; SimulationEnumeratedRequest 2; ProvenanceSource 3; PerStageBlockCoverageIssue 3; StochasticSource 3; PrepPhase 3; InflowNonNegativityMethod 4; RebindOp 5; ResolvedParametersError 5. Command: git grep -n -E '^\s*pub… (sanctioned by attacker-prompt.md § Lens: over-engineering (census requirement)) _[over-engineering]_
- crates/cobre-sddp/src/config.rs — Examined for one-variant enums, constructor-only structs and unconsumed knobs. LoopConfig and CutManagementConfig carry Default impls; LoopParams, EventConfig, EventParams and TrainingConfig are plain parameter carriers with real readers. Nothing survives the… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § Post-lifecycle-walk findings (mirror :749)) _[over-engineering]_
- crates/cobre-sddp/src/validate_phases.rs — the module earns its home — 88 production lines holding 'pub enum PrepPhase { Config, Stochastic, HydroModels }' and 'pub fn prep_phase_metadata', shared by the CLI and the Python binding. Its own module doc gives the structural reason it cannot move down a layer: cobre-io cannot import… (sanctioned by CD-029 doc drift is a dup-of owned by E06; module placement justified in the module's own doc) _[over-engineering]_
- crates/cobre-sddp/src/policy/policy_load.rs::PolicyLoadProof and ::PolicyLoadKind — the sealed type-state credential — A sealed trait with a phantom-typed proof struct is the classic ceremony shape, and it survives the lens on evidence. There are two real kinds (FullFcf, BoundaryInjection), validate_policy_load is the only constructor, and three crates require the proof posit… (sanctioned by two present consumers plus signature-enforced use across three crates; not a one-consumer abstraction under the layering brief's trigger 4) _[over-engineering]_
- crates/cobre-sddp/src/setup/node_graph.rs::TypedVec and ::TypedIndex — A generic newtype over Vec with a hand-rolled index trait would be a one-consumer abstraction if only one index type existed. Two do: StageIdx and NodePos both implement TypedIndex, and the type carries real Index, IndexMut, FromIterator, IntoIterator (three… (sanctioned by two present index consumers; the layering brief's trigger 4 requires the second consumer to precede the abstraction, and here it does) _[over-engineering]_
- crates/cobre-sddp/src/setup/params.rs::BoundaryStateRequirements — The counter-example worth recording in the same crate as a setter wall: private fields, two named constructors (none, present) that encode the '0 reserves none' rule at the boundary, two read accessors, and heavy cross-crate use in cobre-cli, cobre-python and… (sanctioned by examined and clean; recorded so a later station does not mistake it for the setter-wall pattern) _[over-engineering]_
- crates/cobre-sddp/src/policy/scaling_report.rs and crates/cobre-sddp/src/policy/provenance.rs — impl-less nested report trees — Nine impl-less structs across the two files, which reads as noun sprawl until the consumers are checked. Every one is a Serialize node of an artifact that ships: the scaling report is written to training/scaling_report.json by both cobre-cli/src/commands/run/… (sanctioned by examined and clean; consumers verified cross-crate at the pin) _[over-engineering]_
- The superseded cut-sync public methods — Named in the do-not-raise set and honoured. Their anchors are in crates/cobre-sddp/src/cut/cut_sync.rs, outside the 5a manifest, so they are cited here as context rather than anchored, and no candidate touches them. (sanctioned by CD-019 / mirror § Superseded cut-sync public methods) _[over-engineering]_
- crates/cobre-sddp/src/stochastic/ and crates/cobre-sddp/src/policy/ module roots — Both were checked for the empty-wrapper shape and neither is one. stochastic/mod.rs is 24 lines declaring four members with the two intra-crate ones correctly narrowed to pub(crate); policy/mod.rs is 9 lines of seven module declarations with no re-export laun… (sanctioned by examined and clean) _[over-engineering]_
- crates/cobre-sddp/tests/permute_helpers.rs (whole binary) — Examined and clean, and the opposite of harness duplication. Its module doc states the unit tests of common::permute::permute_case were deliberately relocated out of the shared tests/common aggregator 'so `mod common;` contributes no test to its consumers', w… _[test-bloat]_
- The per-file fixture_priced_date and producer_block adapters in tests/boundary_dim_mismatch_reconcile.rs, tests/boundary_reconcile_defaults.rs, tests/boundary_self_describing_clean_break.rs, tests/ri… — Read closely and deliberately NOT raised as duplication. Each one delegates to the shared surface (cobre_sddp::test_support::fixture_priced_date and ..cobre_sddp::test_support::producer_block()) and exists only to bind that file's own epoch. The shared helper… (sanctioned by crates/cobre-sddp/src/test_support.rs:788 caller-owned-epoch doc) _[test-bloat]_
- crates/cobre-sddp/tests/common/builders.rs (StageSpec/make_stage, HydroSpec/make_hydro, BusSpec/make_bus, ThermalSpec/make_thermal) — The correct destination shape and worth protecting: a Spec struct with a neutral Default plus one make_ mapper is exactly the one-place field-addition seam the fixture candidates in this envelope should route through, which is why every fix shape here extends… _[test-bloat]_
- crates/cobre-sddp/src/test_support.rs gated behind crates/cobre-sddp/src/lib.rs:46 cfg(any(test, feature = "test-support")) and Cargo.toml's test-support feature — The consolidation seam already exists and already works: 20 integration binaries reach cobre_sddp::test_support today, and the Cargo.toml comment states the intent ('plain cargo test sees them via test and downstream integration tests see them via this featur… _[test-bloat]_
- crates/cobre-sddp/src/policy/policy_export.rs::reserve_boundary_inflow_lag_slots and ::splice_reserved_state_block — Reserved seam reached while sweeping the policy manifest, examined and left alone. Not dead code and not a test-bloat subject. (sanctioned by mirror :871) _[test-bloat]_
- The post-study-commitment channel for the anticipated family reached from setup/mod.rs::resolve_anticipated_commitments_core and ::resolve_anticipated_commitments — Reserved seam, examined and left alone; the anticipated tests that exercise it are informational for this lens, not bloat. (sanctioned by mirror :871) _[test-bloat]_
- crates/cobre-sddp/tests/deterministic.rs, tests/parity.rs, tests/common/parity_hash.rs, tests/common/permute.rs, tests/scalar_parameters_declaration_order.rs and the slow-tests gated cases — Informational for this lens by directive and never counted as bloat: these are the reproducibility, declaration-order-invariance, parity-golden and rank-invariance gates plus the slow-tests gating, and section 7 of docs/design/testing-architecture.md names re… (sanctioned by inventory.json tests.informationalGates; docs/design/testing-architecture.md section 7) _[test-bloat]_

**5b**

- crates/cobre-sddp/src/lp/builder/patch.rs::fill_col_state_patches — The single owner of incoming-state pinning for all four state families, resolving every column through StateSpace::state_to_lp_incoming_column and pinning with column bounds rather than an equality row - exactly the contract .claude/rules/sddp.md section 'Sta… (sanctioned by .claude/rules/sddp.md section 'State pinning uses column bounds, not equality rows') _[architecture]_
- crates/cobre-sddp/src/lp/indexer/state_space.rs storage family (storage_incoming_col, storage_outgoing_col) — CD-074 class sweep answer for storage: no drift reconciliation exists and none is needed. The incoming storage column is written free at (-INF, +INF) by columns.rs::fill_anticipated_columns' sibling fill_storage_columns and pinned by column bounds each solve;… _[architecture]_
- crates/cobre-sddp/src/lp/indexer/state_space.rs inflow-lag family (lag_incoming_col, state_to_lp_column lag remap) — CD-074 class sweep answer for inflow lags: no drift reconciliation exists and none is needed. Lag columns are written unconstrained and signed by columns.rs::fill_ar_lag_columns, and the lag remap (lag 0 to z_inflow, lag l to the previous stage's lag l-1) cou… _[architecture]_
- crates/cobre-sddp/src/lp/builder/columns.rs::fill_transit_bucket_columns and the transit-bucket family — CD-074 class sweep answer for transit buckets: no live drift reconciliation and none needed at the baseline. Incoming buckets are free at [0, +INF) and pinned by column bounds; the identity carry has no finite cap on the state. The dormant scaling.rs::apply_b… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md; CLAUDE.md 'Unwired config is reserved, not dead') _[architecture]_
- crates/cobre-sddp/src/lp/indexer/state_space.rs::REGION_ORDER and its exhaustive dispatch (state_dim_range, incoming_block_start, classify, classify_incoming_column, set_nonzero_mask) — The single owner of the storage/lag/bucket/commitment-hold walk order, with every consumer matching exhaustively and no _ catch-all, so adding a StateRegion variant fails to compile until every site handles it. This is the pattern the two B-grade candidates a… _[architecture]_
- crates/cobre-sddp/src/lp/indexer/entity_index.rs — Names why BusSys, NcsSys and ContractSys are deliberately NOT introduced (a system-index type for them would have no call site and ship as a dead type) and pins six cross-family index swaps with compile_fail doctests. It is both a correct refusal to over-abst… _[architecture]_
- crates/cobre-sddp/src/lp/builder/delivery_ring.rs (the ring primitive itself) — A genuinely two-consumer shared primitive - the water travel-time ring and the anticipated commitment ring - so it clears the pull-don't-push bar rather than being speculative generality. Only its untyped addressing operand order is raised; the abstraction it… _[architecture]_
- crates/cobre-sddp/src/lp/builder/fpha_cursor.rs::for_each_fpha_plane — The in-crate precedent for the Sev A fix shape: one walker drives both the bounds fill and the coefficient fill, so a one-sided edit landing bounds and coefficients on different rows is impossible. The anticipated ring's three-owner residue is the same proble… _[architecture]_
- crates/cobre-sddp/src/lp/builder/storage_boundary_grid.rs, range_cursor.rs, study_dimensions.rs, hydro_cell.rs — Examined for the same three-owner residue and multi-axis index class and found clean: each is a single owner of its arithmetic (block_storage_col's three-arm formula under an exhaustive Boundary match with no _ arm, RangeCursor::alloc's pos..pos versus 0..0 r… _[architecture]_
- crates/cobre-sddp/src/lp/mod.rs, lp/builder/mod.rs, lp/indexer/mod.rs, lp/indexer/layout.rs, lp/builder/rows.rs, lp/builder/template.rs, lp/builder/generic_constraints.rs — Swept for the 5b questions and clean. builder/mod.rs' shared constants (EVAP_COLS_PER_HYDRO, EVAP_F_PLUS_OFFSET) carry the footgun rationale that justifies them rather than restating the arithmetic, which is the correct shape for a shared-constant module. _[architecture]_
- crates/cobre-sddp/src/lp/builder/patch.rs::PatchBuffer -- fill_forward_patches, fill_load_patches, fill_z_inflow_patches — Named performance target 3 is allocation-free by construction: PatchBuffer::new pre-sizes all six vectors (row capacity hydro_count + n_load_buses * max_blocks + hydro_count; column capacity hydro_count * (1 + max_par_order) + n_buckets + n_anticipated * k_ma… (sanctioned by .claude/architecture-rules.md Sec. State Struct Pattern (scratch allocated once and reused via clear()/resize()/extend(); no allocation on the hot path)) _[performance]_
- the per-opening NCS availability column patch (training/stage_solve_prep.rs::StageSolvePrep::run reaching apply_ncs_col_bounds; cited, not anchored) — The NCS patch is the one part of the per-solve column-bound submission that genuinely varies per opening, and it must stay per opening. It is deliberately excluded from the hoist proposed in the second candidate, which names it as a constraint on the fix shap… (sanctioned by .claude/rules/sddp.md Sec. Lower-bound evaluation must patch NCS) _[performance]_
- crates/cobre-sddp/src/lp/builder template-build path -- template.rs::build_single_stage_template, columns.rs, rows.rs, entries.rs::build_stage_matrix_entries, layout.rs::StageLayout, scaling.rs — Examined and clean for the per-solve lens: every allocation here is setup-time. build_stage_templates has exactly one production caller (setup/mod.rs:983) and loops the stages serially; every other call site is a #[cfg(test)] module or the test/integration wr… _[performance]_
- crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs::CutStateProjection -- field incoming_columns and ::dot_trial_state — Correct and worth protecting on both counts. incoming_columns is filled once in CutStateProjection::new and documented 'LP incoming column per cut slot (extraction hot path)' -- it is the existing sanctioned precedent the first candidate asks the pinning path… (sanctioned by CD-038 (live, keep -- loop-shape duplication, not a hazard)) _[performance]_
- crates/cobre-sddp/src/lp/indexer address and vocabulary types -- block_grid.rs::BlockGrid, range_cursor.rs::RangeCursor, study_dimensions.rs::StudyDimensions, layout.rs, storage_boundary_grid.rs, ind… — Examined and clean for this lens: all are pure index arithmetic or newtype vocabulary with no allocation and no loop of their own on a per-solve path. BlockGrid is the single owner of every production block-major stride (flat / fpha_plane / deficit) and its m… _[performance]_
- crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::BoundRelaxations, delivery_ring.rs, fpha_cursor.rs, builder/mod.rs — Examined and clean for allocation: BoundRelaxations is reused across solves as PatchBuffer::commitment_relax through clear()/push(), fill_bound_relaxations early-returns when anticipated_thermal_indices is empty, and delivery_ring.rs, fpha_cursor.rs and build… (sanctioned by CD-074 (Sev A, live -- sharpen-only; a perf claim on the same code is a different claim per the cell directive)) _[performance]_
- crates/cobre-sddp/src/lp/indexer/state_space.rs::finalize_state_column_map and ::lp_column_for_state — The outgoing direction is the shape to keep, not a smell: a pure cache of the resolver, documented as 'never a reimplementation of its arithmetic', with a debug_assert pinning the finalized length. It is the in-crate reference the first candidate asks the inc… _[performance]_
- PRECONDITION check — the four reserved seams named in the do-not-raise briefing against the 5b manifest — None of the four anchor inside lp/. LipschitzConfig.mode and UpperBoundEvaluationConfig live in cobre-io config/training.rs; splice_reserved_state_block and reserve_boundary_inflow_lag_slots live in policy/policy_export.rs; the anticipated post-study-commitme… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md:54 and :871; register BACKLOG L2272 with the mirror entry owed by E11 (mirror :30 § Reserved-seam register)) _[over-engineering]_
- Enum census over the 5b manifest — six enums, no one-variant enum — git grep -n -E '^\s*(pub |pub\(crate\) )?enum ' 077dbe2c -- crates/cobre-sddp/src/lp/ returns exactly six declarations, and reading each body gives the variant counts: builder/columns.rs::BlockSlackFamily 2 (OutflowBelow, OutflowAbove), builder/columns.rs::Ce… _[over-engineering]_
- #[allow( census over the 5b manifest — 13 files, 68 openers, zero rationale-absent production attributes — git grep -c -E '#!?\[allow\(' 077dbe2c -- crates/cobre-sddp/src/lp/ gives 68 openers across 13 files (columns.rs 13, entries.rs 20, layout.rs 9, layout/tests.rs 2, patch.rs 3, scaling.rs 8, template.rs 6, template/tests.rs 2, test_support.rs 1, generic_constr… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § #[allow(...)] census (Load-bearing, Reserved-seam (Voice 4) and Symmetry-or-test-retention classes); OD-009 i…) _[over-engineering]_
- crates/cobre-sddp/src/lp/builder/scaling.rs::apply_bucket_col_scale — The one #[allow(dead_code)] on a production function in this manifest is an explicit Voice-4 reserved seam naming its future consumer: no production call site wires it in yet, and postprocess_templates activates it once the travel-time bucket LP fill gives bu… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § #[allow(...)] census, Reserved-seam (Voice 4) class) _[over-engineering]_
- crates/cobre-sddp/src/lp/builder/template.rs::StageGeometry fields filling_target and filled_min_storage_floor — Both are #[allow(dead_code)] row ranges whose rationale names them as the per-stage row-shape carriers keeping StageGeometry a faithful mirror of the row shape, with filled_min_storage_floor mirroring the filling_target seam. This is the sanctioned Voice-4 cl… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md § #[allow(...)] census, Reserved-seam (Voice 4) class, which names lp/builder/{layout,template}.rs as its examp…) _[over-engineering]_
- crates/cobre-sddp/src/lp/indexer/entity_index.rs — the ten entity newtypes — HydroSys, HydroCell, ThermalSys, LineSys, FphaLocal, FphaCellLocal, EvapLocal, FillingTargetLocal, FloorLocal and AnticipatedLocal are ten repr(transparent) newtypes whose only methods are new and get, which is exactly the shape a speculative-generality sweep… _[over-engineering]_
- crates/cobre-sddp/src/lp/indexer/storage_boundary_grid.rs::StorageBoundaryGrid and crates/cobre-sddp/src/lp/indexer/block_grid.rs::BlockGrid — This answers the wrapper-layers-that-forward-without-deciding question for the indexer/builder seam. Three one-line forwarders exist (GenericResolverGeom::block_grid, GenericResolverGeom::block_storage_col, StageGeometry::block_storage_col) and each does forw… _[over-engineering]_
- One-consumer sweep of the remaining indexer and builder seams in 5b — Every other indirection in the manifest was consumer-counted and each has at least two, so the lens's one-consumer question yields nothing beyond the two candidates. indexer/range_cursor.rs::RangeCursor has two documented consumers (StageLayout's equipment ch… _[over-engineering]_
- Live dispositions whose anchors fall inside the 5b manifest — CD-034 (builder/entries.rs::fill_operational_violation_entries), CD-038 (indexer/cut_state_projection.rs::dot_trial_state) and CD-007 (builder/entries.rs and builder/columns.rs inline test dominance) all anchor inside this manifest and were reached during the… (sanctioned by prior-register.md live-disposition table (CD-034, CD-038, CD-007) and the CD-074 note) _[over-engineering]_
- crates/cobre-sddp/tests/template_integration.rs already implements the one-integration-binary-per-domain consolidation with #[path] submodules — L4400-4421 declare eleven #[path = "template_integration/..."] submodules, and every one of the eleven members opens with `use super::*;` so the parent owns the fixtures once. This is exactly the mechanism docs/design/testing-architecture.md section 5.1 presc… (sanctioned by docs/design/testing-architecture.md section 5.1) _[test-bloat]_
- The declaration-order and byte-identity gates inside the sweep are informational, never bloat — entries.rs::csc_byte_identical_under_permuted_declaration_order, tests/template_integration/generic_constraints.rs::declaration_order_permutation_is_invariant_in_lp and its assert_lp_byte_identical helper, and entries.rs::k1_chronological_with_travel_time_is_… _[test-bloat]_
- The slow-tests gate in tests/lp_builder.rs is a tier switch, not duplication — The five D33 and D34 non-uniform-block tests carry #[cfg_attr(not(feature = "slow-tests"), ignore = "slow: run with --features slow-tests")], which is the tiering mechanism docs/design/testing-architecture.md section 5.3 asks for: the expensive train-and-simu… (sanctioned by docs/design/testing-architecture.md section 5.3) _[test-bloat]_
- crates/cobre-sddp/src/lp/builder/test_support.rs mirrors production layout resolution instead of hand-rolling a fixture layout — state_layout_for and state_layout_with_resolution reproduce resolve_state_layout's state dimensions and PAR-derived lag counts, and ctx_anticipated_and_mask_inputs routes through the same resolve_anticipated_commitments_core and build_transit_bucket_topology… _[test-bloat]_
- commitment_reconcile.rs plus commitment_reconcile/tests.rs is the clean reference instance of the extracted-sibling convention — A 245-line implementation file whose only test declaration is `mod tests;` on its last line, with a 250-line sibling holding twelve named contract tests. Three more pairs in the same directory (layout, template, generic_constraints) follow it. The homing fix… _[test-bloat]_

**5c**

- crates/cobre-sddp/src/cut/cut_sync.rs::sync_cuts, ::pack_local_records, ::sync_packed_records (live path ::sync_level_records) — Registered and deferred dead public surface, not a finding for this station: the three legacy single-pool exchange methods have no production call site but remain re-exported public API on a published crate, so removal is a breaking change already queued to t… (sanctioned by CD-019 / mirror section Superseded cut-sync public methods) _[architecture]_
- crates/cobre-sddp/src/training/backward_pass_state.rs::resolve_backward_scheduler — Examined and correct. The by-node scheduler's exclusion under dynamic cut selection is not an accident of wiring: resolve_backward_scheduler forces the by-scenario arm whenever DCS is active, the exclusion is stated as a contract rather than left implicit, an… (sanctioned by .claude/rules/sddp.md section By-node scheduler is warm-start-only) _[architecture]_
- crates/cobre-sddp/src/training/backward/mod.rs::SuccessorSpec and ::SuccessorOutcomes — The genuinely-shared part of the retrofitted-variant question, and it is shared properly. All three backward drivers consume one reified successor contract rather than each re-deriving the successor set from the node graph, so the per-variant differences that… _[architecture]_
- crates/cobre-sddp/src/training/stage_solve_prep.rs::StageSolvePrep::run and crates/cobre-sddp/src/training/backward/lp_setup.rs::patch_opening_bounds — The state-pinning pipeline has exactly one owner and no variation point where it matters. StageSolvePrep::run performs pin, then row patches, then the NCS patch, then commit, then reconciliation, and the reconciliation step is unconditional -- it takes no hoo… (sanctioned by .claude/rules/sddp.md section State pinning uses column bounds, not equality rows) _[architecture]_
- The storage, inflow-lag and transit-bucket state families in crates/cobre-sddp/src/lp/builder/columns.rs (fill_storage_columns, fill_ar_lag_columns, fill_transit_bucket_columns) versus the anticipate… — The class sweep for the CD-074 shape finds no second instance, and the reason is structural rather than incidental. The shape requires a pinned incoming state coupled by a no-slack equality to a DIFFERENT column that carries a finite enforced bound, so that s… (sanctioned by .claude/rules/sddp.md section State pinning uses column bounds, not equality rows) _[architecture]_
- crates/cobre-sddp/src/claim_scatter.rs (module doc, ClaimCursor, canonical_scatter) — Worth protecting even though this envelope proposes extending it. The module states its consumers by name and, unusually, states its own boundary -- only genuinely-identical primitives live there, while claim-loop bodies and per-item writes stay at each calle… _[architecture]_
- crates/cobre-sddp/src/gemm.rs — The crate's unsafe_code allowance is honoured as narrowly as the rule intends: the workspace forbids unsafe by default and cobre-sddp's override exists for one call, which is isolated in this single module rather than spread across the cut-selection kernel th… (sanctioned by CLAUDE.md hard rule: cobre-sddp overrides unsafe_code for the matrixmultiply dgemm call, isolated in src/gemm.rs) _[architecture]_
- attacker-prompt performance target 1 names the block constant TRIAL_BLOCK — Drift report, not a finding: no TRIAL_BLOCK symbol exists anywhere in cobre-sddp at the pin. The constant that sizes the cut-selection gemm sweep is crates/cobre-sddp/src/cut/cut_selection.rs::M_BLOCK, and cut/dcs.rs is confirmed as the second crate::gemm cal… _[performance]_
- crates/cobre-sddp/src/training/backward/mod.rs::SuccessorEntry and ::SuccessorOutcomes — Performance target 2's reification half is examined and clean, so it gets a positives entry rather than a candidate. SuccessorEntry is documented as borrow-free precisely so it lives in a reused per-node buffer, and the per-child loop in compute_one_backward_… _[performance]_
- crates/cobre-sddp/src/convergence/risk_measure.rs::aggregate_weighted — Examined and clean. The allocation of a state-dimension coefficient vector here is not a hot-path smell: its own rustdoc marks it an allocating wrapper and directs hot paths to the _into form, the hot-path sibling aggregate_weighted_into is documented as bit-… _[performance]_
- crates/cobre-sddp/src/cut/pool.rs::coefficients_prefix and ::intercepts_prefix — Examined and clean. Both are zero-allocation reslices of the pool's existing storage, so the per-node cut-selection sweep reads the pool without copying it. The per-column maximum over all populated cuts inside the value kernel is documented selection semanti… _[performance]_
- crates/cobre-sddp/src/claim_scatter.rs::canonical_scatter and ::ClaimCursor — Examined and clean. canonical_scatter returns a lazy flat-map over the borrowed counts slice and allocates no scratch of its own, and ClaimCursor is a single relaxed atomic counter. The determinism the pair provides is what makes the borrowed-slice fix shape… _[performance]_
- crates/cobre-sddp/src/training/forward/enumerated.rs::run_enumerated_forward per-run vectors — Not raised, by disposition. The node_seq path buffer, the stage-statistics accumulator, the scenario-cost vector and the path-walk buffer are each allocated once per call over the whole local path range and every stage, so they are per-run allocations of the… (sanctioned by PD-004 deferred pending a profile) _[performance]_
- Modules read at the pin for this cell — Read in full: cut/basis_reconstruct.rs, training/visited_states.rs, training/backward/mod.rs, claim_scatter.rs. Read in the regions that carry the hot loops: cut/cut_selection.rs value kernel and column rule, gemm.rs, cut/pool.rs prefix accessors, training/se… _[performance]_
- Reserved-is-not-dead precondition, run first over the 5c manifest — No candidate below rests on 'this item is unwired'. The four reserved-seam blocks were checked against every symbol I touch: LipschitzConfig.mode plus UpperBoundEvaluationConfig (mirror :54), the policy writer's second-family reserved slot body (splice_reserv… (sanctioned by CLAUDE.md 'Unwired config is reserved, not dead'; mirror :54 and :871; register BACKLOG L2272) _[over-engineering]_
- Enum census over the 5c manifest: 10 production enums, zero one-variant enums — Command: git grep -n -E '^\s*(pub|pub\(crate\))? ?enum ' 077dbe2c -- crates/cobre-sddp/src/cut/ crates/cobre-sddp/src/training/ crates/cobre-sddp/src/solve/ crates/cobre-sddp/src/workspace/ crates/cobre-sddp/src/convergence/ crates/cobre-sddp/src/gemm.rs crat… _[over-engineering]_
- #[allow(...)] census over the 5c manifest: 245 openers across 45 files, and no candidate minted from it — Raw count, reproducible: git grep -c -E '#!?\[(cfg_attr\([^)]*, *)?allow\(' 077dbe2c -- <the eight 5c paths> | awk -F: '{s+=$NF} END{print "files="NR, "openers="s}' gives files=45 openers=245. Classified over all 58 manifest files by whether the site sits at… (sanctioned by mirror :1272 '### #[allow(...)] census'; register L139) _[over-engineering]_
- gemm.rs earns its module — 165 lines, one pub(crate) fn gemm_block, and the module doc states the reason: it is the only call site in cobre-sddp for matrixmultiply::dgemm and it isolates the single unsafe block the cut-selection kernel needs so that call site stays in safe code. CLAUDE… _[over-engineering]_
- claim_scatter.rs earns its module — 89 lines holding two primitives, ClaimCursor and canonical_scatter, with three consumers outside the file: training/backward/by_node.rs:171 and :453, training/forward/enumerated.rs:545 and :596, simulation/enumerated.rs:311 and :340. The module doc says 'the… _[over-engineering]_
- solver_stats.rs earns its module, even though its public surface is over-wide — 1428 lines, 614 before the test module, one responsibility: the SolverStatsDelta value type plus the pack/unpack codec that moves it over MPI and into Parquet rows (pack_delta_scalars, pack_scenario_stats, pack_worker_opening_stats and their inverses, delta_t… _[over-engineering]_
- convergence/ earns its module — 2107 lines across three submodules, roughly 918 before their test modules: convergence.rs owns ConvergenceMonitor, stopping_rule.rs owns MonitorState, StoppingMode, StoppingRule, StoppingRuleSet and the five public rule-name constants, risk_measure.rs owns Ri… _[over-engineering]_
- solve/partition.rs earns its 24 lines — One pub(crate) fn partition with five consumers: simulation/state.rs:50, training/backward/replicated.rs:30, training/forward_pass_state.rs:40, workspace/workspace.rs:20 and the solve/mod.rs re-export. The module doc states the reason a 24-line file exists -… _[over-engineering]_
- PhaseProfiles is a two-implementation trait, not a single-implementation seam — solve/solver_phase.rs:92 declares the trait as three associated constants; solver_phase.rs:107 implements it for HighsProfile under #[cfg(feature = "highs")] and :169 for ClpProfile under the clp feature, and Cargo.toml:21 and :28 document both gates. Two rea… _[over-engineering]_
- The two conditionally-scoped allows in this manifest are already the cleaner form OD-009 asks for — training/forward/enumerated.rs::expected_single_rank_solves carries #[cfg_attr(not(debug_assertions), allow(dead_code))] with the written reason that its only consumer is a debug_assert, so it is legitimately unused in a release build; workspace/mod.rs guards… (sanctioned by OD-009 (the cleaner form its claim names)) _[over-engineering]_
- Constructor-only structs in this manifest are pre-allocated scratch and result holders, not degenerate objects — Nine impl blocks expose only a constructor: RiskMeasureScratch::new, SolverStatsLogEntry::from_raw, LbEvalScratch::new, IterationScratch::new, TrainingResults::new, TrainingResult::new, BackwardAccumulators::new, SolverWorkspace::new and ScratchBuffers::new.… _[over-engineering]_
- The enumerated-sweep duplication has two present consumers, so extracting it is not a one-consumer abstraction — The shared sweep orchestration lives in training/forward/enumerated.rs (run_enumerated_forward, enumerated_stage_worker) and simulation/enumerated.rs (run_sweep, enumerated_sim_stage_worker), and a third module, training/backward/by_node.rs, already consumes… _[over-engineering]_
- Superseded cut-sync public methods are a sanctioned deferral, not a candidate — sync_cuts, pack_local_records and sync_packed_records on CutSyncBuffers are superseded by sync_level_records but are held until the next licensed public-API break. They are cited here as sanctioned rather than emitted as over-engineering debt, and they arrive… (sanctioned by CD-019 / mirror ':334 Superseded cut-sync public methods' (Wave 0 override, register L1995)) _[over-engineering]_
- Golden-set membership in crates/cobre-sddp/tests/parity.rs is justified case by case, and no two of the five cases cover the same LP structure — This is the answer to the golden-set question and it is a positive, not a candidate. The selection rationale is documented at the point of use: the case_dir doc in tests/common/parity_hash.rs states the five labels are deliberately non-sequential and pins eac… (sanctioned by docs/design/testing-architecture.md section 5.3 promotion rule; crates/cobre-sddp/tests/common/parity_hash.rs case_dir documentation) _[test-bloat]_
- The determinism, rank-invariance and wire gates are correctly treated as informational and are not bloat — tests/parity.rs, tests/deterministic.rs, tests/mpi_wire.rs, tests/common/permute.rs and tests/common/parity_hash.rs together are the bit-for-bit contract, so their apparent repetition is the point rather than a cost. The Cobre determinism definition is reprod… (sanctioned by plans/architecture-debt-audit/stations/sddp/inventory.json tests.informationalGates) _[test-bloat]_
- The slow-tests feature gate is the right tier switch and is explicitly not bloat — Long-running cases carry cfg_attr(not(feature = 'slow-tests'), ignore = ...), so the default cargo test run skips them and the full suite is available on demand. This matches the tier taxonomy in testing-architecture.md section 5.3 and its non-goal in section… (sanctioned by docs/design/testing-architecture.md sections 5.3 and 7) _[test-bloat]_
- tests/mpi_wire.rs delegates its setup to the shared pipeline instead of rebuilding it — Each determinism gate in that binary routes through common::fresh_setup_with rather than assembling its own study setup, and the helper's own doc names it as the pipeline shared by every mpi_wire.rs fresh_setup. This is the sharing seam the cut_basis.rs dupli… _[test-bloat]_
- forward_sampler_no_alloc.rs, forward_sampler_integration.rs, node_native_backward_gate.rs and basis_trajectory_probe.rs consume the shared builders rather than re-rolling entity fixtures — These binaries build their entities through the spec types and make functions in tests/common/builders.rs, which is why that module can carry the invariant its doc states, that a new required field on a cobre-core entity is absorbed by editing only that modul… _[test-bloat]_
- The branching and extensive-form oracle harness duplication is already on the register and is cross-referenced rather than re-raised — The registered entry records that extensive_form_oracle.rs carries a verbatim copy of the comparison harness in branching_value_oracle.rs, including the close tolerance helper and its scaffolding, and pays a second static link of the solver against the test c… (sanctioned by docs/design/reserved-seams-and-deferred-debt.md section Oracle test-harness duplication) _[test-bloat]_

**5d**

- crates/cobre-sddp/src/lead_time/mod.rs (resolve_spread, resolve_point, arrival_window, PointResolution) — The lagged-delivery family is ONE skeleton, not two. Both entry points, the spreadable travel-time quantity via resolve_spread and the point anticipated commitment via resolve_point, share a single overlap engine built on cobre_core::window_period_overlaps an… (sanctioned by .claude/rules/sddp.md section Water travel time (the water in-transit bucket ring and the anticipated-thermal ring are one lagged-delivery ring construct; diff…) _[architecture]_
- State families reachable from crates/cobre-sddp/src/simulation and crates/cobre-sddp/src/production (CD-074 class sweep answer for 5d) — No state family in the 5d manifest owns a drift reconciliation, and none pins state itself. git grep for drift_margin over crates/cobre-sddp/src returns only lp/builder/commitment_reconcile.rs and its sibling tests, so the CD-074 specimen stays entirely in 5b… (sanctioned by .claude/rules/sddp.md section State pinning uses column bounds, not equality rows) _[architecture]_
- The anticipated post-study commitment channel surviving in simulation output rows (crates/cobre-sddp/src/simulation/extraction.rs::extract_anticipated_lanes and crates/cobre-sddp/src/production/conve… — Reserved seam, cited not raised. The anticipated lane extraction and its write-record conversion are the surviving surface of the post-study commitment channel, and the delivery_date field on the emitted row is the pre-rename spelling: at this baseline no ant… (sanctioned by prior-register.md:173 reserved-seam note plus the mirror docs/design/reserved-seams-and-deferred-debt.md) _[architecture]_
- crates/cobre-sddp/src/hull (convex_hull_3d, Hyperplane3d, PlaneBuffer, the private ffi submodule) — Examined and clean. The unsafe FFI surface is fenced correctly at module granularity even though the crate-level lint is permissive: every unsafe extern C declaration is confined to the private hull/ffi.rs submodule, convex_hull_3d is the single safe entry po… _[architecture]_
- crates/cobre-sddp/src/test_support.rs and its declaration in lib.rs — Examined and clean. The 3652-line test-support builder module is gated behind #[cfg(any(test, feature = "test-support"))] in lib.rs, so it is not in the default build or the default public surface, yet remains reachable by plain cargo test and by downstream i… _[architecture]_
- crates/cobre-sddp/src/simulation/state.rs (run_worker_scenarios, SimulationState) — Examined and clean for this lens. Pre-reserves worker_costs and worker_stats, resizes raw_noise_buf and corr_scratch once per worker, and builds SimLookups once per worker with an in-code rationale that this is what eliminates the per-(scenario, stage) alloca… _[performance]_
- crates/cobre-sddp/src/simulation/pipeline.rs (solve_simulation_stage, reset_scenario_state, dispatch_scenario_result) — Correct and worth protecting. solve_simulation_stage takes and restores ws.scratch.unscaled_primal and unscaled_dual so capacity survives the call, reset_scenario_state reuses buffers through fill and copy_from_slice rather than reallocating, and dispatch_sce… _[performance]_
- crates/cobre-sddp/src/simulation/enumerated.rs (enumerated_sim_stage_worker, mark_own_paths, run_sweep) — Examined and clean. The worker_captures arena is filled once per call and drained by Option::take, with a documented rationale for moving rather than copying; mark_own_paths reuses node_seq; run_sweep clears and reuses scratch.stage_units; the worker takes an… (sanctioned by PD-001 (refuted)) _[performance]_
- crates/cobre-sddp/src/simulation/extraction.rs (extract_hydro_per_block energy-conversion reads, extract_transit_buckets, extract_anticipated_lanes) — Examined and clean. The EnergyConversionSet lookups are hoisted per hydro rather than per (hydro, block), so conversion and accumulated_productivity cost one lookup per hydro per stage. extract_transit_buckets early-returns when no bucket exists and otherwise… _[performance]_
- crates/cobre-sddp/src/simulation/extraction.rs (stage_release_rate_m3s, build_transit_seed) — Examined and deliberately not raised. The filtered scan over stage release rates looks like a repeated pass but is bounded by the trailing travel-time span: the caller skips any arc whose travel time exceeds the remaining horizon, so the scan is not proportio… _[performance]_
- crates/cobre-sddp/src/simulation/aggregation.rs — Examined and clean for this lens. Its allocations are per-run collective buffers (counts, displacements, send and receive buffers for the two allgatherv calls, resolve_weights, and the gathered collect), not per scenario or per stage. Any future claim here wo… _[performance]_
- crates/cobre-sddp/src/production/fpha_fitting/reduction.rs (reduce_planes_distance, grid_max_gh) — Correct and worth protecting, and the pattern the other fitting stages should adopt. reduce_planes_distance builds the grid once per plant and shares the borrow with grid_max_gh and every per-pair sampler, with grid_max_gh's doc stating the build-once contrac… _[performance]_
- crates/cobre-sddp/src/production/fpha_fitting/geometry.rs (ForebayTable) — Examined and clean. ForebayTable::new pre-reserves both volume and height vectors, and height, locate and locate_tailrace resolve through partition_point binary searches rather than linear scans. _[performance]_
- crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs (TailraceFamilies::from_rows, build_tailrace_families_map) — Correct and worth protecting. Both take pre-sorted rows and isolate each plant's slice in a single contiguity pass without cloning rows, with the sortedness precondition stated in the doc. This is the shape the geometry grouping raised as a candidate lacks. _[performance]_
- crates/cobre-sddp/src/production/fpha_fitting/production.rs (ProductionFunction::net_head, evaluate, evaluate_capped) — Examined and clean. The per-node kernel allocates nothing: net_head resolves the forebay height and the tailrace level through binary searches and the rest is scalar arithmetic. This scopes the fitting candidates correctly: the cost is the repeated grid walk,… _[performance]_
- crates/cobre-sddp/src/production/energy_conversion/builder.rs (build_energy_conversion_set) — Correct and worth protecting. Pre-reserves the per-hydro outer vector and each per-stage row, builds the id-to-index map with capacity, and hoists ForebayTable construction out of the stage loop so it happens once per hydro rather than once per (hydro, stage). _[performance]_
- crates/cobre-sddp/src/production/conversion.rs (with_node) — Examined and clean. Collects through into_iter().map(...), an exact-size iterator that pre-allocates precisely, and moves each record instead of cloning it. _[performance]_
- crates/cobre-sddp/src/production/hydro_models/evaporation.rs (resolve_evaporation_core) — Examined and clean apart from the shared geometry grouping raised as a candidate. The per-hydro stage coefficient and reference volume vectors are pre-reserved with the stage count before the loop fills them. _[performance]_
- crates/cobre-sddp/src/hull/mod.rs and crates/cobre-sddp/src/hull/ffi.rs — Correct and worth protecting. The owned copy of the input points and the canonical sort in and sort out are the declaration-order-invariance contract, not avoidable work; the flattened coordinate buffer and the facet vector both pre-reserve from known counts;… _[performance]_
- crates/cobre-sddp/src/lead_time/mod.rs — Examined and clean. resolve_block_factors, resolve_delivery and cumulative_stage_boundaries all pre-reserve from known counts, and every production caller sits in LP building, bucket topology, policy export or study setup, all outside the per-scenario and per… _[performance]_
- crates/cobre-sddp/src/generic_constraint_echo.rs and crates/cobre-sddp/src/fixed_delivery_echo.rs — Examined and clean. One-shot echo assembly whose allocation is the output itself, run once per study rather than per scenario or per stage. _[performance]_
- crates/cobre-sddp/src/simulation/config.rs, simulation/error.rs, simulation/mod.rs, error.rs, lib.rs, production/mod.rs, production/fpha_fitting/{error,rng,selection}.rs, production/hydro_models/{exp… — Examined and clean for this lens. These hold error enums, re-export surfaces, default constructors, the seeded RNG, selection-mode resolution and one-shot row builders; none carries per-scenario, per-stage or per-node work. The find-by-id inside the hydro mod… _[performance]_
- crates/cobre-sddp/src/test_support.rs — Examined and out of scope by construction: the file is entirely test scaffolding with no production code, so it carries no hot-path claim for this lens. _[performance]_
- crates/cobre-sddp/src/production/fpha_fitting/mod.rs::LipschitzConfig consumption seam and the UpperBoundEvaluationConfig it reserves — Reserved seam, not dead configuration: the vertex-based upper-bound-evaluation config is loaded, validated and schema-exported with no LP consumer yet, which the project's hard rule names as reserved. Examined and deliberately not raised as an over-engineerin… (sanctioned by mirror docs/design/reserved-seams-and-deferred-debt.md line 54 (reserved seam: upper-bound evaluation mode)) _[over-engineering]_
- crates/cobre-sddp/src/production/conversion.rs (delivery_date field pass-through) and crates/cobre-sddp/src/simulation/extraction.rs (the matching simulation output row) — The two 5d touchpoints of the anticipated post-study-commitment channel. The ticket's `delivery_date` anchor was renamed at the pin to reference_date / interval_start / interval_end on the codec side, and what survives in 5d is the simulation output row plus… (sanctioned by mirror docs/design/reserved-seams-and-deferred-debt.md line 871 (anticipated post-study-commitment channel)) _[over-engineering]_
- crates/cobre-sddp/src/cut (the superseded cut-sync public methods) — Settled disposition: examined and deliberately emitted as a positive rather than a redundant-surface candidate. (sanctioned by CD-019 / mirror § Superseded cut-sync public methods) _[over-engineering]_
- crates/cobre-sddp/src/error.rs::SddpError and the 13 other enums declared across the 5d manifest — Enum census over the manifest, run as the mandated probe: 14 enum declarations, zero with a single variant, smallest is two variants, SddpError itself carries 10. No enum in the manifest is a premature variant set of one, so the census yields no candidate. _[over-engineering]_
- crates/cobre-sddp/src/simulation/state.rs and crates/cobre-sddp/src/simulation/pipeline.rs (the 15 rationale-carrying `#[allow(...)]` openers of the manifest) — The `#[allow(` census over the manifest found 69 production-region openers, of which 15 carry a written rationale and 12 carry another explanatory comment. Those 27 are the mirror's sanctioned Load-bearing class and are correctly out of scope; only the 42 com… (sanctioned by mirror docs/design/reserved-seams-and-deferred-debt.md § `#[allow(...)]` census (Load-bearing class)) _[over-engineering]_
- crates/cobre-sddp/src/production/conversion.rs::IntoWriteRecord — The only trait declared in the 5d manifest, and it earns itself: 13 implementors, 13 call sites through `with_node`, correctly module-private visibility, and a written orphan-rule rationale explaining why a local trait beats `From<(S, i32)>`. Not a one-consum… _[over-engineering]_
- crates/cobre-sddp/src/simulation/state.rs::SimulationInputs (its `S: SolverInterface` and `C` type parameters) — The genericity is the crate-wide solver-backend and communicator-backend seam that is instantiated more than once, not speculative generality. Only the constructor is raised; the generic parameters are correct and worth protecting. _[over-engineering]_
- crates/cobre-sddp/src/production/fpha_fitting/reduction.rs::reduce_planes_with — A generic helper with two real present consumers, reduce_planes_angle and reduce_planes_distance, so the second consumer preceded the abstraction exactly as the layering brief requires. _[over-engineering]_
- crates/cobre-sddp/src/simulation/enumerated.rs::EnumeratedSimScratch and crates/cobre-sddp/src/simulation/extraction.rs::SimLookups — Constructor-only impls that are the sanctioned pre-allocated scratch and lookup-context pattern of .claude/architecture-rules.md, not the pass-through-constructor smell: each computes derived state rather than forwarding arguments. (sanctioned by .claude/architecture-rules.md § The Context Struct Pattern) _[over-engineering]_
- crates/cobre-sddp/src/generic_constraint_echo.rs — Earns its module: it is the single chokepoint that keeps the CLI and Python echo outputs in parity, which the project's Python-parity hard rule requires be written once. (sanctioned by mirror docs/design/reserved-seams-and-deferred-debt.md lines 743-751) _[over-engineering]_
- crates/cobre-sddp/src/fixed_delivery_echo.rs — Earns its module on the same parity-chokepoint grounds as generic_constraint_echo.rs, and its two-function split is justified by a stated unit-test seam. Examined explicitly because, unlike its sibling, it is absent from the mirror's pre-cleared list. _[over-engineering]_
- crates/cobre-sddp/src/lib.rs (the re-export surface) — Examined and clean: every re-export shim carries a comment stating why it exists, and the crate-level allows carry written rationales. No pass-through layer in the re-export surface forwards without a stated reason. _[over-engineering]_
- crates/cobre-sddp/src/test_support.rs as a live, consumed test-support surface — The module is gated '#[cfg(any(test, feature = "test-support"))]' at lib.rs:46-47 and 24 of the 40 integration binaries import it at the pin, so the mechanism section 5.2 of the testing architecture adopted on 2026-09-15 works as designed: crate-internal geom… (sanctioned by docs/design/testing-architecture.md section 5.2 and its Adoption (2026-09-15) note) _[test-bloat]_
- crates/cobre-sddp/tests/common/builders.rs Spec-plus-Default-plus-make pattern — StageSpec, HydroSpec, BusSpec and ThermalSpec each pair an exhaustive field list with a Default impl and a make_ constructor, so adding an entity field is one edit in this file instead of one per fixture. HydroPenalties carries no Default in cobre-core, which… (sanctioned by docs/design/testing-architecture.md section 3.1 item 4 and section 5.1 Layer 1) _[test-bloat]_
- Named inner-module consolidation already in use in the large exercising binaries — anticipated_core.rs holds 10 named inner modules, anticipated_scenarios.rs 5, hydro_sim.rs 8 and filling_commissioning.rs 4, each grouping a fixture family inside one integration binary rather than spawning a binary per family. That is the domain-binary shape… (sanctioned by docs/design/testing-architecture.md section 5.1 Layer 1) _[test-bloat]_
- The parity, determinism and rank-invariance gates and the slow-tests attribute tier — The 10 parity-hash golden files and the 13 files carrying slow-tests attributes are a correctness and tier-switch mechanism, not duplication, and none of them is counted as bloat anywhere in this report. Where a candidate above touches one of those files, rig… (sanctioned by plans/architecture-debt-audit/stations/sddp/inventory.json tests.informationalGates disposition and the prompt's informational declaration) _[test-bloat]_
- The 15 manifest files that carry no test module at all — hull/ffi.rs, lib.rs, production/mod.rs, simulation/mod.rs, production/energy_conversion/mod.rs and the fpha_fitting leaves error.rs, geometry.rs, grid.rs, production.rs and selection.rs declare zero #[cfg(test)], and the five sibling tests.rs files hold only… (sanctioned by measured at the baseline across the full 5d manifest) _[test-bloat]_

#### ↩︎ Cleared (dismissed or sanctioned — do not re-raise)

**Dismissed by a defender (16), each with the citation that decides it:**

- **5a-over-engineering-00 — Four production #[allow(clippy::...)] sites in the 5a manifest carry no written rationale, so the census's sanctioned-by-construction claim is false…** — contract *Backward opening order is warm-start-only* (.claude/rules/sddp.md) — the contract is the reason the current shape is correct: Dismissed on the code plus the project's own pinned scope for rationale-on-suppression. comments.md directive D4 (.claude/rules/comments.md:365-371) mandates a rationale for a CLOSED list of refactor-decision lints (too_many_arguments, too_many_lines, type_complexity, dead_code, unused_*) plus borrow-checker workarounds; numeric-cast lints are absent from it. The gate that enforces D4, scripts/ci/check-allow-rationa… Owner question carried: Owner call (prose, not code): reword the allow-census Load-bearing class at docs/design/reserved-seams-and-deferred-debt.md:1282-1288 and the scripts/README.md:36 one-liner so neither reads as an absolute every-site rat…
- **5a-over-engineering-02 — train_inner hand-mirrors the 19-field TrainingContext and the StageContext literal that accessors.rs already constructs, and the training_ctx constru…** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The filed fix does not compile, and that is why the literals exist. accessors.rs:233 and :274 declare stage_ctx(&self) -> StageContext and training_ctx(&self) -> TrainingContext with elided output lifetimes, so each returned value holds a borrow of all of *self for as long as it lives. train_inner takes &mut self (orchestration.rs:109) and needs self.warm_start_basis_cache.take() (:203) and &mut self.fcf (:208) live…
- **5a-over-engineering-03 — setup/scenario_library_set.rs is a 46-line public module of two impl-less structs whose name collides with the sibling setup/scenario_libraries.rs, w…** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): Both load-bearing premises fail against the code and the pinned rules at the baseline. (1) The impl-less-public-module shape is the documented layout, not an accident: .claude/architecture-rules.md section 'StudySetup Sub-Structs' (line 80 onward) tabulates ScenarioLibraries and PhaseLibraries by name with File = cobre-sddp/src/setup/scenario_library_set.rs and Visibility = pub, classifies them under 'New sub-struct…
- **5b-architecture-01 — Pin-round-trip exactness for the four state families is handled by three unrelated per-family mechanisms in three modules, and the commitment family'…** — a sharpening of CD-074 that did not survive — the register entry stays as recorded; contract *Delivered commitments reconcile against solver drift; exactness is unreachable* (.claude/rules/sddp.md) — the contract is the reason the current shape is correct: The delta this candidate adds over CD-074 is that the commitment family's two exactness mechanisms are mutually unaware and their coupling unstated; both halves are false at the pin. The reconciler's module doc (crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs:8-12) names `apply_commitment_hold_col_scale_unscale` by symbol and states the relationship in the contract's own terms: unscaling removes the ring ca…
- **5b-architecture-03 — transit_buckets_in is the only incoming state family whose template bounds are never written, inheriting them from buffer initialization while storag…** — contract *State pinning uses column bounds, not equality rows* (.claude/rules/sddp.md) — the contract is the reason the current shape is correct: The factual half holds (nothing in columns.rs writes state.transit_buckets_in) but the reading does not, on three counts. (1) The asymmetry is a stated rule, not an unexplained omission. The buffer default is (0.0, +INF) at columns.rs:31-32, so a family is written exactly when its open domain differs from that default: storage_in is widened to signed at columns.rs:106-108, the AR lag block at columns.rs:161-164, com…
- **5b-architecture-04 — A production gate predicate takes an unused _state: &StateSpace parameter justified by uniformity with a sibling that is #[cfg(test)]-gated** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The candidate's load-bearing premise fails at the pin: in a release build the gate family has two members, not one. anticipated_gate.rs:123-142 declares anticipated_resolution_for with NO cfg gate, re-exported for production use at indexer/mod.rs:109-111, taking state as its first parameter and reading state.anticipated_resolution.per_plant and state.anticipated_lead_stages at lines 130-133. indexer/mod.rs:44-49 dec…
- **5b-performance-02 — Every generic-constraint term allocates and immediately drops a one- or two-element heap vector during stage-template construction** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The mechanism is real but it is setup-time only, and the allocation policy on this exact path is deliberate and documented in present tense. Frequency: resolve_variable_ref has one production caller, fill_generic_constraint_entries at lp/builder/entries.rs:1363; that function is reached only from lp/builder/template.rs:419 inside build_stage_templates, whose sole production caller is build_energy_and_templates at se…
- **5b-test-bloat-05 — template_integration/generic_constraints.rs silently shadows the parent's `one_hydro_system` builder with a same-named local of a different signature…** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The shadow is real but nothing about it is silent, which is the load-bearing word in the title. The parent declaration at tests/template_integration.rs:251 takes two usize parameters (n_stages, lag_order); the local declaration at tests/template_integration/generic_constraints.rs:1433 takes four (n_blks, BlockMode, Option<GenericConstraint>, ResolvedGenericConstraintBounds). No call can resolve to the unintended bui…
- **5b-test-bloat-06 — Three separate Stage fixture surfaces with divergent defaults serve one crate, and the builder-module one is `#[cfg(test)]`-only so no integration bi…** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): Dismissed on four independent grounds, all read at the pin. (1) The count is wrong: crate::test_support exposes no Stage constructor at all. geometry_stage is a private fn at crates/cobre-sddp/src/test_support.rs:409 with exactly one caller, :554, which feeds StageLayout::new(...).geometry(BlockMode::Parallel) and hands back a StageGeometry, never a Stage; the module's only other stage builder, k_fan_stage at :1228,…
- **5c-architecture-03 — The replicated backward driver never enters the outcome_aggregation module and open-codes a second owner of the Benders intercept derivation, so the…** — contract *The cut intercept dots the trial state through the projection, never positionally* (.claude/rules/sddp.md) — the contract is the reason the current shape is correct: The named fix target does not do what the candidate says it does. solve_replicated_outcome_slice (replicated.rs:83-166) never touches a BackwardOutcome: at :137-138 and :160-161 it pushes the raw objective followed by state_duals[..n_state] into the flat out Vec<f64> whose per-outcome width is outcome_stride = 1 + n_state (:60), which is precisely the wire payload OutcomeExchangeScratch::allgather_outcomes (cut_sync…
- **5c-performance-01 — apply_column_rule walks the row-major gemm output panel column-strided twice for every trial column** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The mechanism as filed ('redundant strided pass', 'full-panel stride') does not hold at the pin, on five grounds read from the code. (1) Nothing is redundant and the interchange removes no load. The per-column nest at crates/cobre-sddp/src/cut/cut_selection.rs:392-403 issues one max-loop load per panel element (line 464) and at most one threshold load per panel element (line 478 or 486), so the whole column loop cos…
- **5c-performance-05 — enforce_basic_count_invariant recounts BASIC statuses with two full filter passes over the status vectors the reconstruction just wrote** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The mechanism rests on the premise that the reconstruction already knew the BASIC counts; at the pin it does not. reconstruct_col_statuses (basis_reconstruct.rs:209-211) is clear + extend_from_slice + resize, and reconstruct_template_row_statuses (:224-232) is the same pair; BasisStatus is a fieldless Copy enum (cobre-solver/src/basis_status.rs:16-32), so both blocks lower to a bulk move plus a fill and never inspec…
- **5d-performance-03 — FPHA fitting rebuilds the shared (V, Q) grid and re-walks the same nodes at four pipeline stages instead of building it once per plant** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): Setup scope is verified at the pin, and it decides this candidate. The only non-test entry into the fitting pipeline is `fit_fpha_planes` (production/fpha_fitting/mod.rs:137), reached through `fit_planes_for_hydro` (production/hydro_models/production.rs:414) and `fit_computed_planes_per_stage` (:508) from the `system.hydros().par_iter()` map at :161-165, i.e. study setup; nothing under `training/` or `simulation/` r… Owner question carried: Process call: the 4t sweep skips setup-time-only items, so the owner must say whether a setup-budget track exists to record the two conceded residues (per-plane grid rebuild in the secant; duplicate deviation walk under…
- **5d-performance-04 — fit_gamma_s_for_planes rebuilds the grid and recomputes the whole min-over-planes envelope once per plane** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The mechanism the candidate names is really present: at secant.rs:74 every call to representative_operating_point re-allocates both axis Vecs through build_grid, and at secant.rs:80-88 the min-over-planes fold and pf.evaluate depend only on the node and the immutable snapshot, so both are recomputed once per plane at each shared node, with the repeated kernel RawPlane::evaluate being a three-term affine dot product…
- **5d-performance-06 — Hull facet dedup uses a linear Vec::contains membership scan whose cost grows with the number of planes already kept** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): The linear membership scan is real as read, but it sits entirely in study setup and its operand set is bounded to a handful of keys, so the named mechanism has no measurable region to act on. Reachability: fit_hull_planes has exactly one non-test caller, production/fpha_fitting/mod.rs:165 inside fit_fpha_planes, reached from fit_planes_for_hydro and fit_computed_planes_per_stage (production/hydro_models/production.r…
- **5d-performance-07 — prepare_hydro_models_from_artifacts groups and per-hydro sorts the same geometry table three times in one call** — dismissed on the code at the pin (mechanism or premise fails; no seam, no contract): This is a study-setup path, not a solve path. The only production call sites of prepare_hydro_models_from_artifacts are crates/cobre-cli/src/commands/run/setup.rs:129 and crates/cobre-python/src/run.rs:981, plus the two validate-only entries crates/cobre-cli/src/commands/validate.rs:419 and crates/cobre-python/src/io.rs:297; each runs the resolver once per run per rank, ahead of any LP build, and production/hydro_mo…

**Reserved seams applied as ingest filters (never candidates, no OD id):**

- **`LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`** — mirror heading at docs/design/reserved-seams-and-deferred-debt.md:54 (register OD-00x KEEP-RESERVED); CLAUDE.md `Unwired config is reserved, not dead`.
- **Writer's reserved second-family slot body beside `splice_reserved_state_block` and the anticipated resolver-derived `delivery_date` channel** — mirror heading *Boundary state-family coupling channels are per-family bespoke* (docs/design/reserved-seams-and-deferred-debt.md:877).
- **Legacy (`None`) cost-scale branch of `rescale_cut_records_for_load` (`LEGACY_COST_SCALE_FACTOR`)** — no mirror entry at the pin (owed to the E11 mirror write-back); sanctioned by the register's 'New documented reserved seam' note (BACKLOG L2272) and the fn's rustdoc at policy/policy_load.rs:52-60.
- **`#[allow(...)]` census — Reserved-seam (Voice 4) class** — mirror heading *`#[allow(...)]` census* (docs/design/reserved-seams-and-deferred-debt.md:1314); the dead-code allows in setup/bucket_topology.rs, setup/mod.rs, lp/builder/{layout,scaling}.rs and production/fpha_fitting/ were never raised.
- **Superseded cut-sync public methods (`sync_cuts`, `pack_local_records`, `sync_packed_records`)** — mirror heading *Superseded cut-sync public methods* (docs/design/reserved-seams-and-deferred-debt.md:334; register CD-019) — E4 handoff, dup-of, no new id; 5c-test-bloat-00 names the methods only through the test fns it deletes.

**Dup-of (folded, no new number):**

- 5c-test-bloat-05 → TD-042 (5a-test-bloat-02).
- 5d-test-bloat-02 → TD-043 (5a-test-bloat-03).
- 5d-test-bloat-03 → TD-044 (5a-test-bloat-04).
- 5d-test-bloat-00 → TD-045 (5a-test-bloat-05).
- 5a-test-bloat-00, 5b-test-bloat-00, 5b-test-bloat-01, 5c-test-bloat-04, 5d-test-bloat-05 → CD-007.
- 5c-over-engineering-00 → OD-009.

**Closed items screened:** the six closed items of the prior register (the two retire-with-commit rows and the retracted, refuted and deferred items on the do-not-touch list) were screened by id and symbol at ingest; no candidate restated any of them (ingest-log.md, re-raise rejections = 0).

#### Queued out

- **Performance → perf-queue.json** (9 Sev-A/B PD entries + PD-004 existence-only): PD-036 (`4t`, requires none, backward_cut_levels), PD-038 (`4t`, requires none, fill_col_state_patches), PD-039 (`4t`, requires none, fill_col_state_patches), PD-040 (`4t`, requires none, select_for_stage), PD-041 (`4t`, requires none, run_cut_management), PD-042 (`4t`, requires none, run_one_backward_level), PD-044 (`4t`, requires none, process_stage_backward), PD-045 (`4t`, requires none, build_row_lower_unscaled), PD-047 (`4t`, requires enumerated, re_expand); PD-004 (`run_enumerated_backward`, exists at the baseline, `2t` enumerated deck, no fix — do-not-touch pending a profile). All UNMEASURED; no timing figure is produced by this epic. Named-target coverage: the cut-selection value sweep and the PatchBuffer fill family carry queued entries; the backward gather carries PD-042, PD-044 while the SuccessorOutcomes reification itself raised no candidate (every performance cell filed it as a positive — the CD-022 dedup owns it); basis reconstruct carries PD-043 at Sev C (not queued) with 5c-performance-05 dismissed. Sev-C PD entries stay on this section only.
- **Test bloat → td-queue.json** (16 TD entries): TD-041, TD-042, TD-043, TD-044, TD-045, TD-046, TD-047, TD-048, TD-049, TD-050, TD-051, TD-052, TD-053, TD-054, TD-055, TD-056; the five CD-007 homing candidates (5a-test-bloat-00, 5b-test-bloat-00, 5b-test-bloat-01, 5c-test-bloat-04, 5d-test-bloat-05) ride the CD-007 row. The determinism and rank-invariance gates (`csc_byte_identical_under_permuted_declaration_order`, tests/common/permute.rs, `opening_order_determinism`, the parity_hash_* goldens) are recorded as informational — a future bit-for-bit phase gate reuses them.
- **Alignment (E9):** I.3-7 — CD-004 (reused) + CD-082; I.3-8 — TD-045; every `advances-*` row above is provisional. The CD-024-successor `conflicts` hold is asked ALONE at the owner gate with an override option.
- **cli+python (E06):** the two cobre-cli contexts cited by 5a-architecture-01/02 (commands/broadcast.rs `BroadcastConfig`, commands/validate.rs `reconcile_boundary`) — CD-002 / CD-009 owners; the CD-004 wire twin is that station's half.
- **test-corpus (E08):** the StubComm / Rank0Of2 destination (TD-053, TD-056) is the E08-2 needs-human; the corpus-wide `case_dir` spread behind TD-050.
- **Reconciliation (E11):** the Legacy cost-scale seam has no mirror entry at the pin (owed to the mirror write-back); OD-040's residue is a cobre-io input-validation gap to re-file against the owning crate; the public-API removals (OD-037, OD-038, CD-083) follow the CD-019 batching precedent.

#### Owner gate — decisions

_(pending — filled by the gate ticket)_

#### Findings by sub-station

One report block per sub-station, each lens either in the register entry template or with an explicit no-finding note. Reused prior ids are not repeated here — they live in the disposition table above.

#### 5a — setup + policy + stochastic + config

**Architecture** — 2 minted; 2 sharpening a live id (see the disposition table)

**CD-082 · Sev B · bad-abstraction · effort M · confidence high**
Narrowed to a missed error class, not a corrupted report: because StudySetup::new and new_with_boundary_requirements take no artifact argument, they cannot satisfy the patch protocol StudyParams::scalar_parameters' own rustdoc mandates, so the two validate paths run build_resolved_parameters on an empty table whenever config.policy.boundary is configured, and validate's documented exit-0 contract is therefore broken for exactly the three ResolvedParametersError variants cobre_io::validation::scalar_parameters does not duplicate (MissingSeason, PerStageBlockCoverage, MissingSpecificProductivity), with is_block_varying returning false on a miss collapsing a PerStageBlock row set silently and with no assert. Conceded from the candidate: no boundary-reconciliation fact validate reports reads ResolvedParameters, so nothing validate prints is wrong; the setup is built and discarded; and the validate.rs:234 mirrors-the-run-path comment concerns boundary-requirement ordering, not the paramete…

- **Station:** cobre-sddp (sddp, sub-station 5a, lens architecture; ingest ref 5a-architecture-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/setup/params.rs::StudyParams`, `crates/cobre-sddp/src/setup/mod.rs::new_with_boundary_requirements`, `crates/cobre-sddp/src/policy/resolved_parameters.rs::ResolvedParameters`
- **Evidence:** The scalar-parameter table is a two-phase init with no type-level enforcement of phase 2. from_config writes Vec::new() and its own comment delegates the patch to 'each setup caller'; new_with_boundary_requirements patches params.boundary and NOTHING else before delegating to from_broadcast_params, and new delegates straight to it. So the only two constructors that take (system, config, stochastic, hydro_models) build the whole StudySetup -- including stage templates -- against an EMPTY table. The consequence is n… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/setup/params.rs | sed -n '174,177p;335,341p'; git show 077dbe2c:crates/cobre-sddp/src/setup/mod.rs | sed -n '342,397p';…`
- **Fix-shape:** Make the table a constructor input instead of a post-construction patch, so the empty case is unrepresentable rather than merely undesirable. Either take the loaded Vec<ScalarParameter> as a parameter of new / new_with_boundary_requirements and drop the field's placeholder role in from_config, or split the type so a StudySetup can only be built from a params value whose scalar table has been supplied (an unpatched value is not accepted by from_broadcast_params). Independently of which, ResolvedParameters::get must stop returning 0.0 on a miss in release: a miss is a build-time defect, not a datum, so it belongs as a construction-time error at the LP-build site rather than a silent sentinel behind a debug_assert -- this is the fail-loud shape already on the open follow-up list. Fixing get alone is not sufficient and fixing the constructors alone is not sufficient: the first without the second turns a silent-wrong validate into an abort, the second without the first leaves the sentinel live for any future caller. Also correct the claim at validate.rs:234 that the call mirrors the run…
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part V §V.1 Phase 0 (the study-block admission-gate carrier) / Part IV §IV.4)
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** a caller-patched placeholder standing in for a constructor input, so the two validate paths build templates against an empty parameter table and validate's exit-0 contract is broken for three error variants; a real smell confined to the validate paths and the placeholder protocol — Sev B, CD-004-adjacent (the fix rides the single admission-gate carrier).
- **Part-I:** I.3-7 (cross-reference; the verdict travels to Epic 9 with partI-handoff.json).
- **Related prior entries:** CD-004, CD-024-successor (distinct claims; adjudicated at ingest).
- **Needs-human (owner gate):** Whether the fail-loud leg changes the public surface of ResolvedParameters::get (a fallible signature or a panic) or is instead enforced as a construction-time check at the LP-build site, leaving get infallible: get is pub and must_use, so the choice is an owner call on public API shape, not a correctness question.
- **Queued to:** alignment

**CD-083 · Sev C · bad-abstraction · effort S · confidence high**
`CutManagementConfig::warm_start_cuts` (config.rs:148) is the struct's only field with no production reader -- its four siblings are read at training/backward_pass_state.rs:164-165 and training/session/mod.rs:262/1042/1215 -- and its rustdoc contract 'contributes to cut-pool capacity' (config.rs:147) is false at the baseline because capacity is derived solely from the per-pool counts consumed by `pool_capacity` (cut/fcf.rs:498) and `CutPool::new_with_warm_start` (cut/pool.rs:893-895). Conceded from the title's framing: no wrong-shape defect is established for this field (that argument belongs to the distinct cobre-io manifest symbol `ProducerBlock::warm_start_cuts`), and the residue does not decide delete-versus-give-it-a-reader, because the field is `pub` on a re-exported type so removal is a breaking API change.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens architecture; ingest ref 5a-architecture-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/config.rs::CutManagementConfig`, `crates/cobre-sddp/src/setup/orchestration.rs::train_inner`
- **Evidence:** The field states a contract its own crate does not honour. Both production construction sites write the literal 0 -- setup/mod.rs:789 and setup/orchestration.rs:136 -- and no production line anywhere under crates/cobre-sddp/src READS cut_management.warm_start_cuts; the only reads are the assertion in config.rs:304 and the test literals. The real owner of the quantity the doc describes is pool_capacity's warm_start_count parameter in cut/fcf.rs, which is per-pool, and the manifest side is per-pool too: policy/orche… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/config.rs | sed -n '136,165p'; git show 077dbe2c:crates/cobre-sddp/src/cut/fcf.rs | sed -n '490,500p'; git grep -n 'war…`
- **Fix-shape:** Delete CutManagementConfig::warm_start_cuts and let cut-pool capacity keep its single owner, the per-pool warm_start_count argument of cut/fcf.rs::pool_capacity. This is a breaking change to a pub struct, so it lands with the projection step of the sibling CD-005 candidate rather than alone: the new CutManagement params projection simply omits the field, which removes the two production literals and the struct-literal reset inside train_inner at the same time. The many test literals that set it are mechanical deletions with no assertion depending on the value, except config.rs's own accessibility test, which goes away with the field. If instead the owner wants the field kept as a reserved seam, it needs the opposite treatment: give it a real reader and reshape it per-pool to match warm_start_counts, and record it in docs/design/reserved-seams-and-deferred-debt.md so a later sweep does not re-raise it -- an unread scalar whose doc asserts a capacity contribution it does not make is the one state that should not persist. Byte-neutrality bar: the value is 0 at both production sites tod…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** one inert `pub` config field with a false rustdoc contract and no production reader; public-API removal or a reserved-seams entry — the delete-or-register disposition (OD-032 precedent) at cosmetic blast radius.
- **Needs-human (owner gate):** Owner must choose between removing `CutManagementConfig::warm_start_cuts` (a breaking change to a re-exported public type) and giving it a real reader plus a reserved-seams entry; the verdict establishes the field is inert and its doc line false, not which of the two lands.

**Performance** — 3 minted

**PD-035 · Sev C · duplication · effort S · confidence high**
Only the symmetric-matrix half of the title survives as a bit-exact free win: `l2_distance_matrix` (noise_key.rs:223-243) evaluates every off-diagonal pair's L2 twice and computes the zero diagonal, and mirroring plus a skipped diagonal leaves every entry bit-identical. The whole-tour `tour_cost` recomputation in `two_opt_improve` (174-191) is real but is demoted from removable redundancy to a semantics-changing perf-sweep item, because the full sum is what keeps the accept test at 182 and the cross-start winner at 212 functions of the tours and the matrix alone, and because a path reversal changes only ONE edge at `j == n_o - 1`. The per-start `nearest_neighbor_tour` buffer allocations (141-142) do not survive at all: they are once-per-study-setup work outside the hot-path allocation ban.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens performance; ingest ref 5a-performance-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/stochastic/noise_key.rs::two_opt_improve`, `crates/cobre-sddp/src/stochastic/noise_key.rs::tour_cost`, `crates/cobre-sddp/src/stochastic/noise_key.rs::shortest_chain_path`, `crates/cobre-sddp/src/stochastic/noise_key.rs::l2_distance_matrix`, `crates/cobre-sddp/src/stochastic/noise_key.rs::nearest_neighbor_tour`, `crates/cobre-sddp/src/stochastic/noise_key.rs::apply_chain_order`
- **Evidence:** `two_opt_improve` calls `tour_cost`, a pass over the whole tour, once per candidate segment reversal, and the reversal loop is the quadratic (i, j) sweep repeated under `while improved`. `shortest_chain_path` runs that whole improvement for each of n_o nearest-neighbour starts, and `apply_chain_order` calls it once per study stage carrying at least three openings. A path 2-opt reversal changes exactly two edges, so the accept/reject test needs two matrix reads rather than a whole-tour sum. `l2_distance_matrix` wri… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/stochastic/noise_key.rs | sed -n '132,146p;174,222p;223,266p'; git grep -n 'build_noise_key_table' 077dbe2c -- crates/c…`
- **Fix-shape:** Two halves with different risk, and they should be queued separately. The byte-exact half: fill the distance matrix symmetrically by writing the mirrored entry instead of recomputing it and skip the zero diagonal, since `(a - b) * (a - b)` equals `(b - a) * (b - a)` exactly in IEEE arithmetic and the per-pair summation order is unchanged, so the matrix is bit-identical; and hoist the nearest-neighbour tour and visited buffers out of the per-start loop, reusing them by clear-and-refill, which changes no value at all. Both leave the winning permutation identical, so the parity goldens in tests/parity.rs and tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and `mpiexec` at one rank versus two ranks all reproduce unchanged and no golden moves. The half that does move goldens: replacing the whole-tour recomputation with the two-edge delta a path reversal actually implies changes the floating-point basis of the accept/reject comparison, can flip a marginal reversal, and so can change the winning permutation. Per .claude/rules/sddp.md section Bac…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Backward opening order is warm-start-only* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** the surviving residue is the bit-exact symmetric-matrix half only, on a once-per-study opening-order chain builder (setup-time by project rule); the tour-cost half moves goldens and is demoted to an owner call — structural calibration only, UNMEASURED.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: redundant pass — a whole-tour cost sum recomputed per candidate reversal where only two edges change, a symmetric distance matrix computed twice per pair, and a per-start buffer allocation; exercising call sites (cited, not anchored): stochastic/noise_key.rs · shortest_chain_path → nearest_neighbor_tour / two_opt_improve (once per study at setup). Sev C — not queued; the owner gate may promote it; no number is asserted here.
- **Needs-human (owner gate):** Owner call: whether the goldens-moving delta reformulation of the 2-opt accept test is worth a parity re-baseline at all, given that the confirmed free win is only the symmetric matrix fill; if not, the perf sweep should carry the symmetric-fill item alone.

**PD-036 · Sev B · duplication · effort S · confidence high**
At `NodeGraph::backward_cut_levels` and its two per-iteration callers only (`run_sampled_backward`, `run_enumerated_backward`), the graph-invariant level partition is re-derived by evaluating the cut-generating predicate once per (stage, node) pair instead of in one bucket pass, and re-collected into a fresh outer `Vec` plus one fresh `Vec` per non-empty level, on every backward pass; `build_node_graph` is a fix site rather than a defect site, and the buffer is distinct in SUBJECT (not in named driver) from the `stage_stats` telemetry pack that the register's deferred backward-scratch item already covers for BOTH drivers.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens performance; ingest ref 5a-performance-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/setup/node_graph.rs::backward_cut_levels`, `crates/cobre-sddp/src/setup/node_graph.rs::NodeGraph`, `crates/cobre-sddp/src/setup/node_graph.rs::build_node_graph`
- **Evidence:** The body walks the whole `nodes` array once to find the maximum cut-generating stage, then walks it again in full for every stage from that maximum down to zero, allocating one `Vec<NodePos>` per non-empty level plus the outer `Vec`. Both production call sites are per-iteration backward drivers: `run_sampled_backward` (the enclosing function at the first hit) and `run_enumerated_backward` (the enclosing function at the second). `NodeGraph` is immutable study-level data reached through `training_ctx.node_graph`, th… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/setup/node_graph.rs | sed -n '774,798p'; git grep -n 'backward_cut_levels()' 077dbe2c -- crates/cobre-sddp/src/training`
- **Fix-shape:** Resolve the level decomposition once where the graph is constructed, in `build_node_graph`, and store it as a `NodeGraph` field the drivers borrow, so each backward pass reads a slice of slices instead of rescanning and reallocating. The crate already has this precedent: the enumerated plan and the nested upper-bound topology are documented as resolved once at construction rather than every training iteration, and `NodeGraph` is exactly the study-level read-only home the architecture rules name for this data. Byte-neutrality: the stored decomposition must be built by a single stable bucket pass that reproduces both orders the current code produces, the outer list descending by stage and each inner list ascending by canonical node position, because the within-level node order is the order the risk aggregation consumes and the index-order tie-break that `assemble_outcome_weights` documents for the nested risk measure depends on it. With both orders preserved the levels are element-for-element what they are today, the level driver's one state exchange and one batched cut exchange per l…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals; NodeGraph is the engine's traversal structure))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Per-level exchange in the backward pass* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** a study-invariant level partition re-derived by whole-graph predicate scans and freshly allocated on every backward pass at two per-iteration callers — the never-allocate-on-hot-paths rule with a one-field precomputation fix; structural calibration only, UNMEASURED.
- **Re-raise-of:** module-map; graph-shape-predicate-grep; NOT a re-raise — distinct claim anchored in a file a file-scoped retired item also cites; NOT a re-raise
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: redundant pass plus allocation per call — a study-invariant level decomposition recomputed by repeated whole-graph scans and freshly allocated on every backward pass; exercising call sites (cited, not anchored): training/backward_pass_state.rs · run_sampled_backward (per iteration); training/backward_pass_state.rs · run_enumerated_backward (per iteration). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Needs-human (owner gate):** Placement is an owner call the architecture rules do not disambiguate: a derived study-invariant consumed only by the backward drivers fits decision-tree rule 2 (a `NodeGraph` field beside `n_pools`/`pool_stage`, the candidate's shape) and rule 5 (a `BackwardPassState` scratch field) equally; the training owner should also decide whether this lands together with the register's open one-owner residue on the interior-…
- **Queued to:** performance-sweep

**PD-037 · Sev C · duplication · effort S · confidence high**
Only the two enumerated per-stage driver loops (training/forward/enumerated.rs:536, simulation/enumerated.rs:302) pay a whole-node-array pass per stage, and only on a declared multi-node graph; the frontier_node / any_stage_node family is a bounded per-pass root-or-terminal lookup with no per-stage amplification (the solve_simulation_stage site is short-circuited behind is_terminal, so it fires once per scenario), the synthesized chain reduces every frontier to a singleton over one node per stage, the pass is amortized against at least one LP solve per stage into a reused stage_units buffer, and any membership index must stay a total function yielding empty for a stage the graph declares no node at, because that emptiness feeds the carries-no-alive-node validation errors.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens performance; ingest ref 5a-performance-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/setup/node_graph.rs::stage_frontier`, `crates/cobre-sddp/src/setup/node_graph.rs::frontier_node`, `crates/cobre-sddp/src/setup/node_graph.rs::any_stage_node`, `crates/cobre-sddp/src/setup/node_graph.rs::build_declared_node_graph`
- **Evidence:** `stage_frontier` filters the whole `nodes` array on every call, so resolving one stage's frontier costs a pass over every node in the graph. Both enumerated drivers call it inside a loop over every stage, so one sweep costs a number of node visits equal to stages times nodes, and the enumerated forward sweep runs once per training iteration while the simulation sweep runs per simulation pass. `frontier_node` and `any_stage_node` are built on the same scan and are the resolution route used by the simulation pipelin… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/setup/node_graph.rs | sed -n '1120,1125p;1133,1135p;1149,1151p'; git grep -n 'stage_frontier(' 077dbe2c -- crates/cobre…`
- **Fix-shape:** Give `NodeGraph` a per-stage membership index built once in `build_node_graph`: an offsets array plus a positions array, the same compressed shape the graph already uses for out-edges, so `stage_frontier` returns an iterator over one contiguous slice and `frontier_node` and `any_stage_node` read that slice's head. The per-stage loop in each enumerated driver then visits only the nodes at that stage. Byte-neutrality: because canonical position is ascending declared node id and not stage-grouped, a stage's members are not contiguous in position space, so the index must be filled by a stable pass over ascending canonical position; each stage's slice then yields exactly the sequence the current filter yields. Same nodes in the same order means the per-stage unit list each driver builds is filled identically, and the claim cursor over it therefore hands the same units to the same workers, so the parity goldens in tests/parity.rs with tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and `mpiexec` at one rank versus two ranks all reproduce bit-for-…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Backward opening order is warm-start-only* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** narrowed to the two enumerated per-stage driver loops on a declared multi-node graph: one whole-node-array pass per stage per sweep, amortised and bounded — not a per-solve cost; structural calibration only, UNMEASURED.
- **Reviewer rating:** B — recalibrated to C because the residue is one scan per stage per enumerated sweep on declared multi-node graphs only; the frontier_node / any_stage_node family has no per-stage amplification and the chain case is a singleton — bounded, not hot.
- **Re-raise-of:** module-map; graph-shape-predicate-grep; NOT a re-raise — distinct claim anchored in a file a file-scoped retired item also cites; NOT a re-raise
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires enumerated; mechanism: redundant pass — a whole-graph node scan per stage to select that stage's nodes, driven once per stage per sweep; exercising call sites (cited, not anchored): training/forward/enumerated.rs · run_enumerated_forward (per stage per sweep); simulation/enumerated.rs · run_sweep (per stage per sweep). Sev C — not queued; the owner gate may promote it; no number is asserted here.

**Over-engineering** — 1 minted

**OD-036 · Sev C · speculative-generality · effort S · confidence high**
The defect is confined to public-surface and naming hygiene on one read-only accessor: `StudySetup::ncs_stochastic_dormant_for_test` (setup/accessors.rs:218) is the workspace's only ungated `pub fn *for_test`, has no production caller, and has a single caller at tests/deterministic.rs:2014. The sibling-parity framing over-reaches and is withdrawn: the four gated hooks in the same file guard config mutation or private-field-layout exposure, which a `&self` method returning owned booleans cannot breach. The candidate's flagged trade-off is also narrower than filed: gating is CI-coverage-neutral because `test-support` is already in NON_SOLVER_FEATURES on both test jobs, so the only coverage that moves is a bare default-feature local `cargo test`, and the D18 rustdoc sentence promising an unconditional default-profile run would have to be corrected with it.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens over-engineering; ingest ref 5a-over-engineering-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/setup/accessors.rs::ncs_stochastic_dormant_for_test`
- **Evidence:** The method allocates a Vec<Vec<bool>> reconstruction of dormancy state that no production path reads, and its own name carries the _for_test suffix, so it is a test seam published to every downstream consumer of the crate. The same file already applies the correct gate four times, which is what makes this a miss rather than a design choice: the pattern exists and one member of the family does not follow it. Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/setup/accessors.rs (read in full, 370 lines) ; git grep -n -w ncs_stochastic_dormant_for_test 077dbe2c -- crates ; git…`
- **Fix-shape:** Either move the method under the same '#[cfg(any(test, feature = "test-support"))]' as its four siblings and declare test-support as a required feature of the one consuming test, or, if the published surface is intended, drop the _for_test suffix and document it as a supported query. Prefer the gate. Note the real trade-off the gate carries: the D18 dormancy test runs today under a plain 'cargo test --workspace', so gating it moves that coverage behind a feature flag and the station should say so rather than let it silently drop. No production caller exists either way, so the solve paths are untouched; parity goldens and the rank-invariance harness bound the change.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** a `pub fn *_for_test` shipped ungated on the public surface with one test caller; naming and surface hygiene on one read-only accessor.

**Test bloat** — 6 minted

**TD-041 · Sev C · duplication · effort S · confidence high**
Two of the three named helpers are a genuine triplication, not all three: the hydro(id, downstream_id, travel_time_hours) constructor and the 16-field all-zero HydroPenalties builder have bodies identical up to a binding name and a callee prefix at bucket_topology.rs:417/442, setup/mod.rs:2981/3002 and setup/tests.rs:9957/9982, and the crate's shared surface carries no reusable equivalent because geometry_hydro pins different bounds, so one pub(crate) pair added to test_support.rs beside ymd removes six declarations, not nine. The date() leg does not survive as duplication: mod.rs:2977 uses unwrap_or_else with unreachable! against .unwrap() at the other two sites, so the trio is not byte-identical, and test_support::ymd already exists, making that leg three call-site rewrites onto an existing helper.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens test-bloat; ingest ref 5a-test-bloat-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/setup/bucket_topology.rs::hydro`, `crates/cobre-sddp/src/setup/mod.rs::hydro`, `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_hydro`, `crates/cobre-sddp/src/setup/bucket_topology.rs::zero_penalties`, `crates/cobre-sddp/src/setup/mod.rs::zero_penalties`, `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_zero_penalties`, `crates/cobre-sddp/src/setup/mod.rs::date`, `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_date`
- **Evidence:** `fn hydro(id, downstream_id, travel_time_hours) -> Hydro` has exactly three declarations in the entire crate and all three are inside this manifest, two of them inside the same file pair (bucket_topology.rs:442, setup/mod.rs:3002, setup/tests.rs:9982). `zero_penalties() -> HydroPenalties` (17 fields, every one 0.0) has 9 declarations crate-wide, 3 of them here at bucket_topology.rs:417, setup/mod.rs:2981 and setup/tests.rs:9957. `date(y,m,d) -> NaiveDate` is a 3-line wrapper with 5 declarations crate-wide, 3 of th… Re-derive: `git grep -n 'fn zero_penalties|fn hydro\(|fn date\(' 077dbe2c -- crates/cobre-sddp/src/setup/; for spec in 'setup/bucket_topology.rs 442' 'setup/mod.rs 3002' '…`
- **Fix-shape:** Hoist one transit-seed fixture trio onto the existing shared surface: add a pub neutral-penalties constructor and a pub travel-time hydro constructor to crates/cobre-sddp/src/test_support.rs beside the ymd date helper already there, under the module's existing cfg(any(test, feature = test-support)) gate, then delete the nine local declarations and call the shared ones. This is L3-internal, adds no crate and no dependency, and per the directive introduces no new fixture crate. Byte-neutral by construction: the shared constructors return the same field values the three local bodies already return, so every fixture that feeds a solve resolves to an identical System, leaving the reproducibility and order-invariance harness, the parity goldens and the rank-invariance map unchanged with no golden moved. Touches no pinned contract in .claude/rules/sddp.md.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** two small fixture helpers declared three times inside one directory; one `pub(crate)` pair on the existing test_support surface removes six declarations — TD-035-scale, not the TD-037 ripple.
- **Reviewer rating:** B — recalibrated to C because two helpers × three copies inside setup/, fix is one pair on test_support.rs (effort S); the crate-wide penalties ripple is the sibling entry.
- **Yardstick:** docs/design/testing-architecture.md §5.2 uniform test-support feature convention (shared fixtures live on the crate's own test-support surface, never a new fixture crate) (td-queue.json).
- **Queued to:** test-corpus

**TD-042 · Sev B · duplication · effort M · confidence high**
cobre-sddp's fixtures never adopt the spread base cobre-core already exports to them: 56 zero-argument `-> HydroPenalties` helpers under 11 names re-implement `HydroPenalties::uniform`, nine of them byte-identical `default_hydro_penalties` copies inside src/setup/tests.rs alone, and no HydroPenalties literal in the crate carries a `..` base, even though `uniform` is pub behind the `cobre-core/test-support` feature the crate's dev-dependencies already enable and whose sibling `declare_mirror_unit_group` those same 35 files already call. The residue is call-site non-adoption of an existing L1 constructor, not a missing one-place constructor, not the absent Default derive, and not the tests/common/builders.rs invariant being defeated; the fix spreads `..HydroPenalties::uniform(v)` at the call sites and adds no constructor to cobre-sddp's test_support.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens test-bloat; ingest ref 5a-test-bloat-02; folds 5c-test-bloat-05)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/setup/tests.rs::minimal_system`, `crates/cobre-sddp/src/setup/tests.rs::bucket_seed_zero_penalties`, `crates/cobre-sddp/src/setup/mod.rs::zero_penalties`, `crates/cobre-sddp/src/setup/bucket_topology.rs::zero_penalties`, `crates/cobre-sddp/tests/common/builders.rs::neutral_hydro_penalties`, `crates/cobre-sddp/tests/common/builders.rs::make_hydro`, `crates/cobre-sddp/tests/common/builders.rs::HydroSpec`
- **Evidence:** 276 HydroPenalties literal sites across 64 files inside cobre-sddp at the pin. HydroPenalties derives no Default (crates/cobre-core/src/entities/hydro.rs:56 derives only Debug, Clone, Copy, PartialEq), and setup/tests.rs contains zero `..Default::default()` spreads, so each of its 46 sites hand-writes all 17 fields; that is roughly 780 lines of field enumeration in one file. tests/common/builders.rs:190 neutral_hydro_penalties() already is the one place the module doc promises, but it is private and reachable only… Re-derive: `git grep -c 'HydroPenalties {' 077dbe2c -- crates/cobre-sddp | sort -t: -k3 -nr | head -6; git grep -c '\.\.Default::default\(\)' 077dbe2c -- crates/cobre-sddp…`
- **Fix-shape:** Promote the neutral penalties constructor from tests/common/builders.rs to a pub fn on crates/cobre-sddp/src/test_support.rs under the module's existing cfg(any(test, feature = test-support)) gate, so both the src-side tests and the integration binaries reach one declaration, then rewrite the literals as struct-update expressions that name only the fields a given test actually varies. Deliberately do NOT add a Default derive to the cobre-core HydroPenalties entity: a Default there would let an omitted field default silently to zero in production construction and deserialization paths, turning a test-ergonomics cleanup into a correctness regression, so the neutral constructor stays behind the test gate. No new fixture crate, per the directive. Byte-neutral by construction: each rewritten site resolves to the identical 17 field values it spells out today, so the reproducibility and order-invariance harness, the parity goldens and the rank-invariance map are unaffected and no golden moves.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** 56 zero-argument `-> HydroPenalties` helpers under 11 names re-implement the spread base cobre-core already exports behind the enabled `test-support` feature, and no literal in the crate carries a `..` base — every new penalty field costs O(copies) edits, the ripple the yardstick's shared-fixture rule exists to prevent (TD-037 precedent, Sev B); folds 5c-test-bloat-05.
- **Yardstick:** docs/design/testing-architecture.md §5.2 uniform test-support feature convention (one fixture owner per type; struct-update spreads over an exported base) (td-queue.json).
- **Needs-human (owner gate):** Whether the corpus should keep at least one literal that still spells all sixteen penalty fields as a canary: spreading `..HydroPenalties::uniform(v)` everywhere means a newly added penalty field silently defaults in every fixture instead of breaking the build, and while `uniform`'s rustdoc endorses the spread and builders.rs places its strictness on the make_<entity> literal, no artifact rules on the crate-wide los…
- **Queued to:** test-corpus

**TD-043 · Sev B · duplication · effort M · confidence high**
penalties() is md5-identical across all four right_boundary_* binaries and bounds() is md5-identical in three with validation's a strict one-parameter superset, and pricing/validation share five ring-prelude functions that are code-identical modulo path qualification and one BETA-const-vs-beta-parameter split. Narrowed on four counts: study_start() is stripped as a 3-line per-fixture calendar anchor whose four calendars genuinely diverge downstream and for which test_support::ymd already exists; the destination narrows from tests/common/builders.rs to the cfg-gated src/test_support.rs that testing-architecture.md section 5.2's Adoption stamp designates, since tests/common/ is itself slated to collapse into that surface; any hoist must carry the union of pricing's three prelude rationale clauses that validation lacks; and the merged twin's premise that right_boundary_validation.rs is a parity-hash golden is false, so the twin's rank-invariance and MPI-reproduction verification burden f…

- **Station:** cobre-sddp (sddp, sub-station 5a, lens test-bloat; ingest ref 5a-test-bloat-03; folds 5d-test-bloat-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::penalties`, `crates/cobre-sddp/tests/right_boundary_output.rs::penalties`, `crates/cobre-sddp/tests/right_boundary_pricing.rs::penalties`, `crates/cobre-sddp/tests/right_boundary_validation.rs::penalties`, `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::bounds`, `crates/cobre-sddp/tests/right_boundary_output.rs::bounds`, `crates/cobre-sddp/tests/right_boundary_cost_semantics.rs::study_start`, `crates/cobre-sddp/tests/right_boundary_pricing.rs::freeze_terminal_template`, `crates/cobre-sddp/tests/right_boundary_validation.rs::freeze_terminal_template`, `crates/cobre-sddp/tests/right_boundary_pricing.rs::inject_ring_boundary`, `crates/cobre-sddp/tests/right_boundary_validation.rs::inject_ring_boundary`
- **Evidence:** Four separate integration binaries, four separate Cargo link units, and the same fixture prelude in each. The 41-line ResolvedPenalties constructor is byte-identical in all four (md5 4b5df6d0ec99); the 40-line ResolvedBounds constructor is byte-identical in three of the four, with the fourth differing only by taking one f64 parameter; study_start is byte-identical in all four. Beyond the prelude, right_boundary_validation.rs:353-470 is a ~125-line clone of right_boundary_pricing.rs:353-477 covering five fns, diffe… Re-derive: `git grep -n 'fn penalties\(\)|fn bounds\(|fn study_start\(\)|fn freeze_terminal_template|fn inject_ring_boundary|fn post_study_ring_slot' 077dbe2c -- crates/co…`
- **Fix-shape:** Extend tests/common/builders.rs with the two resolved-layer constructors it is missing, a neutral ResolvedPenalties and a ResolvedBounds taking the one dimension the validation variant parameterizes, and move the shared ring-injection prelude into a tests/common submodule so pricing and validation call one copy. The four right_boundary_* binaries already declare mod common, so this adds no link unit and no crate, and per the directive introduces no new fixture crate. Byte-neutral by construction: the shared constructors return the same resolved field values the identical local bodies return today and the parameterized bounds variant keeps its argument, so every affected test builds the same LP, leaving the reproducibility and order-invariance harness, the parity goldens and the rank-invariance map unchanged with no golden moved. The boundary-ring pinning that these files exercise is a contract in .claude/rules/sddp.md, so the moved prelude must keep the ring slot and terminal-theta pinning assertions verbatim.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** md5-identical `penalties()` across four right_boundary_* binaries, `bounds()` identical in three, and five ring-prelude functions code-identical between pricing and validation — four binaries drifting in lockstep on one fixture prelude (TD-037 precedent, Sev B); folds 5d-test-bloat-02.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (tests/common owns shared fixture builders; integration binaries do not re-declare them) (td-queue.json).
- **Queued to:** test-corpus

**TD-044 · Sev B · duplication · effort M · confidence high**
Only the byte-identical fixture classes duplicate: in anticipated_core.rs the six default_hydro_penalties bodies (one md5), the 4-plus-2 default_hydro_bounds and 4-plus-2 default_hydro_block_bounds classes, and two build_config classes (1631/2619/6235 and 8322/8674), each collapsible to one file-scope declaration per class in the binary that owns it, with anticipated_scenarios.rs repeating that same penalties body three times and hydro_sim.rs only study_start; build_system's 13 pairwise-distinct bodies, the other nine build_config variants, and any routing through tests/common make_hydro or HydroSpec are not part of the defect.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens test-bloat; ingest ref 5a-test-bloat-04; folds 5d-test-bloat-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_penalties`, `crates/cobre-sddp/tests/anticipated_core.rs::build_config`, `crates/cobre-sddp/tests/anticipated_core.rs::build_system`, `crates/cobre-sddp/tests/anticipated_core.rs::default_hydro_bounds`, `crates/cobre-sddp/tests/anticipated_scenarios.rs::default_hydro_penalties`, `crates/cobre-sddp/tests/anticipated_scenarios.rs::build_config`, `crates/cobre-sddp/tests/anticipated_scenarios.rs::build_system`, `crates/cobre-sddp/tests/common/builders.rs::make_hydro`
- **Evidence:** This is intra-binary duplication, not cross-binary: the six default_hydro_penalties declarations are byte-identical (md5 7d6690cdf19c) and live in six sibling inline modules of the same file, so the compiler already sees all six in one translation unit and one file-level declaration would serve them. Three of the fourteen build_config declarations are byte-identical (md5 aab2fdb27577, 35 lines each) and two more are byte-identical to each other, leaving nine near-variants of the same 27-34 line config fixture in o… Re-derive: `git grep -n 'fn build_config|fn build_system|fn default_hydro_penalties|fn default_hydro_bounds|fn default_hydro_block_bounds' 077dbe2c -- crates/cobre-sddp/te…`
- **Fix-shape:** Collapse each family to one declaration at file scope in the binary that owns it, hoisting the byte-identical default_hydro_penalties, default_hydro_bounds and default_hydro_block_bounds out of the sibling inline modules, and fold the build_config variants into one file-level builder whose parameters are exactly the fields the variants differ in. Route the hydro construction through tests/common/builders.rs make_hydro and HydroSpec, which both binaries already link via mod common, so nothing new is introduced and per the directive no fixture crate is added. Byte-neutral by construction: a hoisted declaration returns the value its six identical copies return, and a parameterized build_config called with each variant's current arguments produces the same Config, so every anticipated test builds the same study and the reproducibility and order-invariance harness, the parity goldens and the rank-invariance map are all unchanged with no golden moved. The anticipated hold-family and ring semantics these tests pin are contracts in .claude/rules/sddp.md, so the parameterization must not mer…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** byte-identical fixture classes re-declared six and four-plus-two times inside one 8.8k-line binary, with one class repeated across a sibling binary — a per-class hoist to file scope removes every copy; the `build_system` family is conceded (13 pairwise-distinct bodies); folds 5d-test-bloat-03.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (one declaration per fixture class per binary; whole-system builders parameterised, not copied) (td-queue.json).
- **Needs-human (owner gate):** Scope call for the owner: the default_hydro_penalties body is one md5 across anticipated_core.rs and anticipated_scenarios.rs, and scenarios build_config at 2954 equals core build_config at 3831, so the owner must decide whether the hoist stays per-binary at file scope or lifts the shared class into tests/common (the test-corpus station owns the cross-binary convention).
- **Queued to:** test-corpus

**TD-045 · Sev B · duplication · effort S · confidence high**
For state_layout_for the surviving residue is narrower than the title in two ways: the seven bodies are identical only after dedent and are semantically the same StateSpace::new call as test_support::state_layout, the unreachability half of the rationale is false in all seven because each file already resolves other cobre_sddp::test_support symbols ungated, but the stale-symbol half is false in six of seven (inflow_nonnegativity.rs:69-71 names no symbol), and the duplicated symbol is imported by NONE of the seven - it is imported by name in three other binaries of the same corpus (mpi_wire.rs:1344, lp_builder.rs:1532, basis_trajectory_probe.rs:282), which is what proves reachability. From the merged 5d twin: the five all_enabled_cut_state_layouts copies survive in full, and there the cited symbol does exist (test_support.rs:657) so only the reachability claim is false, with simulation_pipeline_integration.rs:47 importing it by name while four siblings hand-copy it; the four study_dims…

- **Station:** cobre-sddp (sddp, sub-station 5a, lens test-bloat; ingest ref 5a-test-bloat-05; folds 5d-test-bloat-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/integration.rs::state_layout_for`, `crates/cobre-sddp/tests/load_integration.rs::state_layout_for`, `crates/cobre-sddp/tests/conformance.rs::state_layout_for`, `crates/cobre-sddp/tests/integration.rs::study_dims`, `crates/cobre-sddp/tests/load_integration.rs::study_dims`
- **Evidence:** Seven integration binaries declare the same 12-line fn with md5 27537dfbfaa4, and the shared test_support::state_layout reduces to StateSpace::new with exactly those eight arguments, so all seven clones are literally replaceable by one existing pub symbol. Both halves of the doc comment justifying the duplication are false at the pin. First, no `test_support::state_layout_for` with that signature exists: the crate-level shared surface exposes state_layout(hydro_count, max_par_order) at test_support.rs:572, and the… Re-derive: `git grep -n 'fn state_layout_for|fn study_dims' 077dbe2c -- crates/cobre-sddp/tests crates/cobre-sddp/src/test_support.rs; for f in conformance inflow_nonnegat…`
- **Fix-shape:** Delete the seven state_layout_for clones and the four study_dims clones and call cobre_sddp::test_support::state_layout and cobre_sddp::test_support::study_dims, which are already pub, already gated on the test-support feature these binaries enable, and already imported by the same files. Delete the doc comment with them rather than repairing it, since its two stated reasons are both false at the pin and a corrected copy would only re-justify the duplication. Byte-neutral by construction: the shared state_layout bottoms out in StateSpace::new with the identical eight arguments the clones pass, including the dense per-hydro effective lag vector, so patch-column resolution is unchanged and the reproducibility and order-invariance harness, the parity goldens and the rank-invariance map all hold with no golden moved. Because parity.rs is one of the seven, verify the parity hash is unmoved rather than assuming it.
- **Alignment:** advances-1 (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.5 (I.3-8 shed line) / Part V §V.2 (Phase 1 purification))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** seven integration binaries hand-copy `state_layout_for` (plus `study_dims` / `all_enabled_cut_state_layouts`) under a doc rationale false at the pin while the same files already resolve other `cobre_sddp::test_support` symbols ungated — the exported surface is the owner and the copies are the ripple (TD-037 precedent); folds 5d-test-bloat-00; carries I.3-8.
- **Part-I:** I.3-8 (cross-reference; the verdict travels to Epic 9 with partI-handoff.json).
- **Yardstick:** docs/design/testing-architecture.md §5.2 uniform test-support feature convention (the exported surface already owns these helpers) (td-queue.json).
- **Needs-human (owner gate):** Owner call outside this candidate: invariance-shuffle.yml builds cargo nextest -p cobre-sddp --test parity without the test-support feature, while parity.rs reaches cobre_sddp::test_support ungated at :894 and :1303, so either that workflow cannot build at the pin or the clones never protected a working default-feature configuration; deciding which fixes whether the consolidation must also gate these modules on the…
- **Queued to:** test-corpus, alignment

**TD-046 · Sev C · asymmetry · effort S · confidence high**
Within crates/cobre-sddp/src/stochastic/noise.rs the `#[cfg(test)] pub(crate) fn shift_lag_state` (176-197) is the file's only test-only item declared in its 560-line production region, and both its module scope and its `pub(crate)` visibility exceed every caller: all six call sites lie inside this same file's `mod tests` (561-EOF) and no other module in the workspace calls it. The hot-path-module framing does not survive, since `stochastic/noise.rs` is not one of the files the hot-path rules enumerate; nor does rather than beside the tests, since the oracle is in the same file 384 lines above its callers.

- **Station:** cobre-sddp (sddp, sub-station 5a, lens test-bloat; ingest ref 5a-test-bloat-06)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/stochastic/noise.rs::shift_lag_state`, `crates/cobre-sddp/src/stochastic/noise.rs::accumulate_and_shift_lag_state`, `crates/cobre-sddp/src/stochastic/noise.rs::tests`
- **Evidence:** shift_lag_state at noise.rs:177 is annotated #[cfg(test)], so it cannot be reached from any production path, and all six of its call sites (1347, 1365, 1379, 1397, 1412, 1572) sit inside the `mod tests` block that starts at line 561. It is a reference oracle used to pin the production accumulate_and_shift_lag_state at line 232 by equivalence. The oracle itself is a legitimate and valuable pattern; the observation is only that it is declared at module scope in a hot-path production file, 384 lines above the only mo… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/stochastic/noise.rs | sed -n '174,200p'; git show 077dbe2c:crates/cobre-sddp/src/stochastic/noise.rs | grep -n 'shift_l…`
- **Fix-shape:** Move the oracle inside the `mod tests` block that already holds its only six call sites, dropping the cfg(test) attribute and the pub(crate) visibility that module scope forced on it, so the hot-path production file contains only production code above its test module. Keep the oracle itself and every equivalence assertion verbatim, since the equivalence between the oracle and the production kernel is what pins the lag-shift behavior. Byte-neutral by construction: relocating a cfg(test) fn into the test module that already calls it changes no production symbol, so the reproducibility and order-invariance harness, the parity goldens and the rank-invariance map are untouched and no golden moves.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** one `#[cfg(test)] pub(crate)` oracle declared in a file's production region with all six callers inside that file's own `mod tests`; scope and visibility exceed every caller — cosmetic homing.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (test-only items live with their tests) (td-queue.json).
- **Needs-human (owner gate):** Scope call for this item: whether relocating the oracle should also re-word the three production doc comments that use `shift_lag_state` as the canonical lag-remap vocabulary (the bracketed reference at `noise.rs:211` plus the code spans at `lp/indexer/state_space.rs:443` and `lp/indexer/cut_state_projection.rs:269`), or leave them naming a test-module-private symbol.
- **Queued to:** test-corpus

#### 5b — lp/ (indexer, builder, template, generic constraints)

**Architecture** — 2 minted; 1 sharpening a live id (see the disposition table)

**CD-084 · Sev B (A-risk) · asymmetry · effort S · confidence high**
Narrowed to a LATENT single-site contract violation with a named precondition: fill_anticipated_columns (columns.rs:618) is the only one of four ring-residue owners that keys the deposit slot off the raw delivery axis instead of PointResolution::ring_index, contradicting both its own rustdoc at columns.rs:573-575 and the pinned deposit clause; the two residues diverge only for a plant whose fixed post-horizon width g is not a multiple of k_max AND which also holds a class-3 carried in-study decision above that window, in which case the latched slot stays frozen from fill_anticipated_slot_columns and the commitment is forced to zero (infeasible once min_generation_mw exceeds zero). Such a deck is accepted by cobre-io (thermal.rs:112/:117/:149 exempt a horizon-exceeding lead that reaches post-study), but NO shipped deck declares it - d55 is the only post-study example and resolves g equal to zero - so the defect is latent at the pin, no parity golden is wrong today, and the residue is t…

- **Station:** cobre-sddp (sddp, sub-station 5b, lens architecture; ingest ref 5b-architecture-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/builder/columns.rs::fill_anticipated_columns`, `crates/cobre-sddp/src/lp/builder/entries.rs::fill_anticipated_state_out_def_entries`, `crates/cobre-sddp/src/lp/builder/layout.rs::build_anticipated_slot_row_pos`
- **Evidence:** Three modules own the anticipated deposit-slot residue and only two agree. entries.rs::fill_anticipated_state_out_def_entries and layout.rs::build_anticipated_slot_row_pos both key off the RING axis (`point.ring_index(delivery_stage)` and the `r = stage_idx + depth + 1` ring walk); columns.rs::fill_anticipated_columns keys off the RAW delivery axis (`delivery_stage % layout.k_max`) eleven lines after the identical `point` is already in scope, i.e. the conversion it needs is available and unused. The two agreeing o… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/lp/builder/columns.rs | sed -n '604,624p'; git show 077dbe2c:crates/cobre-sddp/src/lp/builder/entries.rs | sed -n '155,…`
- **Fix-shape:** Give the anticipated ring ONE residue owner in the fpha_cursor.rs style: hoist the delivery-axis to ring-axis conversion (point.ring_index(delivery_stage) with the same excised-window debug_assert entries.rs already carries) into a single walker in lp/builder that the row fill, the column fill and build_anticipated_slot_row_pos all drive off, so a one-sided edit that lands the deposit row and the re-opened column on different slots cannot compile; the change touches only which state column receives (-INF, +INF) rather than [0, 0] and so must cite .claude/rules/sddp.md section 'State pinning uses column bounds, not equality rows' (the re-open is a column-bound write, not a new equality row, and incoming pinning stays with fill_col_state_patches through state_to_lp_incoming_column); byte-neutrality bar is that no deck without a fixed post-horizon window changes a single column bound (ring_index is the identity on a contiguous ring, so parity goldens in tests/parity.rs and tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and mpiexec -n 1 versus -n 2…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals; the anticipated ring is SDDP geometry))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction — asserted at the pin (no shipped deck reaches the precondition); a deck that does moves a golden legitimately — re-baseline decision at the owner gate for that case.
- **Contract:** .claude/rules/sddp.md — *In-LP anticipated ring: definition-row sign, hold carry & asymmetric masking*; *State pinning uses column bounds, not equality rows*; *The ring axis: the delivery axis with the fixed post-horizon window excised* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** a LATENT single-site contract divergence: `fill_anticipated_columns` is the only one of four ring-residue owners keying the deposit slot off the raw delivery axis instead of `PointResolution::ring_index`, contradicting its own rustdoc and the pinned deposit clause; the residues diverge only for a plant whose fixed post-horizon width is not a multiple of k_max AND which carries a class-3 in-study decision above that window — a bounded, wrong-but-compiling divergence that would write a wrong column bound silently, hence the A-risk marker (CD-074 class: pinned state coupled to a bounded column reconciled ad hoc per family).
- **Reviewer rating:** A — recalibrated to B (A-risk) because latent at the pin (no shipped deck satisfies the two-part precondition), a single fill site, and the three sibling residue owners are correct — the blast radius is one function, not a spreading structure.

**CD-085 · Sev B · missing-seam · effort S · confidence high**
Only the two commitment-hold reads in simulation/extraction.rs (:268 incoming, :316 outgoing) are genuine untyped LP-column recompositions: the same file already resolves the storage, lag and bucket families' primal reads through the typed InCol/OutCol accessors, while the commitment family has no StateSpace-side accessor reachable from outside lp/ because its purpose-named owner DeliveryRing::out_col/in_col is only constructible through the pub(super) anticipated_ring helper. The three state-vector sites (commitment_reconcile.rs:193 with :208, setup/mod.rs:2651, :2667) are StateDim indices and not columns, and the DeliveryRing out_col versus slot_target transposition hazard is unreachable at every production call site.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens architecture; ingest ref 5b-architecture-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/indexer/state_space.rs::commitment_hold_in_study_offset`, `crates/cobre-sddp/src/lp/indexer/state_space.rs::bucket_incoming_col`, `crates/cobre-sddp/src/lp/builder/delivery_ring.rs::out_col`
- **Evidence:** Every other state family carries a typed resolver returning InCol or OutCol, and entity_index.rs pins six cross-family swaps with compile_fail doctests, so the crate's stated discipline is that a resolved column is a newtype and not a usize. The commitment-hold family is the exception, and the exception has a concrete cost visible at the call sites. First, setup/mod.rs:2651 indexes a STATE VECTOR with `commit_out.start + commitment_hold_in_study_offset(..)` while simulation/extraction.rs:316 uses the identical exp… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/lp/indexer/state_space.rs | sed -n '565,640p'; git grep -n 'commit_out\.start\|commit_in\.start' 077dbe2c -- crates/cob…`
- **Fix-shape:** Add the missing typed resolvers next to their siblings in state_space.rs - a commitment_hold_incoming_col returning InCol and a commitment_hold_outgoing_col returning OutCol, both routing through state_to_lp_incoming_column / state_to_lp_column exactly as bucket_incoming_col and storage_incoming_col already do - then convert the LP-column call sites (commitment_reconcile.rs, simulation/extraction.rs both sides) to them, leaving setup/mod.rs on an explicitly state-dimension-named accessor so the two axes stop sharing one expression; keep commitment_hold_in_study_offset as the shared slot arithmetic both resolvers call. Cites .claude/rules/sddp.md section 'State pinning uses column bounds, not equality rows' because the incoming resolver is precisely what that section requires every pin and dual read to resolve through, and no equality row is added. Byte-neutrality bar: each new resolver returns the same integer the arithmetic it replaces already produced, proven by an equivalence test asserting resolver equals hand-computed offset over every plant and slot, so parity goldens in tests…
- **Alignment:** advances-0b (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (lp/ indexer is engine-neutral emission logic the 0b carve lifts))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *State pinning uses column bounds, not equality rows* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** the commitment-hold family is the only one of five state families without a typed `InCol`/`OutCol` resolver reachable from outside lp/, so the two extraction-side reads recompose the LP column by hand — CD-035 precedent (missing typed-role seam), Sev B; NOT a one-consumer seam: both consumers exist at the pin and the accessors complete an existing five-member family (owner question carried).
- **Needs-human (owner gate):** Alignment-epic owner call: whether conflicts trigger 4 (one-consumer abstraction) reaches a purpose-named single-consumer StateSpace accessor that completes an existing five-member family, or is bounded to seams, traits and crates as its text states.
- **Queued to:** alignment

**Performance** — 2 minted

**PD-038 · Sev B · asymmetry · effort S · confidence high**
Narrowed to one loop: `fill_col_state_patches` (patch.rs:257-260) re-runs `classify`'s four-way region membership test and region select once per state element on every prepared solve, for a state-dim-to-incoming-column map that is fixed at `StateSpace::new` and that both other production consumers read from a precomputed table. Stripped from the claim: (i) the twin-cache symmetry as evidence, since `state_to_lp_column_map` amortizes the outgoing lag arm's `offset % n` / `offset / n` (state_space.rs:478-479) that the incoming resolver does not contain; (ii) the filed per-element cost model of a Range reconstruction per probe plus two further boundary derivations, since those are loop-invariant through `#[inline]` accessors on non-aliasing `&StateSpace` fields; (iii) the extraction wrappers `storage_incoming_col` / `lag_incoming_col` / `bucket_incoming_col` at simulation/extraction.rs:376,920,924,2065, which run per extracted stage rather than per solve.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens performance; ingest ref 5b-performance-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/builder/patch.rs::fill_col_state_patches`, `crates/cobre-sddp/src/lp/indexer/state_space.rs::state_to_lp_incoming_column`, `crates/cobre-sddp/src/lp/indexer/state_space.rs::classify`
- **Evidence:** patch.rs:259 calls state_to_lp_incoming_column once per state element inside fill_col_state_patches, and StageSolvePrep::run executes fill_col_state_patches on every solve on all four hot paths (forward solve/stage_solve.rs, backward training/backward/lp_setup.rs::patch_opening_bounds, simulation/pipeline.rs, training/lower_bound.rs). Each call runs classify(), a linear find over the four REGION_ORDER regions that reconstructs a Range<usize> per probe, then derives the offset with a second state_dim_range(region).… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/lp/builder/patch.rs | sed -n '243,262p'; git show 077dbe2c:crates/cobre-sddp/src/lp/indexer/state_space.rs | sed -n '42…`
- **Fix-shape:** Give the incoming direction the cache the outgoing direction already has: an incoming twin of state_to_lp_column_map on StateSpace, filled by an incoming twin of finalize_state_column_map that calls state_to_lp_incoming_column for every j in [0, n_state) -- a pure cache of the resolver, never a second copy of its arithmetic, which is the discipline finalize_state_column_map's own rustdoc already states -- and let fill_col_state_patches read it, with storage_incoming_col / lag_incoming_col / bucket_incoming_col following the same route so the simulation extraction path benefits without a second cache. The debug_assert on the finalized length that lp_column_for_state carries transfers unchanged. .claude/rules/sddp.md Sec. State pinning uses column bounds, not equality rows is untouched: the same incoming columns are pinned with the same column bounds, and the resolver remains the single authority for the arithmetic. Byte-neutral by construction -- every offset is a pure function of (N, L, B, A, k_max) as the state_space.rs module doc states, so a table lookup returns the identical col…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *State pinning uses column bounds, not equality rows* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** one per-solve loop re-runs the four-region `classify` membership test per state element for a map fixed at `StateSpace::new`, while the two other production consumers read a precomputed table — an uncached twin on the per-solve pin path; structural calibration only, UNMEASURED.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: redundant per-element index recomputation on every solve: a four-region linear scan with a Range reconstruction per probe plus two further region-boundary derivations per state element, where a precomputed table already exists for the twin direction and for the extraction path; exercising call sites (cited, not anchored): training/stage_solve_prep.rs · StageSolvePrep::run (every prepared solve); training/backward/* · patch_opening_bounds (per opening). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Queued to:** performance-sweep

**PD-039 · Sev B · duplication · effort M · confidence high**
For a Generated child run carrying more than one opening, the per-opening backward prep redundantly re-fills and re-submits the opening-invariant column-bound state pin from the unchanged x_hat and col_scale, costing one fill over n_state plus one set_col_bounds per opening that the CLP backend expands into a full-length col_lower and col_upper FFI push; the opening-invariant commitment relaxation is additionally re-filled per opening only when n_anticipated is nonzero and the stage has geometry, and re-submitted only when the relaxation set is nonempty. The claim is redundant call frequency alone: no pinned contract is violated, patch.rs's 'once per stage-visit' rustdoc is a value-semantics contract the current code already satisfies, and the External single-opening branch carries no redundancy.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens performance; ingest ref 5b-performance-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/builder/patch.rs::fill_col_state_patches`, `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::fill_bound_relaxations`
- **Evidence:** load_backward_lp runs once per child ('The LP structure is identical across a child's openings, so only bound patching runs per opening') and patch_opening_bounds runs once per opening with the SAME x_hat -- x_hat is loop-invariant in replicated.rs's opening loop and in the by_node.rs / by_scenario.rs loops, which is the whole per-opening entry point (load once at by_node.rs:245, by_scenario.rs:85, replicated.rs:101; patch at by_node.rs:284, by_scenario.rs:187 and :305, replicated.rs:143). patch_opening_bounds del… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/training/backward/replicated.rs | sed -n '141,144p'; git show 077dbe2c:crates/cobre-sddp/src/training/stage_solve_prep.…`
- **Fix-shape:** Hoist the opening-invariant work to per-child: fill and submit the column-bound state pin (fill_col_state_patches plus its set_col_bounds) and fill the commitment relaxation once alongside load_backward_lp, and narrow the per-opening entry point to what actually varies -- the row patches from the transformed inflow/load noise and the NCS availability column patch. All three backward drivers reach this through the single helper patch_opening_bounds, so the blast radius is that helper plus a parameterization of StageSolvePrep::run (a per-child prep and a per-opening prep), with no driver-by-driver edit. Three pinned contracts govern it. .claude/rules/sddp.md Sec. State pinning uses column bounds, not equality rows: the pin stays a column-bound pin at the same incoming columns and merely recovers the once-per-stage-visit identity patch.rs's rustdoc already declares. Sec. Lower-bound evaluation must patch NCS: the NCS patch stays per opening and must not be hoisted with it. And the ordering rule reconcile_commitments carries in its rustdoc -- it runs last because the template reload and…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction — asserted (bit-identical bounds already in force are simply not re-pushed); the owner gate decides whether dropping an idempotent re-push counts as a warm-start-chain change that must clear the `opening_order_determinism` gate before it lands.
- **Contract:** .claude/rules/sddp.md — *Backward opening order is warm-start-only*; *Lower-bound evaluation must patch NCS*; *State pinning uses column bounds, not equality rows* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** for a child run with more than one opening, the opening-invariant column-bound pin is re-filled and re-submitted per opening (one fill over n_state plus one `set_col_bounds` the CLP backend expands to a full-length FFI push), and the commitment relaxation likewise when anticipated geometry is present — redundant per-opening work on the backward hot path; the three governing contracts hold under the hoist (same columns, same bounds, NCS patch still per opening, opening order untouched); structural calibration only, UNMEASURED.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: redundant per-opening recomputation plus a repeated FFI bound-submission crossing for an opening-invariant column-bound pin and an opening-invariant commitment relaxation; exercising call sites (cited, not anchored): training/backward/by_scenario.rs · patch_opening_bounds (:187, :305); training/backward/by_node.rs · patch_opening_bounds (:284); training/backward/replicated.rs · patch_opening_bounds (:143). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Needs-human (owner gate):** Owner call: whether removing a bit-identical repeat submission of already-in-force column bounds counts as a warm-start-chain change that must clear the parity goldens, the rank-invariance harness and the opening_order_determinism gate before the hoist lands, given that the CLP backend's set_row_bounds rustdoc asserts factorization/basis preservation across a bound patch but nothing in this repo pins that property f…
- **Queued to:** performance-sweep

**Over-engineering** — 2 minted

**OD-037 · Sev C · speculative-generality · effort S · confidence high**
Among the types declared in lp/indexer/index.rs, only Col and Row have no non-test, non-doc production consumer at the pin — their sole references are the mod.rs:119 re-export, three in-file round-trip tests and two rustdoc links — and Col's own doc names set_col_bounds as its seam although that seam is usize-only at cobre-solver/src/trait_def.rs:90; Boundary is excluded from the residue because block_storage_col consumes it, and because lib.rs:69 makes both names public items the removal is a semver-major deletion that batches onto the 0b carve-out of lp/, not a standalone break.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens over-engineering; ingest ref 5b-over-engineering-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/indexer/index.rs::Col`, `crates/cobre-sddp/src/lp/indexer/index.rs::Row`
- **Evidence:** Nothing in the repository constructs or names Col or Row as a type outside index.rs itself. The only non-declaration mention of Col in the crate is the re-export at lp/indexer/mod.rs:119; the only typed Row matches repo-wide are the unrelated cobre_io PriceStrategy::Row variant; the one remaining Col hit is prose inside a patch.rs doctest comment. Three tests in index.rs exist solely to give the two unused types a test: col_is_zero_cost_and_round_trips, row_is_zero_cost_and_round_trips and col_equality_is_value_ba… Re-derive: `git grep -n -E '\b(Col|Row)::new\b' 077dbe2c -- crates/ ; git grep -n -E '\b(Col|Row)::new\b|:\s*(Col|Row)\b|->\s*(Col|Row)\b' 077dbe2c -- crates/ | grep -v 'l…`
- **Fix-shape:** Delete the Col and Row declarations and their two impl blocks from lp/indexer/index.rs, delete the three tests that exist only to exercise them (col_is_zero_cost_and_round_trips, row_is_zero_cost_and_round_trips, col_equality_is_value_based_not_identity), drop the two names from the pub use in lp/indexer/mod.rs, and rewrite the index.rs module-doc sentence so the typed vocabulary it introduces lists only the types that exist: StateDim, CutSlot, BlockIdx, the InCol/OutCol role split and Boundary. Byte-neutral by construction, because the census proves zero production call sites, so no LP column, row, coefficient or bound moves; still assert the parity goldens in tests/parity.rs through tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and mpiexec -n 1 versus -n 2 reproduction, and move no golden. The blast radius is the public API: lib.rs re-exports the module (pub use lp::indexer), so both names are public items of cobre-sddp and the deletion is a public-API removal with no in-repo consumer; the natural batching is the next public-API break, which…
- **Alignment:** advances-0b (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (lp/ indexer moves with the cobre-model carve-out))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** two `pub` newtypes with no non-test, non-doc consumer at the pin (three round-trip tests and a re-export are their whole footprint); the delete-or-register disposition is one edit — OD-032 precedent, Sev C; the semver-major removal is a scheduling call (CD-019 batching precedent), not severity.
- **Reviewer rating:** B — recalibrated to C because zero consumers, three in-file tests and two rustdoc links are the whole blast radius; OD-032 (zero-consumer public predicate) calibrated C.
- **Needs-human (owner gate):** Scheduling only: whether the public-API removal of Col and Row ships as its own semver-major break or waits for the 0b cobre-model carve-out of lp/, as the CD-019 deferral precedent suggests — the defect itself is settled.
- **Queued to:** alignment

**OD-038 · Sev C · speculative-generality · effort S · confidence high**
FphaRowRange (lp/indexer/layout.rs:34-40) has no construction or read site anywhere under crates/ outside its own cfg(test) Debug/Copy smoke test, yet is public API through lib.rs:69, and the satellite-types carrier sentence at layout.rs:5-7 is false for it because StageGeometry exposes FPHA rows only as the flat fpha: row_fpha_start()..fpha_rows_end range (builder/layout.rs:1905), never as a Vec<FphaRowRange>. Narrowed from the title on two counts: the documented row formula does NOT contradict the walker's stride, since start + k * planes_per_block + p reproduces BlockGrid::fpha_plane (block_grid.rs:95-104) exactly and only the granularity of start diverges, per-plant in the doc against the per-CELL re-base in for_each_fpha_plane (fpha_cursor.rs:80-97); and that divergence is latent rather than a live mis-address, because the hydro-cell partition is the identity for every shipping study (hydro_cell.rs:5-10, builder/layout.rs:1282-1284), so no in-repo deck can be mis-addressed by it.…

- **Station:** cobre-sddp (sddp, sub-station 5b, lens over-engineering; ingest ref 5b-over-engineering-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/indexer/layout.rs::FphaRowRange`, `crates/cobre-sddp/src/lp/builder/fpha_cursor.rs::for_each_fpha_plane`
- **Evidence:** All eight repo-wide mentions of FphaRowRange are the declaration, its own Debug/Clone smoke test, two module-doc lines, the re-export at lp/indexer/mod.rs:120 and one prose cross-reference. No code anywhere constructs or reads the type. Two separate claims in the doc are false. First, the carrier claim: the module doc says these satellite types are produced by StageLayout and carried on its StageGeometry snapshot, which holds for the sibling EvaporationIndices (four production consumer files: builder/layout.rs, bu… Re-derive: `git grep -n '\bFphaRowRange\b' 077dbe2c -- crates/ ; git grep -c 'FphaRowRange' 077dbe2c -- crates/cobre-sddp/src/lp/builder/template.rs crates/cobre-sddp/src/…`
- **Fix-shape:** Delete the FphaRowRange declaration and the fpha_row_range_debug_clone_copy smoke test from lp/indexer/layout.rs, drop the name from the pub use in lp/indexer/mod.rs, narrow both module docs (the lp/indexer/layout.rs header and the satellite-types bullet in lp/indexer/mod.rs) to describe only EvaporationIndices, and repoint the sentence in lp/builder/fpha_cursor.rs that says it matches FphaRowRange::start at StageLayout::row_fpha_start, the value for_each_fpha_plane actually seeds its cursor from. Contract-first: the change touches no FPHA row emission, so neither the FPHA uses average storage section nor the Hydro-cell aggregation assumes one production map per cell section of .claude/rules/sddp.md is weakened; the cell-major nesting those sections rest on stays owned by for_each_fpha_plane, which is exactly why the dead type's per-plant formula must not survive as documentation of it. Byte-neutral by construction, since the type has zero construction sites and only prose changes accompany the deletion; assert the parity goldens in tests/parity.rs through tests/common/parity_hash.r…
- **Alignment:** advances-0b (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (lp/ indexer moves with the cobre-model carve-out))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *FPHA uses average storage*; *Hydro-cell aggregation assumes one production map per cell* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** one `pub` type with no construction or read site outside its own smoke test, plus a module-doc carrier sentence that is false for it; the row-formula contradiction is conceded — OD-032 precedent, Sev C; public-API removal batching per CD-019 is the owner's call.
- **Reviewer rating:** B — recalibrated to C because a single zero-consumer type and one false doc sentence; the formula contradiction the reviewer weighed did not survive the defender.
- **Needs-human (owner gate):** Owner call carried over from the attacker: deleting FphaRowRange is a public-API removal (reachable as cobre_sddp::indexer::FphaRowRange via lib.rs:69) with no in-repo consumer; decide whether it lands now as ordinary cleanup or is batched with the Phase 0b cobre-model carve-out that relocates lp/indexer anyway, per the CD-019 deferral precedent. Scope call: the fix as filed leaves the same per-plant framing live in BlockGrid::advance_fpha_base's rustdoc (block_grid.rs:106-107), so the reader hazard is only half retired; decide whether that sibling doc correction joins this item or gets its own.
- **Queued to:** alignment

**Test bloat** — 4 minted

**TD-047 · Sev C · god-module · effort M · confidence high**
Scoped to crates/cobre-sddp/src/lp/builder/entries.rs alone: its inline module pumping_water_tests (3518-10093) is named for the least-exercised subject it contains and carries no internal partition, evidenced at the pin by 3 call sites of fill_pumping_water_entries against 25 of build_stage_matrix_entries, 36 helper fns declared from module offset 46 to 6440, and one CSC accessor duplicated about 2000 lines apart (coeff_at at 2431, csc_at at 4395), while sibling columns.rs in the same directory partitions 72 tests across 12 subject-named modules. Not conceded: the 'crate's catch-all' framing, and the fixShape's 'pure test relocation' premise, because the 72-reference module-private PumpFixtures harness must gain sibling visibility and the read_dir-based lp_builder_never_references_dual_extraction gate's scan surface depends on where the split files land.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens test-bloat; ingest ref 5b-test-bloat-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lp/builder/entries.rs::pumping_water_tests`, `crates/cobre-sddp/src/lp/builder/entries.rs::csc_byte_identical_under_permuted_declaration_order`, `crates/cobre-sddp/src/lp/builder/entries.rs::zero_cost_tests`
- **Evidence:** pumping_water_tests is a single brace block with no nested submodules holding 65 percent of entries.rs and 78 percent of the file's 99 test fns. Its contents span at least eight subjects that have nothing to do with pumping water entries: contract import and export rows, split-plant cell water balance, FPHA per-bus generation crediting, transit-bucket definition rows, chronological travel time, filling-phase sigma_fill rows, generic-constraint column resolution, and the CSC declaration-order byte-identity gate. It… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/lp/builder/entries.rs | grep -n '^mod '; git show 077dbe2c:crates/cobre-sddp/src/lp/builder/entries.rs | sed -n '3518,1…`
- **Fix-shape:** When candidate two converts entries.rs to a directory module, do not carry pumping_water_tests across as one file. Split it along the subject boundaries its own fn names already mark, one sibling file per subject (contracts, split-plant cells, FPHA crediting, transit buckets, chronological water, filling-phase rows, generic constraints, declaration-order determinism), and lift the shared fixture constructors it declares into a single sibling shared by them. Each resulting file gets a name that predicts its contents so a new test has one obvious home. The determinism gate csc_byte_identical_under_permuted_declaration_order moves verbatim and stays a gate. Pure test relocation: no non-test line of entries.rs changes, the library stays byte-identical, and the parity goldens, the rank-invariance harness and the mpiexec -n 1 versus -n 2 reproduction are unaffected. Moves no golden.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (the 0b carve lifts lp/builder emission logic; its tests move with it))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** one inline test module of 6.5k lines and 77 tests named for its least-exercised subject with no internal partition; the split shape is the CD-007 convention call — inline-test giants calibrate C (CD-007 precedent).
- **Related prior entries:** CD-007 (distinct claims; adjudicated at ingest).
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (subject-named test homes; the section 5.1 threshold is a proposal the test-corpus station ratifies) (td-queue.json).
- **Needs-human (owner gate):** Target shape for the split is a convention call CD-007 owns: one extracted tests.rs matching commitment_reconcile/layout/template, or N subject-named files matching columns.rs's 12 inline modules. The unratified section 5.1 threshold cannot settle it.
- **Queued to:** test-corpus

**TD-048 · Sev C · duplication · effort S · confidence high**
build_classical_fixture (par_a_lag12_lp_coefficient.rs:347-580, 234 lines) repeats 229 lines of build_par_a_fixture verbatim to vary only the annual field at its two InflowModel sites, and the structural sameness that the control test classical_par_has_no_lag_11_column depends on is carried solely by the doc comment at 344-346; the drift exposure is confined to the raw InflowModel, LoadModel, HydroPenalties and bounds/penalties spec literals that bypass tests/common/builders.rs, since Stage, Hydro and Bus field additions are already one-place changes there, and any consolidation must additionally preserve the two per-variant builder-internal .expect labels that the fixShape mistook for test-body messages.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens test-bloat; ingest ref 5b-test-bloat-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs::build_classical_fixture`, `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs::build_par_a_fixture`, `crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs::classical_par_has_no_lag_11_column`
- **Evidence:** Roughly 230 of 243 lines are byte-identical between the two fixture builders, and the file's own doc comment declares the duplication. The only behavioural difference is whether the two inflow models carry an AnnualComponent. Both builders already draw their entities from tests/common/builders.rs via make_bus, make_hydro and make_stage, so the duplication is not a missing shared builder; it is a copy taken instead of adding one parameter. The cost is paid on every edit: any change to the shared 230 lines, for inst… Re-derive: `git show 077dbe2c:crates/cobre-sddp/tests/par_a_lag12_lp_coefficient.rs > /tmp/par.rs; sed -n '101,346p' /tmp/par.rs > /tmp/p1; sed -n '347,589p' /tmp/par.rs >…`
- **Fix-shape:** Collapse the pair into one private builder taking the annual component as a parameter, for example an `Option<AnnualComponent>` argument, with the two existing names kept as two-line wrappers that pass Some(...) and None so the three test bodies and their assertion messages are untouched. This is test-binary-local: it changes no library code, so the parity goldens under tests/parity.rs with tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and the mpiexec -n 1 versus -n 2 reproduction are all unaffected, and the three tests in this binary assert the same LP coefficients as before. Moves no golden. Keep the builders in this binary rather than promoting them to tests/common, since no other binary consumes the PAR-A lag-12 shape.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** one 229-line verbatim clone pair inside one binary whose only delta is one field; parameterising in place is a local S fix with drift confined to the raw spec literals.
- **Reviewer rating:** B — recalibrated to C because one clone pair, one binary, one-parameter delta — TD-035-scale blast radius, not the TD-037 cross-binary ripple.
- **Yardstick:** docs/design/testing-architecture.md §3.2 sustainability (a fixture earns its second copy only by a delta the copy cannot express as a parameter) (td-queue.json).
- **Needs-human (owner gate):** Whether this binary's classical arm should be parameterized in place or instead folded onto the existing two_hydro_par_system helper when template_integration and par_a_lag12_lp_coefficient are grouped under the testing-architecture section 5.1 layout - a test-corpus station call, since that homing layout is still a proposal at the pin.
- **Queued to:** test-corpus

**TD-049 · Sev B · duplication · effort M · confidence high**
Narrowed to two verified copy-paste families and a corrected cost basis: the anticipated-thermal trio (one_anticipated_thermal_system, two_thermal_one_anticipated_system, two_anticipated_thermal_system, 127 and 130 of 144 lines identical, differing only in the thermal list and the n_thermals and k_max derived from it) and the one_bus_system_n_blks / one_bus_system_n_blks_with_generic pair (111 of 117 identical, the second being the first plus two builder calls) are genuine sibling copies that one list-parameterized builder would subsume. The title's stated cost is wrong and does not survive: StageSpec plus the Default spread at all 20 sites keeps a new Stage field O(1) per the invariant documented at tests/common/builders.rs:1-9, so the per-copy fan-out is confined to the non-Spec literals, namely the four bounds and penalties sizing structs (20 copies, 55 of 80 sites without a Default spread) and the InflowModel and LoadModel literals (32 sites, none with one), which the proposed fam…

- **Station:** cobre-sddp (sddp, sub-station 5b, lens test-bloat; ingest ref 5b-test-bloat-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/template_integration.rs::build_hydro_one_ant_system`, `crates/cobre-sddp/tests/template_integration.rs::one_hydro_one_ant_system`, `crates/cobre-sddp/tests/template_integration.rs::two_anticipated_thermal_system`, `crates/cobre-sddp/tests/template_integration.rs::one_anticipated_thermal_system`, `crates/cobre-sddp/tests/template_integration.rs::one_bus_system_n_blks_with_generic`
- **Evidence:** These nineteen builders are about 3400 of the file's 4421 lines, so the parent of the eleven #[path] submodules is roughly three quarters fixture scaffolding. They are not duplicating tests/common/builders.rs: every one of them calls make_bus, make_hydro, make_stage or make_thermal (two to six calls each) and constructs zero Stage, Hydro, Bus or Thermal literals, so the shared builder layer is being used as intended. The duplication is one level up, between siblings. The build_hydro_one_ant_system body is the one_… Re-derive: `git show 077dbe2c:crates/cobre-sddp/tests/template_integration.rs > /tmp/ti.rs; awk '/^fn [a-z_]/{n=$0;sub(/^fn /,"",n);sub(/\(.*/,"",n);s=NR} /^}$/{if(s){prin…`
- **Fix-shape:** Collapse each family into one parameterized builder in the parent module and keep the existing names as thin wrappers so the eleven submodules that reach these through `use super::*` need no edits. The three families the diffs identify are the anticipated-hydro pair, the anticipated-thermal trio and the one-bus trio; parameterize on the values the copies already differ by, namely lead stages, discount rate, block count and the optional generic constraint. Do this inside tests/template_integration.rs rather than moving the builders to tests/common/builders.rs, since only this binary consumes whole-system shapes at this granularity and common/builders.rs owns the per-entity layer below them. No library code changes, so the parity goldens under tests/parity.rs with tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs and the mpiexec -n 1 versus -n 2 reproduction are unaffected; the wrappers must reproduce their current field values exactly so every assertion in the eleven submodules keeps its present expected numbers. Moves no golden.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** two verified copy-paste families among nineteen whole-system builders in one binary (a trio 127-130/144 lines identical, a pair 111/117), each one list-parameterised builder away; the remaining fan-out through the raw spec literals is the owner's scope call — TD-037 precedent (ripple on every field change), Sev B.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (parameterised builders, one per shape family) (td-queue.json).
- **Needs-human (owner gate):** Scope call: whether the ticket is the family collapse as filed or the higher-leverage extension of the O(1) Spec layer to InflowModel, LoadModel and the four bounds/penalties sizing structs, which owns the 20-copy fan-out the collapse leaves at fifteen.
- **Queued to:** test-corpus

**TD-050 · Sev C · duplication · effort S · confidence high**
Only the single-file half survives, and it is the base-path construction rather than a shared corpus helper: tests/lp_builder.rs repeats Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic") three times in one binary, as the two parameter-name-only variants of case_dir at lines 65-69 and 573-577 and as deterministic_root at line 478, all of which one file-scope base-path helper would collapse. The tests/common/parity_hash.rs site is conceded and excluded: its case_dir is already exported (pub fn inside pub mod parity_hash at common/mod.rs:27, linked from parity.rs:48) and is a closed-set golden-label-to-directory mapper with a panic! guard, not a copy of the suffix-join helper.

- **Station:** cobre-sddp (sddp, sub-station 5b, lens test-bloat; ingest ref 5b-test-bloat-07)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/lp_builder.rs::case_dir`, `crates/cobre-sddp/tests/common/parity_hash.rs::case_dir`, `crates/cobre-sddp/tests/common/mod.rs::build_setup_for_case`
- **Evidence:** tests/common/mod.rs is the fixture root and every one of its case helpers takes a case directory as `&Path` (boundary_requirements, build_setup_for_case, fresh_setup_with), yet it exports no way to name one, so each consumer re-derives the examples/deterministic path itself. tests/common/parity_hash.rs, inside the shared directory, holds its own private copy rather than a shared one. The clearest instance is entirely within one file in scope: lp_builder.rs declares the identical helper twice, once per inline modul… Re-derive: `git show 077dbe2c:crates/cobre-sddp/tests/lp_builder.rs | sed -n '65,70p;573,578p'; git grep -c 'fn case_dir' 077dbe2c -- crates/cobre-sddp/tests`
- **Fix-shape:** Export one case-directory helper from tests/common/mod.rs next to the case helpers that already take that path as an argument, and have lp_builder.rs's two inline modules and common/parity_hash.rs call it instead of declaring their own. Keep it in tests/common rather than creating a fixture crate. The wider corpus-level spread across the other binaries belongs to the test-corpus station; the two in-file copies here are the part anchored in this sub-station. Pure test-harness change resolving to the same directory, so the parity goldens under tests/parity.rs with tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs and the mpiexec -n 1 versus -n 2 reproduction all read the same cases and stay byte-neutral. Moves no golden.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** one base-path construction spelled three times inside one binary; the tests/common half is conceded (already exported) and the corpus-wide spread belongs to the test-corpus station.
- **Cross-station:** corpus-wide `case_dir` spread belongs to the test-corpus station (E08); only the in-file copies are anchored here.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (one file-scope helper per binary; corpus-wide helpers live in tests/common) (td-queue.json).
- **Queued to:** test-corpus

#### 5c — cut + training + solve + workspace

**Architecture** — 1 minted; 3 sharpening a live id (see the disposition table)

**CD-086 · Sev B (A-risk) · asymmetry · effort M · confidence high**
Under `Traversal::Enumerated`, `run_enumerated_backward` neither folds a `slot_increments` contribution nor calls `sync_stage_metadata`, so `CutPool::record_binding` never fires and every populated slot keeps `last_active_iter == iteration_generated` with `active_count == 0`; the enumerated forward's `build_initial_resident_set` seed therefore reduces to active-and-generated-within-k2, on a pairing no validation rejects. Strictly narrower than the title on two counts: the per-opening bump is NOT a hand-copied per-driver duty (both sampled schedulers call the single shared `accumulate_opening_outcome`; only the `slot_increments`-to-`metadata_sync_contribution` fold and the `sync_stage_metadata` call are driver-local), and the DCS consequence is confined to inner iterations and solve-count telemetry — never a wrong cut, bound or objective, because the lazy loop is exact and the seed is only a warm-start hint.

- **Station:** cobre-sddp (sddp, sub-station 5c, lens architecture; ingest ref 5c-architecture-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward`, `crates/cobre-sddp/src/training/backward/replicated.rs::run_backward_node_replicated`, `crates/cobre-sddp/src/cut/dcs.rs::build_initial_resident_set`
- **Evidence:** CutPool::record_binding has exactly one production caller (backward_pass_state.rs:1063, inside sync_stage_metadata), and sync_stage_metadata has exactly one production call site (:1673, inside run_one_backward_level), which is reached only from run_sampled_backward (:647). BackwardPassState::run routes Traversal::Enumerated to run_enumerated_backward (:508); that function commits cuts (inputs.fcf.add_cut at :880) but never pushes a PoolRegion -- the sole push is :1810 inside compute_one_backward_node, on the sampl… Re-derive: `git grep -n 'record_binding' 077dbe2c -- crates/ ; git grep -n -i 'dcs' 077dbe2c -- crates/cobre-sddp/src/setup/mod.rs ; git show 077dbe2c:crates/cobre-sddp/sr…`
- **Fix-shape:** Two roads, and the pick is the owner's because it turns on whether enumerated-plus-DCS is a supported combination. Road (a), make the exclusion explicit: reject the combination in the study admission gate exactly as the by-node scheduler's DCS exclusion is already handled, and pin it with a named test plus a rules entry, so an unsupported combination is a rejection rather than a silent degradation. Road (b), make the channel driver-independent: fold the slot_increments contribution into the shared outcome-aggregation step every backward driver already calls, so the replicated driver cannot omit it, and give the enumerated backward the per-stage allreduce that sync_stage_metadata performs on the sampled path. Road (a) is the smaller and the reversible one and matches the existing precedent; road (b) is the durable one if enumerated-plus-DCS is meant to work. Byte-neutrality bar: road (a) is byte-neutral by construction on every currently-passing configuration, since it only converts a silently-degraded run into a rejected one. Road (b) is NOT byte-neutral in solver statistics, becaus…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (cut selection and the backward decomposition are SDDP geometry that stays in the engine))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *By-node scheduler is warm-start-only*; *Cut pool is append-only; basis matches by slot identity* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** an instance of the owner-elevated retrofitted-variant asymmetry class (live instances CD-013, CD-015, CD-027): the enumerated traversal is the Nth variant on the traversal axis while the sampled original keeps the cut-binding metadata channel bespoke, so under `Traversal::Enumerated` `CutPool::record_binding` never fires and the DCS resident-set seed and the budget eviction key both degrade SILENTLY on a pairing no admission gate rejects — Sev B by the class, A-risk carried by the silence of the divergence (CD-004 marker rule); reviewer proposed B.
- **Reviewer rating:** B — recalibrated UP to B (A-risk); house rating above the reviewer's — see calibrationBasis.
- **Related prior entries:** CD-022 (distinct claims; adjudicated at ingest).
- **Re-raise-of:** PD-004; performance-debt-follow-ups-first-performance-pass-2026-08-18; NOT a re-raise — distinct claim anchored in a file a file-scoped retired item also cites; NOT a re-raise
- **Needs-human (owner gate):** Carried over from the attacker and still decisive: is enumerated traversal combined with dynamic cut selection a SUPPORTED configuration? Road (a) rejects the pairing beside the existing enumerated preconditions in `setup/mod.rs`; road (b) wires the binding contribution plus a per-stage metadata reduction into the enumerated driver. Both are byte-neutral on today's goldens, so only owner intent decides which road. Scope and severity call: `CutPool::enforce_budget`'s `(last_active_iter, active_count)` eviction key is a second production reader that degrades the same way under enumerated traversal, and unlike the DCS seed it changes which cuts are deactivated and therefore the bound. The owner must decide whether that reader belongs inside this item's scope and whether it lifts the severity above B; I did not mint a separate it…

**Performance** — 5 minted

**PD-040 · Sev B · asymmetry · effort S · confidence high**
Only two of the sweep's allocations are removable by the mechanism as filed: `m_block_starts` (cut_selection.rs:359) materialises a stride range that the following .par_iter() could take as a parallel range with no Vec at all, and `eligible` (:344) is call-local and dead at return so it can ride the TrainingSession::cut_selection_state_scratch field the caller already reuses across the same per-node loop; the Vec::new() deactivation/reactivation pair, the two per-fold-task identity buffers, the zero-fill and the run_cut_management no-allocation-contract framing do not survive.

- **Station:** cobre-sddp (sddp, sub-station 5c, lens performance; ingest ref 5c-performance-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/cut/cut_selection.rs::select_for_stage`, `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management`, `crates/cobre-sddp/src/cut/cut_selection.rs::M_BLOCK`
- **Evidence:** select_for_stage allocates eligible, m_block_starts, deactivations and reactivations once per call, plus a populated*M_BLOCK value panel and a populated-long bitmap per rayon fold task and one more bitmap per reduce identity. None of them is workspace-owned, so every one is a fresh heap allocation on each invocation. run_cut_management calls select_for_stage once per interior node per cut-management iteration while its own rustdoc promises no heap allocation once the pools have stopped growing, so the caller's sta… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/cut/cut_selection.rs | sed -n '344,372p;405,421p' && git show 077dbe2c:crates/cobre-sddp/src/training/session/mod.rs |…`
- **Fix-shape:** Move the four per-call vectors and the two per-fold-task buffers onto the per-worker scratch the crate already owns for exactly this purpose, reusing them through clear/resize in the State Struct shape that .claude/architecture-rules.md prescribes for hot-path drivers, and borrow the fold identity from that scratch rather than materialising a fresh vec per task. Drop the zero-fill of the value panel, because gemm_block writes it with beta zero before any read. Byte-neutrality: the rayon fold-and-reduce keeps the same commutative associative OR-merge over the same bitmaps, so the selected slot set, the deactivation list and its order are unchanged, and cut activity therefore stays identical against the parity goldens in tests/parity.rs and tests/common/parity_hash.rs, against the rank-invariance harness in tests/common/permute.rs, and between mpiexec -n 1 and -n 2 because selection is rank-local. .claude/rules/sddp.md section 'Cut pool is append-only; basis matches by slot identity' governs: the fix must not renumber or compact slots, only reuse the scratch that scores them.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (cut selection is SDDP geometry that stays in the engine))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Cut pool is append-only; basis matches by slot identity* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** two removable allocations per call inside the per-node cut-selection value sweep (`m_block_starts`, `eligible`) on a path whose caller documents a no-allocation contract; the other four allocations and the zero-fill are conceded — never-allocate-on-hot-paths rule, structural calibration only, UNMEASURED.
- **Related prior entries:** CD-018, CD-012 (distinct claims; adjudicated at ingest).
- **Re-raise-of:** performance-debt-follow-ups-first-performance-pass-2026-08-18; NOT a re-raise — distinct claim anchored in a file a file-scoped retired item also cites; NOT a re-raise
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: allocation per call and per rayon fold task inside the per-node cut-selection value sweep, plus a redundant zero-fill of a panel the gemm kernel fully overwrites; exercising call sites (cited, not anchored): training/session/mod.rs · run_cut_management (per node per cut-management pass). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Needs-human (owner gate):** Owner call: whether the inaccurate rustdoc sentence at training/session/mod.rs:1026-1027 ('no heap allocation when the cut pools have not grown') gets its own doc-accuracy item — it is false because of run_cut_management's own per_stage / deactivations / record_by_pool allocations, independent of this candidate's mechanism claim, so it is stripped here rather than fixed by it.
- **Queued to:** performance-sweep

**PD-041 · Sev B · asymmetry · effort S · confidence high**
Only the absent contiguous-copy path at the single gather in `training/session/mod.rs` L1112-1121 survives, on the ground that the archive is flat and packed at one global stride so an identity pool's trial row is already the wanted contiguous run. Stripped: the fixShape premise that a dedicated projection identity predicate is mandatory because a reduced pool could match the state dimension in slot count, which cannot arise at the pin since `REGION_ORDER`'s four ranges partition [0, n_state) contiguously and `CutStateProjection::new` walks them ascending with only Storage and Lag gated, making `n_slots == n_global` already equivalent to the identity index vector and strictly better than the rustdoc `storage && inflow_lags` test, which misses an identity pool with an empty lag range. Also stripped: Sev-B, because the gather reads each archive element once while its sole consumer `select_for_stage` reads the same buffer once per populated cut inside `gemm_block`, making the residue a l…

- **Station:** cobre-sddp (sddp, sub-station 5c, lens performance; ingest ref 5c-performance-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management`
- **Evidence:** The gather runs one push per (trial, slot) pair for every interior node, each push routed through global_state_index, an inline lookup into the projection's slot-index table. The comment on the loop names the identity case itself: the projection is the identity when n_slots equals n_global, which is the all-enabled pool, and the projection's own rustdoc in the sibling sub-station states s equals global_state_index(s) in that case. In that case the trial row in global_states is already exactly the wanted contiguous… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/training/session/mod.rs | sed -n '1104,1124p'`
- **Fix-shape:** Give the gather a guarded fast path: when the projection is the identity, extend scratch from the contiguous trial row of global_states in one extend_from_slice, and otherwise keep the indexed loop exactly as it stands. The guard must test the projection's own identity predicate, not the slot count alone, so a reduced pool that happens to have as many slots as the state dimension cannot take the fast path. .claude/rules/sddp.md section 'The cut intercept dots the trial state through the projection, never positionally' is why the guard is mandatory rather than optional: a positional prefix is wrong for a reduced pool, and that section is the pinned reason the slow path must remain the default. Byte-neutrality: on the fast path the same f64 values land in the same buffer positions in the same order, so the value sweep, the selected set and the deactivations are bit-identical; parity goldens, the rank-invariance harness and mpiexec -n 1 versus -n 2 are unaffected.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (cut management is SDDP geometry in the engine))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *The cut intercept dots the trial state through the projection, never positionally* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** the single gather in `run_cut_management` pushes one scalar at a time through the slot-index table although the archive is flat and packed at one global stride, so an identity pool's trial row is already the wanted contiguous run; the mandatory-predicate premise is conceded — structural calibration only, UNMEASURED.
- **Related prior entries:** CD-012 (distinct claims; adjudicated at ingest).
- **Re-raise-of:** performance-debt-follow-ups-first-performance-pass-2026-08-18; NOT a re-raise — distinct claim anchored in a file a file-scoped retired item also cites; NOT a re-raise
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: scalar-at-a-time gather through an indirection table per (node, trial, slot), with no contiguous copy path for the identity projection the loop's own comment names; exercising call sites (cited, not anchored): training/session/mod.rs · run_cut_management L1112-1121 (per node, trial, slot). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Queued to:** performance-sweep

**PD-042 · Sev B · duplication · effort M · confidence high**
Only the MEMORY half survives, and only on the sampled level-driver path when the archive exists (cut selection enabled or export_states) and a declared nodes[] graph puts more than one cut-generating node at a stage: each sibling's NodeStates.data holds its own extend_from_slice copy of an identical level block, removable byte-neutrally by one shared per-level block viewed per node. The value-sweep half does not survive: each interior node owns its own pool and projection, so the per-sibling walk evaluates different cuts rather than recomputing the same values, and the proposed fix leaves n_trials and the per-node projection gather unchanged.

- **Station:** cobre-sddp (sddp, sub-station 5c, lens performance; ingest ref 5c-performance-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::run_one_backward_level`, `crates/cobre-sddp/src/training/visited_states.rs::archive_gathered_states`, `crates/cobre-sddp/src/training/visited_states.rs::append`
- **Evidence:** archive_gathered_states delegates to NodeStates::append, which extend_from_slice copies the entire real_states_buf into that node's own data vector. The loop runs that copy once per node in the level with the same buffer, so on a level with sibling nodes the identical state block is duplicated into every node's bucket every iteration. The code's own comment calls this a conservative over-inclusion. The cost is twofold: archive bytes grow as nodes-per-level times the level's real forward-pass count times the state… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/training/backward_pass_state.rs | sed -n '1560,1574p' && git show 077dbe2c:crates/cobre-sddp/src/training/visited_state…`
- **Fix-shape:** Store the level's gathered block once and give every node in that level a view of it, so states_for_node returns the shared block instead of a per-node copy: the values every reader sees are unchanged, only the duplication disappears. The per-node retention window must be preserved, since trim_to_window currently trims each node's own vector, so a shared block needs a level-keyed trim with the same window arithmetic. .claude/rules/sddp.md section 'Per-level exchange in the backward pass' governs the level's gather and must keep owning it, and section 'Cut pool is append-only; basis matches by slot identity' is untouched because no slot is renumbered. Byte-neutrality: the trial states read back per node are identical in value and order, so cut selection, cut generation and the reported bounds are bit-identical against the parity goldens, the rank-invariance harness and mpiexec -n 1 versus -n 2. Narrowing the archive to each node's OWN routed states is a separate and larger change: it would alter which trial states each pool is scored against and therefore which cuts survive selection…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (the backward decomposition stays in the engine))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Cut pool is append-only; basis matches by slot identity*; *Per-level exchange in the backward pass* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** on the sampled level-driver path with an archive (cut selection enabled or export_states) and a declared multi-node level, every sibling node holds its own copy of one identical level block — dense replication removable by one shared per-level block viewed per node; the value-sweep half is conceded; structural calibration only, UNMEASURED.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: dense replication: the level's state block is copied into every sibling node's archive bucket, so both archive memory and the per-node cut-selection value sweep scale with nodes per level; exercising call sites (cited, not anchored): training/backward_pass_state.rs · run_one_backward_level → training/visited_states.rs · archive_gathered_states (per level per iteration). queued to the performance sweep (perf-queue.json); no number is asserted here. exercised only on a declared multi-node graph under SAMPLED traversal; neither protocol deck (chain sampled 4t; branching enumerated 2t) has that shape — stays UNMEASURED until the sweep adds such a deck.
- **Needs-human (owner gate):** Owner call, needed only to scope the LARGER change and not the claim confirmed here: is the level-wide over-inclusion a required correctness margin, or is narrowing each node's archive to its own routed states licensed as a golden-moving change? Only that narrowing reduces the per-node value sweep; the shared-block fix confirmed above does not depend on the answer.
- **Queued to:** performance-sweep

**PD-043 · Sev C · duplication · effort S · confidence high**
Narrowed to the training path: `build_slot_lookup`'s unconditional clear is a FIXED-length pass over the run-lifetime pool capacity (`warm_start_count + max_iterations * visit_bound`, set once by `ScratchBuffers::new` from `max_pool_capacity`), not over the current pool length, so the redundancy is proportionally largest in the FIRST iterations and shrinks as the pool fills rather than diverging monotonically; the simulation path, whose buffer the `run_stage_solve` resize sizes to `populated()` (the LP's own cut-row count), carries no residue, and the clear is the only capacity-keyed pass in a call that is otherwise already several LP-width passes.

- **Station:** cobre-sddp (sddp, sub-station 5c, lens performance; ingest ref 5c-performance-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/cut/basis_reconstruct.rs::build_slot_lookup`, `crates/cobre-sddp/src/solve/stage_solve.rs::run_stage_solve`
- **Evidence:** fill(None) writes every entry of slot_lookup, and run_stage_solve resizes that buffer up to inputs.pool.populated() before each warm-started solve while noting that the populated count only grows. The scatter that follows writes reconcilable_slots.len() entries. The clear therefore costs the full append-only pool length on every warm-started LP solve while the useful work stays proportional to the reconcilable slot count, and the two diverge monotonically as the pool grows over the training run. Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/cut/basis_reconstruct.rs | sed -n '240,260p' && git show 077dbe2c:crates/cobre-sddp/src/solve/stage_solve.rs | sed -n '…`
- **Fix-shape:** Replace the sentinel clear with a generation stamp: store a solve epoch alongside each position and treat an entry whose epoch is stale as absent, so no bulk reset is needed; the alternative is to record the slots written by the previous call and reset only those. Either way every lookup reconstruct_basis performs returns the same answer it does today. .claude/rules/sddp.md section 'Cut pool is append-only; basis matches by slot identity' governs and is preserved exactly, because the slot-to-position mapping is unchanged and no slot is compacted; section 'A stored basis warm-starts only at its own node (node-tag)' is untouched because the node-tag filter runs before reconstruction. Byte-neutrality: the reconstructed basis bytes are identical, so the parity goldens, the rank-invariance harness and mpiexec -n 1 versus -n 2 all reproduce. The debug assertion that the caller pre-sized the buffer stays; the workspace construction test that asserts every entry is None would assert a zero epoch instead.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (basis reconstruction is SDDP geometry in the engine))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *A stored basis warm-starts only at its own node (node-tag)*; *Cut pool is append-only; basis matches by slot identity* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** a fixed-length clear over the run-lifetime pool capacity on every warm-started training solve where only the reconcilable slots are written; proportionally largest early and shrinking as the pool fills — bounded; structural calibration only, UNMEASURED.
- **Related prior entries:** CD-023 (distinct claims; adjudicated at ingest).
- **Re-raise-of:** organizational-quality-follow-ups-low-priority; NOT a re-raise — distinct claim anchored in a file a file-scoped retired item also cites; NOT a re-raise
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: full-length clear per warm-started LP solve, over a buffer whose length tracks the monotonically growing append-only pool, where the useful scatter touches only the reconcilable slots; exercising call sites (cited, not anchored): solve/stage_solve.rs · run_stage_solve (per warm-started solve). Sev C — not queued; the owner gate may promote it; no number is asserted here.

**PD-044 · Sev B · asymmetry · effort S · confidence high**
On the default ByScenario branch, and only for a worker owning at least one routed trial point, process_stage_backward pays one payload-proportional heap allocation plus a copy in, a copy out and a free per (backward node, worker, iteration) for the owned Vec<StagedCut> it drains into at its return, which the counts-plus-canonical_scatter read-back already used by process_stage_backward_by_node removes; the workspace buffer's own capacity reuse is NOT defeated (drain leaves capacity intact and within-stage pushes stay allocation-free), and the constant-size per-node outer result vector is excluded because the sibling scheduler allocates the same shape plus its own counts vector.

- **Station:** cobre-sddp (sddp, sub-station 5c, lens performance; ingest ref 5c-performance-06)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/training/backward_pass_state.rs::process_stage_backward`, `crates/cobre-sddp/src/training/backward/by_node.rs::process_stage_backward_by_node`
- **Evidence:** The per-worker closure pushes each staged cut into ws.backward_accum.staged_cuts_buf, a SolverWorkspace-owned buffer that exists to be reused, then returns drain(..).collect(), which allocates a brand new owned Vec<StagedCut> while leaving the reused buffer empty. Rayon then collects those into a fresh Vec<Result<..>>. The caller immediately re-extends its own reused state.staged_cuts_buf from the temporaries and drops them, so both allocations are pure intermediates on a path called once per backward node per ray… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/training/backward_pass_state.rs | sed -n '2036,2038p;2164,2170p;1950,1958p' && git show 077dbe2c:crates/cobre-sddp/src/…`
- **Fix-shape:** Adopt the sibling scheduler's return shape: have process_stage_backward return per-worker counts and leave each staged cut in the producing worker's own staged_cuts_buf, then let the caller read them back in ascending worker-then-item order through the shared canonical_scatter primitive and extend state.staged_cuts_buf from borrowed slices, so no per-node vector is allocated. Error propagation is preserved because the by-node path already returns a Result per worker and propagates the first failure by value. Byte-neutrality: the merge already sorts by cut.trial_state_idx, the sole globally unique key, so the order in which cuts are added to the pool is unchanged and remains independent of worker count; .claude/rules/sddp.md section 'Cut pool is append-only; basis matches by slot identity' governs that add order and section 'Joint risk is applied once over the flattened successor x opening vector' is untouched because no aggregation moves. Parity goldens, the rank-invariance harness and mpiexec -n 1 versus -n 2 stay byte-identical.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (the backward decomposition stays in the engine))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Cut pool is append-only; basis matches by slot identity* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** an instance of the retrofitted-variant asymmetry class (CD-013/015/027): the by-node arm already returns per-worker counts read back through `canonical_scatter`, while the original by-scenario arm drains the reused workspace buffer into a fresh owned `Vec<StagedCut>` per (backward node, worker, iteration) — Sev B by the class; structural calibration only, UNMEASURED.
- **Related prior entries:** CD-016 (distinct claims; adjudicated at ingest).
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: allocation per (backward node, rayon worker) at a return boundary: a reused workspace buffer is drained into a fresh owned vector that the caller immediately copies into another reused buffer and drops; exercising call sites (cited, not anchored): training/backward_pass_state.rs · compute_one_backward_node (ByScenario arm, per node per iteration). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Queued to:** performance-sweep

**Over-engineering** — 1 minted

**OD-039 · Sev C · over-parameterization · effort S · confidence high**
The method must not exist while `actual_per_rank` accepts `total_forward_passes` as a parameter (rank_distribution.rs:60) that shadows the `num_total_forward_passes` the same struct already stored in `new` (:53): the vector view can be built from a total other than the one `my_actual_fwd` / `my_fwd_offset` / `max_local_fwd` were derived from (:37-41), with no check rejecting the disagreement. The residue is the parameter, not the method - the deletion half of the title is dropped, since the mirror at :933 declares this method the SDDP-side owner and prescribes routing cut_sync.rs:182 and stats_aggregation.rs:118 through `RankDistribution` rather than the reverse.

- **Station:** cobre-sddp (sddp, sub-station 5c, lens over-engineering; ingest ref 5c-over-engineering-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/training/session/rank_distribution.rs::actual_per_rank`, `crates/cobre-sddp/src/training/session/rank_distribution.rs::RankDistribution`
- **Evidence:** RankDistribution::actual_per_rank is a two-line method whose whole body forwards to cobre_comm::per_rank_counts with one field. It decides nothing: it neither validates, adapts, reorders nor reshapes the result. It has exactly one production caller (session/mod.rs:245) plus one unit test (:210), while the two sibling sites in this same crate that need the identical vector call the free function directly (cut/cut_sync.rs:182, training/forward/stats_aggregation.rs:118), so the codebase already prefers the unwrapped… Re-derive: `git grep -n 'actual_per_rank\|per_rank_counts' 077dbe2c -- crates/cobre-sddp/src crates/cobre-comm/src | grep -v 'state_exchange.rs'; git grep -n 'num_total_fo…`
- **Fix-shape:** Delete the method and let session/mod.rs:245 call cobre_comm::per_rank_counts directly, matching the two sibling sites that already do. If a named owner on the struct is preferred, keep the method but drop the parameter and read self.num_total_forward_passes, so the vector and the per-rank constants cannot disagree by construction. Do not promote the split arithmetic into a new trait or type - cobre_comm::per_rank_counts is already the single owner, and a seam over one consumer would trip conflicts-trigger 4. Byte-neutrality: both variants evaluate the same base/remainder split over the same two inputs, because the sole call site already passes the value that new stored, so the returned Vec<usize> is element-wise identical and the ExchangeBuffers::with_actual_counts sizing it feeds is unchanged. This is rank-distribution arithmetic, so the bar is the one that binds it: mpiexec -n 1 vs -n 2 vs -n 4 must reproduce bit-for-bit, which tests/parity.rs already asserts by comparing 1-rank against 2-rank and 4-rank hashes, plus tests/common/parity_hash.rs and the tests/common/permute.rs ran…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** a parameter that shadows a total the struct already stores, letting the per-rank view be built from a different total than the one its siblings were derived from; the deletion half is dropped because the mirror declares the method the owner of the partition arithmetic.
- **Related prior entries:** OD-009 (distinct claims; adjudicated at ingest).

**Test bloat** — 4 minted

**TD-051 · Sev C · duplication · effort S · confidence high**
Narrowed to one deletion and stripped of its two supporting claims: at the pin sync_cuts_rejects_mismatched_local_cut_count is the weakest of THREE copies of the count-mismatch assertion, not one of two, because the unit test sync_cuts_invariant_rejects_local_mismatch (src/cut/cut_sync.rs:1672) already holds the strongest copy with unreachable! collective bodies pinning that none fires; the byte-identity is confined to the struct plus six-method impl Communicator bodies (22-68 versus 25-71), the enclosing rustdoc differing; the defensible residue is deleting the single solver-linking binary test_mpi_allgatherv_nonuniform_workers.rs, whose remaining test is strictly weaker than both other copies and whose titular nonuniform-workers assertion lives at src/training/backward/tests.rs:5443. Rejected as over-reach: the section 6 count-parity gate cannot certify a deletion that drops the nextest count by one, and the fallback hoist into test_support.rs is unwarranted because tests/common/mod…

- **Station:** cobre-sddp (sddp, sub-station 5c, lens test-bloat; ingest ref 5c-test-bloat-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/test_mpi_allgatherv_nonuniform_workers.rs::sync_cuts_rejects_mismatched_local_cut_count`, `crates/cobre-sddp/tests/test_mpi_allgatherv_nonuniform_workers.rs::StubComm2Rank`, `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs::sync_cuts_invariant_rejected_when_cut_count_mismatches`, `crates/cobre-sddp/tests/test_mpi_sync_cuts_invariant.rs::StubComm2Rank`
- **Evidence:** Two integration binaries, two solver links, one assertion. The nonuniform binary's own module doc states the assertion it is named for has moved out: 'The separate `n_workers_local` uniformity handshake is covered by the unit test `handshake_rejects_nonuniform_workers`.' What remains in it is therefore only the duplicate of the sibling's first test. The sibling additionally carries the real-communicator K-fan rank-invariance gate, which the nonuniform binary does not, so the sibling is the strict superset. Per tes… Re-derive: `git show 077dbe2c:crates/cobre-sddp/tests/test_mpi_allgatherv_nonuniform_workers.rs > /tmp/a.rs; git show 077dbe2c:crates/cobre-sddp/tests/test_mpi_sync_cuts_i…`
- **Fix-shape:** Delete the test_mpi_allgatherv_nonuniform_workers.rs binary and keep the surviving sync_cuts count-mismatch assertion in test_mpi_sync_cuts_invariant.rs, which is a strict superset of it. This is coverage-neutral, not a coverage reduction: the deleted file's titular nonuniform-workers assertion already lives in the handshake_rejects_nonuniform_workers unit test by its own module doc, and its only remaining test is the byte-for-byte same assertion as the survivor's. It removes exactly one integration binary and one static solver link, which is the per-binary cost testing-architecture.md section 7 names. No CI edit is needed: neither file is enumerated by name under .github or scripts, so the binary roster is discovered by cargo. If the owner instead wants both file names retained, then the alternative is to hoist one strict 2-rank Communicator stub into the existing cobre_sddp::test_support surface behind its current cfg(any(test, feature = 'test-support')) gate per testing-architecture.md section 5.2 and have both binaries consume it; a dedicated fixture crate is explicitly out of b…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** one redundant integration binary whose only assertion is the weakest of three copies of the same count-mismatch check (the strongest sits in the unit test with `unreachable!` collective bodies); deleting it is coverage-neutral and removes one solver link — TD-039 precedent (retire a redundant binary), Sev C.
- **Reviewer rating:** B — recalibrated to C because a single binary with one duplicated assertion; TD-039 (redundant clp-only binary) calibrated C; the sequencing against the superseded sync_cuts family is the owner's call.
- **Related prior entries:** Superseded cut-sync public methods (distinct claims; adjudicated at ingest).
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (integration binaries are expensive; one owner per assertion) (td-queue.json).
- **Needs-human (owner gate):** Sequencing call for the cut-pool/training owner: delete the redundant binary now, or let it retire together with the superseded sync_cuts family at the next licensed public-API break, since all three copies of this assertion die with the method.
- **Queued to:** test-corpus

**TD-052 · Sev C · duplication · effort S · confidence high**
Only the three constant-valued helpers duplicate without justification: d01_case_dir at 41, 917 and 1580, d03_case_dir at 908 and 1884, and ascending_stage_end_dates at 54, 1593 and 2745 are fixed deck paths and one pure function of n_pools with no per-group tuning surface, and the d01 copies have already drifted unforced (`.unwrap()` at 41 and 1580 versus `.expect(...)` at 917, with both lint styles allowed file-wide at lines 8-9); each should be declared once at cut_basis.rs file scope. write_test_checkpoint and build_setup are excluded: the former is the assertion-feeding fixture whose baked metadata constants are per-site pinning covered by the module doc at lines 3-5 and the Layer-1 fixtures-untouched invariant, and the latter genuinely diverges in return type at 113 versus 1652. The claim that the wired `mod common` seam is the unused sharing route does not survive, since that seam is used at seven sites and its build_setup_for_case drives from_broadcast_params with patched scal…

- **Station:** cobre-sddp (sddp, sub-station 5c, lens test-bloat; ingest ref 5c-test-bloat-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/cut_basis.rs::write_test_checkpoint`, `crates/cobre-sddp/tests/cut_basis.rs::d01_case_dir`, `crates/cobre-sddp/tests/cut_basis.rs::d03_case_dir`, `crates/cobre-sddp/tests/cut_basis.rs::ascending_stage_end_dates`, `crates/cobre-sddp/tests/cut_basis.rs::build_setup`, `crates/cobre-sddp/tests/common/mod.rs::build_setup_for_case`
- **Evidence:** Every copy is in the same compilation unit as every other, so this duplication buys nothing at all; a single file-scope declaration is visible to all nine inline modules. The binary already wires `mod common;`, so the sharing seam is present and unused for these five helpers. Two of the five are genuinely divergent in signature and must stay distinguishable: build_setup returns a tuple at one site and a bare StudySetup at the other. write_test_checkpoint is NOT the same fixture as test_support::write_synthetic_bou… Re-derive: `git grep -nE '^ fn (write_test_checkpoint|d01_case_dir|d03_case_dir|ascending_stage_end_dates|build_setup)' 077dbe2c -- crates/cobre-sddp/tests/cut_basis.rs; t…`
- **Fix-shape:** Declare each of the five helpers once at cut_basis.rs file scope, above the nine inline modules that consume them, since all copies already share one compilation unit and no cfg gate separates them. Keep the two genuinely divergent variants distinguishable rather than forcing one signature: the tuple-returning setup builder and the bare-StudySetup one differ in return type and both have callers, so give the second a distinct name or have it delegate to the first and discard the System. Do not collapse the multi-pool checkpoint writer into test_support::write_synthetic_boundary, which writes a single-cut boundary checkpoint and is a different fixture. Only if a second binary later needs these helpers does ownership move to tests/common; a new fixture crate is out of bounds per testing-architecture.md section 5.2. Byte-neutrality bar: the change is confined to one file under crates/cobre-sddp/tests, so the parity goldens in tests/parity.rs and tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and mpiexec -n 1 versus -n 2 are all untouched. The reloca…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** three constant-valued helpers declared two and three times inside one binary, one pair already drifted unforced (`.unwrap()` vs `.expect(...)`); a file-scope declaration each removes them — TD-035 scale.
- **Reviewer rating:** B — recalibrated to C because three constant helpers inside one binary; the two divergent setup builders are conceded as genuine variants.
- **Yardstick:** docs/design/testing-architecture.md §5.1 canonical per-crate layout (one declaration per helper per binary) (td-queue.json).
- **Queued to:** test-corpus

**TD-053 · Sev B · duplication · effort M · confidence high**
Narrowed to three bodies and no destination: session/mod.rs:1654 StubComm, training/training/tests.rs:173 StubComm and session/mod.rs:2767 Rank0Of2 are token-identical re-declarations of tests/common/mod.rs:32 and :86 that no src unit test can reach because cobre-sddp's EXPORTED test-support surface holds no Communicator double, so one shared pair is warranted. Excluded from the claim: the 'zero impl Communicator in test_support.rs' premise (a seventh token-identical copy sits at src/test_support.rs:3555-3598, private to mod trunk_fan_tests); the fix-shape's destination (testing-architecture.md section 5.2 assigns StubComm/Rank0Of2 to cobre-comm, not cobre-sddp/src/test_support.rs, so the home is an open owner call); the other three src-side bodies, which are strictness variants to preserve rather than fold (backward/tests.rs:233's unreachable broadcast AND the full-slice allreduce equal-length assertion that backward_pass_state.rs:2216 and :2258 drop); and all four cut/cut_sync.rs du…

- **Station:** cobre-sddp (sddp, sub-station 5c, lens test-bloat; ingest ref 5c-test-bloat-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/common/mod.rs::StubComm`, `crates/cobre-sddp/tests/common/mod.rs::Rank0Of2`, `crates/cobre-sddp/src/training/backward/tests.rs::StubComm`, `crates/cobre-sddp/src/training/backward_pass_state.rs::StubComm`, `crates/cobre-sddp/src/training/session/mod.rs::StubComm`, `crates/cobre-sddp/src/training/training/tests.rs::StubComm`, `crates/cobre-sddp/src/training/backward_pass_state.rs::Rank0Of2`, `crates/cobre-sddp/src/training/session/mod.rs::Rank0Of2`, `crates/cobre-sddp/src/cut/cut_sync.rs::ThreeRankComm`, `crates/cobre-sddp/src/cut/cut_sync.rs::TwoRankStubComm`
- **Evidence:** tests/common is only reachable from crates/cobre-sddp/tests, whereas crates/cobre-sddp/src/test_support.rs is declared at src/lib.rs under cfg(any(test, feature = 'test-support')) and is therefore reachable from BOTH src unit tests and integration binaries. It is the sanctioned shared-fixture home named by testing-architecture.md section 5.2, yet it carries no Communicator double at all, which is exactly why the src side re-rolls one. The two verbatim same-file duplicates inside cut/cut_sync.rs are duplication wit… Re-derive: `git grep -nE '^\s*(pub )?struct (StubComm|Rank0Of2|ThreeRankComm|TwoRankStubComm)\b' 077dbe2c -- crates/cobre-sddp/src crates/cobre-sddp/tests; git grep -nE 'i…`
- **Fix-shape:** Move one strict single-rank Communicator double and one rank-0-of-2 double into crates/cobre-sddp/src/test_support.rs behind the cfg(any(test, feature = 'test-support')) gate it already carries, then have the src-side unit tests and tests/common/mod.rs consume that one pair instead of re-declaring it; collapse the two verbatim same-file stub pairs in cut/cut_sync.rs to one declaration each. The consolidation must preserve per-site strictness rather than average it: at least one existing copy implements broadcast as unreachable with the message that broadcast is not used in backward pass tests, and rewriting that to Ok(()) would silently weaken a live assertion, so the shared double keeps the strict body and any site that genuinely needs a permissive broadcast declares that permissiveness explicitly at the call site. Leave FailingComm, FailingBcastComm, NonUniformStubComm, Rank1Of2, StubCommN, LbReconcileStub, ReconcileStub, Rank0Of2Outcome, Rank0Of2Preserve and DualRankStubComm alone; each encodes a different failure or outcome shape and is not a duplicate. Do not introduce a dedica…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals; the double is an L0 communicator contract))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** three token-identical `StubComm` / `Rank0Of2` re-declarations in src unit tests that cannot reach the tests/common owner because the crate's EXPORTED test-support surface holds no Communicator double — a fixture with no shared owner (TD-037 precedent), Sev B; the destination (cobre-comm test-support per §5.2 vs the §6 phase-3 fold) is the test-corpus station's E08-2 call.
- **Cross-station:** E08-2 needs-human (StubComm/Rank0Of2 duplication: tests/common/mod.rs + private copies) and E04 TD-037 (tests/common vs test-support gate) — the src-side half is this station's; tests/common is the test-corpus station's.
- **Yardstick:** docs/design/testing-architecture.md §5.2 uniform test-support feature convention / §5.8 (StubComm and Rank0Of2 are the kept harness doubles) (td-queue.json).
- **Needs-human (owner gate):** Destination owner call for the shared StubComm/Rank0Of2 pair: testing-architecture.md section 5.2 assigns them to cobre-comm, which has no test-support feature at the pin, while section 6 phase 3 collapses tests/common into cobre-sddp's test-support surface; and Rank0Of2's load-bearing forward_passes == 1 caveat names cobre-sddp's RankDistribution, which the infrastructure-genericity hard rule would keep out of cobr…
- **Queued to:** test-corpus

**TD-054 · Sev B · duplication · effort M · confidence high**
Only the inert subset of the eleven required methods is genuine duplication across the eight sites — the ActiveProfile associated type, the empty apply_profile, the constant solver_name_version returning the literal MockSolver 0.0.0 and the no-op set_row_bounds (identical at all eight, differing only in ignored parameter names), plus the SolverStatistics::default() statistics/statistics_into pair at six of the eight — and it is that inert restatement alone that makes a new required trait method an eight-site edit here. The behavioral bodies are not duplication and do not reduce to one or two knobs: solve is four structurally distinct modes, get_basis splits seven-to-one at backward_pass_state.rs:2363, statistics is three-way, and two sites carry live strictness a shared default would erase (the copy_from_slice length check at forward_pass_state.rs:1138 and the set_col_bounds_calls == 2 * n_openings NCS per-opening assertion at lower_bound.rs:2093). The 'no solver double in test_suppor…

- **Station:** cobre-sddp (sddp, sub-station 5c, lens test-bloat; ingest ref 5c-test-bloat-03)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/training/backward/tests.rs::MockSolver`, `crates/cobre-sddp/src/training/backward_pass_state.rs::MockSolver`, `crates/cobre-sddp/src/training/forward/tests.rs::MockSolver`, `crates/cobre-sddp/src/training/forward_pass_state.rs::MockSolver`, `crates/cobre-sddp/src/training/lower_bound.rs::MockSolver`, `crates/cobre-sddp/src/training/session/mod.rs::MockSolver`, `crates/cobre-sddp/src/training/training/tests.rs::MockSolver`, `crates/cobre-sddp/src/workspace/workspace.rs::MockSolver`
- **Evidence:** One name, eight bodies, 381 lines of boilerplate whose only purpose is to satisfy the 11 required methods so a test can vary a single return. Adding a required method to SolverInterface is therefore an eight-site edit inside this crate alone. The shared home already exists and is reachable from src unit tests: crates/cobre-sddp/src/test_support.rs is gated cfg(any(test, feature = 'test-support')), but it holds no solver double. Genuinely distinct doubles that are not part of this claim, because each records or fai… Re-derive: `git grep -nE '^\s*(pub )?struct MockSolver' 077dbe2c -- crates/cobre-sddp/src/cut crates/cobre-sddp/src/training crates/cobre-sddp/src/solve crates/cobre-sddp/…`
- **Fix-shape:** Put one configurable solver double in crates/cobre-sddp/src/test_support.rs behind the cfg(any(test, feature = 'test-support')) gate it already carries, implementing the 11 required SolverInterface methods once, and let each of the eight sites configure the one or two behaviors it actually varies instead of restating the whole surface. Keep it a plain struct with fields, not a trait-object or closure-per-method design, so the crate-wide ban on Box<dyn Trait> is respected and the double stays enum-or-field dispatched. Preserve each site's current strictness exactly: where a body panics or asserts on an unexpected call today, the shared double must keep a mode that still panics there rather than returning a default, because turning a panic into a silent success removes a live assertion. Leave the genuinely distinct recording and probing doubles alone, specifically the two-phase mock in cut/dcs.rs, the recording mock in the row builder, the recording solver in training/stage_solve_prep/tests.rs and the per-child probe solver in training/backward/tests.rs. Byte-neutrality bar: the chang…
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** eight same-named `MockSolver` doubles restate the inert subset of the eleven required `SolverInterface` methods identically, so a new required trait method is an eight-site edit — the TD-037 ripple, Sev B; the strict per-site behaviours are conceded as genuine and must be preserved by any consolidation.
- **Yardstick:** docs/design/testing-architecture.md §5.2 uniform test-support feature convention (one configurable double on the crate's test-support surface; no `Box<dyn Trait>`) (td-queue.json).
- **Needs-human (owner gate):** Owner call on the shape of the inert-only consolidation: a cfg-gated declarative macro in test_support.rs versus the candidate's field-configured shared struct, since a single struct cannot express the seven-to-one get_basis split and the two strict sites without adding modes no test reads.
- **Queued to:** test-corpus

#### 5d — simulation + production + support

**Architecture** — 2 minted

**CD-087 · Sev C · asymmetry · effort M · confidence high**
Narrowed to the internal-import half of the alias block at lib.rs:56-99: the crate-root aliases give each cluster submodule a second, cluster-blind in-crate path, and internal code mixes both spellings inside a single use crate tree (workspace/ as context plus workspace at simulation/enumerated.rs:27 and :43; solve/ as crate::solver_phase at simulation/state.rs:33 plus solve::partition at :50; cut/ as cut plus cut_sync at training/backward_pass_state.rs:33-34), including inside one cluster where production/hydro_models/production.rs pairs super::types at 27 with crate::energy_conversion at 32 and crate::fpha_fitting at 35 and types.rs:27 repeats it. Conceded and stripped: no alias is dead, so the five-zero-use finding and the delete-outright half of the fix do not survive (four are reached from braced use crate trees, and orchestration is load-bearing for cobre-cli, cobre-python, the integration corpus and the literal string the python-parity gate matches); and the cluster-blindness i…

- **Station:** cobre-sddp (sddp, sub-station 5d, lens architecture; ingest ref 5d-architecture-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/lib.rs::lp`, `crates/cobre-sddp/src/lib.rs::production`, `crates/cobre-sddp/src/lib.rs::workspace`, `crates/cobre-sddp/src/lib.rs::solve`, `crates/cobre-sddp/src/simulation/pipeline.rs::solve_simulation_stage`, `crates/cobre-sddp/src/simulation/enumerated.rs::run_enumerated_simulation`, `crates/cobre-sddp/src/simulation/extraction.rs::extract_stage_result`, `crates/cobre-sddp/src/production/hydro_models/production.rs::resolve_production_models_from_artifacts`, `crates/cobre-sddp/src/production/hydro_models/mod.rs::prepare_hydro_models`
- **Evidence:** The crate has two path namespaces for the same modules and internal code standardizes on the flat one, so a cluster boundary the directory tree was created to express is invisible at the import site. The sharpest form is one file reaching one cluster twice by different names: simulation/enumerated.rs names workspace::context as 'context' at line 27 and workspace::workspace as 'workspace' at line 43 in the SAME use tree, and simulation/state.rs does the same for solve, naming solve::solver_phase as 'solver_phase' a… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/lib.rs | sed -n '52,99p' ; for f in simulation/enumerated.rs simulation/state.rs simulation/pipeline.rs; do git show 07…`
- **Fix-shape:** Separate the two roles the alias layer currently conflates. Keep the curated public re-export surface in lib.rs, which is what downstream and the published API depend on, and retire the crate-root module aliases as an INTERNAL import path, rewriting internal call sites to name the owning cluster, so an import reads production::energy_conversion, lp::indexer, lp::builder, stochastic::noise, solve::stage_solve and workspace::context. Delete outright the five aliases with zero internal uses. Do the lp aliases first and in isolation, because lp::builder and lp::indexer carry the largest internal reference counts and are exactly the surface the cobre-model carve-out has to move, so every remaining crate::lp_builder or crate::indexer import is a site that carve-out must touch blind. Byte-neutrality bar: import-path-only change with no item moved, so cargo fmt --all --check plus the parity goldens in crates/cobre-sddp/tests/parity.rs and the rank-invariance harness in tests/common/permute.rs are sufficient; no .claude/rules/sddp.md contract is touched.
- **Alignment:** advances-0b (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.2 (the lp/ aliases are exactly the surface the cobre-model carve-out moves))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** the crate-root alias block gives each cluster submodule a second, cluster-blind in-crate path and internal code mixes both spellings inside single `use crate` trees — an import-path organisational asymmetry with no type or behaviour change; CD-021 precedent (module-shape asymmetry against the directory-module convention), Sev C; the public half of the block is the owner's call.
- **Reviewer rating:** B — recalibrated to C because import-path spelling only: no type, behaviour or dependency changes, and the public re-export surface is untouched — the CD-021 organisational precedent is C.
- **Needs-human (owner gate):** Whether the public half of the alias block is also retired is an owner call: lib.rs:23-26 declares the pub mod namespaces non-semver-stable and the nested paths already resolve, but removing pub use policy::orchestration would touch cobre-cli, cobre-python, the integration corpus and the literal match in scripts/ci/check_python_parity.py.
- **Queued to:** alignment

**CD-088 · Sev C · leaky-boundary · effort S · confidence high**
Narrowed to a documentation-and-organization misfit: production/mod.rs:3-6 states a cluster purpose that does not cover production/conversion.rs, whose sole source domain is simulation/types.rs, making it the one engine-to-cobre_io projection in cobre-sddp homed outside the domain that owns its source types (unlike training/training_output.rs, policy/policy_export.rs, production/hydro_models/export.rs, fixed_delivery_echo.rs and generic_constraint_echo.rs). Conceded and dropped from the claim: the file breaks no layering or dependency rule (production/ imports cobre_io elsewhere), costs nothing at any call site or to Python parity (both front ends reach it by From coherence without naming the module), and the doc sentence is judged as written rather than as an amendment.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens architecture; ingest ref 5d-architecture-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/production/conversion.rs::IntoWriteRecord`, `crates/cobre-sddp/src/production/conversion.rs::with_node`, `crates/cobre-sddp/src/production/mod.rs::conversion`, `crates/cobre-sddp/src/simulation/types.rs::SimulationStageResult`
- **Evidence:** The file has no production-modeling content and no reference to any sibling in its own cluster: it reads simulation/types.rs and writes cobre_io::output::simulation_writer, nothing else. Because its entire surface is trait impls, callers reach it by type inference and no call site ever names production::conversion, so its home is invisible and therefore arbitrary. The evidence that the placement is wrong is written in production/mod.rs itself: the cluster doc states a single coherent purpose, reservoir geometry an… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/production/mod.rs ; git show 077dbe2c:crates/cobre-sddp/src/production/conversion.rs | grep -n -E '^use |^impl |^trait…`
- **Fix-shape:** Re-home the file beside the types it converts, as a simulation-owned write-payload projection module, so the production cluster doc states one purpose again and the cluster boundary is not carrying an unrelated member; the private IntoWriteRecord trait and with_node helper move with it unchanged. Governing contract: .claude/rules/sddp.md section Terminal boundary FCF is booked in the reported total cost governs the cost fields these records carry, so the move must stay field-for-field with no re-derivation of any cost component, and the Python-parity hard rule means the CLI and cobre-python write paths must continue to reach the same conversion. Byte-neutrality bar: a pure move of trait impls with no signature change, verified by the parity goldens in crates/cobre-sddp/tests/parity.rs with tests/common/parity_hash.rs, which hash the written simulation output. Defer the decision on whether the destination is the engine or the Phase 0a cobre-io output orchestration to the owner, since one target makes the other move redundant.
- **Alignment:** advances-0a (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part V §V.1 Phase 0 (shared output orchestration in cobre-io) / Part IV §IV.1)
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Contract:** .claude/rules/sddp.md — *Terminal boundary FCF is booked in the reported total cost* — the contract is the reason the current shape is correct; the fix-shape preserves it.
- **Calibration:** a documentation-and-organisation misfit: one engine-to-cobre_io projection homed in the production cluster whose module doc does not cover it, while every sibling projection sits beside its source types; the destination (intra-engine move vs the Phase-0a shared output orchestration) decides the hint.
- **Needs-human (owner gate):** Destination is an owner call and decides the hint: re-home inside the engine beside simulation/types.rs (a pure local cleanup, hint degrades to neutral) or leave it until Phase 0a's shared output orchestration in cobre-io absorbs the engine-to-io projection wholesale (target-layering-brief.md section 2, 0a bullet), which keeps advances-0a but makes the intra-engine move wasted work.
- **Queued to:** alignment

**Performance** — 4 minted

**PD-045 · Sev B · duplication · effort S · confidence high**
Narrowed to the `row_scale` non-empty branch (every production template after `postprocess_templates`): the removable per-(scenario, stage) work in `build_row_lower_unscaled` is the element-wise division and its length/zero guard, not the buffer fill (the candidate's copy keeps a full-length pass over a capacity-retaining workspace buffer), and only the load-balance span of the result is ever read (simulation/extraction.rs:1420), so the defensible residue is a per-stage cache of that span's unscaled values under absolute row indexing, not of the whole row vector.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens performance; ingest ref 5d-performance-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/simulation/pipeline.rs::build_row_lower_unscaled`, `crates/cobre-sddp/src/simulation/pipeline.rs::extract_sim_stage_result`
- **Evidence:** build_row_lower_unscaled divides every element of template_row_lower by the matching row_scale entry. Both scaling inputs at the only production call site are ctx.template(t).row_lower and ctx.template(t).row_scale, per-stage constants read off StageContext. The one scenario-dependent input is load_rhs_buf, which overwrites the n_load_buses * n_blks load-balance span after the pass. The call sits inside extract_sim_stage_result, which solve_simulation_stage invokes once per (scenario, stage), so the whole row vect… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/simulation/pipeline.rs | sed -n '223,247p;664,673p'`
- **Fix-shape:** Compute the unscaled row_lower once per stage and, on each call, copy the cached vector into the scratch buffer and overwrite only the load-balance span that actually varies per scenario. The cache is per-stage read-only data, so it belongs on StageContext, or on SimulationState scratch if built lazily per worker, per the context-struct decision tree in .claude/architecture-rules.md; keep build_row_lower_unscaled within its recorded argument budget. Byte-neutral because each element's value is the same quotient val / scale computed from the same operands, division is deterministic, and the row order is untouched: the parity goldens in tests/parity.rs with tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs (the cache is keyed by stage index, never by entity order), and mpiexec -n 1 against -n 2 all see identical bytes since the cache derives from already-broadcast templates.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** an element-wise divide over the entire per-stage `row_lower` template repeated on every (scenario, stage) simulation solve although both operands are per-stage constants and only the load-balance span is ever read — the residue is a per-stage cache of that span; structural calibration only, UNMEASURED.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: Redundant pass: an element-wise divide over the entire per-stage row_lower template is repeated on every (scenario, stage) solve, although both scaling operands are per-stage constants and only the load-balance rows differ per scenario.; exercising call sites (cited, not anchored): simulation/pipeline.rs · extract_sim_stage_result ← solve_simulation_stage (per scenario per stage). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Needs-human (owner gate):** Owner picks where the per-stage cache lives (shared `StageContext` vs per-worker simulation scratch); the span-only residue shrinks the memory side of that trade from a full row vector to the load-balance rows.
- **Queued to:** performance-sweep

**PD-046 · Sev C · asymmetry · effort S · confidence high**
Narrowed to a reservation asymmetry with a logarithmic residue: the three per-block branches (extract_hydros' turbine branch, extract_exchanges, extract_buses) and, only when state.max_par_order is positive, extract_stub_collections' inflow_lags grow their result vector by amortized doubling from a capacity seeded at one entity's block group, because Vec's nested from_iter reads the FlatMap size_hint after pulling the first element and the live frontiter then reports that group; the cost is one reallocation per doubling with a wide-record prefix copy, not the zero-capacity cold start the title's 'no reservation' implies, and the fix removes only the redundant regrowth, so the item is an asymmetry against the five sibling functions that already reserve the exact product, not a breach of the no-allocation-on-hot-paths rule that Vec::with_capacity per call would equally violate.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens performance; ingest ref 5d-performance-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/simulation/extraction.rs::extract_hydros`, `crates/cobre-sddp/src/simulation/extraction.rs::extract_exchanges`, `crates/cobre-sddp/src/simulation/extraction.rs::extract_buses`, `crates/cobre-sddp/src/simulation/extraction.rs::extract_stub_collections`
- **Evidence:** The four flat_map collect sites fall inside extract_hydros (declared at 1167, per-block branch), extract_exchanges (1314), extract_buses (1367) and extract_stub_collections (2043). Their output lengths are known before the walk: hydro_ids.len() * n_blks, line_ids.len() * n_blks, bus_ids.len() * n_blks, and hydro_ids.len() * state.max_par_order. A FlatMap reports a zero size_hint lower bound before consumption and is not TrustedLen, so collect starts at zero capacity and grows by repeated reallocation. The pre-rese… Re-derive: `git grep -n "flat_map" 077dbe2c -- crates/cobre-sddp/src/simulation/extraction.rs; git grep -n "Vec::with_capacity" 077dbe2c -- crates/cobre-sddp/src/simulatio…`
- **Fix-shape:** Build each of the four vectors with Vec::with_capacity over the same product its pre-reserving sibling already reserves and extend from the flat_map, or drive the nested loop directly in the shape extract_thermals uses. Byte-neutral by construction: capacity is not observable in the output, push order is unchanged, so the emitted row order and every parity golden hash are identical, the rank-invariance harness in tests/common/permute.rs is unaffected, and MPI reproduction is unaffected because each worker extracts only its own scenarios. This also brings the four sites into line with the crate hard rule against allocating on hot paths.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** a reservation asymmetry with a logarithmic residue: three per-block extraction vectors (four when PAR lags are present) grow by amortised doubling from a first-group seed while six siblings pre-reserve the same product — a bounded number of reallocations per (scenario, stage); structural calibration only, UNMEASURED.
- **Reviewer rating:** B — recalibrated to C because the residue is O(log n) reallocations per vector per (scenario, stage), a reservation asymmetry rather than an unbounded allocation pattern; bounded to one file.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: Allocation per call: collect over a FlatMap begins at zero capacity and grows by reallocation, so each of these per-(scenario, stage) result vectors performs a sequence of reallocations and copies where one exact allocation would do.; exercising call sites (cited, not anchored): simulation/extraction.rs · extract_hydros / extract_exchanges / extract_buses / extract_stub_collections (per scenario per stage). Sev C — not queued; the owner gate may promote it; no number is asserted here.

**PD-047 · Sev B · duplication · effort M · confidence high**
The avoidable copy in re_expand is one deep clone per owned distinct arena node, not one per visiting leaf path: dispatch_scenario_result takes the stage-result Vec by value and moves it into the channel, so each path must own its materialized rows and only a node's terminal visit can become a move. The bookkeeping-free subset is the final-stage clone at enumerated.rs:401, where paths.leaf is injective over the single-predecessor tree so a path's own leaf result has exactly one reader. The title's immutable-borrow causation does not survive, because the caller already holds the scratch mutably at enumerated.rs:494-499.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens performance; ingest ref 5d-performance-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/simulation/enumerated.rs::re_expand`, `crates/cobre-sddp/src/simulation/types.rs::SimulationStageResult`
- **Evidence:** re_expand receives scratch as an immutable &EnumeratedSimScratch, so no visitor can move a captured result out; its only consumer of a captured node is stage_results.push(visit.result.clone()). SimulationStageResult declares thirteen owned Vec fields (costs, hydros, hydro_bus_generation, thermals, exchanges, buses, pumping_stations, contracts, non_controllables, inflow_lags, transit_buckets, generic_violations, anticipated_lanes), so each clone is a thirteen-vector deep copy of a result the census sweep already so… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/simulation/enumerated.rs | sed -n '367,374p;398,402p'; git show 077dbe2c:crates/cobre-sddp/src/simulation/types.rs | se…`
- **Fix-shape:** Take the scratch as &mut and move each node's captured result out on its last visiting path, cloning only for the earlier visitors; under the canonical enumeration the leaf paths beneath a node are contiguous, so the last visitor is identifiable from the walk itself without extra bookkeeping. Do not convert the field to a shared handle: an Arc inside SimulationStageResult would drag the serde rc feature into the output types. Byte-neutral because the value pushed is identical whether cloned or moved and the push order into stage_results is untouched, so parity goldens and the rank-invariance harness observe the same bytes; MPI reproduction is unaffected since re_expand runs per worker over its own scenario range.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals; enumerated orchestration))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** one deep clone of a thirteen-vector stage result per owned distinct arena node in the enumerated re-expansion (not per leaf path as titled), with the final-stage clone the bookkeeping-free subset; dense reification on the enumerated simulation path; structural calibration only, UNMEASURED.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires enumerated; mechanism: Dense reification: an already-captured per-node result carrying thirteen owned entity vectors is deep-copied once per leaf path that visits the node, instead of being moved out on its final visit.; exercising call sites (cited, not anchored): simulation/enumerated.rs · run_enumerated_simulation → re_expand (per owned arena node per sweep). queued to the performance sweep (perf-queue.json); no number is asserted here.
- **Queued to:** performance-sweep

**PD-048 · Sev C · duplication · effort S · confidence high**
Narrowed to a one-time study-setup cost whose SIZE, not its once-per-plant recurrence, is the defect: each `long_term_mean_inflow` call is sized by the whole shared inflow-history table while it retains only one plant's rows, so setup scan work grows as computed-FPHA plant count times total history rows instead of total history rows once. Only plants reaching the `ComputedFromGeometry` branch at production.rs:289 pay it, so a precomputed-FPHA or constant-productivity case scans nothing, and no per-solve, per-iteration or per-stage path is affected.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens performance; ingest ref 5d-performance-05)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/production/hydro_models/production.rs::long_term_mean_inflow`, `crates/cobre-sddp/src/production/hydro_models/production.rs::fit_one_hydro`
- **Evidence:** long_term_mean_inflow iterates system.inflow_history() in full and keeps only the rows matching one hydro_id. Its call site is inside fit_one_hydro on the ComputedFromGeometry branch, and fit_one_hydro is the body of system.hydros().par_iter().map(...), so the whole history table is scanned once per hydro that computes planes. Every other per-hydro row table that fit_one_hydro consumes is pre-grouped exactly once before the parallel map and passed in as a map: config_map, geometry_map built by build_geometry_map,… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/production/hydro_models/production.rs | sed -n '390,398p;286,290p;161,166p'`
- **Fix-shape:** Accumulate per-hydro sums and counts in one pass over inflow_history() before the parallel map, into a container indexed by the hydro's canonical declaration position, and pass the resolved mean into fit_one_hydro the way the other pre-grouped maps are already passed. This preserves the determinism note on the function: a single sequential pass in stored canonical order adds each hydro's rows in exactly the order the current per-hydro scan does, no add is reordered and no partitioned reduction is introduced, so every mean is bit-identical. Declaration-order invariance holds because the accumulator is indexed by declaration position rather than by first appearance in the history table. Guarded by the parity goldens over the exported FPHA planes plus the rank-invariance harness.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** a study-setup cost whose SIZE scales as computed-FPHA plant count × total history rows (each per-plant scan reads the whole shared table) but which runs once per run and only for plants on the `ComputedFromGeometry` branch — setup-time recomputation is acceptable by project rule, recorded for the size scaling only; structural calibration, UNMEASURED and not queued.
- **Reviewer rating:** B — recalibrated to C because once per run at study setup, never per solve or per iteration; the project rule accepts setup-time recomputation — the rule-11 scope question (superlinear setup cost) goes to the owner gate.
- **Measurement:** UNMEASURED — claim-type single-process, layout `4t`, requires none; mechanism: Redundant pass: a full linear scan of the inflow-history table filtered to a single hydro_id, repeated once per computed-FPHA hydro, where one grouped pass before the parallel map would serve all of them.; exercising call sites (cited, not anchored): production/hydro_models/production.rs · fit_one_hydro (once per computed-FPHA hydro at setup). Sev C — not queued; the owner gate may promote it; no number is asserted here.
- **Needs-human (owner gate):** Rule-11 scope call the owner may want to settle once for all similar candidates: does the setup-time per-plant carve-out cover a per-plant recomputation whose cost is sized by a whole shared table (superlinear in plant count), or only one sized by the plant's own data? I read it as the latter and narrowed accordingly.

**Over-engineering** — 3 minted

**OD-040 · Sev B · missing-seam · effort S · confidence high**
Only 4 of the 42 bare openers survive, and not as a comment gap: at the four `#[allow(clippy::cast_sign_loss)]` sites in `resolve_fitting_bounds` (`geometry.rs:157-164`) the suppressed sign loss is unmitigated, because no non-negativity guarantee for `volume_discretization_points` and its three siblings exists in cobre-io's raw struct, in `validate_model_fields`, or in the exported schema, and the `< 2` / `< 1` guards test the already-cast `usize`, so a declared negative count wraps past every guard into `build_grid` instead of raising `InsufficientDiscretization`. The 42-site rationale sweep and the census-premise refutation do not survive: comments.md D4 and the E4 gate's `IN_SCOPE_LINTS` both exclude numeric-cast lints, so the mirror sentence is a prose overclaim, not 42 sites of code debt.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens over-engineering; ingest ref 5d-over-engineering-00)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/geometry.rs::resolve_fitting_bounds`, `crates/cobre-sddp/src/production/fpha_fitting/grid.rs::build_grid`, `crates/cobre-sddp/src/production/fpha_fitting/secant.rs::fit_gamma_s`, `crates/cobre-sddp/src/simulation/aggregation.rs::aggregate_simulation`, `crates/cobre-sddp/src/simulation/extraction.rs::extract_hydro_per_block`, `crates/cobre-sddp/src/simulation/state.rs::run_worker_scenarios`
- **Evidence:** The mirror's census clearing rests on a stated premise: every numeric-cast allow on production code carries a `// Rationale:` naming the non-obvious choice. Over the 5d manifest that premise does not hold for 42 of 69 production-region openers, which carry no comment of any kind. All 42 are numeric-cast lints (cast_possible_truncation 30, cast_sign_loss 6, cast_precision_loss 6) spread over 11 files. This is not the settled "this allow exists" claim the do-not-touch entry clears, and it is disjoint from OD-009: ze… Re-derive: `python3 - <<'EOF' import re,subprocess files=subprocess.run(['git','ls-tree','-r','--name-only','077dbe2c','crates/cobre-sddp/src/'],capture_output=True,text=T…`
- **Fix-shape:** Sharpen, do not sweep. For each of the 42 bare openers either narrow the attribute to the expression it guards or add the single clause that records the fact living outside the file, which for the fpha_fitting cluster is the non-negativity validated on the cobre-io config fields. Never delete an allow that CI's zero-warning bar still needs. Under the comment-discipline Deletion Test the clause qualifies precisely because the justifying fact is cross-crate. The work is comment-and-attribute-only and must be byte-neutral: assert the parity goldens in tests/parity.rs through tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and mpiexec -n 1 versus -n 2 reproduction, and move no golden. Blast radius is 11 files under production/fpha_fitting/, production/hydro_models/ and simulation/, with no signature or control-flow change. Touches no pinned contract in .claude/rules/sddp.md, so no section citation applies.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L1 cobre-io owns input validation; the L3 consumer trusts it))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** of 42 bare `#[allow]` openers only the four `cast_sign_loss` sites in `resolve_fitting_bounds` survive, and not as a comment gap: no non-negativity guarantee for the four discretisation counts exists in cobre-io's raw struct, `validate_model_fields` or the schema, and the `< 2` / `< 1` guards test the already-cast `usize`, so a declared negative count wraps past every guard — a real input-boundary hazard confined to one fn, Sev B; the residue is a cobre-io validation gap routed to reconciliation (E11) for re-filing against the owning crate.
- **Needs-human (owner gate):** Doc-owner call: the mirror sentence at reserved-seams-and-deferred-debt.md:1278-1284 asserts every numeric-cast allow on production code carries a `// Rationale:`, which comments.md D4 and the E4 gate's IN_SCOPE_LINTS both contradict; decide whether that sentence is corrected down to the D4 scope or D4 is widened to cover cast_* lints. Routing call: the surviving residue is a missing input-boundary validation owned by cobre-io, not an over-engineering item in cobre-sddp; decide whether it stays on this station's over-engineering ledger or is re-filed against the owning crate.
- **Queued to:** reconciliation (re-file the cobre-io validation gap)

**OD-041 · Sev C · speculative-generality · effort S · confidence high**
Narrowed to the tailrace trio only, and to a marker-level cleanup rather than a defect: QuarticSegment (tailrace.rs:60), TailraceSegments (:86) and TailraceFamily (:189) hold pub(crate) that no signature forces and that contradicts the module's own declared crate surface at fpha_fitting/mod.rs:66, and they must be narrowed as one unit because TailraceFamily's pub segments field (:194) types TailraceSegments. SimWorkerParams drops out (mirror of ForwardWorkerParams, whose pub(crate) is forced by pub(crate) fn run_forward_worker) and DEFAULT_REFERENCE_VOLUME_FRACTION drops out (its 'sole owner' doc is about the 0.65 literal, held in production code, with the cross-file protocol at types.rs:441-443 routed through resolve_reference_volume_hm3). The trio is 3 of 25 single-file pub(crate) declarations out of 397 in the crate, so it belongs to a crate-wide visibility pass with per-site forcing checks, not to a standalone three-file edit.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens over-engineering; ingest ref 5d-over-engineering-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs::QuarticSegment`, `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs::TailraceSegments`, `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs::TailraceFamily`, `crates/cobre-sddp/src/production/hydro_models/production.rs::DEFAULT_REFERENCE_VOLUME_FRACTION`, `crates/cobre-sddp/src/simulation/state.rs::SimWorkerParams`
- **Evidence:** Five items are declared `pub(crate)` and have zero mentions outside their defining file, so the wider visibility publishes a crate-wide surface no crate-wide consumer wants. The counter-signal that would justify it is absent in each case: Rust's private_interfaces lint forces `pub(crate)` only when a type appears in a wider-visibility signature, and here `TailraceSegments.segments`, `TailraceFamilies.families` and every field of `SimWorkerParams` are private while `run_worker_scenarios` is a private fn, so nothing… Re-derive: `for s in QuarticSegment TailraceSegments TailraceFamily DEFAULT_REFERENCE_VOLUME_FRACTION SimWorkerParams; do echo "== $s"; git grep -n -w "$s" 077dbe2c -- 'cr…`
- **Fix-shape:** Narrow each of the five to the visibility its single defining file actually needs, which is private for QuarticSegment, TailraceSegments, TailraceFamily, DEFAULT_REFERENCE_VOLUME_FRACTION and SimWorkerParams, keeping TailraceFamilies and build_tailrace_families_map at pub(crate) because mod.rs re-exports them. Compilation is the verifier: if private_interfaces fires on any of the five, that item is structurally forced and stays pub(crate) with the reason recorded. A visibility narrowing emits no code, but still assert byte-neutrality through the parity goldens in tests/parity.rs via tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and mpiexec -n 1 versus -n 2, and move no golden. Blast radius is three files. No pinned contract in .claude/rules/sddp.md is touched.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** three tailrace types carry `pub(crate)` that no signature forces and that contradicts the module's declared crate surface; marker-level cleanup narrowed to one unit (the family's `segments` field types the segments type); the two other items are conceded as forced.
- **Needs-human (owner gate):** Scope call: whether crate-internal single-file items should be narrowed to module-private at all, or whether pub(crate) is the ratified uniform internal marker for this workspace; if narrowing is wanted it should be one crate-wide pass over the 25 sites with a per-site forcing check, not this five-anchor slice.

**OD-042 · Sev C · redundant-wrapper · effort S · confidence high**
The removable residue is exactly the one-item impl block at simulation/state.rs:87-118 — `SimulationInputs::new`, its `#[allow(clippy::too_many_arguments)]` and its RATIONALE comment — justified not by a duplicated suppression pair but by `run_simulate` at simulation/pipeline/tests.rs:38-66 being a signature-for-signature twin of `simulate` that already constructs the bundle with the struct literal at lines 54 and 92, proving the literal compiles at that exact call shape; the suppression on `simulate` itself is independently earned by its own public ten-parameter signature and survives.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens over-engineering; ingest ref 5d-over-engineering-02)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/src/simulation/state.rs::SimulationInputs`, `crates/cobre-sddp/src/simulation/pipeline.rs::simulate`
- **Evidence:** `SimulationInputs::new` takes ten positional parameters and forwards all ten unchanged into a struct literal, deciding nothing: no defaulting, no validation, no derived field. Every one of the ten fields is `pub`, so the struct literal is available to every caller, and the crate's own sibling tests already use it twice at simulation/pipeline/tests.rs lines 54 and 92 rather than calling `new`. There is exactly one production call site, inside `simulate`, which itself already carries the identical suppression. The c… Re-derive: `git show 077dbe2c:crates/cobre-sddp/src/simulation/state.rs | sed -n '52,118p'; git show 077dbe2c:crates/cobre-sddp/src/simulation/pipeline.rs | sed -n '1040,1…`
- **Fix-shape:** Delete `SimulationInputs::new` and have `simulate` build the value with the struct literal the sibling tests at simulation/pipeline/tests.rs already use, which removes the state.rs suppression and its duplicated rationale. The suppression on `simulate` itself stays, because its own ten parameters are the public signature and are not in scope here. No field visibility changes, since all ten fields are already pub. The change is call-shape only and must be byte-neutral: assert the parity goldens in tests/parity.rs through tests/common/parity_hash.rs, the rank-invariance harness in tests/common/permute.rs, and mpiexec -n 1 versus -n 2 reproduction, and move no golden. Blast radius is two files plus the two sibling test sites that already compile against the literal form. Touches no pinned contract in .claude/rules/sddp.md.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** asserted against the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction.
- **Calibration:** a one-item impl block (`SimulationInputs::new`, its `too_many_arguments` allow and rationale) whose struct-literal replacement is proven by the test twin `run_simulate` already constructing the bundle at the same call shape.

**Test bloat** — 2 minted

**TD-055 · Sev C · duplication · effort S · confidence high**
Narrowed from the whole file to two of its six tests. lead_ge2_rebaseline_contains_the_named_k2_k3_fixtures (hold_k1_byte_stability_probe.rs:217-229) requires three string literals to appear in a const declared 45 lines above it in the same file, and it therefore passed green when commit 19ecadbd renamed simulation_ring_buffer_shifts_anticipated_state_k2 to simulation_commitment_hold_carries_anticipated_state_k2 in the same commit that also edited this probe, leaving lines 172 and 222 as the only two occurrences of a name that resolves to no test fn at the pin. k1_byte_stability_verdict (lines 231-250) has zero assert sites and five println!, so it has no failing path. Explicitly conceded and NOT part of the claim: the three DeliveryRing tests at lines 32-94 exercise production code and must stay; k1_survivors_and_lead_ge2_rebaseline_are_disjoint (lines 197-215) enforces a real cross-list partition invariant over all 46 entries and is not confirmed for deletion; and the k2 coverage wa…

- **Station:** cobre-sddp (sddp, sub-station 5d, lens test-bloat; ingest ref 5d-test-bloat-01)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::lead_ge2_rebaseline_contains_the_named_k2_k3_fixtures`, `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::LEAD_GE2_REBASELINE`, `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::K1_SURVIVORS`, `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::k1_survivors_and_lead_ge2_rebaseline_are_disjoint`, `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::k1_byte_stability_verdict`, `crates/cobre-sddp/tests/hold_k1_byte_stability_probe.rs::AUDITED_FIXTURE_FILES`, `crates/cobre-sddp/tests/anticipated_scenarios.rs::simulation_ring_buffer_shifts_anticipated_state_k1`
- **Evidence:** Three of the nine tests exercise production code through the single DeliveryRing import and are legitimate. The other structure is self-referential: lead_ge2_rebaseline_contains_the_named_k2_k3_fixtures asserts that a hand-written &[&str] in the same file contains three string literals the same file writes, so the assertion can only fail if someone edits both sites inconsistently; it cannot detect that simulation_ring_buffer_shifts_anticipated_state_k2 has no corresponding test fn anywhere in the repo, which is th… Re-derive: `git grep -n 'simulation_ring_buffer_shifts_anticipated_state' 077dbe2c ; for each of the 46 quoted names in K1_SURVIVORS/LEAD_GE2_REBASELINE/EXCLUDED_NOT_RING_…`
- **Fix-shape:** Keep the three DeliveryRing primitive tests, which are the only part of this binary that touches production code. Delete k1_byte_stability_verdict, whose five println! lines carry no assertion and are invisible under a normal cargo test run, and delete the two list-shape assertions that compare in-file literals to each other. Relocate the audited k_max partition and its rationale to prose that no green test can misrepresent, either the module doc that already holds the argument or the sddp rules file, and fix or drop the rotted simulation_ring_buffer_shifts_anticipated_state_k2 entry while doing so. If the partition must stay executable, make the assertion resolve each name against the test tree rather than against a sibling literal in the same file, so a renamed or deleted fixture fails the guard. Byte-neutrality: the probe writes no output and reads no golden, and the three retained tests construct DeliveryRing directly, so nothing in the parity or rank-invariance sets is touched; the reproduction pair still runs to confirm.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** two of six tests in one probe binary assert string literals against a const declared in the same file and passed green through the rename that rotted them — tautological tests (TD-038 / TD-040 precedents: a test asserting its own transcription), Sev C; the rot is the evidence, not the blast radius.
- **Reviewer rating:** B — recalibrated to C because two tautological tests in one binary; the precedent for a test asserting its own literal (TD-038, TD-040) is C — the false-confidence guard is the finding, the fix is S.
- **Yardstick:** docs/design/testing-architecture.md §3.2 sustainability (a test that restates its own literals is invisible and therefore dead) (td-queue.json).
- **Needs-human (owner gate):** Owner call on whether the audited k_max partition should survive in any executable form now that the shift-to-hold switchover has landed and there are zero k_max >= 2 tier-1 goldens: retire the checklist to prose, or re-home it as a script-backed gate that resolves each name against a nextest listing, since Rust cannot reflect over another integration binary's tests (the probe's own module doc, lines 25-28).
- **Queued to:** test-corpus

**TD-056 · Sev C · asymmetry · effort S · confidence high**
Confined to tests/simulation_pipeline_integration.rs:78 - the local double reuses the harness's StubComm identifier for a type whose collective semantics are incompatible with it, inside a file that compiles `mod common;` at line 51; its rustdoc at line 77 says Single-rank while line 1173 instantiates it with `size: 2`; and its `rank` field is 0 at all 20 sites. A naming, stale-rustdoc and dead-field defect only. The three-way consolidation, the premise that the parameterized type subsumes the two harness unit structs, and the anchors common/mod.rs::StubComm, common/mod.rs::Rank0Of2, conformance.rs::LocalComm and integration.rs::ShutdownComm do not survive.

- **Station:** cobre-sddp (sddp, sub-station 5d, lens test-bloat; ingest ref 5d-test-bloat-04)
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Anchors:** `crates/cobre-sddp/tests/common/mod.rs::StubComm`, `crates/cobre-sddp/tests/common/mod.rs::Rank0Of2`, `crates/cobre-sddp/tests/simulation_pipeline_integration.rs::StubComm`, `crates/cobre-sddp/tests/integration.rs::ShutdownComm`, `crates/cobre-sddp/tests/conformance.rs::LocalComm`
- **Evidence:** The same concept is spelled three ways in one crate's test surface: a fixed 1-rank unit struct and a fixed 2-rank unit struct in the shared harness, plus a parameterized rank/size struct local to one binary that covers both shapes and shadows the shared name inside a file that imports the same common module. The collective bodies are not interchangeable as written, so this is fragmentation rather than a redundant copy: the local one panics via unreachable! on allgatherv, allreduce and broadcast, common::StubComm c… Re-derive: `git grep -n -E '^ *(pub )?struct StubComm' 077dbe2c -- crates/cobre-sddp/tests/ ; git grep -l 'common::StubComm' 077dbe2c -- crates/cobre-sddp/tests/ | wc -l ;…`
- **Fix-shape:** Replace the three shapes with one parameterized stub communicator in the shared harness that takes rank and size and carries an explicit mode for the collectives a given path must not touch, so the pipeline tests keep their assert-unused semantics as a stated mode rather than as a separate type. The two existing unit structs become constructor calls at their 20 and 4 call sites and the local copy in simulation_pipeline_integration.rs disappears along with the name shadowing. This is not a blind delete: the unreachable! bodies are load-bearing for the pipeline tests that assert no collective fires on the frozen path, and the displs-offset copy in the 2-rank shape must survive verbatim. Byte-neutrality: communicator behaviour on every currently exercised path stays identical, so no golden moves; verify against the tier-1 parity goldens, the rank-invariance permutation harness, and a single-rank versus two-rank MPI reproduction, which is the direct check for this candidate.
- **Alignment:** neutral (provisional; Epic 9 adjudicates against the L0-L4 target layering — plans/generalizing/beyond-sddp-generalization.md Part IV §IV.1 (L3 engine internals))
- **Byte-neutrality:** n/a — test-only, visibility-only or doc-only change; no rendered LP byte can move (bounded by the parity goldens (`parity_hash_highs` / `parity_hash_clp` in the sddp parity binary), the rank-invariance harness (tests/common/permute.rs) and `mpiexec -n 1/2` reproduction as a formality).
- **Calibration:** one local double reuses the harness's `StubComm` name for a type with incompatible collective semantics inside a file that compiles `mod common;`, with a stale Single-rank rustdoc and a dead `rank` field — naming, stale-doc and dead-field only; the three-way consolidation is conceded and the harness pair stays per §5.8.
- **Cross-station:** E08-2 needs-human (communicator doubles in tests/common) — kept distinct from 5c-test-bloat-02 (harness shape vs src-side re-declaration).
- **Yardstick:** docs/design/testing-architecture.md §5.8 (StubComm / Rank0Of2 are the kept harness doubles) / §5.1 (td-queue.json).
- **Needs-human (owner gate):** Owner call: whether testing-architecture.md section 5.8 (keep StubComm/Rank0Of2) plus the section 5.2 move into cobre-comm's test-support surface is the final disposition for the harness pair, or whether the test-corpus station's E08-2 needs-human item may still merge them - my dismissal of the merge half rests on section 5.8 being ratified. Owner call: whether conformance.rs:40 LocalComm's no-copy collectives (Ok(()) without writing recv, unlike common::StubComm) are deliberate; I read the divergence as semantics-by-design, but no rustdoc states why.
- **Queued to:** test-corpus

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

### Tier-1 fix wave (2026-09-11) — validated at `1baeeadb`, fixed on `fix/quality-tier1` + `fix/quality-tier1-followups`

The 112 ratified findings were re-derived against source (`PRIORITIES.md`, c03c0336: 0 refuted,
26 partial, 86 verified) and the user-visible correctness tier was fixed first. `fix/quality-tier1`
(a729a259, 4af5e4c7, 8375e63b, 2d7c7cd7, 0d4c8c22) is merged to `develop`; the two follow-up
fixes found while executing it (19521701, 3b363161 on `fix/quality-tier1-followups`) are pending
merge. Each closed entry carries a `- **Status:** fixed` bullet with current-tree evidence.

| Finding    | Resolution                                                                                                                                                                                                                                                                               |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OD-011** | FIXED — the NCS column fill reads the resolved per-(source, stage) curtailment penalty like its three siblings.                                                                                                                                                                          |
| **CD-056** | FIXED — one table-driven stage-axis rule (rule 49) over all six bound families, declared-id set membership; rule 16 retired. Its "padded region makes thermal legitimately family-specific" clause was wrong (correction bullet on the entry). CD-053's NCS half deliberately untouched. |
| **CD-051** | FIXED — `validate_config` resolves both scenario sources; validate and run agree; Python parity test added.                                                                                                                                                                              |
| **CD-058** | FIXED — atomic payload/manifest/CSV writes; a rewrite removes the old manifest, then every stale payload, before writing. The reader's directory enumeration made stale payloads a release-build silent hazard (correction bullet on the entry).                                         |

Register corrections from `PRIORITIES.md` §3 were folded into their entries as
`- **Correction (2026-09-11):**` bullets (PD-007, PD-023/024/025, CD-045, CD-047, CD-054, CD-058,
CD-060, CD-066, CD-067, OD-017, OD-021, OD-029, TD-004, TD-008, TD-022, TD-028, TD-029, TD-031);
`stations/stochastic/perf-queue.json` is a station artifact and was left as written.

New observations recorded for the sddp / cli-python stations (no IDs minted outside a gate):
`EventConfig.checkpoint_interval` has no production consumer; `SobolPrecomputed::new` had zero
callers at `1baeeadb`; `sampling/out_of_sample.rs` carries a false "No heap allocation" doc claim.

Mirror: `docs/design/reserved-seams-and-deferred-debt.md` gained its first 2026-09 section
(ID-free, by tier) in the same commit as this entry.

### New finding minted at reconciliation (2026-09-11) — found while mapping the Tier-2 seams

**CD-072 · Sev A · asymmetry · effort S · confidence high**
Every out-of-sample class sampler was seeded from the study's root forward seed with no class discriminator anywhere in the derivation, so a deck that set two or more of `inflow`, `load`, `ncs` to `out_of_sample` drew bit-identical noise for the k-th entity of each class under every noise method (SAA, LHS, Sobol, Halton) — an undeclared perfect cross-class correlation; the shipped decks and every test set only the inflow class, so nothing pinned it.

- **Station:** cobre-stochastic (sub-station sampling) — minted at reconciliation, not at the station gate
- **Baseline:** `0d4c8c22` (develop after the Tier-1 merge)
- **Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs::build_forward_sampler`, `crates/cobre-stochastic/src/sampling/mod.rs::build_class_sampler`, `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated`, `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_saa`, `crates/cobre-stochastic/src/noise/seed.rs::derive_forward_seed_grouped`
- **Evidence:** the three `build_class_sampler` calls pass the same `forward_seed`; `ClassSampler::fill`'s `OutOfSample` arm builds `FreshNoiseSpec` from `(forward_seed, iteration, scenario, noise_group_id, dim)` only; `fill_saa` seeds `derive_forward_seed_grouped(forward_seed, iteration, scenario, noise_group_id)` and each QMC/LHS point spec maps the same tuple. Reproduced with a throwaway test calling `ClassSampler::OutOfSample { forward_seed: 99, dim: 4 }` and `{ dim: 2 }` with one `ClassSampleRequest`: `output[..2]` identical for all four methods. Supported configuration: `config/mod.rs` tests assert `simulation.load_scheme == OutOfSample`; `stochastic_summary.rs` builds a context with `load: OutOfSample`.
- **Fix-shape:** derive a per-class forward seed at `build_forward_sampler`; keep the inflow class on the root seed so every existing inflow-only deck reproduces bit-for-bit, derive load and NCS with a class tag under a new domain prefix.
- **Alignment:** neutral
- **Status:** fixed (2026-09-11) — `derive_class_forward_seed(root, "load" | "ncs")` in `noise/seed.rs` (`0x02` prefix); inflow unchanged; `sampling::tests::test_out_of_sample_classes_draw_distinct_streams` pins it (fails on the old seeding). `fix/out-of-sample-class-seed` ff2221d1 (merged to develop 2026-09-12). Results change only for decks with two or more out-of-sample classes.

### Tier-2 fix wave (2026-09-12) — `plans/quality-tier2-hotpath`, base `5cc0a042` (develop after both follow-ups and CD-072 merged), merged to `develop` at `fe439fc0`

Fourteen tickets in two epics, every ticket bit-for-bit neutral (goldens captured on the pre-change
code in `crates/cobre-stochastic/tests/forward_sampler_golden.rs`). Each closed entry carries a
`- **Status:** fixed` bullet with current-tree evidence.

| Finding                    | Resolution                                                                                                                                                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PD-023, PD-024, PD-025** | FIXED — per-iteration noise tables (`ForwardNoiseTables`, one table per class and `(noise_group_id, noise_method)`), rebuilt by each driver and shared by reference through the sample request; Sobol wires its dormant precompute, Halton and LHS gain one each. |
| **PD-026, PD-028**         | FIXED — positions resolved once at `DecomposedCorrelation::build`; profile and group slice resolved once per draw / per stage.                                                                                                                                    |
| **PD-027**                 | FIXED — caller-owned correlation scratch (`ScratchBuffers.corr_scratch`) replaces the permutation scratch; no allocation at any group width, pinned by a counting-allocator guard.                                                                                |
| **CD-068**                 | FIXED — the full-vector applier, the scan fallback and their differential oracles are deleted; one applier remains.                                                                                                                                               |
| **CD-069**                 | FIXED — one `NoisePointSpec`.                                                                                                                                                                                                                                     |
| **CD-067**                 | FIXED — `EntityClass` parsed once at build; the correlation side is typed end to end.                                                                                                                                                                             |
| **CD-066**                 | PARTIAL — the enum exists; the sampler-side `class_name: &str` gate is still open (status bullet on the entry).                                                                                                                                                   |

Found and fixed inside the wave (not register findings): the hoist moved Sobol construction ahead
of the draw-time dimension check, so a class wider than the direction table panicked in the rebuild
instead of returning `DimensionExceedsCapacity` — `ClassNoiseTables::refill` is fallible and both
drivers propagate; `WorkspaceSizing.total_forward_passes` lost its only reader when the permutation
scratch was retyped and is removed; the allocation guard's `mod common;` pulled the aggregator's own
`#[cfg(test)]` tests into its binary and failed under the threaded `cargo test` harness (nextest's
process-per-test isolation masked it) — the guard now includes only `common/builders.rs`.

Observations from the 2026-09-11 list now resolved: `SobolPrecomputed::new` has production callers;
the false "No heap allocation" claim in `sampling/out_of_sample.rs` is gone; `cargo machete` is
clean (the unused `postcard` dependency left `crates/cobre-sddp/Cargo.toml`). Still open:
`EventConfig.checkpoint_interval` has no production consumer.

Partial progress on the test-corpus tier (not closed): `crates/cobre-stochastic/tests/common/mod.rs`
now homes the correlation-model and inflow order/dimension fixtures for the three QMC/LHS binaries
(part of TD-029/TD-030); `make_bus`, `make_hydro`, `make_inflow_model`, `approx_erf` and `norm_cdf`
remain copied per binary, and `saa_golden_value.rs` / `forward_sampler.rs` / `forward_sampler_golden.rs`
keep local helpers whose shapes differ. The cobre-sddp side already had `tests/common/builders.rs`.

### New findings minted at reconciliation (2026-09-12) — found while executing the Tier-2 wave

**CD-073 · Sev C · bad-naming · effort S · confidence high**
`NoisePointSpec.stage_id` is fed `noise_group_id` at every production call site (`fill_uncorrelated`, `ClassNoiseTables::refill` → `build_table`), and the `stage_id` position on the three precompute constructors receives the group id too; the name predates the wave and was preserved deliberately so the precomputed-vs-direct equivalence tests stayed trustworthy.

- **Station:** cobre-stochastic (sub-station tree-noise) — minted at reconciliation
- **Baseline:** `fe439fc0`
- **Anchors:** `crates/cobre-stochastic/src/tree/point_spec.rs::NoisePointSpec`, `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated`, `crates/cobre-stochastic/src/sampling/tables.rs::build_table`
- **Evidence:** every producer writes `stage_id: spec.noise_group_id` or passes `group` into the `stage_id` parameter; the opening-tree callers pass a real stage id. One name, two meanings.
- **Fix-shape:** rename the field (and the constructor parameter) to the seed-tuple role it plays (`noise_group_id`, or a neutral `group_id`), or split the tree-side and forward-side spec constructors so each names its own key. Bit-neutral; a rename touches every equivalence test and the goldens' helper, so it is its own narrow ticket, not folded into another.
- **Alignment:** neutral
- **Status:** fixed (2026-09-14) — `NoisePointSpec.stream_id` (doc: forward-pass producers pass the `noise_group_id`, opening-tree producers the stage id) and `stream_id` on the three `*Precomputed::new` constructors; `generate_*`'s `stage_id`, `derive_stage_seed`, `derive_opening_seed` and `FreshNoiseSpec.stage_id` untouched by the rename; `build_table`'s doc deleted (the naming tension it explained is gone). ticket-009.

**PD-031 · Sev C · asymmetry · effort S · confidence high**
`fill_uncorrelated`'s `Selective` and `HistoricalResiduals` arms emit `tracing::warn!` on every draw before falling back to the sample-average method; under those (unsupported-in-forward) methods the warning fires once per (iteration, scenario, stage) on the hot path.

- **Station:** cobre-stochastic (sub-station sampling) — minted at reconciliation
- **Baseline:** `fe439fc0`
- **Anchors:** `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated`
- **Evidence:** the two arms log with the stage id and fall through to `fill_saa`; nothing rate-limits or hoists the warning to sampler construction.
- **Fix-shape:** reject or warn once at `build_forward_sampler` (where the per-stage methods are known) and make the two arms silent fallbacks, or reject the combination at config validation so the arms become unreachable.
- **Alignment:** neutral
- **Status:** fixed (2026-09-14) — `warn_unsupported_forward_noise_methods(class, scheme, stages, noise_methods)` runs once per class in `build_forward_sampler` (only for `OutOfSample`, naming the class via `EntityClass::as_str`, the method(s) and the affected `Stage.id`s); the two draw arms are one silent `Selective | HistoricalResiduals => fill_saa` arm; four `WarnRecorder` tests, one of which draws after construction and asserts no further WARN. `FreshNoiseSpec.stage_id` lost its only production readers and was deleted (owner rejected keeping it alive through error-string reads); the test-only `sample_fresh` takes the stage id explicitly. Allocation guard unchanged. ticket-010.

**TD-034 · Sev C · asymmetry · effort S · confidence high**
`crates/cobre-sddp/tests/common/permute.rs` carries its own `#[cfg(test)] mod tests`, so every integration binary that declares `mod common;` compiles and runs those tests inside its own harness; a binary that owns process-global state (the counting-allocator guard) cannot use the aggregator at all and includes `common/builders.rs` by `#[path]` with an include-level `dead_code` allow instead.

- **Station:** cobre-sddp (test corpus) — minted at reconciliation
- **Baseline:** `fe439fc0`
- **Anchors:** `crates/cobre-sddp/tests/common/permute.rs`, `crates/cobre-sddp/tests/common/mod.rs`, `crates/cobre-sddp/tests/forward_sampler_no_alloc.rs`
- **Evidence:** `cargo test -p cobre-sddp --features test-support --test forward_sampler_no_alloc -- --list` showed four tests before the `#[path]` include; nextest's process-per-test isolation hid the coupling on every per-ticket gate, and only the threaded `cargo test` harness (which `ci.yml` runs) exposed it.
- **Fix-shape:** move `permute.rs`'s tests to a binary of their own (or to a `#[cfg(test)]` module gated behind a cargo feature the aggregator does not enable), so `mod common;` never adds tests to a consumer; then the guard can use the aggregator and drop its include-level allow. Fold into the test-support surface work (Tier 4) that already owns `common/`.
- **Alignment:** neutral
- **Status:** fixed (2026-09-15) — `permute.rs`'s tests moved to their own binary, `crates/cobre-sddp/tests/permute_helpers.rs`, so `mod common;` adds no tests to a consumer; the allocation guard `forward_sampler_no_alloc.rs` declares `mod common;` like every other binary and its `--list` shows the one guard test. Closed by `plans/quality-tier45-closeout` ticket-013 (epic-03), commit range `3e90024f..54ab986a` on `develop`; recorded at the Tier-4/5 reconciliation after the register merge exposed the id.

### Tier-3 fix wave (2026-09-14) — `plans/quality-tier3-footguns`, base `fe439fc0`, merged to `develop` at `3e90024f`

Ten tickets in two epics (`epic-01-core-io-invariants`, `epic-02-sampler-typing`), every ticket
bit-for-bit neutral for every deck cobre-io can load; two BREAKING CHANGELOG entries (a removed
public type; a new validation error plus a fallible setter). Waves W2 and W4 of `PRIORITIES.md` §5
are closed. Each closed entry carries a `- **Status:** fixed` bullet with current-tree evidence.

| Finding            | Resolution                                                                                                                                                                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **CD-040**         | FIXED — `HydroStagePenalties` deleted outright (owner chose deletion over the alias); the resolved table stores `HydroPenalties`; the sixteen-field copy is a move.                                                                                    |
| **CD-043**         | FIXED — the builder assigns `Stage.index` after its sort; the loader loop is gone. The boundary review caught a pre-build reader (inflow-seeding validation → lag transitions); the window is now position-based, which also fixes a latent over-skip. |
| **CD-045**         | FIXED — validate-not-sort: `ValidationError::UnsortedModelTable`, `with_scenario_models` fallible. Exposed 31 stage-major test fixtures and an unsorted `run_partial_estimation` table, both corrected.                                                |
| **CD-048**         | FIXED — `SystemBuilder::build` is the one doc owner; sixteen sites are pointers (seven parser docs added with owner approval).                                                                                                                         |
| **CD-057, OD-019** | FIXED — keyed `InputFile` registry; `FileManifest::present(InputFile)` is the single read; the positional zip and the 43-field struct are gone.                                                                                                        |
| **CD-066**         | FIXED — `EntityClass` carried through the factory; the class seed derives from `EntityClass::as_str`, both pinned constants unchanged.                                                                                                                 |
| **OD-028**         | FIXED — per-class `InflowSource` / `ClassSource` built inside the factory from the unchanged public config; `MissingScenarioSource` text and order byte-identical; diagnostics stay in the factory (owner).                                            |
| **OD-029**         | FIXED — `DerivedSeed<'a>` through six signatures (11→8, 13→10 ×5); all six suppressions kept, rationales true.                                                                                                                                         |
| **CD-073**         | FIXED — `NoisePointSpec.stream_id`, named for the seed-tuple slot; the genuine stage-id carriers are untouched.                                                                                                                                        |
| **PD-031**         | FIXED — one warning per out-of-sample class at sampler construction; the draw arms are silent; `FreshNoiseSpec.stage_id` deleted with its last production reader.                                                                                      |

Found and fixed inside the wave (not register findings): (1) `cobre-io`'s semantic validation runs
`precompute_stage_lag_transitions` on pre-build `StagesData`, so removing the parser's index loop
zeroed every index it read — `compute_period_transition` now derives the "later stages" window from
slice position (`par/lag_transition.rs`), a change that also corrects the pre-existing over-skip on
the pre-study-filtered `study_stages` slice in `stochastic_pipeline.rs` (the three shipped decks
with pre-study stages, d26/d30/d43, reproduce their goldens unchanged); (2) `run_partial_estimation`
appended pre-study seasonal rows after the in-study rows without a re-sort, a contract violation
the new check turned into a hard error — sorted, with a regression test through
`estimate_from_history`; (3) 31 test fixtures across 13 cobre-sddp/cobre-stochastic files built
`inflow_models` / `load_models` stage-major — reordered, rows and values unchanged; (4) ticket-007's
fold made the `class` field ticket-006 had just added unread, so it was dropped rather than kept.

Register corrections folded into the entries as part of the status bullets: CD-040's PRIORITIES
fix-shape (alias, "~130 sites") — deletion touched 34 files / 224 occurrences and compiled after one
`cargo check --all-targets` sweep; CD-043's PRIORITIES re-rating ("still latent") — the live reader
was on the cobre-io validation path, not only post-build; OD-029 — six signatures, not three.
CD-066's PRIORITIES re-rating (C) stands; it is closed as part of the same epic.

Process observations carried to memory (not register findings): a ticket that deletes a field's
writer must enumerate every _caller of every reader_ across crates (the missed `validate_inflow_seeding`
path), and a ticket that changes a `pub` signature must grep `tests/` binaries for direct callers
(`par_a_historical_replay.rs` was named in ticket-008's Integration Tests section but omitted from
its frontmatter). Two owner-approved scope widenings and one rejected workaround (keeping
`FreshNoiseSpec.stage_id` alive through manufactured error-string reads) are recorded in the plan's
`boundary-notes.md` files.

Boundary simplification (owner gate G4c each time): 21 + 19 + 13 proposals applied, 7 declined with
owners; the two epic-boundary code reviews found the two regressions above (NEEDS_ATTENTION → fixed
before each epic commit); the plan-level review was PASS_WITH_WARNINGS (two stale doc fragments, fixed).
Two follow-ups closed on the same branch (`3e90024f`): the two byte-identical duplicate-id validators
in `cobre-io/src/stages.rs` are one, and `seeds.rs` casts the k=0 season projection directly (its
`unreachable!` is gone). Still open from the 2026-09-11 list: `EventConfig.checkpoint_interval` has
no production consumer (cobre-sddp station). TD-034 stays in W5.

**Baseline for the next reconciliation: `develop` @ `3e90024f` (the Tier-3 merge).**

### Tier-4/5 fix wave (2026-09-15) — validated at `3e90024f`, fixed on `feat/quality-tier45-closeout`, merged to `develop` at `54ab986a`

Waves W5 (test-support surface, seventeen ids with TD-034) and W6 (dead-surface sweep, twenty-four
ids) of `PRIORITIES.md` §5 were executed as one plan of thirty-nine code tickets in six epics on
`feat/quality-tier45-closeout` (branched from `develop` @ `3e90024f`, fast-forwarded into `develop`
at `54ab986a` on 2026-09-15 and merged back into this register branch at `6ea69a02`). The plan ran
against `develop`, which does not carry this branch's Tier-2/Tier-3 reconciliation commits, so two
of its closeout observations were written from a stale register and are corrected below. Each closed entry carries a `- **Status:** fixed (2026-09-15)` bullet naming the
mechanism, any deliberate non-doing and the closing epic's completion and boundary commits. D1(a)
ratified `docs/design/testing-architecture.md` §5.2 (the uniform `test-support` convention) and
only §5.2; §5.1's binary consolidation and homing threshold stay a proposal.

| Finding                                                                                                                                   | Resolution                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **W5 / epic-01** — TD-002, TD-005, TD-004                                                                                                 | FIXED — `cobre-core` `test_support`: uniform penalty constructors, one spec-plus-`make_*` entity/stage builder family, `..`-free exhaustive bit comparators over the shared `f64_bits_eq`/`opt_f64_bits_eq`. `7c28de14` + `5e06c30e` + `16ca7c66`.                                                                                                                                                                                                |
| **W5 / epic-02** — TD-023, TD-016, TD-007, TD-014, TD-018, TD-020, TD-009                                                                 | FIXED — `parquet_helpers` contract tests first; `crates/cobre-io/src/test_support.rs` behind `test-support` owns the writers, the minimal-case corpus, the phase fixtures, the output trio, `read_first_batch` and the stats-parser template. `ac2e608a` + `bd29eb5e` + `9c5bbf5a`.                                                                                                                                                               |
| **W5 / epic-03** — TD-028, TD-025, TD-030, TD-029, TD-024, TD-032, TD-034                                                                 | FIXED — stochastic entity literals route through `cobre-core`'s builders; `cobre-stochastic` gains its own `test-support` surface (`uniform_tree`, season maps, `InflowModelSpec`); `tests/common/mod.rs` is the integration prelude for eight of nine binaries; goldens byte-identical. `2bf30ece` + `59c5c09f`.                                                                                                                                 |
| **W6 / epic-04** — OD-010, CD-041, OD-012, OD-013, PD-008                                                                                 | FIXED — the two unconstructed `ValidationError` variants and the unread `buses` argument gone; the population-statistics arm of `WelfordAccumulator` gone; `NetworkTopology` deleted and `cascade` serde-skipped with a bit-equal rebuild guard. `dd03c153` + `8a72c69d`.                                                                                                                                                                         |
| **W6 / epic-05** — OD-014, OD-015, OD-018, OD-016, OD-017, OD-020, OD-021, OD-022, PD-017, CD-052, OD-024, OD-025, CD-047, CD-063, CD-049 | FIXED — the `cobre-io` dead surface (free serializers, scalar-parameter loaders, scenario entry point, `default_severity`, `_config` parameter, serde defaults privatised, setup-timing columns, `partitions_written`, `CrossReferenceError`, f32 decoder, pipeline wrappers) removed; `open_record_batch_reader`, `ensure_parent_dir` + `write_batch_atomic` and the generic factor resolver are the one-owner helpers. `09381faa` + `883c022e`. |
| **W6 / epic-06** — OD-026, OD-027, OD-030, OD-031                                                                                         | FIXED — `ParValidationReport`/`ParWarning`, the three bare season-map forwarders, `SweepDirection` and the three unconstructed `StochasticError` variants deleted; the fatal PAR check and the descending solve order unchanged. `719ed408` + `858e3ad7`.                                                                                                                                                                                         |

Register corrections found while refining and executing the plan (the seam map's
`[SPEC CORRECTION]`s and the tickets' completion notes) were folded into their entries as
`- **Correction (2026-09-15):**` bullets (TD-002, TD-004, TD-005, TD-007, TD-009, TD-016, TD-024,
TD-028, TD-029, TD-030, OD-013, OD-014, OD-015, OD-016, OD-018, OD-020, OD-030, PD-008, PD-017,
CD-047, CD-052, CD-063, and CD-046, which this wave narrowed but did not close). Where a line anchor
drifted between the entry baseline and `3e90024f`, or moved again mid-plan when a `test_support`
module declaration landed above it, the bullet names the commit at which each value held.

New observations recorded at this reconciliation (no IDs minted outside a gate):

- **`TD-034` is a register id after all — closed by ticket-013.** The closeout recorded it as a
  phantom because `develop` @ `3e90024f` carried neither the entry (minted 2026-09-12 at the Tier-2
  reconciliation on this branch) nor the `PRIORITIES.md` W5 row that lists it. The relocation of
  `permute.rs` out of the `cobre-sddp` test aggregator is exactly its fix-shape; the entry now
  carries its status bullet and the W5 count above includes it.
- **The `identity_correlation` fixture family still has five in-`src` copies** in
  `crates/cobre-stochastic/src`: `provenance.rs`, `context.rs` and `sampling/mod.rs` return
  `CorrelationModel`; `tree/generate.rs` and `sampling/out_of_sample.rs` return
  `DecomposedCorrelation`; a sixth copy sits inside a `context.rs` rustdoc example. The integration
  binaries already share one pair from `crates/cobre-stochastic/tests/common/mod.rs`. An in-`src`
  fixture-sharing item for the test-corpus sweep (W9).
- **`crates/cobre-core/README.md`'s `Bus` row still says "connected generators"** although the bus
  adjacency types were removed with `NetworkTopology`. Prose drift for the doc-drift sweep (W7).
- **A pre-existing acceptance-criteria tag survives in shipped test code.** `// ── AC: … ──` banners
  sit in `crates/cobre-io/src/extensions/{tailrace_curves,hydro_geometry,fpha_hyperplanes,production_models}.rs`.
  `AC: ` is in `scripts/ci/check-no-plan-leaks.sh`'s banned pattern, but every banner sits after
  its file's first `#[cfg(test)]` line, where the script's `.rs` pass stops emitting, so the gate
  does not see them. No `AC: ` occurs under `crates/*/tests` or `crates/*/benches`, where the pass
  scans in full. A sweep of those tags is its own item, not this plan's.
- **`EventConfig.checkpoint_interval`** (`crates/cobre-sddp/src/config.rs`) still has no production
  reader — only the field, its serde-skip note, the doctests, the in-crate tests and `None`
  constructions. It is a config field, so the "unwired config is reserved" hard rule applies: a
  record, never a deletion. Re-confirmed at this baseline (first recorded at the Tier-1
  reconciliation).
- **The plan-completion verification checklist has lost its upstream owner.** The checklist every
  epic of this plan ran is reproduced verbatim in
  `plans/quality-tier45-closeout/epic-06-stochastic-dead-surface/00-epic-overview.md`, which credits
  `plans/quality-tier3-footguns/` as where it is maintained. That directory no longer exists
  (`plans/` holds only `architecture-debt-audit` and `quality-tier45-closeout`), so this plan's copy
  is the only live one and whoever next reproduces it must copy from there. The previously recorded
  follow-up — that the Tier-3 spec pointed its checklist at a deleted Tier-2 path — is moot for the
  same reason and is closed, not carried.
- **Two generic uses of the word "ticket" survive in the mirror's prose**
  (`docs/design/reserved-seams-and-deferred-debt.md`: "do not attempt inside a feature ticket" in
  the data-model trigger, and "a dedicated ticket owns" in the `#[allow(deprecated)]` census note),
  outside every section this wave edited. A prose sweep item for the doc-drift wave (W7).
- **Anchor drift on entries this plan did not close**, from a HEAD-resolving pass
  (`python3 plans/architecture-debt-audit/tools/check-anchors.py core-io --baseline HEAD --allow-drift`
  → `checked 345 anchors, 82 failing` at `858e3ad7`; `stochastic` → `checked 118 anchors, 0 failing`;
  the 40 closed entries' anchors are missing by design and are not drift; after the register merge,
  `6ea69a02`, the same pass reads `checked 365 anchors, 87 failing`, and CD-046 and TD-017 are the
  only open entries among the failures): CD-056
  (`check_thermal_bounds_override_stage_range`, retired with rule 16), CD-057 and OD-019
  (`manifest_fields_mut`, `FILE_ENTRIES`, deleted by the Tier-3 wave), CD-040 (`HydroStagePenalties`,
  deleted by the Tier-3 wave), TD-017 (all five thermal test anchors, retired by the rule-49 rewrite
  — re-anchor or close as moot before the test-corpus sweep), and CD-048 (its two
  `resolution/{load,ncs}_factors.rs` anchors moved to `resolution/factors.rs` by this wave's CD-049
  fold — re-anchor). The closeout also recorded CD-040, CD-057 and OD-019 as closed without a
  status bullet; that was the stale-register view — all three carry `fixed (2026-09-14)` bullets
  from the Tier-3 reconciliation on this branch, so no status-bullet pass is owed. TD-017 is the one
  entry in this list still without a bullet (re-anchor or close as moot before the test-corpus sweep).

Withdrawn — two previously queued follow-ups that do not reproduce:

- **`PD-023`'s anchor prefix drift.** All three anchors carry the `crates/cobre-stochastic/` prefix
  (`grep -n -A6 '^\*\*PD-023 ·' BACKLOG.md | grep Anchors`), no `Anchors:` line in the
  register names `forward_sampler_golden.rs` (the Tier-2 status bullet mentions it in prose only), and
  the HEAD-resolving `stochastic` anchor pass reports zero failures at `6ea69a02`. Not carried.
- **`Severity` is not dead.** The Tier-5 removal (ticket-024) took `ErrorKind::default_severity`
  only; `Severity` keeps production readers across `crates/cobre-io/src/validation/mod.rs` and its
  crate-root export in `crates/cobre-io/src/lib.rs` (`grep -rn Severity crates/cobre-io/src/validation/mod.rs`
  → 13 hits). Not carried.

Mirror: `docs/design/reserved-seams-and-deferred-debt.md`'s "Structural lever" and "Dead-surface
sweep" sections became two ID-free `### Fixed — … (2026-09-15)` sections in the same commit as this
entry; `docs/design/testing-architecture.md` reads `Partially adopted (§5.2); the rest Proposal`
with `docs/design/README.md`'s row matching.

**Baseline for the next reconciliation: `develop` @ `54ab986a` (the Tier-4/5 fast-forward; every earlier fix branch is already in). Waves W1–W6 are closed; W7–W10 remain.**

### Doc-and-table-drift wave (W7, 2026-09-15) — fixed directly on `develop` at `eb0b82ef` + `2a14fe56`

Wave W7 of `PRIORITIES.md` §5 (eleven ids) was executed without a plan directory: three
specialists on disjoint file sets (cobre-core + cobre-stochastic; cobre-io validation; cobre-io
output), one guardian each (0.96 / 0.98 / 0.95), the full workspace gate chain (nextest 6377,
MPI gate 87, pins 117, schemas, release builds, doctests, parity, machete), two commits. Each
closed entry carries a `- **Status:** fixed (2026-09-15)` bullet naming the mechanism.

| Finding    | Resolution                                                                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CD-042** | FIXED — one `cell_index` owner beside `thermal_cell_index`, 14 sites routed.                                                                                                      |
| **CD-044** | FIXED (doc contract) — engine-private symbol gone from the docs; the type's crate home is a W10 layering decision, not a sweep item.                                              |
| **CD-046** | FIXED — the one remaining wire map is a `BTreeMap`; rule stated once on `System`; byte-identity test.                                                                             |
| **CD-050** | FIXED — parse layer owns rules 4/5; semantic copies, their tests and the rows retired.                                                                                            |
| **CD-053** | FIXED — Layer 5a wins (owner): seventh `FamilyMeta` row; deck-visible, in the CHANGELOG; NCS negative-value branches deleted as unreachable (deviation accepted by the guardian). |
| **CD-054** | FIXED — one descriptor-driven emit helper carries 39 of 48 sites; 9 differently shaped sites stay open-coded; messages byte-identical.                                            |
| **CD-055** | FIXED — row 26 retired; both tables audited row by row (7 rows added, 3 reworded); every dispatched function tagged `Rule N`.                                                     |
| **CD-059** | FIXED — `write_results` contract stated truthfully in the module, function and crate docs.                                                                                        |
| **CD-060** | FIXED — all 34 schemas in `schemas.rs`; one registry drives the axis gate and the dictionary rows.                                                                                |
| **CD-062** | FIXED — one 14-row family table drives directory creation and per-scenario writes.                                                                                                |
| **CD-064** | FIXED — six phrases reworded; genericity gate green.                                                                                                                              |

Register corrections folded into the status bullets: CD-046 had already narrowed to one field by
the Tier-4/5 wave; CD-054's second `s.id >= 0` copy in `referential.rs` left with CD-053's deleted
loop; CD-060's live count was 25/9, not 26/10; CD-053's NCS half could not use the descriptor
table (`NcsBoundsRow` has no `block_id`). Two things this wave deliberately did not do: relocate
`StageLagTransition` (CD-044; W10 adjudicates) and consolidate the output orchestration (CD-059
points at CD-025, which stays open in W8's neighbourhood). The remaining `s.id >= 0` copies
outside `referential.rs` and the `Rule N` convention's extension to future checks are the
doc-drift residue; no new ids minted.

**Baseline for the next reconciliation: `develop` @ `2a14fe56`. Waves W1–W7 are closed; W8–W10 remain.**

## ★ POST-PLAN RECONCILIATION (2026-09-17) — `develop` `2a14fe56..077dbe2c` merged into the register branch at `e3535a47`

Two workstreams landed on `develop` since the W7 baseline; neither was a scheduled debt wave, so this
section records their debt impact and refreshes the picture. **No new ID minted; no entry closed outright;
one entry moves to partial.**

**What landed (23 commits, 140 files, +14,956 / −10,299):**

- **Boundary policy by calendar date** (`220f96b9` … `d4dcef72`, cobre-sddp +8.6k/−3.4k, cobre-io
  +1.8k/−0.9k): dated self-describing checkpoint wire format (pools and slots stamped with real dates,
  transit buckets on the extended arrival calendar), boundary cuts selected and reconciled by date,
  `policy.boundary.source_stage` replaced by a strict-superset switch with report-only tallies, the season
  descriptor validated at decode time and the season gate relaxed to referenced seasons. Station 2 (policy
  load) stays the positive reference; CD-039's writer half (per-family slot-reservation channel) is
  untouched by construction — the change is on the read/reconcile path.
- **CLI simplification + cobre-python review** (`797ba443` … `24e76cdb`, cobre-cli +0.8k/−3.1k,
  cobre-python +2.3k/−1.9k): `cobre report`/`cobre summary` and their Python and cobre-io mirrors deleted
  (`commands/report.rs`, `commands/summary.rs`, `output/convergence_reader.rs`, four public readers, three
  CLI test binaries); `cobre.run.run` re-implemented over the `Study` lifecycle; typed error leaves
  (`InternalError`), `threads=0` rejected, boundary reconciliation in the Python validator, checkpoint
  `season_manifest` round-trip, CLI simulation metadata `solver_version`, Python `setup` timings, golden
  CLI-vs-Python determinism test, run-summary `Option` fields tightened (12 of 25) with the test-only
  renderer oracle deleted, dead-code audit clean, CHANGELOG/README/notebook refreshed.

**Debt impact:**

| Entry | Effect |
| --- | --- |
| **CD-025** (CLI/Python output hand-mirror, Wave 5) | OPEN, unchanged in structure; detection hardened (golden parity test + import-resolving gate). Note bullet on the entry. |
| **CD-029** (`PrepPhase` doc + boundary bypass + Python parity) | **PARTIAL** — Python-parity half fixed (phase 11); abstraction half open. Status bullet on the entry. |
| **CD-009** (policy-dir guard ×3) | OPEN, unchanged: `run/policy.rs:173`, `:202`, `:335` still three `!exists()` guards. |
| **CD-011** (`PolicyStageManifest` naming) | OPEN, unchanged (9 occurrences in `policy_load.rs`). |
| **CD-047 / TD-018 / TD-020** (fixed 2026-09-15) | Their bullets cite `output/convergence_reader.rs`, deleted by the CLI plan (its two readers had no production caller). Historical anchors; nothing to re-open. |
| **CD-051** (fixed 2026-09-11) | Its parity mechanism held: the CLI plan's Python validator gained phase 11 through the shared reconciler, not a second admission gate. |
| Anchor drift (`tools/check-anchors.py core-io --baseline HEAD`) | 31 entries fail at HEAD, 30 of them FIXED entries whose baseline anchors moved with their fixes (expected). The one OPEN entry with drifted anchors is **TD-017** (thermal boundary tests), pre-existing since the Tier-4/5 merge. `stochastic`: 126/126 anchors resolve. |

**Candidates for the pending stations (recorded, not minted — the station mints):**

- *cli-python:* the plan's own audits are ready-made station inputs — `plans/cli-simplification-python-review/`
  `parity-arguments.md` (48 rows, 6-verdict vocabulary), `parity-behaviour.md` (30 rows + output-file-set,
  gate-coverage and error-mapping sections), `docstring-audit.md` (128 claims) and `fix-list.md` (26 items,
  all fixed; 16 justified-as-is rows that the station should re-adjudicate). Residual observations: the
  `training/hydro_models.json` / `training/model_provenance.json` sidecars were written for the deleted
  `cobre summary` and now have no in-repo consumer (kept under the parity rule — an OD candidate);
  `write_training_outputs` ↔ `write_training_artifacts` still mirror by hand (CD-025); the run-summary
  `TrainingSummary` keeps nine `Option` timing fields beside twelve tightened counters by owner decision
  (recorded so the station does not re-raise it as an asymmetry).
- *sddp:* `EventConfig.checkpoint_interval` (`cobre-sddp/src/config.rs:183`) still has no production
  consumer (re-confirmed at `077dbe2c`; only docs and doctests read it).
- *build-ci:* the Python CI job now builds the CLI and requires it (`--require-cli-binary`), runs the
  bindings crate's Rust tests with `LD_LIBRARY_PATH` from `sysconfig`, and `check-comment-line-refs.sh`
  scans `.py`/`.pyi`; `cobre-python` has `doc = false`, so `cargo doc -D warnings` never checks its
  intra-doc links (a gate-coverage gap in the §3.6 sense).

**Self-accounting (the two plans' own contribution):** new test infrastructure only (`tests/_cobre_cli.py`,
`conftest.py`, the golden module, four checkpoint round-trip tests); no new production abstraction beyond
the `PySeasonManifest` mirror of the existing graph-manifest pair; one new dependency edge (`chrono` in
cobre-python, already a sibling dependency); `cargo machete` clean; no `#[allow(dead_code)]` in the plan's
file set. No new CD / OD / PD / TD.

**Baseline for the next reconciliation: `develop` @ `077dbe2c`.** Waves W1–W7 closed; W8–W10 open;
the 2026-08-22 Waves 4–7 (setup redesign, output single owner, retrofit sweep, C-tier batch) unscheduled.

## ★ USER-REPORTED BUG (2026-09-17) — anticipated-commitment delivery reconciliation aborts on sub-kW drift

Minted from a user report ahead of the `sddp` station (unrun; the station inherits this entry and does not
re-mint it). The "no new ID minted" accounting of the post-plan reconciliation above predates this section.

**Report (user log, MPI run, rank 8, iteration 37 of 50, 5.5 h in):**

```
Training   37/50 iter  LB: 3.03102e12  UB: 1.87012e11  gap: -93.8%  fwd: 32035ms / bwd: 493339ms  [05:34:34 < 01:57:33]
rank 8:
error: training error: anticipated commitment -0.00015040481625625826 MW for thermal 70 at stage 47 block 0 lies 0.00015040481625625826 MW outside its delivery generation bound 0 — beyond the solver-drift margin, so this is a genuine over-commitment, not numerical drift
-> this may indicate a software or environment problem
-> report this at https://github.com/cobre-rs/cobre/issues
Abort(4) on node 8 (rank 8 in comm 0): application called MPI_Abort(MPI_COMM_WORLD, 4) - process 8
```

**CD-074 · Sev A · bad-abstraction (correctness bug, user-visible abort) · effort M (hotfix S + redesign M) · confidence high**
The delivery-stage reconciliation of a carried anticipated commitment tells solver drift from a modelling
error with two hard-coded constants (`1e-7` relative to the commitment, `1e-5` MW absolute). A carried value
of `−1.5e-4` MW against a `0` MW floor is fifteen times the absolute floor, so a physically impossible
negative commitment — which under the ring's own construction can only be solver-side noise — is classified
as a "genuine over-commitment", the rank returns `SddpError::AnticipatedCommitmentOutOfBounds`, and the whole
MPI job dies through `MPI_Abort` with a user-facing message asserting the opposite of what happened. The
discrimination line has now been recalibrated by production failure three times; the mechanism, not the
constant, is the defect. **Redesign flagged.**

- **Station:** cobre-sddp (LP builder `commitment_reconcile` + `training/stage_solve_prep.rs`) — minted from a user report, not at the station gate
- **Baseline:** `develop` @ `077dbe2c` (mechanism unchanged since `7628d3d0`)
- **Anchors:** `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::drift_margin`, `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::reconcile_commitment`, `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs::fill_bound_relaxations`, `crates/cobre-sddp/src/training/stage_solve_prep.rs::StageSolvePrep::reconcile_commitments`, `crates/cobre-sddp/src/error.rs::SddpError::AnticipatedCommitmentOutOfBounds`, `crates/cobre-sddp/src/lp/builder/columns.rs::fill_anticipated_slot_columns`, `crates/cobre-sddp/src/lp/builder/columns.rs::fill_anticipated_state_columns`, `.claude/rules/sddp.md` § "Delivered commitments reconcile against solver drift; exactness is unreachable"
- **Evidence:**
  - Arithmetic of the report: `under = 0 − (−1.5040e-4) = 1.5040e-4` MW; `drift_margin(−1.5040e-4) = |c|·1e-7 + 1e-5 = 1.0000e-5` MW; `under / margin = 15.04` → `Reconciliation::Violation` → `SddpError::AnticipatedCommitmentOutOfBounds` → exit 4 → `MPI_Abort`. The refused quantity is 150 W of thermal output.
  - The value is negative. Delivery-anchoring bounds the decision column to the delivery stage's `[min_gen, max_gen]`, and the ring carries it through equalities on free `(−inf, +inf)` slot columns (`fill_anticipated_slot_columns`, `fill_anticipated_state_columns`) that are always basic and therefore always factorization-computed — the rule's own explanation of where the drift is born. In this report the floor is `0`, so the negative sign alone proves noise — but that is the instance, not the class. The same factorization error lands on **any** side of **any** delivery-stage bound: a must-run plant (`min_generation_mw > 0`, `fill_thermal_columns`) drifts below a positive floor, a plant at its cap drifts above `max_generation_mw`, and both bounds are **per delivery stage and per block** (`thermal_bounds_at_block`; a commissioning-dormant stage forces `[0, 0]`). The classifier has no notion of the box it is protecting: it measures distance from one crossed bound against a margin sized from the commitment, so it misjudges a positive floor and a cap exactly as it misjudged zero.
  - Third production-driven recalibration of the same guard: the original relax-for-any-overshoot widen (deleted, then found load-bearing); `54882130` reinstated it with margin = the raw `1e-9` primal tolerance and it refused `3.8e-6` MW over a `1593` MW cap in production; `7628d3d0` widened to `1e-7` rel + `1e-5` abs, promising "kilowatt-scale modelling errors are refused exactly as before"; now `1.5e-4` MW is refused at a zero floor. Each incident moved a constant; none changed the classifier's model of where the error comes from.
  - The relative term is anchored to the wrong magnitude. Factorization error on a basic variable scales with the LP's row/RHS magnitudes and the basis conditioning, not with the carried value itself, so near zero only the `1e-5` MW floor acts — 10 W — while the observed noise is 150 W. A commitment of exactly zero (an idle anticipated plant, the common case at most stages) gets the tightest margin of all, which is why this surfaced at a `0` bound and not at a cap.
  - The `Relaxed` arm lowers the generation column's scaled lower bound to `(commitment − margin)/scale < 0`, admitting negative thermal generation into the LP rather than projecting the pinned state back onto the physical box.
  - Severity mismatch: a numerical-noise class terminates the whole MPI job (one rank aborts, every rank dies), losing 37 iterations; the message text ("genuine over-commitment, not numerical drift", "report this at …/issues") is confidently wrong and the users did exactly as told. Re-derive: `sed -n 23,40p crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs`; `sed -n 140,180p crates/cobre-sddp/src/lp/builder/columns.rs`; `git show -s 7628d3d0 54882130`.
- **Fix-shape:** two layers; the second is the point of this entry — **redesign the mechanism, do not move the constants a fourth time.**
  1. *Hotfix (S, unblocks the reporter, byte-neutral for every in-bounds solve so D34's golden and both backends' parity hold):* **project the pinned commitment onto the actual enforced delivery box** instead of relaxing the LP bound. The box is the delivery stage's per-block `[col_lower·scale, col_upper·scale]` for that thermal — the same enforced (scaled, round-tripped) values `reconcile_commitment` already reads — intersected over the stage's blocks, since one commitment pins every block column; a positive must-run floor and a cap are handled by the same clamp as the zero floor, and a dormant `[0, 0]` stage clamps to zero. Drift past the box on either side is absorbed by the projection up to a margin sized from the box/LP magnitude (never from the commitment); the `Relaxed` arm stops lowering a generation bound below its floor or raising it above its cap.
  2. *Redesign (M, owner-gated):* retire the runtime "genuine over-commitment" verdict. As far as delivery-anchoring holds, a genuine over-commitment can only enter through declared inputs (`past_anticipated_commitments` above the delivery cap), so that check belongs to input validation (cobre-io / setup) and fails loud there, at load time. At solve time the pinned commitment is **projected onto the delivery box** — the intersection over blocks of the delivery stage's enforced per-block `[min, max]` for that thermal, both sides, never a hard-coded zero — a pure deterministic function of the value and the template, so reproducibility and order-invariance hold — and the projection distance is **tallied** (max absolute and relative drift per run, surfaced in the run summary, one `tracing::warn!` past a diagnostic threshold sized from the LP scale rather than the commitment). Nothing on this path aborts. Delete `AnticipatedCommitmentOutOfBounds` or re-scope it to the input validator. Rewrite — never delete — the `.claude/rules/sddp.md` contract: the invariant to keep is "a hair of drift never produces a false `Infeasible`" plus its two forbidden alternatives (unscaling-makes-it-redundant; opt-in hook); the clause this entry retires is "the margin is the discrimination line between solver noise and a modelling error". Tests: keep `anticipated_commitment_drifted_over_cap_is_absorbed`; move `anticipated_commitment_over_cap_seed_is_refused` to an input-validation reject; add regressions seeded a hair below a `0` floor (the report), a hair below a positive must-run floor, a hair above a cap, and against a dormant `[0, 0]` delivery stage, plus one with block-varying bounds so the intersection is pinned. An empty intersection over blocks is an input defect (the fishing equality can never hold) and belongs to the same load-time validator.
- **Alignment:** neutral
- **Status:** open — reported 2026-09-17 by users on an MPI production run; not yet reproduced locally (needs the reporter's deck or a synthetic case: an anticipated thermal idle at its delivery stage, long ring, production-scale LP). Owner to choose hotfix-then-redesign vs redesign-only; either way the redesign gets its own ticket.
- **Side observation (not part of this finding):** the same log shows `LB 3.03e12 > UB 1.87e11` (gap `−93.8%`) at iteration 37. Under a risk-neutral objective the lower bound cannot sit 16× above the forward-pass estimate beyond sampling noise; under CVaR the printed forward statistic is not an upper bound, so this may be benign. Ask the reporter for the risk configuration before treating it as a second defect.

## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — unified-roadmap

_(no entries yet)_
