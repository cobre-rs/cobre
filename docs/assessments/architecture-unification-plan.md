# Architecture Unification & Hot-Path Simplification

**Date:** 2026-04-18
**Branch:** `feat/highs-integration` (starting point)
**Scope:** `cobre-sddp`, `cobre-solver`, `cobre-cli`, `cobre-python`
**Status:** Bucket A decisions confirmed 2026-04-18; ready to promote
to a plan under `plans/architecture-unification/` for epic breakdown.

## Executive Summary

Epic-01 (B.1 non-alien `setBasis`) and Epic-02 (A.1 `clear_solver_state`)
successfully landed measurable performance wins, but the hot path now
carries several orthogonal runtime switches that multiply into a
combinatorial mesh:

- `WarmStartBasisMode::{AlienOnly, NonAlienFirst}` — rollback lever from
  epic-01 ticket-010.
- `CanonicalStateStrategy::{Disabled, ClearSolver}` — rollback lever
  from epic-02.
- `baked.ready: bool` — dual-path for structural cut baking.
- Three near-duplicate LP-solve orchestrations (forward / backward /
  simulation) that share most logic but drift on safety invariants. The
  recent 900 non-alien rejections in simulation
  ([`warm-start-observability-findings.md`](warm-start-observability-findings.md))
  was produced by exactly this drift: the forward path applies
  `enforce_basic_count_invariant`; simulation was overlooked.

This document proposes a clean-break simplification for cobre **v0.5.0**
that collapses the mesh into a single path and makes the most common
classes of bugs structurally impossible.

**The core bet**: the rollback levers (alien fallback, `Disabled`
canonical state, non-baked templates) were justified as safety nets
during landing. They have now served their purpose. Keeping them
permanently would double the test surface and let the
forward/backward/simulation drift continue indefinitely.

**Cost of the refactor**: ~4-6 weeks focused engineering, backwards
incompatible (v0.5.0 breaking release), loss of the in-tree ability to
A/B-compare the old paths. **Benefit**: one LP-solve entry point, strict
invariants enforced by types, hot-path reads as straight-line code.

---

## Current Surface Area — What We're Simplifying Away

### 1. LP-solve orchestration — three near-duplicate paths

| File                                           |  Lines | Role                 |
| ---------------------------------------------- | -----: | -------------------- |
| `crates/cobre-sddp/src/forward.rs`             | ~5,200 | Forward-pass driver  |
| `crates/cobre-sddp/src/backward.rs`            | ~5,200 | Backward-pass driver |
| `crates/cobre-sddp/src/simulation/pipeline.rs` | ~5,000 | Simulation driver    |

All three call `reconstruct_basis` → `solve_with_basis`, but:

- **Forward** calls `enforce_basic_count_invariant` after reconstruction.
- **Backward** skips it (ticket-008 proved `delta == 0` by construction
  for the baked path only).
- **Simulation** skips it and the assumption silently fails → the
  observability bug.

Additionally, each has its own way of handling "no stored basis"
(forward: cold solve; backward: cold solve; simulation: cold solve —
same outcome, three sites).

### 2. Hot-path branches per LP solve

Reading `HighsSolver::solve_with_basis` plus the `process_stage_backward`
worker closure today requires holding ~8 combinations in mind:

```
(AlienOnly | NonAlienFirst) × (baked | legacy) × (Disabled | ClearSolver)
  × (has_stored_basis | cold)
```

Only one combination is the production path (`NonAlienFirst × baked ×
ClearSolver × has_stored_basis`). The rest are either rollback levers
or cold-start fallbacks.

### 3. Invariants maintained by conventions, not types

- **`TrainingResult` struct-literal parity** — CLAUDE.md requires grep
  after every struct change because there are multiple literal
  construction sites.
- **`#[allow(clippy::too_many_arguments)]` budget** — a pre-commit
  script enforces "no more than 10" because the context-struct pattern
  hasn't been consistently applied.
- **`CapturedBasis` 4-broadcast wire format** — documented in CLAUDE.md
  as an invariant that "regressing either silently degrades warm-start
  quality without a compile error."
- **Declaration-order invariance** — repeated across multiple modules;
  enforced by D-case fixtures rather than by API.

Each of these is a design gap papered over with process.

### 4. Dual CLI/Python wiring

`run_training_phase` threads `warm_start_basis_mode` into its solver
factory. `run_simulation_phase` does not (the bug from
`warm-start-observability-findings.md`). The same duplication exists in
cobre-python. Every new config flag this session had to be wired
four times (CLI training, CLI simulation, Python training, Python
simulation). At least one wiring has failed in every such ticket.

---

## Design Principles (from the user)

### P1 — One hot path. No combinatorial mesh

The hot LP-solve path as seen from the driver supports exactly one shape:

```
load_model       (per worker, per stage; re-called when cuts change)
  → set_bounds   (per trial point)
  → solve(basis) (per trial point)
```

Inside the HiGHS implementation of `solve`, the sequence is
`Highs_clearSolver` → basis install → `Highs_run`, but that is an
implementation detail. The driver does not see it, call it, or branch
on it.

No `match warm_start_mode`, no `if canonical_state ==`, no `if
baked.ready`. The hot-path code reads top-to-bottom with no branches on
backend capability or rollback lever.

### P2 — `NonAlien` only, no fallback

**Cobre owns basis correctness.** The solver is not responsible for
salvaging a malformed basis; a basis we hand it must already be
consistent. No solver-level workaround is permitted.

- `cobre_highs_set_basis_non_alien` is the only warm-start entry point.
- If `isBasisConsistent` fails, we return
  `Err(SolverError::BasisInconsistent { ... })` — no silent rewrite via
  `accommodateAlienBasis`, no fallback to the alien setter.
- **Every caller that supplies a basis is responsible for proving it
  consistent.** `enforce_basic_count_invariant` becomes load-bearing
  (runs on every reconstruction, period) rather than conditional.
- **`AlienOnly` is deleted from `WarmStartBasisMode`**. In fact the
  whole enum goes away — there is no mode.
- **No escape hatch in any build**, not even `#[cfg(debug_assertions)]`.
  If reconstruction produces an inconsistent basis, we fix the
  reconstruction code; we do not mask it with a flag.

### P3 — Baked templates only (no incremental row mutation)

The trait has **one** method for topology: `load_model(&StageTemplate)`.
There is no `add_rows`, no `update_appended_rows`, no row-delete surface.
Cut-set changes are handled driver-side: SDDP tracks active cuts, assembles
the current `RowBatch` (CSR), calls `bake_rows_into_template(&base, &rows,
&mut baked_scratch)`, then `solver.load_model(&baked_scratch)`.

Rationale: HiGHS always receives a single homogeneous `Highs_passLp(COLWISE)`
per cut-set change. Under dual simplex the internal representation stays
aligned; no COLWISE+ROWWISE merge, no forced transpose, no factorization
invalidation beyond what a fresh model naturally requires.

- `baked.ready: bool` goes away. Templates are always baked.
- `stored_cut_row_offset` parameter in `reconstruct_basis` becomes
  always-zero; the parameter is removed.
- Cold-start only for iteration 0 (no cuts yet). From iteration 1 on,
  every solve uses baked templates.
- `basis_row_capacity` simplifies to `baked_templates[t].num_rows`
  everywhere.
- The `SolverInterface::add_rows` method is deleted outright. Its removal
  is the forcing function: with no `add_rows` on the trait, the non-baked
  path is not representable.

### P4 — Solve-to-solve independence contract

The trait expresses what SDDP needs in behavioral terms, not in the shape
of any one backend's API:

> **Each `solve` call is self-contained.** The result depends only on
> (a) the loaded model, (b) the current column/row bounds, (c) the
> `basis: Option<&Basis>` argument. It does NOT depend on state left
> over from prior `solve` calls — not on the previous trial point's
> optimum, not on the order trial points were submitted, not on any
> hidden cached data the solver accumulated.

Implementations achieve this however they like: clearing internal state
before each solve, relying on the underlying solver's natural
independence, or re-loading the model per solve. What matters to SDDP
is the guarantee.

- `CanonicalStateStrategy` enum goes away.
- `SolverInterface::clear_solver_state` is **not on the trait**. The
  contract is on `solve()`; how the implementation fulfills it is an
  implementation detail. For HiGHS that means calling `Highs_clearSolver`
  inside `solve()` before basis install and `Highs_run`. For other
  backends, whatever delivers the same guarantee.
- Work-distribution-invariant output is guaranteed by the contract, not
  by configuration.
- `basis: None` means "warm-start from whatever basis this solver
  instance currently holds." This does not violate independence — the
  current basis is itself a deterministic function of the prior
  (model, bounds, basis-in) tuple on this instance.

### P5 — Unified run path

- One driver function for forward/backward/simulation passes,
  parameterized by the **minimum** that genuinely differs:
  - what to capture from each solve (trajectory? cut dual? cost?)
  - what to do with captured state (feed next stage? generate a cut?
    write to parquet?)
- Shared primitives (solver workspace, basis reconstruction, invariant
  enforcement) live in **one** module and are called by all three
  drivers through a single entry point.
- Default for "no stored basis" is cold solve. One code path, one
  place.

### P6 — Invariants enforced by types, not by process

- Remove the `TrainingResult` grep rule by making the type impossible
  to construct incorrectly (builder pattern or exhaustive matches on
  fields).
- Replace the `too_many_arguments` suppression budget with an API
  where context structs are the only way to pass > 4 parameters
  (linted by a stricter clippy config).
- `CapturedBasis` wire format behind a single
  `try_from_broadcast_payload` constructor that owns the 4-broadcast
  protocol invariants; callers can't mis-pack.

### P7 — Comments explain behavior, not history

Production comments have accumulated a "changelog in the source code"
pattern: `// Forward-path invariant fix (ticket-009): after cut-set
churn, ...`, `// Post ticket-008: cut-value evaluation is no longer
performed ...`, `// backward path: delta==0 by construction
(ticket-008); no demotion pass`. A grep across `crates/cobre-sddp/src`
and `crates/cobre-solver/src` surfaces **113 such historical-reference
comments in production code** (not counting test modules, where
traceability to acceptance criteria can be legitimate).

Rules:

- **No references to epics, tickets, findings, assessments, or plans in
  production code comments.** These rot — a future reader has no way to
  verify what "ticket-009" claimed, whether it still applies, or
  whether the code has since been refactored beyond recognition. The
  git history (`git blame` / `git log -S`) carries this information
  authoritatively and is the single source of truth for "why was this
  written".
- **Comments answer "why is this non-obvious?", not "what does this
  do?"**. If a well-named function does the obvious thing, no comment
  is needed. If there is a subtle invariant (concurrency assumption, a
  HiGHS-internal quirk, a numerical workaround), that comment stays —
  but it describes the invariant itself, not the ticket that
  discovered it.
- **Module and function doc-comments describe the contract.** Not the
  journey. A rustdoc block that says "This module extracts study
  params. Originally lived in `setup/mod.rs`, extracted in epic-05,
  see `plans/setup-extraction/` for context" is deleted down to
  "This module extracts study params."
- **Delete "NOTE:" / "TODO:" stacks older than the current session.**
  If a `NOTE` references a ticket that already shipped (e.g.,
  `// NOTE: new_tight is always 0 after ticket-008`), the note is
  obsolete — either the code is correct as-is (delete the note) or
  the observation is now a load-bearing invariant (keep the
  observation, drop the ticket reference).

The guardrail: during the refactor, a reviewer should be able to read
any hot-path file and understand what it does without knowing any
project history. If understanding requires prior knowledge of an epic
or ticket, that's a comment-shape bug to fix.

### P8 — Test suite right-sizing

The test footprint has grown to the point where it measurably slows
development:

| Metric                                            |       Value |
| ------------------------------------------------- | ----------: |
| Total lines under `cobre-sddp/src`                |      83,691 |
| Total lines under `cobre-sddp/tests/`             |      17,263 |
| `#[test]` annotations in `fpha_fitting.rs` alone  |         138 |
| `#[test]` annotations in `lp_builder/template.rs` |         105 |
| Per-file test:production ratio (hot-path files)   | 2.1x – 3.6x |

For example, `backward.rs` carries ~1,130 lines of production code and
~4,040 lines of tests in the same file; `forward.rs` is 1,520 prod /
3,670 test. This is not inherently wrong — SDDP is numerically
sensitive and D01-D30 deterministic regression is load-bearing — but
after years of growth the suite has redundancies and opportunities
that should be audited.

Principles:

- **Every test earns its place.** A test is justified if (a) it
  documents a non-obvious contract, (b) it is a regression guard for
  a real past bug, (c) it is an acceptance criterion for a specific
  requirement, or (d) it is part of a coverage matrix (D-case sweep,
  cross-MPI determinism). Tests that exist because they were written
  during a ticket and nobody deleted them afterward are candidates
  for removal.
- **Categorize tests by intent.** Unit (property of one function) /
  Integration (multi-module contract) / E2E (full pipeline) /
  Regression (pinned past bug) / Conformance (D-case numerical
  snapshot). Category drives retention policy.
- **Remove redundant coverage.** When three unit tests differ only in
  input parameters, parameterize them into one. When an integration
  test duplicates a unit test's coverage, keep the integration test
  (it also checks wiring) and delete the unit duplicate.
- **Delete tests for deleted code paths.** The refactor deletes entire
  rollback-lever paths (`AlienOnly`, `Disabled`, non-baked templates).
  Every test asserting behavior specific to those paths is also
  deleted. This is expected to be a meaningful fraction of the suite.
- **Slow tests need justification.** Tests that take > 10 seconds
  must either (a) be mission-critical (D-cases qualify) or (b) be
  moved behind a `--features slow-tests` flag so they don't run in
  default `cargo test`. The `fpha_*` tests in `fpha_fitting.rs` are
  the current canonical "slow" group (already gated as `fpha_*` by
  convention; formalize the gating).
- **Tests in test modules (`#[cfg(test)]` in src files) must not
  exceed the production code in the same file.** When they do, split
  the tests into `crates/<crate>/tests/<module>_tests.rs`. This
  prevents a reader from having to scroll past 4,000 lines of tests
  to find the next production function.

The audit is a separate phase in the sequencing below; it runs
_after_ the deletions of phases 2-4 (which naturally eliminate many
tests mechanically), so the human review is only applied to what
remains.

---

## Target Architecture (after the refactor)

### Solver trait

```rust
pub trait SolverInterface {
    // Model topology (cold, called on initial setup and whenever the
    // driver re-bakes a new template — e.g., after a cut-set change).
    fn load_model(&mut self, template: &StageTemplate);

    // Per-trial-point setup (hot, frequent).
    fn set_col_bounds(&mut self, cols: &[usize], lower: &[f64], upper: &[f64]);
    fn set_row_bounds(&mut self, rows: &[usize], lower: &[f64], upper: &[f64]);

    /// Solve the LP.
    ///
    /// # Contract — solve-to-solve independence
    ///
    /// The result depends only on the currently-loaded model, the current
    /// column/row bounds, and the `basis` argument. It does NOT depend on
    /// state left over from prior `solve` calls.
    ///
    /// `basis = Some(&b)` installs `b` before running the simplex.
    /// `basis = None` warm-starts from whatever basis this instance holds
    /// (itself a deterministic function of the prior inputs on this
    /// instance).
    ///
    /// Implementations fulfill the contract however they like — internal
    /// state clearing, natural solver independence, per-solve reload.
    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError>;

    // Extraction — driver-owned buffers, zero allocation.
    fn copy_primal_into(&self, out: &mut [f64]);
    fn copy_duals_into(&self, out: &mut [f64]);
    fn copy_basis_into(&self, out: &mut Basis);
    fn objective(&self) -> f64;

    // Observability — borrowed refs / static strings, zero allocation.
    fn statistics(&self) -> &SolverStatistics;
    fn name(&self) -> &'static str;
    fn solver_name_version(&self) -> &str;  // cached at construction
}
```

Removed methods relative to today's trait:

- `add_rows` — cut-set changes are driver-side re-bakes + `load_model`.
  Eliminating this method is what makes "baked only" unfalsifiable.
- `clear_solver_state` — contract lives on `solve`; implementation detail.
- `solve_with_basis` — folded into `solve(Option<&Basis>)`.
- `reset` — replaced by `load_model` for topology resets; the
  solve-independence contract handles per-solve state.
- `record_reconstruction_stats` — rolled into driver-side telemetry.

**Crate-independence follow-up (not blocking this trait work):** the
current `StageTemplate` carries SDDP-specific semantic fields
(`n_state`, `n_transfer`, `n_dual_relevant`, `n_hydro`, `max_par_order`)
in `cobre-solver`. These are pre-existing leaks of algorithm semantics
into the solver crate. Worth a dedicated cleanup ticket to move them to
an SDDP-side side-car type keyed by the stage template, but out of scope
for the unification refactor.

### LP-solve entry point

```rust
pub enum Phase { Forward, Backward, Simulation }

pub fn run_stage_solve(
    workspace: &mut SolverWorkspace,
    ctx: &StageContext,
    phase: Phase,
    trial: &TrialPoint,
    basis: Option<&CapturedBasis>,  // None → cold solve
) -> Result<StageOutcome, SddpError>;
```

Single entry point. The `Phase` enum is used only inside to gate:

- **Forward**: capture basis after solve.
- **Backward**: compute dual values, enqueue cut generation.
- **Simulation**: serialize outcome for parquet.

No other branching. The bound updates, basis reconstruction, invariant
enforcement, and solve call are identical across phases — only the
post-solve extraction differs.

### Counters

```rust
pub struct SolverStatistics {
    // core
    pub lp_solves: u64,
    pub lp_failures: u64,
    pub simplex_iterations: u64,
    pub solve_time_ns: u64,
    pub load_model_count: u64,     // how often the template was re-baked
    pub load_model_time_ns: u64,

    // warm-start observability — one rule, no mode dependency
    pub basis_offered: u64,
    pub basis_consistency_failures: u64,  // basis-in rejected
    pub basis_solves_successful: u64,     // accepted + solved
}
```

Deleted: `basis_rejections` (misleading under alien), `basis_non_alien_rejections`
(no longer a separate path), the `basis_preserved` / `basis_new_tight` /
`basis_new_slack` / `basis_demotions` reconstruction breakdown (rolled into a
single `basis_reconstructions` + optional detail via a debug build),
`clear_solver_count` and `add_rows_count` (no longer trait-level operations).

### Config schema

Removed keys:

- `training.solver.warm_start_basis_mode` (gone — no mode to select)
- `training.solver.canonical_state` (gone — solve-independence is always on)
- `training.cut_selection.basis_padding` (seen in `convertido_before`;
  appears to be legacy — audit and remove if unused)

The config becomes strictly smaller. Parsers that encounter the removed
keys emit a clear migration error.

---

## Explicit Deletions (what goes away)

### Code

- `WarmStartBasisMode` enum (`crates/cobre-solver/src/highs.rs`)
- `cobre_highs_set_basis` C API wrapper usage for warm-starts (kept
  only for initial cold-start if needed; likely removable)
- `CanonicalStateStrategy` enum (`crates/cobre-sddp/src/config.rs`)
- `BakedTemplates::ready: bool` + all `if baked.ready` branches
- `reconstruct_basis`'s `stored_cut_row_offset` parameter (always 0)
- The fallback (non-baked) arm in `simulation/pipeline.rs:446-467`
- The "legacy path" comments scattered across forward.rs/backward.rs
- `SolverInterface::add_rows` method (trait-level) and all call sites.
  Driver-side cut baking + `load_model` replaces it.
- `SolverInterface::clear_solver_state` method (trait-level). The
  solve-independence contract subsumes it; implementations handle state
  clearing internally.
- `SolverInterface::solve_with_basis` method (folded into
  `solve(Option<&Basis>)`).
- `SolverInterface::reset` method (no caller after `load_model` covers
  topology reset).
- `cobre_highs_add_rows` C FFI wrapper and the underlying `Highs_addRows`
  call site (unused after trait deletion).
- `add_rows_count` / `total_add_rows_time_seconds` / `load_model_count`
  bookkeeping specific to the dual-path are consolidated into the new
  `SolverStatistics` shape.

### Wire / broadcast fields

- `BroadcastConfig::warm_start_basis_mode`
- `BroadcastConfig::canonical_state_strategy`

### Tests

- Any D-case regression case whose sole purpose is asserting
  alien-only or disabled-canonical-state equivalence. A/B tests
  against the legacy path become meaningless once the legacy path
  doesn't exist.
- Parameter-sweep unit tests that collapse into one parameterized
  test.
- Tests asserting behavior of code paths deleted under P2-P4
  (rollback levers, non-baked reconstruction, `Err(Unsupported)`
  handling for `clear_solver_state`).
- Target retention: suite runs in under half the current wall time
  once the dead tests are removed.

### Comments

- The 113+ production-code comments referencing `ticket-XX`,
  `epic-XX`, `finding-XX`, `plan-XX`, `assessment-XX`. Each is either
  rewritten to describe the invariant without the historical marker
  or deleted outright if the underlying behavior is now obvious.
- `// NOTE:` / `// TODO:` lines referencing already-shipped tickets.
- Rustdoc module-level prose that describes the refactoring history
  of the module (e.g., "extracted from `setup/mod.rs` in epic-05").
- Inline `// SAFETY:` comments stay (they describe load-bearing FFI
  invariants). Historical narrative inside them gets trimmed.

### Documentation

- Anything explaining the rollback levers.
- The `TrainingResult` struct-literal parity rule.

---

## Risks & Open Questions

### R1 — Hard error on `BasisInconsistent` may surface latent bugs

Today's fallback to the alien path + `accommodateAlienBasis` silently
hides basis-reconstruction bugs. After P2, any reconstruction error
becomes an observable test failure.

**Mitigation**: Land P2 _after_ the existing simulation
`enforce_basic_count_invariant` fix, and with D01-D30 passing as
acceptance. If the hard-error mode surfaces new inconsistencies,
fix them before wider rollout.

**Decision**: No escape hatch in any build. Not an env var, not
`#[cfg(debug_assertions)]`, not a build feature. The hot path is
literally the same code in debug and release. If reconstruction is
buggy, fix reconstruction. Consequence of P2's "Cobre owns basis
correctness" — there is nowhere to hide an inconsistent basis.

### R2 — Baked-only removes config-less flexibility for structural changes

Cobre currently does not support structural changes (new constraints)
mid-run. Baked-only codifies this assumption. If the project ever needs
to support, e.g., dynamic network topology changes, the baked-template
approach would need rework.

**Mitigation**: Not a regression — today's "legacy non-baked path" also
doesn't support mid-run structural changes. Deleting it only removes an
illusion of flexibility.

### R3 — Backend plurality

The trait's only behavioral requirement is **solve-to-solve
independence**: `solve(basis)` results depend only on (loaded model,
bounds, basis-in) and never on prior solves. How a backend delivers this
is its business — clearing internal state, relying on natural
independence, re-loading the model per solve, or any combination.

**Mitigation**: Document the contract. Any LP solver — HiGHS, Gurobi,
CPLEX, COIN-OR CLP, a future open-source backend — can back Cobre as
long as it can provide independent solves. This is a far weaker
restriction than "must expose a cheap explicit state-clear call."

**Decision**: No `Err(Unsupported)` escape. A backend either provides
the independence guarantee or it is not a Cobre backend. For HiGHS the
implementation is `Highs_clearSolver + Highs_run` inside `solve`; other
backends pick their own equivalent.

### R4 — Breaking change for existing users

Existing config files with `warm_start_basis_mode` or `canonical_state`
will fail to parse. Existing policy checkpoints may have been captured
with the legacy path.

**Mitigation**:

- Release as **v0.5.0** with a clear CHANGELOG entry.
- Config parser emits a migration error: `"key X was removed in v0.5.0;
delete from config.json"`.
- Policy checkpoints carry a schema version; v0.5.0 rejects checkpoints
  from v0.4.x (they were captured under different invariants). Users
  re-train, or we write a one-time migration tool if demand justifies
  it.

### R5 — Test-surface collapse may hide regressions

If we delete the `AlienOnly` D-case sweeps, we lose the A/B signal that
ticket-010 used to validate epic-01. Future regressions could go
undetected.

**Mitigation**: Keep the D01-D30 determinism sweeps, but only against
the single remaining path. Add a cross-worker / cross-MPI determinism
matrix (which is ticket-007/008 work anyway). If a regression appears,
it must be debugged on the unified path — which is exactly the goal.

### R6 — `TrainingResult` and other API types need breaking changes

Making these types impossible-to-misconstruct means removing public
fields, introducing builders, or restructuring. Downstream callers
(cobre-bridge, cobre-python, any user code) break.

**Mitigation**: v0.5.0 is the breaking-change window. Document the new
builder-based API in the migration guide.

---

## Proposed Sequencing

**Phase 0a — Finish in-flight work** ✅ complete (2026-04-18)

- Simulation bugs closed (simulation warm-start mode +
  `enforce_basic_count_invariant`).
- Epic-02 tickets 006-010 landed.
- `feat/highs-integration` branch merged to `develop`.
- v0.4.5 remains available to cut when needed; ships with the rollback
  levers still live and observable.

**Phase 0b — Test inventory and categorization**

Before any deletion, label every test in `cobre-sddp/src` and
`cobre-sddp/tests/` (plus related crates) with two tags:

- **Intent**: `unit` / `integration` / `regression-for-<X>` /
  `conformance` / `parameter-sweep` / `coverage-matrix`.
- **Guards**: which code path the test depends on — e.g., `AlienOnly`,
  `Disabled`, `non-baked`, `unified-path`, `generic`.

Output is an inventory committed to
`docs/assessments/test-inventory.md`. No deletions in this phase.

This makes phase 2-4 deletions mechanical ("tests tagged
`guards: AlienOnly` go with phase 2") and gives phase 8 a clean
starting point for judgment-based consolidation. Forcing the pass now
— while context is fresh — is cheaper than re-deriving test intent
six months later.

**Phase 1 — Unified run path (P5) on top of existing switches**

- Introduce `run_stage_solve` that calls the current multi-path code.
  No behavior change, but now every site goes through one function.
- Remove duplicate LP-solve orchestration in forward/backward/simulation.
- Deliverable: same behavior; sum of production LoC across the three
  drivers strictly decreases.

**Phase 2 — Drop `WarmStartBasisMode::AlienOnly` (P2)**

- Flip all call sites to `NonAlienFirst`.
- Run D01-D30 + convertido on `NonAlienFirst`-only.
- If determinism holds (it should, based on epic-01 evidence), delete
  `AlienOnly` entirely.
- Hard-error on `isBasisConsistent` failure.
- Deliverable: `cobre_highs_set_basis` disappears from the hot path.

**Phase 3 — Collapse to the solve-to-solve independence contract (P4)**

- Flip all call sites to `ClearSolver`.
- Re-run determinism and convertido.
- Delete `CanonicalStateStrategy::Disabled` variant; delete
  `SolverInterface::clear_solver_state` from the trait; move the
  equivalent work (currently `Highs_clearSolver` + basis install +
  `Highs_run`) inside the HiGHS implementation of `solve`.
- Fold `solve_with_basis` into `solve(Option<&Basis>)`.
- Deliverable: trait exposes one solve method; hot path has no
  configuration branches; the independence contract is enforced by the
  API surface, not by driver orchestration.

**Phase 4 — Drop the legacy (non-baked) template path (P3)**

- Delete `SolverInterface::add_rows` from the trait. This is the forcing
  function — without the method, the non-baked path cannot compile.
- Make baked-template construction unconditional in the driver.
- Delete `baked.ready` checks and the legacy `cut_batches` rebuild.
- Delete the `cobre_highs_add_rows` FFI wrapper and its call sites.
- Deliverable: `basis_row_capacity = baked[t].num_rows` everywhere;
  HiGHS only ever sees `Highs_passLp(COLWISE)`, never `Highs_addRows`.

**Phase 5 — Type-level invariant enforcement (P6)**

- Replace the `TrainingResult` grep rule with a builder/non-exhaustive
  type.
- Replace the `too_many_arguments` budget with context-struct-only
  signatures (stricter clippy).
- Consolidate `CapturedBasis` broadcast behind a single safe
  constructor.
- Deliverable: CLAUDE.md shrinks; the pre-commit script is deleted.

**Phase 6 — Counter hygiene + config cleanup**

- Rename/consolidate `SolverStatistics` fields per the proposed target.
- Remove deprecated config keys with clear migration errors.
- Update parquet schemas and book docs.
- Deliverable: `SolverStatistics` shape finalized for v0.5.0.

**Phase 7 — Comment hygiene pass (P7)**

- Strip production-code historical references (tickets, epics,
  findings, plans, assessments). Rewrite each comment to describe
  the invariant or quirk it encodes, or delete it outright.
- Strip module-level rustdoc that narrates refactoring history.
- Delete stale `NOTE:` / `TODO:` annotations referencing shipped
  tickets. Keep only ones describing currently-active future work.
- Deliverable: `rg "(ticket|epic|finding|plan|assessment)-?[0-9]"
crates/cobre-{sddp,solver,io,cli,python}/src` returns zero hits
  in production code (test modules may retain AC traceability).
- This phase is purely a read-through exercise; no behavior change.
  Sized for one focused review session per hot-path file.

**Phase 8 — Test suite audit (P8)**

- Run the suite once to establish a baseline wall time (for both
  `cargo test --workspace` and the `D01-D30` sweep).
- For each `#[cfg(test)] mod tests` block in production files, apply
  the P8 rules: categorize, collapse parameter-sweep duplicates,
  extract oversized blocks into `tests/<module>_tests.rs`.
- For each `crates/*/tests/*.rs` file: review against the
  "every test earns its place" criterion. Delete regressions against
  deleted code paths from phases 2-4. Parameterize duplicates.
- Gate slow suites (fpha\_\*, full D01-D30) behind an opt-in cargo
  feature `slow-tests`; default `cargo test` runs the fast tier.
- Deliverable: `cobre-sddp/tests/` reduced in line count by a
  concrete target (see Success Criteria); default suite wall time
  reduced by ≥ 50%; D01-D30 still passes under `--features
slow-tests`.

Phases 1-6 deliver the architectural change. Phases 7 and 8 are
dedicated post-cleanup passes; they can run in parallel with each
other once phases 1-6 land, and they can be paused and resumed
without blocking anything.

Each phase is individually reviewable and individually revertible.
Phases 1, 5, and 7 are mostly mechanical; phases 2, 3, 4 require
determinism validation on real workloads; phase 8 requires human
judgment per test and is the longest-wall-clock of the set.

---

## Success Criteria

- **Hot path audit**: `rg "match.*WarmStartBasisMode|if.*canonical_state|if.*baked\.ready"
crates/cobre-{sddp,solver}/src` returns zero results after phase 4.
- **Single entry point**: exactly one `run_stage_solve` function;
  `forward.rs`, `backward.rs`, and `simulation/pipeline.rs` contain
  only phase-specific capture logic (what to do with the solution),
  never orchestration (load, basis, solve, clear).
- **Straight-line hot path**: reading `run_stage_solve` top-to-bottom
  contains no `match` on `WarmStartBasisMode`, no `if
canonical_state ==`, no `if baked.ready`. Configuration does not
  enter the hot path.
- **Monotonic LoC**: sum of production LoC across the three driver
  files decreases at every phase 1-4 boundary. No target number — the
  commitment is direction, not magnitude.
- **Read-through time**: a new engineer can trace one LP solve from
  `run_stage_solve` to `highs::solve` without a cross-reference table.
- **No hidden invariants**: `CLAUDE.md` loses all three of the
  "struct-literal parity", "too-many-arguments budget", and
  "basis-cache metadata integrity" items. They become unneeded because
  the types enforce them.
- **Determinism preserved**: D01-D30 + convertido produce byte-identical
  results under the unified path, matching the v0.4.5 `ClearSolver +
NonAlienFirst + baked` reference.
- **Observability honest**: `basis_consistency_failures` is the single
  source of truth for warm-start rejections. `basis_rejections` (the
  misleading counter) is gone.
- **Historical references purged (P7)**:
  `rg "(ticket|epic|finding|plan|assessment)-?[0-9]"
crates/cobre-{sddp,solver,io,cli,python}/src` returns zero results
  outside `#[cfg(test)]` blocks after phase 7. A sampled reviewer can
  read any hot-path file without knowing project history and still
  understand what it does.
- **Test suite right-sized (P8)**:
  - `cobre-sddp/tests/` line count strictly decreases from its
    phase-0b baseline; the concrete target is set at the phase-8
    boundary once the inventory is in hand.
  - Default `cargo test --workspace` wall time strictly decreases;
    the concrete target is likewise set at phase 8.
  - No in-source test module exceeds the production code of its host
    file; oversized modules have been extracted to
    `tests/<module>_tests.rs`.
  - D01-D30 + `fpha_*` still pass, gated behind `--features
slow-tests` (or equivalent) so default runs stay snappy.

---

## What This Is Not

- **Not a rewrite.** The simplex, cut pool, stochastic pipeline, and
  MPI plumbing are unchanged. This is surgery on orchestration,
  trait boundaries, and config surface.
- **Not a performance ticket.** The wall-time delta from this refactor
  is expected to be near zero (all rollback-lever branches already
  run in the fast path on one side or the other). Performance work
  continues separately (B.2, Phase C).
- **Not a deprecation window.** v0.4.5 is the last release with the
  rollback levers. v0.5.0 removes them outright. Users who need the
  legacy paths pin to 0.4.x.

---

## Alternative Considered: Keep levers, unify orchestration only

A partial-refactor that collapses forward/backward/simulation
duplication (P5) but keeps the rollback levers. This would fix the
simulation bug class (same code paths) while preserving the A/B
capability.

**Rejected because**: the rollback levers are the primary source of
cognitive load in the hot path. Deleting the orchestration duplication
without deleting the levers leaves the worst of both worlds — a unified
driver that still does 8-way branching on every solve. Either commit
to one path or keep the duplication; half-measures produce the current
state.

---

## Alternative Considered: `#[cfg(debug_assertions)]`-gated levers

Delete the rollback levers from release builds but keep them behind
`#[cfg(debug_assertions)]` for in-tree A/B debugging. Zero
release-build cost; preserves the ability to re-enable `AlienOnly` or
`Disabled` for a single comparison run.

**Rejected because**: the hot path must be literally the same code in
debug and release. Two shapes drift — the debug shape rots silently,
and the moment we need it for comparison we discover it doesn't match
release anymore. If a post-refactor A/B is ever needed, capture a
convertido or D-case artifact from v0.4.x and diff out-of-tree; the
in-tree escape hatch costs more maintenance than it saves.

---

## Decision Points Requiring User Approval

Decisions are bucketed by when the answer is needed and how reversible
the commitment is. The upfront approval gate is exactly three
questions (Bucket A); the rest are answered as their phase opens.

### Bucket A — Architectural, decide before phase 1 starts

These change the trait shape and failure model. Every subsequent phase
assumes them. Irreversible without a second v0.5-class refactor.

1. **Hard-error on `BasisInconsistent`?** — **YES** (confirmed
   2026-04-18). No escape hatch in any build. Cobre owns basis
   correctness.
2. **Solve-to-solve independence as the trait contract?** — **YES**
   (confirmed 2026-04-18). The trait has no `clear_solver_state`
   method; instead, `solve(basis)` is contractually self-contained and
   implementations deliver the guarantee however they like. Any solver
   that can provide independent solves can back Cobre.
3. **Baked templates only, no `add_rows` on the trait?** — **YES**
   (confirmed 2026-04-18). Cut-set changes are driver-side re-bakes +
   `load_model`. Deleting `add_rows` from the trait is the forcing
   function that makes the non-baked path unrepresentable. Mid-run
   structural changes remain unsupported (no regression — today's
   legacy path doesn't support them either).

### Bucket B — Phase-boundary calibration

Set when the relevant phase opens, after phase-0b inventory data is
in hand. Pre-committing specific numbers now is guessing.

4. **Comment-hygiene scope (P7)**: production only, or also
   `crates/*/tests/`? Default recommendation: production only. Test
   modules legitimately use AC traceability as a design note and
   aren't hot-path reading. Revisit at phase 7 if the test modules
   turn out to carry rot too.
5. **Test-audit aggressiveness (P8)**: set reduction targets once
   phase-0b categorization reports what's actually in the 17,263
   lines of `cobre-sddp/tests/`. No pre-commitment to 30% / 50%.
6. **Slow-test gating**: gate `fpha_*` and full D01-D30 behind a
   `slow-tests` cargo feature? Default recommendation: yes — they're
   the primary drag on default `cargo test`. Finalize at phase 8
   based on what the categorization surfaces.

### Bucket C — Release-time decisions

Stable enough to wait until release planning.

7. **Target release: v0.5.0 (breaking)?** Default: yes. Confirm at
   release-branch cut, not now.
8. **Phase ordering**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 as proposed,
   unless phase-0b reveals test-suite drag is blocking iteration
   speed; in that case, hoist phase 8 earlier.

Once Bucket A is confirmed, the refactor becomes a plan under
`plans/architecture-unification/` with concrete epic breakdowns.
Buckets B and C are revisited as their phases open.
