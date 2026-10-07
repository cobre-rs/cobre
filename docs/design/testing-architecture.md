# Uniform Testing Architecture

> **Status:** Partially adopted (§5.2); the rest Proposal — target standard, not yet implemented. Snapshot figures are calibration-time context; re-measure before acting.

**Scope:** the whole workspace's test strategy — layering, per-crate structure,
shared fixtures, comparators, runner, CI tiering, and correctness hardening.
**Relationship to other docs:**

- **Layer 1 (binary structure)** of this standard — the concrete
  `#[path]`-submodule grouping mechanics and the domain-binary grouping approach
  for consolidating the per-file integration binaries — is specified in §5.1
  below. This document subsumes and generalizes it.
- `.claude/rules/testing.md` is the authoritative tier taxonomy; this document
  **formalizes and extends** it (adds property / compile-fail / hardening tiers,
  a decision rule, and a comparator standard) without weakening any existing
  contract.

The figures below are a point-in-time calibration snapshot, each paired with a
command to re-measure it; the standard itself is stated as invariants, not
counts.

---

## 1. Problem statement

The suite is **strong on correctness and weak on uniformity/sustainability.**
The advanced machinery — the four test tiers, golden baselines, the determinism
gate binary, the shared fixture harness — lives almost entirely in `cobre-sddp`;
every other crate tests ad hoc. The workspace links **one integration binary per
`tests/*.rs` file** — most of them statically linking the C++ solver — homes unit
tests inconsistently (giant inline modules next to extracted `tests.rs` siblings
_in the same directory_), runs the entire slow suite on every PR, and has no uniform
cross-crate fixture-sharing convention, no nextest configuration, and no
correctness-hardening tier (miri / sanitizers / fuzz). None of this is broken; all of it scales badly and
resists being taught to a new crate or contributor. The objective is a **single,
uniform, per-crate testing standard** that preserves the correctness depth while
making the suite cheap to build, consistent to navigate, and safe to extend.

---

## 2. Current state

### 2.1 Shape

The counts below rot silently, so they are stated as the mechanism that measures
each rather than a frozen census. Re-measure before acting on any of them.

| Metric                               | How to re-measure                                                                                          |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Integration-test binaries, per crate | `for d in crates/*/tests; do printf '%s %s\n' "$d" "$(find "$d" -maxdepth 1 -name '*.rs' \| wc -l)"; done` |
| …that statically link the solver     | the subset of the above whose crate depends on `cobre-solver` (`cobre-sddp`, `cobre-cli`, `cobre-solver`)  |
| Unit + integration test count        | `cargo nextest list --features test-support`                                                               |
| Doctests                             | `cargo test --doc`                                                                                         |
| pytest (`cobre-python`)              | `pytest crates/cobre-python --collect-only -q`                                                             |
| Golden bit-exact cases               | a small deliberate set × 2 backends (HiGHS/CLP); enumerate the `tests/fixtures/parity_baselines*` decks    |
| `to_bits`/ULP determinism assertions | `git grep -c 'to_bits' -- 'crates/**/*.rs'`                                                                |
| `proptest!` sites                    | `git grep -l 'proptest!' -- 'crates/**/*.rs'`                                                              |
| Slow-gated (`slow-tests`) attributes | `git grep -c 'slow-tests' -- 'crates/**/*.rs'` — concentrated entirely in `cobre-sddp`                     |

The invariants those numbers evidence — which do **not** rot — are: unit tests
outnumber integration tests by roughly an order of magnitude (a healthy
pyramid), so the cost is not test _count_ but integration-binary _count_ ×
static-solver-link; and the solver-linking crates (`cobre-sddp`, `cobre-cli`,
`cobre-solver`) own most of the integration binaries and therefore most of the
link cost.

### 2.2 Where the sophistication lives

`cobre-sddp` is the sole home of: all four test tiers, the only large
`tests/common/` harness (`StubComm`/`Rank0Of2`, `build_setup_*`, `make_*`
builders, `parity_hash`, `permute`), the only `tests/fixtures/` (dual golden
baselines + deterministic case decks), the `mpi_wire.rs` determinism-gate binary
(with its power self-checks), and the only `benches/`. Every other crate is unit

- behavioral only, constructing fixtures per-file.

### 2.3 Tooling & CI topology

- **Runner is inconsistent**: the HiGHS `Test` job uses `cargo test`; the CLP
  job and the shuffle-matrix job use `cargo nextest`. There is **no
  `.config/nextest.toml`** — no profiles, retries, partitioning, JUnit, or
  archive.
- **No CI cadence tiering**: `NON_SOLVER_FEATURES` (used by `Test`, `CLP`, and
  `Coverage`) **includes `slow-tests`**, so the full slow suite runs on **every
  PR**. The `slow-tests` cargo feature therefore functions only as a _local-dev_
  convenience, not as a CI tier. The one lighter job is `Check`.
- **Order-invariance shuffle matrix** (`invariance-shuffle.yml`) is
  **`workflow_dispatch`-only** — its nightly cron is commented out — so a
  _hard-rule_ determinism guarantee is exercised in automation only on manual
  dispatch.
- **Coverage** via `cargo-llvm-cov` → codecov (HiGHS backend). Good.
- **Real multi-rank MPI** via a SLURM Docker cluster on `examples/4ree`
  (`mpi-slurm.yml`). Good. In-process MPI via `StubComm`/`Rank0Of2` across the
  in-process rank-shape tests.
- **Absent**: miri, sanitizers/valgrind, fuzzing, snapshot tooling
  (`insta`/`expect-test`), a uniform cross-crate fixture convention (the
  `test-support` feature exists but is applied inconsistently), and any
  `cargo test`/doctest run for the workspace-excluded `cobre-python` (its Rust
  unit tests and doctests never compile in CI).

---

## 3. Assessment (test-engineer lens)

### 3.1 What is genuinely strong — keep and standardize, do not disturb

1. **An explicit four-tier taxonomy, including an analytical-derivation tier.**
   Golden-bit-exact / behavioral / LP-structural / analytical closed-form. The
   analytical tier is the Rust analog of the scientific-computing **Method of
   Manufactured Solutions** — proving _correctness against a derived truth_, not
   "same as last time." Most Rust projects have nothing like it; this is ahead of
   the field and must be preserved and named.
2. **Contract-pinning.** Every invariant in `.claude/rules/sddp.md` is tied to a
   named regression test. This is the single best property of the suite.
3. **Determinism discipline at HPC grade.** Pervasive bit-exact assertions, the
   `mpi_wire` gates with _power self-checks_ (a gate proves it exercises the
   condition it guards), dual HiGHS/CLP golden baselines, an order-invariance
   shuffle matrix, and real multi-rank MPI. This matches the strict end of the
   parallel-numerics reproducibility spectrum.
4. **O(1) fixture field-add builder** (`tests/common/builders.rs`).
5. **Coverage measured; compile-fail tested (via doctests); property tests for
   ordering.**

### 3.2 Sustainability & uniformity problems (ranked by leverage)

1. **Binary sharding — the dominant structural cost.** Rust links one binary per
   `tests/*.rs`; with `n` binaries and `m` libraries the linker does `m·n` work,
   and Cargo runs integration binaries _sequentially_. Cobre amplifies this by
   statically linking a C++ LP solver into every solver-linked binary. (When
   Cargo itself moved to a single integration binary, compile time dropped **3×**
   and artifacts **5×** — matklad.) This is Layer 1; its mechanics are in §5.1.
2. **Per-crate non-uniformity.** There is no crate testing _standard_: tiers,
   the shared harness, fixtures, and golden management exist only in `cobre-sddp`.
   A contributor to `cobre-io` or `cobre-stochastic` has no template to follow.
   This is the user's central concern and the reason the suite "isn't
   sustainable" — it cannot be taught, only imitated from one crate.
3. **Fixture / test-support fragmentation.** The cross-crate mechanism is a
   `test-support` _cargo feature_ that gates internal symbols behind
   `#[cfg(any(test, feature = "test-support"))]`; the reusable builders live in
   `cobre-sddp/tests/common/`; `cobre-python` (workspace-excluded) shares nothing
   and its Rust tests never run in CI. Three different sharing mechanisms, none
   uniform.
4. **Inline-giant vs extracted-sibling asymmetry — intra-directory.**
   `lp/builder/entries.rs` and `columns.rs` keep multi-thousand-line test
   modules inline while their siblings `template.rs`/`layout.rs` use extracted
   `tests.rs`. The homing rule is a coin-flip.
5. **No CI cadence tiering.** Everything (including the entire slow suite) runs
   on every PR, while the order-invariance shuffle — a hard-rule guarantee —
   runs only on manual dispatch. This is backwards: the cheap high-signal gate is
   under-run and the expensive sweep over-runs. It also contradicts the project's
   own recorded intent that slow tests not burden routine CI.
6. **Runner inconsistency + zero nextest configuration.** `cargo test` vs
   `nextest` across jobs; no profiles, no per-test process isolation guarantees
   documented, no partitioning/sharding, no JUnit dashboard feed, and — most
   costly for an HPC project — **no use of `nextest archive`** to build on a CI
   node and run on Slurm/MPI compute nodes.
7. **Correctness-hardening gaps for an FFI/HPC solver.** No miri on the pure-Rust
   `unsafe` (the isolated `gemm` kernel, raw-buffer reuse); no sanitizer/valgrind
   job for the C++ solver / MPI / PyO3 FFI; no fuzzing of the parser-heavy
   `cobre-io`; property testing at only 5 sites despite pervasive
   sort/canonicalization/reduction invariants that are its ideal target.
8. **Ad hoc golden and matrix management.** Golden re-baselining is prose
   discipline; the HiGHS×CLP × mpi/numa/shared-memory × slow feature matrix is
   uncodified (the CLP suite can silently rot, per the register's
   backend-scoped-parity caveat); MPI tests carry no rank-weight, so concurrent
   multi-rank tests can oversubscribe cores.

---

## 4. How mature projects test (benchmark)

### 4.1 Large Rust projects

- **One integration binary.** `tests/it/main.rs` with everything else a `mod`
  submodule; libtest still parallelizes the `#[test]`s within it. Adopted by
  Cargo and rust-analyzer (matklad, _Delete Cargo Integration Tests_ /
  _Fast Rust Builds_). The `m·n` link argument is exactly cobre's amplified case.
- **cargo-nextest.** Process-per-test isolation (a solver segfault or MPI
  deadlock kills one process, not the binary's whole result set), `hash:`
  partitioning for sharding, retries with flaky-marking, JUnit XML, and
  **`archive`** (build on one node, ship a `.tar.zst`, run on compute nodes with
  no toolchain — the standout HPC feature).
- **Snapshot testing.** `insta` (file + inline, `cargo insta review`) and
  rust-analyzer's `expect-test` (`UPDATE_EXPECT=1` patches expectations inline) —
  for verbose structured output (report tables, cut-pool dumps, LP structure).
- **Property + compile-fail.** `proptest` (regression seeds committed to VCS) for
  invariants; `trybuild` for API-misuse diagnostics (only if proc-macros ship).
- **Hardening.** tokio runs dedicated `miri`, `loom`, `valgrind`, and
  `-Zsanitizer=address` jobs; miri via `cargo miri nextest` (isolation disabled,
  strict provenance).
- **Centralized shared fixtures — a `*-test-support` crate _or_ a `test-support`
  feature.** tokio-test, rust-analyzer `test-utils`, polars-testing, datafusion
  `test_util` are dedicated crates; the lighter, equally-common variant is a
  `test-support` cargo feature on existing crates. Either centralizes
  fixtures/comparators workspace-wide; cobre uses the feature variant (§5.2).
- **CI tiering.** tokio/DataFusion/polars all split PR-fast from an extended
  main-branch/nightly tier (the full edge-case sweep runs off the PR path).

### 4.2 HPC / scientific computing (C++, Julia)

- **CTest labels + timeouts + `PROCESSORS` weights** (Trilinos/TriBITS, deal.II,
  PETSc): one test registry sliced at _run time_ by label regex into
  fast/slow/perf; per-test timeouts kill hung MPI/solver tests; each MPI test
  declares a rank weight so the scheduler never oversubscribes cores, and a
  max-rank budget auto-excludes over-budget tests so the same suite runs
  laptop→cluster.
- **Tolerance-aware golden diff with "alt" outputs** (deal.II `numdiff`, PETSc
  `petscdiff` + `alt` files): scientific codes _cannot_ demand byte-equality
  across machines, so they diff with FP tolerance and allow _multiple legitimate
  outputs_. Cobre sits at the strict end (bit-exact within a mode); the PETSc
  `alt`-file idea is the clean precedent for encoding "different-but-valid
  optimal vertex" (the hot≠cold case cobre already exempts).
- **Verification & Validation.** The **Method of Manufactured Solutions** and
  **order-of-accuracy** tests (Sandia SAND2000-1444; deal.II `ConvergenceTable`)
  prove correctness against a _derived_ truth — exactly cobre's analytical tier,
  and the strongest correctness signal a numerical code can have.
- **Nightly dashboards + promotion gates** (CDash; Trilinos requires a clean
  nightly before `develop`→`master`) — maps onto cobre's `develop`→`main`
  discipline; nextest JUnit is the Rust-native dashboard feed.
- **Julia optimization (closest domain).** SDDP.jl seeds `Random.seed!(12345)` in
  its test runner and runs `docs/examples` as tests; JuMP.jl runs `Aqua.jl`
  package-hygiene as a dedicated gate over an explicit toolchain matrix.
- **Reproducibility comparator split** (ReproBLAS, Intel MKL CNR, Collange et
  al.): exact bit/hash equality for within-mode + cross-rank-count invariance
  (seeded RNG, fixed reduction order); ULP tolerance for cross-mode equivalence.
  This is _precisely_ cobre's stated contract.

---

## 5. Proposed standard — uniform per-crate testing architecture

Every crate adopts the same layout, the same sharing mechanism, the same runner
config, and the same tier vocabulary. The standard is the following eleven rules.

### 5.1 Canonical per-crate layout

```
crates/<crate>/
  src/<module>.rs            # unit tests INLINE (#[cfg(test)] mod tests) …
  src/<module>/tests.rs      # … OR an extracted sibling above the threshold
  tests/it.rs                # THE single integration binary (mod-declares below)
  tests/it/<domain>.rs       # one submodule per domain (boundary, cut, sim, …)
  benches/                   # criterion — only where a hot path is measured
```

- **One integration binary per crate** (`tests/it.rs` or `tests/it/main.rs`),
  submodules by domain via plain (`#[path]`-free) `mod` in the greenfield case.
  This generalizes to **every** crate the binary consolidation that Layer 1
  (below) applies first to the solver-linking crates. Kills the `m·n` re-link.
- **One deterministic unit-test homing rule**: inline when the test module is
  below a fixed threshold (proposal: ~500 test-LOC or ~40 test fns), extracted to
  a sibling `tests.rs` above it. No more intra-directory coin-flip. (Pick the
  threshold once here; it is a lint, not a judgment call.)

#### Layer 1 — consolidating the existing per-file integration binaries

The greenfield layout above is the end state; Layer 1 is the mechanism that gets
the current solver-linking crates there without rewriting test bodies — the
concrete grouping mechanics and target groupings follow.

**Why it is the dominant cost.** Every crate depending on `cobre-solver`
statically links the vendored HiGHS / CLP / CoinUtils / qhull C++ into _each_
`tests/*.rs` binary, because Cargo compiles one executable per integration-test
file. Each such binary embeds the solver object code + debug info + the crate
rlib — on the order of hundreds of MB in a debug build — and pays a full C++
static link (several seconds) at build time. This is the dominant steady-state
contributor to `target/` size and CI link wall-time _after_ a clean build.
(Cause separation: the multi-hundred-GB `target/` that first motivated the work
was mostly stale accumulated artifacts, addressed separately by build hygiene —
`CONTRIBUTING.md` → "Reclaiming disk space". Layer 1 addresses only the
steady-state cost that survives `cargo clean`: binary count × solver-link
size/time.) Consolidation also brings the suite back into compliance with
`.claude/rules/testing.md`, which mandates grouping related tests with `mod`
submodules and treats a new test _binary_ as needing justification proportional
to its link cost.

**Grouping mechanics (`#[path]`-submodule).** Each current `tests/<file>.rs`
becomes a submodule of a domain binary, included by path so the file contents
move with minimal edits:

```
tests/
  <domain>.rs                # the new binary root
  <domain>/
    <file_a>.rs              # was tests/<file_a>.rs
    <file_b>.rs
  common/                    # unchanged, shared harness
```

`tests/<domain>.rs`:

```rust
mod common;                              // declared ONCE per binary, at the root
#[path = "<domain>/file_a.rs"] mod file_a;
#[path = "<domain>/file_b.rs"] mod file_b;
```

Per-file edits when moving `tests/file_a.rs` → `tests/<domain>/file_a.rs`:

1. **`mod common;` → remove.** The domain root owns the single `mod common;`.
   Rewrite submodule references from `use common::…` / bare `common::…` to
   `use crate::common::…` (Rust resolves `common` at the crate root, not in the
   submodule's own namespace).
2. **Crate-inner attributes stay, as module-inner attributes.** A file's leading
   `#![allow(clippy::…)]` / `#![allow(unused)]` remains valid unchanged inside
   `mod file_a { #![allow(…)] … }`: with a `#[path]` include the file _is_ the
   module body, so its `#![…]` inner attributes are still legal at the top of
   that body — no rewrite needed.
3. **Symbol collisions.** Because each file is its own `mod`, free items with the
   same leaf name (`fn setup`, `const CASE`, a local `struct Fixture`) do **not**
   collide — they are namespaced under their submodule. Only items a file
   declares at crate scope (rare in these files) collide; audit and qualify per
   merge.
4. **Feature/slow gates ride along** on the individual `#[test]` / `mod`; no
   change. Files that are entirely `#[cfg(feature = "mpi")]` are grouped with
   each other so the whole domain binary is coherently gated.

Prefer the `#[path]` include over physically concatenating files into one `.rs`:
concatenation would collide the per-file `#![allow]` inner attributes and
free-item names, and destroy `git mv` blame. The `#[path]`-submodule layout is
the low-diff way to apply the `testing.md` "group related tests with `mod`"
prescription to existing large files without rewriting them.

**Domain-binary grouping approach.** Priority order is the solver-linking crates;
group by subject so a contributor still finds tests by domain. Exact membership
is a starting proposal, refined during migration:

- **`cobre-sddp`** → a handful of domain binaries: `deterministic` (stays
  standalone — already one large domain file), `parity` (**stays standalone** —
  it owns the slow-gated `parity_regen` ignored tests and the golden baselines,
  which §6 / the parity fixtures leave untouched), `anticipated`, `boundary`,
  `simulation`, `cut_backward`, `lp_structural`, `pipeline_io`, and `mpi` (all
  `mpi`-gated files grouped so the binary is coherently gated).
- **`cobre-cli`** → domain binaries by subject: `cli_run`, `cli_validate`,
  `cli_reporting`, `cli_metadata`, `cli_basics`.
- **`cobre-solver`** → ~2–3 binaries grouped by concern (backend FFI,
  warm-start / basis, determinism) — a lower absolute win, but each of its
  binaries links the solver.
- **Non-solver crates** (`cobre-io`, `cobre-stochastic`, `cobre-comm`,
  `cobre-core`) are **out of scope**: their binaries do not link the solver, so
  the per-binary cost is small. Consolidate opportunistically only, never on this
  effort's critical path.

Expected outcome: the solver-linking integration binaries drop by roughly 3–4×,
and clean-build test-binary disk and cumulative link time drop roughly
proportionally.

**Migration invariants (a refactor, not a coverage change).**

- **No test is deleted, skipped, renamed, or weakened**; test-function count is
  identical before and after, verified mechanically by the §6 count-parity gate.
- **No test logic changes** — bodies, assertions, fixtures, and the tier a test
  belongs to are untouched.
- **No change to feature or slow gating** — `#[cfg(feature = "mpi")]`,
  `#[cfg_attr(not(feature = "slow-tests"), ignore = …)]`, and every other gate
  are preserved verbatim.
- **The shared `tests/common/` harness stays the single source** of fixture
  builders; it is not duplicated or forked per domain binary. (This is the
  interim state; §5.2 later collapses `tests/common/` into `cobre-sddp`'s
  `test-support` surface.)
- **Determinism gates keep their power** (e.g. `mpi_wire.rs`'s self-checked
  thresholds). Integration submodules share no mutable global state today and
  must not start: grouping leaves each binary a single process running its tests
  on separate threads, introduces no shared `static mut`, and must not reorder or
  share state a gate depends on.

**Migration procedure (incremental, one domain at a time — do _not_ move all
files at once).** Per domain binary:

1. Create `tests/<domain>.rs` and `tests/<domain>/`, `git mv` each member file
   in (preserving blame), apply the per-file edits above.
2. `cargo build --tests -p <crate>` under both `highs` and, where relevant,
   `clp` — fix `use crate::common::…` paths and any crate-scope collisions.
3. Run the new binary; confirm every moved test still runs and passes.
4. Verify test-count parity for the crate (§6) before moving on. A drop means a
   `#[test]` was lost in the move — stop and fix.
5. Commit per domain (`refactor(test): consolidate <domain> integration tests`)
   so a regression bisects to one domain.

Order within `cobre-sddp`: start with a small, self-contained family (`boundary`
— already grouped by subject) to validate the mechanics end-to-end, then the
larger families. The `mpi` domain binary must compile and run under
`--features mpi` in the SLURM job and compile (its tests gated out) without it.
The headline payoff — a measurable drop in the crate's test-build link wall-time
— is worth capturing once, before vs after each crate is fully migrated.

**Complementary lever (not part of Layer 1).** Dynamically linking the solver for
dev/test builds (static only for `dist`) would shrink every test binary
regardless of count, but it changes the vendored-static-reproducible build
contract and adds an `LD_LIBRARY_PATH` runtime dependency. Tracked separately; it
does not block consolidation and the two compose.

### 5.2 A uniform `test-support` feature convention (not a dedicated test crate)

Keep the `test-support` cargo-feature mechanism the repo already uses
(`cobre-core`, `cobre-solver`, `cobre-sddp`) and make it uniform and complete —
do **not** introduce a dedicated test crate. Cross-crate test sharing in Rust is
either a crate or a `#[cfg(any(test, feature = "test-support"))]` feature; a
separate crate adds a workspace member, a version-bump and crates.io-publish
surface, and a dev-dependency cycle (`cobre-io` → support → `cobre-sddp` →
`cobre-io`), and forces an artificial layering split to respect the
infra-genericity hard rule — for no capability the feature does not already
provide. The `#[cfg(...)]` gate compiles to nothing in a normal build, so the
"test feature in a production `Cargo.toml`" cost is cosmetic.

The convention:

- **Helpers live with the type they build** — a cleaner ownership model than a
  catch-all crate. Generic scaffolding (the golden-SHA helper, the two-tier
  comparator of §5.4, the `permute` order-shuffle helper, RNG-seed and `TempDir`
  conventions) lives in `cobre-core` (the universal base dependency) behind its
  `test-support` feature; `StubComm`/`Rank0Of2` live in `cobre-comm` (they impl
  its `Communicator`); the entity/`StudySetup` builders stay in `cobre-sddp`.
- **Every crate exposes its shareable fixtures the same way** — a `test-support`
  feature gating them behind `#[cfg(any(test, feature = "test-support"))]`, enabled
  by consumers as a dev-dependency feature. `cobre-sddp/tests/common/` collapses
  into `cobre-sddp`'s `test-support` surface so `cobre-cli` and `cobre-python`
  reach it, not just sddp's own `tests/`.
- **`cobre-python` dev-depends on `cobre-sddp`'s `test-support`**, which (with the
  §5.11 CI wiring) lets its Rust tests share the same fixtures.

**Adoption (2026-09-15).** The convention is the adopted standard for the crates that own
shareable fixtures: each exposes them behind its own `test-support` feature under the gate
above, and consumers enable it as a dev-dependency feature. Two of its bullets are not yet
carried out: `crates/cobre-sddp/tests/common/` still holds its own fixture directory rather
than collapsing into the crate's `test-support` surface, and `crates/cobre-python/Cargo.toml`'s
`[dev-dependencies]` does not yet name `cobre-sddp` with that feature. Until both land, the
status line above reads "partially adopted" rather than "adopted".

A dedicated crate wins in exactly one case — heavy test-only dependencies you
want kept out of every production crate's dev-graph, or a pristine
crates.io-published surface — neither of which applies today; revisit only if the
helper surface grows its own heavy deps. This convention also resolves the
register's oracle-harness-duplication item (hoist the `close`/tolerance helpers
into `cobre-core`'s `test-support` surface).

### 5.3 The tier taxonomy, formalized with a decision rule

Keep the four tiers from `testing.md`; name two more that already exist implicitly
(property, compile-fail) and one to add (hardening). The **decision rule** a new
test follows, cheapest-sufficient-tier first:

| Tier                               | Proves                                                                         | Use when                                             | Cost         |
| ---------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------- | ------------ |
| Unit                               | a function's local contract                                                    | always, by default                                   | cheap        |
| Property                           | an invariant over generated inputs (order-invariance, reduction commutativity) | a claim quantifies over "all orderings/inputs"       | cheap        |
| LP-structural                      | column/row/coefficient wiring without a solve                                  | a wrong count would be numerically smooth            | cheap        |
| Analytical (MMS/order-of-accuracy) | correctness vs a **hand-derived** value/cut/rate                               | a _new_ correctness claim                            | medium       |
| Behavioral                         | LB==UB / known cost to tolerance, backend-agnostic                             | a deterministic case's end result                    | medium       |
| Golden bit-exact                   | byte-stability of the final artifact                                           | the small, deliberate cross-feature set only         | high         |
| Compile-fail (doctest/`trybuild`)  | misuse fails to compile                                                        | a type-state contract (e.g. `ValidatedBoundaryCuts`) | cheap        |
| Hardening (miri/sanitizer/fuzz)    | absence of UB / crash on adversarial input                                     | code with `unsafe`/FFI or untrusted parsing          | high, off-PR |

Rule of promotion: **write the analytical test before adding a behavioral case,
and a behavioral case before promoting anything into the golden set** (golden
membership needs justification — it is the only tier that breaks on a faster path
to the same answer).

### 5.4 The two-tier comparator standard

Encode the determinism contract in shared comparators (`cobre-core`'s
`test-support` surface, §5.2), not ad hoc per test:

- **Exact (`to_bits`/SHA)** for the _within-mode reproducibility_ and
  _cross-rank/thread invariance_ tier — the guarantee cobre actually makes.
- **ULP-tolerance** for _cross-mode / cross-algorithm equivalence_ (objective
  agreement where duals legitimately differ). A `assert_equivalent_vertex`
  comparator asserts equal objective + primals within ULP and _permits_ different
  duals — the PETSc `alt`-output idea, typed. This stops contributors from either
  over-asserting bit-equality across modes or under-asserting with a loose `abs`
  tolerance.

### 5.5 Runner standard — nextest everywhere, configured

Adopt `cargo-nextest` as the sole runner (keep a separate `cargo test --doc` for
doctests, which nextest does not run) and commit a `.config/nextest.toml`:

- `[profile.ci]` with `retries` (exponential backoff) + flaky-marking, and
  `[profile.ci.junit]` for the dashboard feed.
- `[test-groups]` + overrides to bound MPI/solver-heavy tests (the rank-weight /
  oversubscription control, §5.8), and per-test `slow-timeout` to kill hung
  MPI/solve tests.
- Standardize CI shards on `--partition hash:m/n` (deterministic pinning).
- **`nextest archive`** for the SLURM/MPI path: build the archive on the CI node,
  ship the `.tar.zst` to compute nodes, run without a toolchain there.

### 5.6 CI cadence tiering

Split the monolithic per-PR run into cadence tiers (this is the change that most
improves sustainability):

| Tier          | Contents                                                                                                                                                    | Cadence                      |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 0 — smoke     | fmt, clippy, `Check`, quality-script gates                                                                                                                  | every push                   |
| 1 — PR-fast   | unit + property + structural + **determinism gates (`mpi_wire`)** + doctests, both backends (check/clippy), coverage                                        | every PR                     |
| 2 — extended  | `slow-tests` sweep (D-case, FPHA, forward-sampler convergence), golden parity sweep, **order-invariance shuffle matrix**, real multi-rank MPI (`mpi-slurm`) | merge-to-`develop` + nightly |
| 3 — hardening | miri (pure-Rust unsafe), valgrind/ASan (FFI), fuzz smoke                                                                                                    | nightly / weekly             |

The load-bearing moves: (a) **remove `slow-tests` from the per-PR feature set**
(`NON_SOLVER_FEATURES`) and run it on the extended tier — PR CI gets dramatically
faster and the `slow-tests` gate finally _means_ a CI tier, not just a local
convenience; (b) **put the shuffle matrix on the nightly cadence** (uncomment the
cron) so the hard order-invariance rule is actually exercised in automation. This
is a policy change with a tradeoff — a slow-test regression is caught at
merge/nightly rather than on the PR — and should be ratified explicitly; it
matches DataFusion/tokio norms and the project's own recorded intent that slow
tests not gate routine CI.

### 5.7 Feature/backend matrix policy

Declare the `{backend} × {feature set} × {CI job}` matrix in one place (a table in
this doc + a CI matrix), so the CLP path cannot silently rot: HiGHS is the golden

- primary tier; CLP re-runs behavioral + structural + its own golden baseline;
  `mpi`/`numa`/`shared-memory` gate only the tests that need them. State the
  backend-scoping of the parity roster (the register's open caveat) inline in the
  roster.

### 5.8 MPI & determinism testing standard

- Keep `StubComm`/`Rank0Of2` (in `cobre-comm`'s `test-support` surface) as the
  in-process rank-shape harness for PR-tier determinism gates; keep the SLURM job for real
  multi-rank on the extended tier.
- Give every multi-rank test an explicit **rank weight** (nextest test-group
  `max-threads`) and a **max-rank budget**, the CTest `PROCESSORS` /
  `MPI_EXEC_MAX_NUMPROCS` idea, so concurrent multi-rank tests don't oversubscribe
  and the suite runs laptop→cluster unchanged.
- Frame the forward-sampler convergence test explicitly as an
  **order-of-accuracy** verification (observed vs theoretical rate).

### 5.9 Correctness-hardening additions (right-sized)

Add Tier-3, off the PR path, scoped to where it pays:

- **miri** on the pure-Rust `unsafe` only (the isolated `gemm` kernel, raw-buffer
  reuse). Miri **cannot** cross the C++ FFI boundary — do not expect it to cover
  the solver.
- **valgrind / `-Zsanitizer=address`** for the C++ solver + MPI + PyO3 FFI, where
  miri cannot reach (tokio runs both as dedicated jobs).
- **One `cargo-fuzz` target for the `cobre-io` parsers** (config/Parquet/JSON
  ingest is the untrusted-input surface); commit the corpus.
- **Expand `proptest`** from 5 sites to cover the declaration-order-invariance and
  reduction-order invariants directly (permute → assert identical bits), with
  `proptest-regressions/` committed to VCS.

### 5.10 Golden / snapshot standard

- Keep SHA-bit-exact for the small deliberate golden set (unchanged).
- Adopt **`insta` (or `expect-test`)** for the _verbose text_ goldens that are
  currently hand-maintained strings — the reconciliation diagnostic block, CLI
  report tables, LP-structure dumps, exported schemas — so regeneration is
  `cargo insta review` / `UPDATE_EXPECT=1`, not manual editing.
- One `just`/script regen entrypoint per golden family; the "goldens are the final
  artifact, never the trajectory" rule (already in `testing.md`) stays.

### 5.11 cobre-python parity

Wire the crate's Rust tests into CI (a `cargo test -p cobre-python` / `cargo check
--tests` step), closing the register's "Python-Rust tests invisible to CI" item.
Have it dev-depend on `cobre-sddp`'s `test-support` feature so its Rust fixtures
match the CLI's by construction; keep the pytest output-parity suite unchanged.

---

## 6. Migration path (incremental, coverage-neutral)

Each phase is behavior-neutral and gated on `cargo nextest list` count parity
(no test added, removed, or renamed except the module-path prefix). Order by
leverage:

1. **Layer 1 — binary consolidation** (§5.1): group each solver-linking crate's
   per-file integration binaries into a handful of domain binaries via the
   `#[path]`-submodule mechanics and target groupings specified there. Biggest
   single win.
2. **`.config/nextest.toml` + nextest as the sole runner** (+ `--doc` step). Cheap,
   immediate CI-time and observability win; unlocks archive/partitioning.
3. **Standardize the `test-support` feature** (§5.2): collapse `tests/common/`
   into `cobre-sddp`'s `test-support` surface, move the generic comparators into
   `cobre-core`'s, and dev-depend `cobre-python` on it — no new crate.
4. **CI cadence tiering** (§5.6): move `slow-tests` off the PR feature set;
   enable the shuffle-matrix nightly. Ratify the tradeoff first.
5. **Unit-test homing lint** (§5.1 threshold) + resolve the inline-giant
   asymmetry as files are touched.
6. **Tier-3 hardening** (miri / sanitizer / fuzz jobs) — additive, off-PR.
7. **`insta` for text goldens** — opportunistic, as each hand-maintained golden is
   next touched.

Phases 1–3 are the sustainability core; 4 is the policy decision; 5–7 are
steady-state hygiene.

## 7. Non-goals / what NOT to over-adopt

- **`loom`** — only if hand-rolled lock-free/atomic code appears; MPI
  message-passing is out of its scope. Not warranted today.
- **`trybuild`** — the doctest `compile_fail` blocks already cover the type-state
  contracts; adopt only if a proc-macro/DSL ships.
- **Blanket miri** — the `unsafe` surface is mostly FFI, which miri can't enter;
  scope it to pure-Rust unsafe and lean on sanitizers/valgrind for the rest.
- **Reducing coverage to shrink the suite** — the cost is per-_binary_ and
  per-_feature-combo_, never per-_test_; deleting tests saves nothing structural
  and forfeits the suite's best asset.

## 8. References

- matklad, _Delete Cargo Integration Tests_ — https://matklad.github.io/2021/02/27/delete-cargo-integration-tests.html
- matklad, _Fast Rust Builds_ (the `m·n` link argument) — https://matklad.github.io/2021/09/04/fast-rust-builds.html
- cargo-nextest: archiving — https://nexte.st/docs/ci-features/archiving/ · partitioning — https://nexte.st/docs/ci-features/partitioning/ · JUnit — https://nexte.st/docs/machine-readable/junit/
- `insta` — https://insta.rs/docs/ · `expect-test` — https://github.com/rust-analyzer/expect-test
- `proptest` failure persistence — https://altsysrq.github.io/proptest-book/proptest/failure-persistence.html
- miri — https://github.com/rust-lang/miri · tokio miri→nextest — https://github.com/tokio-rs/tokio/pull/6885
- CMake CTest `PROCESSORS` — https://cmake.org/cmake/help/latest/prop_test/PROCESSORS.html · `FindMPI` — https://cmake.org/cmake/help/latest/module/FindMPI.html · LABELS — https://cmake.org/cmake/help/latest/prop_test/LABELS.html
- deal.II testsuite — https://dealii.org/current/developers/testsuite.html · PETSc testing — https://petsc.org/release/developers/testing/
- Method of Manufactured Solutions, Sandia SAND2000-1444 — https://www.osti.gov/biblio/759450 · deal.II `ConvergenceTable` (order of accuracy) — https://dealii.org/current/doxygen/deal.II/step_7.html
- SDDP.jl tests — https://github.com/odow/SDDP.jl/tree/master/test · Aqua.jl — https://github.com/JuliaTesting/Aqua.jl
- ReproBLAS — http://bebop.cs.berkeley.edu/reproblas/ · Intel MKL CNR — https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2024-2/numerical-reproducibility.html
- DataFusion testing (PR vs extended) — https://datafusion.apache.org/contributor-guide/testing.html · tokio CI — https://github.com/tokio-rs/tokio/blob/master/.github/workflows/ci.yml
