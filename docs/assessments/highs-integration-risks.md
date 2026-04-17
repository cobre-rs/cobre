# HiGHS Integration Risk Assessment

**Date:** 2026-04-16
**Branch:** `feat/performance-optimization-pipeline`
**Scope:** All low-level HiGHS usage in `cobre-solver` and `cobre-sddp`

## Executive Summary

Cobre uses several advanced HiGHS features to squeeze maximum performance out of
repeated LP solves:

- **Dual simplex** as the default strategy.
- **CSC template baking** — pre-assembled structural LP shared across scenarios,
  with cut rows merged into the template at each iteration.
- **Warm-started basis** — a basis from a previous solve is passed back to HiGHS
  to skip the initial feasibility phase.
- **Basis-aware padding** — when new cut rows are added between iterations, the
  extra row statuses are filled with `NONBASIC_LOWER` / `BASIC` based on
  cut-tightness at the warm-start state.
- **Presolve and internal scaling both disabled** (`presolve=off`,
  `simplex_scale_strategy=0`) — Cobre's own prescaler is trusted to normalise
  matrix entries.
- **Batched bound patching** via `Highs_changeColsBoundsBySet` /
  `Highs_changeRowsBoundsBySet`.

These choices are correct for the scenario-loop workload, but each one relies on
HiGHS's low-level C API with minimal safety nets. This document catalogues the
failure modes that each trick exposes, grouped by severity and by whether they
produce **silent wrong answers**, **crashes**, or **performance regressions**.

The analysis is based on reading:

- `crates/cobre-solver/src/{ffi.rs, highs.rs, baking.rs, types.rs}`
- `crates/cobre-solver/csrc/highs_wrapper.{h,c}`
- `crates/cobre-sddp/src/{forward.rs, backward.rs, basis_padding.rs,
lp_builder/{template.rs, patch.rs, baking}}`
- `crates/cobre-solver/vendor/HiGHS/highs/{lp_data, util, interfaces}/*` — the
  vendored HiGHS source itself.

---

## Cobre's HiGHS Usage At A Glance

| Step                 | Rust API                          | HiGHS C API                                                   | HiGHS C++ entry point                                          |
| -------------------- | --------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- |
| Construct solver     | `HighsSolver::new`                | `Highs_create` + 8× `Highs_set*OptionValue`                   | `Highs::setOptionValue`                                        |
| Load structural LP   | `load_model(template)`            | `Highs_passLp`                                                | `Highs::passModel` (CSC)                                       |
| Append cut rows      | `add_rows(batch)`                 | `Highs_addRows`                                               | `Highs::addRowsInterface`                                      |
| Patch row/col bounds | `set_row_bounds`/`set_col_bounds` | `Highs_changeRowsBoundsBySet` / `Highs_changeColsBoundsBySet` | `Highs::changeRowBoundsInterface` / `changeColBoundsInterface` |
| Install basis        | `solve_with_basis(basis)`         | `Highs_setBasis`                                              | `Highs::setBasis(const HighsBasis&)`                           |
| Solve                | `solve()`                         | `Highs_run`                                                   | `Highs::run`                                                   |
| Extract solution     | —                                 | `Highs_getSolution` / `Highs_getBasis`                        | `Highs::getSolution` / `getBasis`                              |

The Rust side pre-allocates `col_value`, `col_dual`, `row_value`, `row_dual`,
`basis_col_i32`, and `basis_row_i32` buffers inside `HighsSolver` and grows them
inside `load_model`/`add_rows`. FFI calls pass raw pointers into these Rust-owned
buffers; correctness of the buffer sizes is the sole defense against the UB
described in **C2**.

---

## Critical Risks (Correctness)

### C1. Silent zero-drop for small matrix coefficients

**Severity:** Critical — silent wrong answers.

**Mechanism.** HiGHS's `assessMatrix`
(`util/HighsMatrixUtils.cpp:145-247`) iterates every CSC non-zero; any entry
with `|value| ≤ options.small_matrix_value` (default **1e-9**, configurable in
the range 1e-12..∞ — see `HighsOptions.h:769`) is **removed from the LP**. The
compressed `matrix_start` / `matrix_index` / `matrix_value` vectors are
rewritten in place. HiGHS returns `HighsStatus::kWarning` and logs
`"... |values| in [..., ...] less than or equal to ...: ignored"`.

**Why Cobre cannot see it.**

- Cobre sets `output_flag = 0` (`highs.rs:113-115`), suppressing all HiGHS log
  output.
- The Rust wrapper treats any status except `HIGHS_STATUS_ERROR` as success:
  `assert_ne!(status, HIGHS_STATUS_ERROR, …)` at `highs.rs:872-876, 935-939`.
  `kWarning` therefore passes silently.

**Amplifiers in Cobre.** `apply_col_scale` / `apply_row_scale`
(`lp_builder/scaling.rs`) multiply values by inverse column/row norms. A
near-zero coefficient (very small FPHA `γ_v/2`, a near-zero coupling term, or
anything that passes through `COST_SCALE_FACTOR = 1_000`) can land below 1e-9
after scaling. It vanishes from the LP that HiGHS actually solves.

**Consequence.** Training / simulation converges to the wrong answer with no
visible signal. `SolverStatistics.success_count` increments normally.
Deterministic regression tests catch it only if a dropped coefficient happens
to be on a tight constraint for one of the test cases.

**Defenses today.** None. `simplex_scale_strategy = 0` means HiGHS's internal
equilibration is off, so the prescaler is the only line of defense and any
prescaler bug propagates directly.

**Remediation.**

1. Tighten `small_matrix_value` to `1e-12` (the minimum allowed) via
   `cobre_highs_set_double_option`.
2. Elevate `HighsStatus::kWarning` to a hard error in `load_model` and
   `add_rows` — return a typed `SolverError` rather than silently proceeding.
3. Add a prescaler post-check that no scaled coefficient is smaller than
   `2 × small_matrix_value` in magnitude.

---

### C2. `Highs_setBasis` reads past caller buffers without bounds check

**Severity:** Critical — undefined behaviour.

**Mechanism.** `Highs_setBasis` (`interfaces/highs_c_api.cpp:664-706`) loops

```c
const HighsInt num__col = Highs_getNumCol(highs);
for (HighsInt i = 0; i < num__col; i++) {
  if (col_status[i] == ...) ...
}
```

The C API has no way to know the length of the caller's `col_status` /
`row_status` arrays. **Any basis shorter than the current LP dimensions
triggers an out-of-bounds read — undefined behaviour via FFI.**

**Why it matters for Cobre.** The LP size inside HiGHS is updated by
`passLp`, `addRows`, `addCols`, `changeObjectiveSense`, etc. The Rust
`HighsSolver` mirrors this with `self.num_cols` / `self.num_rows` and keeps
`self.basis_col_i32` / `self.basis_row_i32` at least `num_cols` / `num_rows`
long (`highs.rs:884-893, 947-950`). The FFI call in `solve_with_basis`
(`highs.rs:1206-1212`) passes pointers into those Rust-owned buffers. Buffer
sizes are correct **only if** `load_model`/`add_rows` bookkeeping stays in sync
with HiGHS's internal state.

**Failure modes that break the invariant.**

- Direct use of `test-support` helpers (`lib.rs:58-68` exposes
  `cobre_highs_set_*_option`; `highs.rs:1250` exposes `raw_handle()`). A test
  could call `Highs_passLp` directly on the raw handle, resize HiGHS's LP, and
  leave `num_cols`/`num_rows` stale on the Rust side. The next
  `solve_with_basis` reads past the end of the `Vec`.
- A new row/column-shrinking wrapper (e.g. `Highs_deleteRowsBySet`, not yet
  bound but easy to add) that reduces HiGHS's row count without updating
  `basis_row_i32`. `solve_with_basis` passes the too-large pointer, HiGHS reads
  bytes beyond the `Vec`'s valid region.

**Consequence.** UB. In practice the random bytes rarely match a valid
`HighsBasisStatus` enum value (`kLower (0) … kNonbasic (4)`), so `Highs_setBasis`
returns `kError` and Cobre increments `basis_rejections`. But **nothing prevents
the garbage bytes from matching a valid enum**, in which case HiGHS silently
installs a nonsense basis and solves from it.

**Defenses today.** The assertion at `highs.rs:1174-1179`:

```rust
assert!(basis.col_status.len() == self.num_cols, …);
```

This validates the _caller-supplied Basis struct_. It does **not** validate that
`basis_col_i32` / `basis_row_i32` (the scratch buffers actually passed to FFI)
are correctly sized, and it does **not** validate that HiGHS's own `num_col_` /
`num_row_` agree with `self.num_cols` / `self.num_rows`.

**Remediation.**

1. Add `debug_assert!(self.basis_col_i32.len() >= self.num_cols &&
self.basis_row_i32.len() >= self.num_rows)` immediately before the FFI call.
2. In debug builds, call `cobre_highs_get_num_col`/`_row` and assert agreement
   with `self.num_cols`/`_rows` before every basis-touching FFI call.
3. Move `raw_handle()` from `#[cfg(feature = "test-support")]` to `#[cfg(test)]`
   so it cannot leak into downstream crates by accident.

---

### C3. `bake_rows_into_template` heap corruption in release builds

**Severity:** Critical — silent wrong LP.

**Mechanism.** `baking.rs:136-138` and `209-216` compute

```rust
let j = rows.col_indices[k] as usize;
let pos = (col_list_start[j] + write_cursor[j]) as usize;
col_list_row[pos] = row_i32;
col_list_val[pos] = rows.values[k];
```

The invariant `col < base.num_cols` is only a `debug_assert!` at
`baking.rs:108-114`. In release builds, a malformed `RowBatch` (from a cut pool
bug, a `patch.rs` refactor, or corrupted serialized state) can write past the
end of `col_list_row[rows_nnz]` — silent heap corruption of the adjacent
`col_list_val` or `write_cursor` buffers.

The resulting CSC template can have wrong non-zero values, wrong start offsets,
or (if `write_cursor[j]` overflows) a completely scrambled column-major layout.

**Why HiGHS will not catch it.** After the bake, Cobre hands a _structurally
consistent_ CSC template (correct sizes, non-decreasing starts, in-range
indices) to `passLp`. HiGHS's `assessMatrix` checks that indices fall in
`[0, num_rows)` and that start offsets are non-decreasing — but it cannot check
whether each non-zero value came from the intended row or column. The LP loads
cleanly and solves silently.

**Remediation.** Promote the `col < base.num_cols` check and the
sanity-checks in `baking.rs:60-115` to release asserts. They are inexpensive
relative to the overall bake cost.

---

## High Risks (Crashes, Aborted Training, Variable CPU)

### H1. Every warm-started basis forces a full factorization

**Severity:** High — per-solve performance cost on the hottest path.

**Mechanism.** `Highs_setBasis` (`interfaces/highs_c_api.cpp:664`) constructs

```cpp
HighsBasis basis;  // struct HighsBasis { ... bool alien = true; ... };
```

with `alien = true` as the struct default (`lp_data/HStruct.h:68`). The subsequent
`Highs::setBasis(const HighsBasis&, ...)` (`Highs.cpp:2559-2592`) takes the
alien path, which runs

```cpp
HighsLpSolverObject solver_object(...);
HighsStatus return_status = formSimplexLpBasisAndFactor(solver_object);
```

every time — an expensive symbolic + numeric factorization with repair for
singular or inconsistent bases.

**Consequence.** A "bad" basis (singular or inconsistent with bounds) does
**not** corrupt HiGHS state — it is rejected, `Highs_setBasis` returns `kError`,
and Cobre falls back to cold-start. However:

- **Every** `solve_with_basis` call pays factor cost, even for a numerically
  sound basis that HiGHS already has internally.
- A bad basis can stall for tens of seconds before rejection.

**Defenses today.** Retry escalation wall-clock budget (15 s per Phase 1 level,
30 s per Phase 2). No warning when a "successful" solve takes anomalously long.

**Remediation options.**

1. Investigate whether HiGHS exposes a non-alien API — `Highs_setHotStart`,
   `putIterate`/`getIterate`, or a direct way to signal
   `basis.alien = false`. **This is the follow-up investigation this document
   frames.**
2. If no such API exists, periodically cold-start (e.g. every N iterations or
   when the basis hit rate falls below a threshold) to break out of bad chains.
3. Surface per-solve wall-clock outliers via a new `SolverStatistics` field.

---

### H2. `assert_ne!` panics on `passLp` / `addRows` error

**Severity:** High — parallel worker crash on recoverable data errors.

**Mechanism.** `load_model` ends with
`assert_ne!(status, HIGHS_STATUS_ERROR, "cobre_highs_pass_lp failed …")`
(`highs.rs:872-876`); `add_rows` has the same pattern (`highs.rs:935-939`).
Inside a Rayon `par_iter_mut()` forward/backward loop, a panic in one worker
unwinds and propagates out of the parallel region — aborting the whole
iteration and losing all partial results.

**Triggers (all HiGHS `kError` returns).**

- A coefficient with `|value| ≥ large_matrix_value` (default **1e15** —
  `HighsOptions.h:776`). `assessMatrix` returns `kError`.
- A `matrix_start[0] ≠ 0`, non-monotone starts, or duplicate indices within a
  column — `assessMatrix` rejects all three (`HighsMatrixUtils.cpp:66-118,
175-184`).
- A usize index overflowing `i32::MAX` (`convert_to_i32_scratch` has a
  `debug_assert!` only; release builds silently truncate to a negative value,
  which `assessMatrix` then rejects).

**Severity rationale.** These are all _data-shape_ errors, not genuinely
unsolvable LPs. Crashing in the middle of a long MPI training run loses hours
of work, and the error (`panic!` with a HiGHS status code) does not identify
which entity / constraint / stage produced the bad coefficient.

**Remediation.** Replace `assert_ne!` with a returnable `Err` variant, and add
a pre-flight validator inside `bake_rows_into_template` that scans for
`|value| ≥ large_matrix_value` and tags the offending entity.

---

### H3. `debug_assert!(false, "raw basis rejected; …")` panics debug builds

**Severity:** High in debug, zero in release — test brittleness.

**Location.** `highs.rs:1218`. When `Highs_setBasis` returns `kError`, release
builds increment `basis_rejections` and silently cold-start. Debug builds panic.

**Impact.** Any test that intentionally or accidentally supplies a stale /
malformed basis (e.g. after MPI basis transfer with a mismatched dimension,
after cut-selection churn) crashes under `cargo test`. Not a production risk,
but a source of false-positive CI failures.

**Fix.** Remove the `debug_assert!(false, …)`; keep `basis_rejections` tracking
as the observability mechanism.

---

### H4. Retry option leakage into subsequent solves

**Severity:** High — latent, difficult to diagnose performance regression.

**Mechanism.** `apply_retry_level_options` (`highs.rs:646-776`) mutates options
globally: `presolve = on`, `simplex_scale_strategy = 3|4`,
`user_objective_scale = -10|-13`, `user_bound_scale = -5|-8`,
`primal_feasibility_tolerance = 1e-6`, `dual_feasibility_tolerance = 1e-6`,
`solver = ipm`, …. Most of these are option names HiGHS does not reset on
`clearSolver`.

`retry_escalation` calls `restore_default_settings` (replays the 8 named
defaults) plus explicit resets of `user_objective_scale` and `user_bound_scale`
(lines 601-606). Any future retry level that touches an option outside that
restore list will silently persist into subsequent solves.

**Consequence.** A single successful retry late in training could leave the
solver in a slower configuration (e.g. IPM instead of dual simplex, 1e-6
tolerances instead of 1e-7) for thousands of subsequent "normal" solves. The
per-solve regression is too small to attribute without instrumentation.

**Defenses today.** Explicit reset of the two user\_\* options. Nothing else.

**Remediation.**

1. Maintain a single canonical "full option set" that is replayed on every
   retry boundary, not the current 8-element subset.
2. Snapshot option values into a `HashMap<&CStr, OptionValue>` on entry to
   `retry_escalation` and restore from that snapshot on exit.

---

## Medium Risks

### M1. `(lower > upper)` bounds → warning, not error

`assessBounds` (`HighsLpUtils.cpp:420-426`) emits `kWarning` and lets the LP
through. HiGHS will detect infeasibility downstream via the simplex.
`SolverError::Infeasible` propagates to the caller, which treats it as a
genuine modeling issue. The real cause — a buggy template — is hidden.
`output_flag = 0` hides the warning log.

**Fix.** Elevate `kWarning` returns from `passLp`/`addRows` to hard errors, or
add a pre-bake validator for `col_lower[j] > col_upper[j]` and
`row_lower[i] > row_upper[i]`.

### M2. Finite bounds ≥ 1e15 silently become infinite

`assessBounds` (`HighsLpUtils.cpp:396-410`) normalises `lower ≤ -infinite_bound`
to `-kHighsInf` and `upper ≥ infinite_bound` to `+kHighsInf` (default
`infinite_bound = 1e15`, range 1e15..1e20). Cobre uses `f64::INFINITY` for
unbounded variables, so this path is only hit if a legitimate large-but-finite
bound arises. Silent loss of the upper bound makes a variable free, which
changes duals significantly.

**Fix.** Raise `infinite_bound` to 1e20, or validate the template post-scale.

### M3. `change*BoundsBySet` allocates three vectors per call

`Highs::changeColsBounds(num_set_entries, set, lower, upper)`
(`Highs.cpp:2950-2985, 3031-3066`) copies `set`, `lower`, `upper` into three
`std::vector` locals, then `sortSetData` reorders them. Caller data is
unchanged (good) but there are three heap allocations per call.

Every bound change also runs `setNonbasicStatusInterface` and
`ekk_instance_.updateStatus(kNewBounds)` (`HighsInterface.cpp:991-1002,
1042-1053`) which invalidates derived simplex data. Fewer, larger batched
calls are always faster than more frequent smaller ones.

**Fix.** Today Cobre batches well. Any future refactor that breaks the
row/column patches into multiple smaller calls would regress.

### M4. Matrix-format constant flip silently transposes on every load

`Highs::passModel` (`Highs.cpp:358`) calls `lp.ensureColwise()` regardless of
the caller's format. Cobre passes `HIGHS_MATRIX_FORMAT_COLWISE` literally
(`highs.rs:858`), matching the CSC template. If a refactor ever flips that
constant to `ROWWISE` while the template stays CSC, HiGHS transposes on every
`load_model` — an O(nnz) reshuffle per scenario with a ~2–3× slowdown and no
correctness error.

**Fix.** Keep the constant literal close to the CSC builder. Name a single
`const MATRIX_FORMAT: i32 = …` with a comment explaining the dependency.

### M5. `reset()` + no `load_model` → `solve_with_basis` panics

`Highs::clearSolver` → `invalidateSolverData` → `invalidateBasis` +
`invalidateEkk`. On the Rust side, `reset` (`highs.rs:1118-1135`) sets
`has_model = false`, `num_cols = 0`, `num_rows = 0`. The
`assert!(self.has_model, …)` in `solve_with_basis` (`highs.rs:1170-1173`) then
fires if no subsequent `load_model` is made.

Not a HiGHS misuse — a Cobre contract the compiler does not enforce. Training
and simulation never call `reset` mid-scenario today.

---

## Low Risks (Acknowledged, Contained)

- **`Send + !Sync`** correctly scoped. One solver per Rayon worker, no concurrent
  access.
- **`HighsInt` width assumption.** The static_assert in `highs_wrapper.c:18`
  catches a HIGHSINT64 build mismatch at compile time.
- **Iteration limits set to `i32::MAX`.** `kHighsIInf =
std::numeric_limits<HighsInt>::max()` (`HConst.h:28`), so passing `i32::MAX`
  is the canonical "infinite" value.
- **`set_iteration_limits` / `restore_iteration_limits` FFI overhead.** Two
  `cobre_highs_set_int_option` round-trips per solve for limits that almost
  never matter. Small, but on the hottest path.

---

## Risks That Cause Performance Regressions

Extracted from the above, ranked by likely cumulative impact:

| Rank | Risk                                                                | Dominates when                                           |
| ---- | ------------------------------------------------------------------- | -------------------------------------------------------- |
| 1    | **H1** — alien factor on every `setBasis`                           | Every warm-start — the single largest fixed cost         |
| 2    | **H1b** — basis-padding misclassifications                          | After cut-selection swaps or stale warm-start state      |
| 3    | **M3** — per-call `std::vector` allocations in `change*BoundsBySet` | Steady-state training, cumulative                        |
| 4    | **H4** — option leakage into post-retry solves                      | Invisible without diffing retry vs non-retry timings     |
| 5    | **M4** — matrix-format flip transpose cliff                         | Only if someone changes the constant                     |
| 6    | Iteration-limit FFI round-trips per solve                           | Hot-path microoverhead                                   |
| 7    | **C1** — silent zero-drop altering LP conditioning                  | Only with very small coefficients; manifests as variance |

**H1 is the single largest lever.** Every `solve_with_basis` call factors the
supplied basis anew via `formSimplexLpBasisAndFactor`, regardless of whether
Cobre's basis is already consistent with HiGHS's most recent solve. If HiGHS
exposes a non-alien warm-start path, switching to it is the highest-value
change possible here.

---

## Follow-up Investigation

Whether HiGHS exposes a non-alien warm-start path is the subject of the next
section of this report (see below). Candidate APIs to check in the vendored
HiGHS source:

- `Highs_setHotStart` / `Highs_getHotStart` — if present, these typically
  preserve simplex state without forcing a refactor.
- `Highs::putIterate` / `Highs::getIterate` — visible at `Highs.cpp:2635-2660`.
- Direct access to `basis_.alien` via a supplementary setter.
- `Highs::setBasis()` (no-arg) — seen at `Highs.cpp:2623` — invalidates the
  basis rather than installing one; not useful for warm-start.
- Private `ekk_instance_` entry points exposed via a debug flag.

The investigation in the next section walks through each and reports whether
it provides a genuine non-alien warm-start.

---

# Non-Alien Warm-Start Path — Investigation

This section documents the inventory of HiGHS APIs that could avoid or
mitigate the factor-on-every-setBasis cost identified in **H1**.

## Finding 1 — `setHotStart` is deprecated

`Highs.h:1487-1496` exposes `getHotStart()` (returns
`ekk_instance_.hot_start_`) and `setHotStart(const HotStart&)`. The setter
is **deprecated** and hard-coded to fail:

```cpp
HighsStatus setHotStart(const HotStart& hot_start) {
  this->deprecationMessage("setHotStart", "None");
  return HighsStatus::kError;
}
```

`freezeBasis`, `unfreezeBasis`, and `frozenBasisAllDataClear` (same file,
lines 1504-1525) are similarly deprecated no-ops. These APIs are not viable.

## Finding 2 — `setBasis(HighsBasis&)` with `alien=false` saves one full LU factor

`Highs_setBasis` (`interfaces/highs_c_api.cpp:664-706`) locally constructs

```cpp
HighsBasis basis;   // struct default: alien = true
```

and passes it to `Highs::setBasis(const HighsBasis&, ...)`
(`lp_data/Highs.cpp:2557-2620`). The two branches are:

### Alien branch (current behaviour)

```cpp
if (basis.alien) {
  if (model_.lp_.num_row_ == 0) { ... }
  else {
    if (!isBasisRightSize(model_.lp_, basis)) return kError;
    HighsBasis modifiable_basis = basis;
    modifiable_basis.was_alien = true;
    HighsLpSolverObject solver_object(...);
    HighsStatus return_status = formSimplexLpBasisAndFactor(solver_object);
    if (return_status != kOk) return kError;
    basis_ = std::move(modifiable_basis);
  }
}
```

`formSimplexLpBasisAndFactor` (`lp_data/HighsSolution.cpp:1811-1872`) with
`alien=true` goes through `accommodateAlienBasis`
(`HighsSolution.cpp:1875-1934`) which builds a **full LU factorization**
(`HFactor::build()` at line 1898) purely to detect rank deficiency and
complete the basis with logicals. The factorization is **discarded** — it is
not reused for the subsequent solve.

### Non-alien branch

```cpp
} else {
  if (!isBasisConsistent(model_.lp_, basis)) {
    highsLogUser(...,  "setBasis: invalid basis\n");
    return kError;
  }
  basis_ = basis;
}
basis_.valid = true;
basis_.useful = true;
...
newHighsBasis();   // → invalidateBasis → has_invert = false
```

`isBasisConsistent` (`HighsSolution.cpp:1950-1963`) enforces:

1. `basis.col_status.size() == lp.num_col_` _and_
2. `basis.row_status.size() == lp.num_row_` _and_
3. Total number of `kBasic` statuses equals `lp.num_row_`.

**No factorization is performed during setBasis** on this branch. The LP's
`basis_` is updated in place; the simplex-level basis and factorization are
built only when `run()` is called.

### Savings

For a basis that is already consistent (e.g. a basis extracted via
`getBasis` after a successful solve), the alien check-factorization is
redundant. Switching the wrapper to `alien = false` when the caller can
guarantee consistency eliminates **one full LU factor per
`solve_with_basis` call**.

`run()` still performs its own factorization via
`ekk_instance.initialiseSimplexLpBasisAndFactor` — that one is unavoidable
because `Highs::setBasis` always calls `newHighsBasis()`, which fires
`updateStatus(LpAction::kNewBasis)` → `invalidateBasis()` → `has_invert =
false`. The non-alien gain is therefore "one factor" not "two factors", but
it is the cheapest correctness-preserving change possible inside the current
`solve_with_basis` flow.

### Applicability to Cobre's basis shapes

Cobre supplies bases of size `num_col × num_row` where `num_row = base_rows

- active_cut_count`. After a successful solve, `get_basis`returns a basis
with **exactly`num_row` basic variables\*\* (simplex invariant). Non-alien
  would accept it directly.

The wrinkle is basis **padding**. `pad_basis_for_cuts`
(`basis_padding.rs:72-148`) extends the row-status vector for new cut rows
with either `kBasic` or `kNonbasic_kLower`:

- **Old structural basis contributes** exactly `base_rows_basic_count` basic
  rows — a number ≤ `num_row_old` that equals the number of basic slacks at
  the previous solve.
- **Padded cut rows contribute** `basis_padding_slack` additional basic
  rows.

The total basic count after padding is
`old_basic_count + basis_padding_slack`, which is _not_ guaranteed to equal
`new_num_row`. For the non-alien check to pass, the padding routine must
deterministically produce exactly `new_num_row` basic variables — typically
by padding all new cut rows as `kBasic` (i.e. switching off the
"informed tight/slack" discrimination for the purpose of counting).

In the **baked-template** flow (epic-03), cut rows are already structural
rows of the template — no padding is needed between scenarios, and the
basis from the previous solve has exactly `num_row` basic variables by
construction. **Non-alien works directly on the baked path.**

## Finding 3 — `putIterate`/`getIterate` preserves the full iterate (including invert)

`Highs.h:1317-1324` exposes

```cpp
HighsStatus putIterate();   // save the current iterate
HighsStatus getIterate();   // restore the saved iterate
```

The implementations at `Highs.cpp:2635-2660` delegate to
`HEkk::putIterate` and `HEkk::getIterate` (`simplex/HEkk.cpp:3806-3832`):

```cpp
void HEkk::putIterate() {
  assert(this->status_.has_invert);
  SimplexIterate& iterate = this->simplex_nla_.simplex_iterate_;
  this->simplex_nla_.putInvert();
  iterate.basis_ = this->basis_;
  if (this->status_.has_dual_steepest_edge_weights) {
    iterate.dual_edge_weight_ = this->dual_edge_weight_;
  } else {
    iterate.dual_edge_weight_.clear();
  }
}

HighsStatus HEkk::getIterate() {
  SimplexIterate& iterate = this->simplex_nla_.simplex_iterate_;
  if (!iterate.valid_) return HighsStatus::kError;
  this->simplex_nla_.getInvert();
  this->basis_ = iterate.basis_;
  if (iterate.dual_edge_weight_.size())
    this->dual_edge_weight_ = iterate.dual_edge_weight_;
  else
    this->status_.has_dual_steepest_edge_weights = false;
  this->status_.has_invert = true;
  return HighsStatus::kOk;
}
```

**This is the only path in HiGHS that preserves the factorization across
solver calls.** `has_invert = true` after `getIterate` means the next
`run()` **skips factoring entirely** and proceeds directly to simplex
pivoting. For LPs where the bounds change but the matrix structure does
not, this is the ideal warm-start.

### Constraints

These APIs are **single-instance and single-slot**:

- The saved state lives inside `simplex_nla_.simplex_iterate_` of the same
  `Highs` instance that called `putIterate`. Crossing instances / threads
  is not supported; `get`/`put` must be paired on the same instance.
- Only one iterate is stored. Subsequent `putIterate` calls overwrite the
  slot. Cobre's per-scenario basis cache (`BasisSlice::get_mut(m, t)`) does
  not map onto this — it would need to be replaced with a single canonical
  iterate per (worker, stage).

### Invalidation rules

The saved iterate survives:

- `changeColBoundsInterface` / `changeRowBoundsInterface` — `updateStatus`
  is called with `LpAction::kNewBounds`, which only clears
  `has_fresh_rebuild` and the cached objective values
  (`simplex/HEkk.cpp:325-329`). **Basis, invert, and saved iterate are
  preserved.**
- Option changes that do not re-solve.

The saved iterate is **destroyed** by:

- `passLp` / `load_model` — calls `clearSolver` → `invalidateSolverData` →
  `invalidateEkk` → `HEkk::invalidate`. `invalidate` itself does not wipe
  `simplex_iterate_`, but it sets `status_.initialised_for_new_lp = false`,
  and `Highs::getIterate` at `Highs.cpp:2648` returns `kError` when that
  flag is false.
- `addRows` with `kExtendInvertWhenAddingRows = false` (the default in
  `lp_data/HConst.h:49`) — triggers `updateStatus(kNewRows)` →
  `this->clear()` → `simplex_nla_.clear()` → `simplex_iterate_` is
  destroyed.
- Direct calls to `clear_solver` / Cobre's `reset`.

### Applicability to Cobre

`putIterate`/`getIterate` **cannot cross a `load_model` or `add_rows` call
for non-baked templates**. In Cobre's current flow (`load_model` per
scenario at `forward.rs:1257, 1259`), every call wipes the saved iterate.

**To make `putIterate`/`getIterate` useful, Cobre must stop reloading the
model per scenario.** The baked-template flow already bundles cuts into the
template, making this feasible — the template only changes between
iterations (when cuts are added/removed/reordered), not between scenarios.

## Finding 4 — `Highs_setLogicalBasis` is misnamed and does not install a logical basis

`Highs_setLogicalBasis` (`interfaces/highs_c_api.cpp:708-710`) calls
`Highs::setBasis()` (no-arg, `Highs.cpp:2623-2633`) which **invalidates** the
current basis. The comment explicitly says

```cpp
// Don't set to logical basis since that causes presolve to be skipped
this->invalidateBasis();
```

A true "install a logical basis" is done lazily by
`HEkk::initialiseSimplexLpBasisAndFactor` (`HEkk.cpp:1471-1472`) when
`!status_.has_basis`. This just builds the identity basis internally — an
intentional cold start. Not useful for warm-starting.

## Finding 5 — The EKK-level `has_invert` is the real control lever

The flag that determines whether HiGHS will or will not factor on the next
`run()` is `ekk_instance_.status_.has_invert`. It is set to `true` only by:

- `ekk_instance_.getIterate()` (line 3830).
- Internal simplex factor / refactor operations at the end of a successful
  iteration.

It is set to `false` by a long list of `updateStatus` actions
(`HEkk.cpp:313-378`). Notably, `updateStatus(LpAction::kNewBasis)` —
triggered by **every** `Highs::setBasis` call (alien or not, via
`newHighsBasis()`) — always invalidates the invert.

This means no public API can supply a basis AND preserve the invert in a
single call. The only way to keep `has_invert = true` is to avoid calling
`setBasis` at all — i.e., let HiGHS's internal basis and factor carry over
from one `run()` to the next, optionally checkpointed with
`putIterate`/`getIterate`.

## Finding 6 — `kExtendInvertWhenAddingRows` is compile-time false

`lp_data/HConst.h:49`:

```cpp
const bool kExtendInvertWhenAddingRows = false;
```

With this default, `addRows` always triggers `updateStatus(kNewRows)` →
`this->clear()` → full EKK wipe (`HEkk.cpp:337-346`, `31-43`). Even though
`appendBasicRowsToBasisInterface` (`HighsInterface.cpp:1403-1432`) tries to
extend `simplex_basis` with basic slacks, `HEkk::addRows` immediately
follows up with the clear. The NLA, the invert, the dual edge weights — all
gone.

This is why the **baked-template flow** (epic-03) is load-bearing for
warm-start performance: it avoids `addRows` on the hot path. On the legacy
(non-baked) path, every iteration's `add_rows` forces a cold simplex
factorization on the next solve regardless of what Cobre does with bases.

## Summary of Available Warm-Start Mechanisms

| Mechanism                              | Preserves basis? | Preserves invert? | Usable today in Cobre?                                                 | Gain                                |
| -------------------------------------- | ---------------- | ----------------- | ---------------------------------------------------------------------- | ----------------------------------- |
| `Highs_setBasis` (alien=true, current) | Yes              | No                | Yes                                                                    | Baseline                            |
| `Highs::setBasis` (alien=false)        | Yes              | No                | Only on baked path                                                     | Skip one LU factor per setBasis     |
| `putIterate`/`getIterate`              | Yes              | **Yes**           | Only if `load_model` is removed from the per-scenario loop             | Skip the run()-time factor entirely |
| `setHotStart`                          | n/a              | n/a               | No (deprecated)                                                        | n/a                                 |
| Implicit carry-over (no `setBasis`)    | Yes              | Yes               | Requires eliminating per-scenario `load_model` and avoiding `add_rows` | Skip the run()-time factor entirely |

## Recommended Plan

### Phase 1 — Non-Alien SetBasis (low risk, moderate gain)

Add a new wrapper function to `csrc/highs_wrapper.{h,c}`:

```c
int32_t cobre_highs_set_basis_non_alien(
    void* highs,
    const int32_t* col_status,
    const int32_t* row_status);
```

Implementation:

```cpp
HighsBasis basis;
basis.alien = false;    // key line
/* populate col_status and row_status exactly as Highs_setBasis does */
return (int32_t)((Highs*)highs)->setBasis(basis);
```

On the Rust side, expose a second `solve_with_basis_non_alien` method on
`HighsSolver` that uses the new FFI function. Use it from the **baked**
forward/backward paths, where the supplied basis is guaranteed to have
exactly `num_row` basic variables. Fall back to the existing alien path
on the non-baked path (where padding may produce inconsistent counts) and
as the retry-escalation cold path.

Expected gain: **one full LU factor per warm-started solve** on the baked
path.

Risks:

- If a non-alien basis fails `isBasisConsistent`, `setBasis` returns `kError`
  and Cobre must fall back to the alien path or cold-start. Instrumentation
  via a new `basis_non_alien_rejections` counter in `SolverStatistics` is
  needed to detect this regression.
- Padding correctness becomes load-bearing. Any future change to
  `pad_basis_for_cuts` that breaks the invariant "total basic count =
  num_row" silently falls back to alien.

### Phase 2 — Eliminate per-scenario `load_model` (higher risk, higher gain)

The larger win is to stop reloading the model between scenarios. With a
baked template the structural LP is identical across scenarios at a given
(worker, stage) pair; only bounds change. Keeping the HiGHS instance warm
across scenarios lets the natural simplex warm-start kick in (no `setBasis`
call needed).

The blocker is determinism: Cobre's hard rule is bit-identical results
regardless of thread assignment. A scenario executed after scenarios A, B,
C on one thread may reach a different optimal vertex than when executed
after X, Y, Z on another thread, in the presence of primal/dual
degeneracy.

Two feasible sub-strategies:

1. **Canonical iterate per (worker, stage)**: call `putIterate` once after
   an initial solve, then `getIterate` at the start of every scenario. Every
   scenario starts from the same snapshot.
2. **Bound the cross-scenario chain**: accept that scenarios executed in
   deterministic order within a worker form a warm-start chain, but ensure
   the work partition is deterministic (Cobre's partition function already
   is). Validate bit-identicality with the existing D01–D30 deterministic
   regression suite.

### Phase 3 — Speculative

Flipping `kExtendInvertWhenAddingRows` to `true` in the vendored HiGHS copy
would enable invert extension on `addRows`, making the non-baked path
warm-startable as well. This requires validating that HiGHS's
`HFactorExtend` code path is production-quality — a pre-requisite is a
careful read of `util/HFactorExtend.cpp`.

## Immediate Next Action

Implement Phase 1. It is:

- Low-risk (falls back to alien path on any inconsistency).
- Isolated to `cobre-solver` (`highs_wrapper.{h,c}`, `ffi.rs`, `highs.rs`).
- Covered by existing tests if we assert the fallback behaviour.
- Instrumented via a new statistic (`basis_non_alien_rejections`).

Benchmarking harness: pick a representative case from the D01–D30 suite,
run training for a fixed iteration count, record `total_solve_time_seconds`
and `success_count`. Apply Phase 1, re-run. The expected delta is a
reduction in per-solve factor time proportional to the size of the
`solve_with_basis` calls that used the baked path.

---

# Amendment — Cross-Work-Distribution Reproducibility (2026-04-17)

This amendment **supersedes the "Recommended Plan" section above**. It
incorporates a constraint that the original plan implicitly relaxed, and a
C-API-exposure finding that changes the shape of the minimum viable fix.

## The Constraint That Invalidated the Previous Phase 2

Cobre's reproducibility invariant is stronger than "HiGHS is deterministic
given identical input":

> Results must be **bit-for-bit identical regardless of how scenarios /
> trial points are partitioned across MPI ranks × OpenMP threads**. A
> 4-rank × 8-thread run and a 1-rank × 32-thread run must produce the same
> cuts, policy, and simulation output.

This is why the backward pass calls `load_model` per trial point today
(`backward.rs:699`) and the forward pass calls it per scenario
(`forward.rs:1325, 1327`). `passLp` was the only resource available to
guarantee that every solve starts from an identical cold state.

The original Phase 2 ("eliminate per-scenario `load_model`, rely on
`kNewBounds`") breaks this invariant. Worker A solving scenarios
`{0,1,2,3}` sequentially arrives at scenario 2 with three prior solves of
accumulated simplex state (basis, invert, DSE weights). Worker B solving
`{2,3}` arrives at scenario 2 with zero prior state. HiGHS is deterministic
given identical inputs, but the _inputs to scenario 2's simplex_ differ
between the two workers. Under primal/dual degeneracy the resulting vertex
— and therefore the dual multipliers fed into cut construction — can
differ. The cobre team has observed this regression empirically.

**`changeBounds` alone is not a reproducibility-preserving reset.** It
preserves the invert and basis for performance, but those are exactly the
bits that carry scenario-history bias.

## What We Actually Need

A mechanism that gives every scenario solve an **identical starting
simplex state across workers**, at a fraction of the cost of `passLp`.
Three properties are required:

1. **Deterministic** — byte-identical starting state across workers for
   the same (iteration, stage).
2. **Cheap** — materially cheaper than a full structural reload.
3. **Correctness-preserving** — produces the same pivot path a cold
   start-from-template would.

`kNewBounds`-only fails (1). `passLp` fails (2). What remains is to
identify the cheapest HiGHS operation that restores **all** solver state
flags cobre's determinism depends on — not just the basis — so that every
trial point's solve starts from an equivalent "canonical" state
regardless of the prior solve history on the same worker. The exhaustive
audit below identifies that operation: `Highs_clearSolver`.

## Finding 7 — Exhaustive `setBasis`-as-Reset Determinism Audit

Before committing to a reset strategy we performed a line-by-line audit
of HiGHS to answer a single question: given a HiGHS instance that has
previously solved many LPs, does `setBasis(B) + changeBounds(Δ) + run`
produce byte-identical output to `passLp(template) + setBasis(B) +
changeBounds(Δ) + run` on a fresh instance? The answer is **NO** —
`setBasis` alone is not a sufficient deterministic reset. Two specific
HiGHS internal state fields are cleared by `passLp` (which calls
`clearSolver` internally) but **not** by `setBasis`:

### Finding 7.1 — `HEkk::random_` PRNG state leaks across solves

`setSimplexOptions()` (`HEkk.cpp:1623-1645`) reseeds
`random_.initialise(options_->random_seed)` at line 1639. But
`setSimplexOptions` is invoked only from `initialiseEkk()`
(`HEkk.cpp:1574-1581`), which is guarded at line 1574:

```cpp
if (this->status_.initialised_for_new_lp) return;
```

`setBasis` → `newHighsBasis()` → `updateStatus(LpAction::kNewBasis)` →
`invalidateBasis()` → `invalidateBasisArtifacts()` does **not** clear
`initialised_for_new_lp`. This flag is cleared only by
`HEkk::invalidate()` (`HEkk.cpp:277-284`), which `setBasis` never
invokes.

Consequence: on a reused instance, the PRNG is seeded exactly once
(on the first `passLp`). Every subsequent `run()` calls
`initialiseForSolve()` → `initialiseSimplexLpRandomVectors()`
(`HEkk.cpp:1601`), which advances `random_` to produce
`numTotPermutation_`, `numColPermutation_`, `numTotRandomValue_`.
These permutation vectors feed into simplex pricing tie-breaks. Two
runs that differ in the number of prior solves on the same instance
see different permutation vectors on identical input, producing
different pivot sequences and (on degenerate LPs) different optimal
vertices.

### Finding 7.2 — `bad_basis_change_` taboo list accumulates across solves

The `bad_basis_change_` vector holds cycle-avoidance taboo entries
written during degenerate solves. `tabooBadBasisChange()`
(`HEkk.cpp:665`) reads it to prevent repeating a pivot that led to
cycling. The vector is cleared only by `clearBadBasisChange()`, called
solely from `initialiseEkk()` (same `initialised_for_new_lp` guard).

Consequence: a reused instance carries taboos from prior degenerate
solves. A subsequent `run()` avoids pivots that a fresh instance
would freely choose. On degenerate LPs — which backward-pass
subproblems frequently are — this produces divergent trajectories.

### Finding 7.3 — What IS safely reset by `setBasis`

The exhaustive field audit confirmed these properties hold for cobre's
option set (`simplex_strategy = kSimplexStrategyDualPlain`,
`parallel = off`, `simplex_scale_strategy = 0`, `presolve = off`):

- **Dual edge weights.** `invalidateBasisArtifacts()` sets
  `has_dual_steepest_edge_weights = false`. `HEkkDual.cpp:146`
  unconditionally `assign(solver_num_row, 1.0)` before any DSE
  computation — stale buffer values are never read.
- **Perturbation state.** `costs_perturbed` / `bounds_perturbed` are
  reset to `false` by `initialiseCost()` / `initialiseBound()`
  (`HEkk.cpp:2443, 2571`) on every `initialiseForSolve()`.
- **Factorization (`simplex_nla_.factor_`).** `has_invert = false`
  after `invalidateBasisArtifacts` forces a full fresh
  `HFactor::build()` on the next `run`, which calls
  `refactor_info_.clear()` (`HSimplexNla.cpp:126`). Stale factor data
  is overwritten.
- **Crash / logical basis.** A valid consistent `setBasis(B)`
  installs `basis_ = B` and sets `has_basis = true`, bypassing the
  crash heuristic entirely (`HApp.h:191-215`).
- **Synthetic clock / timer-based gating.** All algorithmic decisions
  are gated on deterministic counters (iteration count, synthetic
  tick derived from FLOP estimates). No wall-clock decisions remain
  in the production path — the obsolete synthetic-tick check in
  `HEkkDual.cpp:2258-2265` is commented out.
- **Parallel simplex.** With `simplex_strategy =
kSimplexStrategyDualPlain` and `parallel = off`,
  `chooseSimplexStrategyThreads()` (`HEkk.cpp:1726`) never upgrades
  to `kSimplexStrategyDualMulti`. No thread-local state, no task
  scheduling.

### Finding 7.4 — The correct reset is `Highs_clearSolver`

`Highs::clearSolver()` (`Highs.cpp:63-68`) calls
`invalidateSolverData()`, which includes `invalidateEkk()` →
`HEkk::invalidate()` (`HEkk.cpp:277-284`). This sets
`initialised_for_new_lp = false` and `initialised_for_solve = false`.
On the next `run()`:

- `initialiseEkk()` fires (because `initialised_for_new_lp == false`)
- `setSimplexOptions()` reseeds `random_` from `options.random_seed`
- `clearBadBasisChange()` zeros the taboo list
- `initialised_for_new_lp = true` is restored

After `clearSolver`, the sequence `setBasis(B) + changeBounds(Δ) +
run` produces byte-identical output to `passLp + setBasis(B) +
changeBounds(Δ) + run` on a fresh instance, given identical options.

**Crucially, `clearSolver` does NOT touch the LP model itself.** The
CSC matrix (`lp_.a_matrix_`) and bounds vectors persist. No matrix
re-copy, no `assessMatrix` re-scan. The work is O(1) in flag resets
plus a few small vector clears — compared to `passLp`'s O(nnz) matrix
transfer and assessment.

This reconciles the apparent tension with HiGHS issue #1598 (user
complaints about non-determinism on reuse): the reported symptom is
consistent with `bad_basis_change_` taboo accumulation across solves.
The issue would be closed by exactly the reset we need — which
`clearSolver` already provides.

### Finding 7.5 — The FFI is already in place

`Highs_clearSolver` is in the HiGHS C API
(`interfaces/highs_c_api.h:396`, `.cpp:378`). Cobre's wrapper already
exposes it: `cobre_highs_clear_solver` (`csrc/highs_wrapper.h:166`,
`.c:196-197`), and it's declared on the Rust FFI at
`crates/cobre-solver/src/ffi.rs:204`. It is currently invoked inside
`HighsSolver::reset` (`highs.rs:1123`) and the retry-escalation path
(`highs.rs:652`) — **no new FFI is required**. The missing piece is a
trait-level entry point that triggers the state reset without forcing
a subsequent `load_model` (which the current `reset()` does via
`has_model = false` at `highs.rs:1132`).

---

# Implementation Roadmap

The roadmap splits into two shipping phases (B.1 → A.1) and two gated
follow-ups (B.2 → possibly C). No new types, no new storage pattern, no
`SolverState` abstraction. One new trait method, one new FFI function.

## Phase B.1 — Non-alien `setBasis`

**Goal.** When `HighsSolver::solve_with_basis` installs a caller-provided
basis, route through `Highs::setBasis` with `alien = false` instead of
the current default-alien path. Saves one LU factor per basis install
(the throwaway factor inside `formSimplexLpBasisAndFactor`).

**Scope.**

1. Add `cobre_highs_set_basis_non_alien(void*, const int32_t*, const int32_t*)`
   to `csrc/highs_wrapper.{h,c}`. Implementation constructs a `HighsBasis`,
   sets `alien = false`, populates `col_status` / `row_status` from the
   raw `i32` inputs, calls `((Highs*)h)->setBasis(basis)`.
2. Mirror declaration in `crates/cobre-solver/src/ffi.rs`.
3. In `HighsSolver::solve_with_basis`, try the non-alien path first.
   On `HIGHS_STATUS_ERROR` (meaning `isBasisConsistent` rejected the
   basis), fall back to the existing alien `cobre_highs_set_basis`.
4. New counter `basis_non_alien_rejections` on `SolverStatistics`.

**Acceptance.** D01-D30 bit-identical. Instrumentation shows >99% of
warm-start calls succeed on the non-alien path for baked-template LPs.

**Risk.** Low. Fallback to alien preserves current behavior on any
inconsistency. **No trait changes, no caller changes.**

## Phase A.1 — Per-stage `load_model`, per-trial-point `clear_solver_state`

**Goal.** Eliminate the O(nnz) `passLp` matrix re-copy currently
performed per trial point in the backward pass (`backward.rs:699`).
Replace with per-stage `load_model` (the template is identical across
all trial points at a given stage) plus per-trial-point
`clear_solver_state` + basis install + bound patch + solve.

### Trait addition

```rust
/// Clears the solver's derived state (factorization, warm-start weights,
/// PRNG state, cycle-avoidance taboos, all simplex status flags) while
/// keeping the loaded LP intact. After this call the next solve behaves
/// as if it were the first solve on a fresh instance with the same LP.
///
/// This is the deterministic-reset primitive for warm-start chains that
/// span work-distribution variations — specifically, the backward pass
/// reusing a solver across trial points at a single stage.
///
/// Contract: caller does NOT need to call `load_model` before the next
/// solve. Bounds should be set (via `set_row_bounds` / `set_col_bounds`)
/// and a warm-start basis may be installed via `solve_with_basis`.
///
/// # Errors
/// `SolverError::Unsupported` on backends without an equivalent cheap
/// reset. Such backends should be invoked via the `reset` +
/// `load_model` path instead.
fn clear_solver_state(&mut self) -> Result<(), SolverError> {
    Err(SolverError::Unsupported("clear_solver_state not implemented for this backend"))
}
```

HighsSolver implementation: two FFI lines wrapping
`ffi::cobre_highs_clear_solver`. Does not touch `has_model`, `num_cols`,
`num_rows` — the LP remains loaded and usable.

### Backward-pass restructuring

The backward pass outer loop is already stage-major
(`backward.rs:806`). The structural change is moving `load_model` from
per-trial-point to per-worker-per-stage:

```rust
// Pseudocode — replaces the per-trial-point load_backward_lp.
for t in (0..num_stages - 1).rev() {
    // Barrier implicit in process_stage_backward.
    let workers = /* ... */;
    workers.par_iter_mut().for_each(|ws| {
        ws.solver.load_model(&templates[t]);          // ONCE per worker per stage
        while let Some(trial) = claim_next_trial_point() {
            ws.solver.clear_solver_state()?;          // per-trial-point deterministic reset
            let stored_basis = state_store.get(trial.scenario, t);
            // For opening 0:
            apply_bounds_for_opening(ws, trial, 0);
            if let Some(b) = stored_basis {
                ws.solver.solve_with_basis(b)?;
            } else {
                ws.solver.solve()?;                   // cold fallback
            }
            ws.solver.get_basis(&mut captured.basis);
            // For openings 1..K-1: natural warm-start chain.
            for k in 1..num_openings {
                apply_bounds_for_opening(ws, trial, k);
                ws.solver.solve()?;                   // invert preserved via kNewBounds
            }
        }
    });
}
```

Key invariants:

- `load_model` per worker per stage. One call per `(worker, stage)`
  pair, not per `(worker, trial_point)`.
- `clear_solver_state` per trial point — resets RNG, bad-basis-change,
  and all simplex status flags without re-copying the LP matrix.
- `solve_with_basis` per trial point (opening 0) — deterministic basis
  install (the caller provides the stored basis from the state store).
- Natural warm-start chain for openings 1..K-1 via `changeBounds` + `run`,
  as today.

### Scope

1. Add `clear_solver_state` to `SolverInterface` (`trait_def.rs`) with
   default `Err(Unsupported)` implementation.
2. Implement on `HighsSolver` via `ffi::cobre_highs_clear_solver`. Do
   not touch `has_model` / `num_cols` / `num_rows`. Return
   `SolverError::InternalError` only on FFI failure.
3. Restructure the backward-pass worker loop: detect stage transitions
   (first trial point claimed at each stage), hoist `load_model` out of
   the trial-point loop. Per trial point, call `clear_solver_state`
   before applying opening 0's bounds and solving.
4. Gate the new path behind `CanonicalStateStrategy::{Disabled,
ClearSolver}` config option, defaulting to `Disabled`.
5. Instrument: `clear_solver_count` and `clear_solver_failures` counters
   on `SolverStatistics`.

### Acceptance criteria (blocking)

**Cross-work-distribution reproducibility is not optional.** The prior
team hit reproducibility bugs on a similar optimization. The audit
argues `clearSolver` is a complete reset, but empirical validation
across parallel configurations is the authority.

The invariant we test is specifically: **a single binary built with
the A.1 changes must produce identical outputs regardless of how work
is distributed across workers or MPI ranks.** We do NOT require the
new binary to reproduce pre-A.1 reference outputs — the algorithmic
restructuring (per-stage `load_model` + per-trial-point
`clear_solver_state`, replacing per-trial-point `passLp`) may produce
numerically different but equally valid solutions by choosing a
different path through the simplex. What is non-negotiable is that
the new binary's output depends only on the problem, not on how many
workers or ranks happen to be running.

**What is required.**

1. For each D01-D30 case, the binary run under `ClearSolver` must
   produce byte-identical outputs across worker counts 1, 2, 4, 8 on
   a single rank. Any divergence across worker counts is a hard
   blocker.
2. Same invariant across MPI configurations: 2, 4, 8 ranks × 1, 2, 4
   threads-per-rank. Outputs must be byte-identical across all
   rank/thread combinations. Covered on at least three representative
   D-cases.
3. `work_stealing_produces_identical_results_across_worker_counts`
   (`backward.rs:4516`) must pass under `ClearSolver`. This is the
   existing in-suite probe for the exact invariant A.1 must preserve.

**What is NOT required.**

- Byte-identical match against pre-A.1 reference outputs. D-case
  fixtures may need to be regenerated from the new binary once the
  cross-distribution invariant holds. Update fixture files as part of
  the A.1 PR; note the fixture rebaseline in the PR description.
- Any specific match against the fixtures currently in the repo. If
  the new solver path lands on a different-but-equivalent optimum, we
  accept that — as long as all distributions land on the same new
  optimum.

**Post-flip expectation.** Once `ClearSolver` becomes the default,
the regenerated D-case fixtures become the new frozen reference. All
future PRs must then preserve byte-identicality against those
fixtures AND across work distributions. Any future change that breaks
either is a regression.

**Divergence across distributions at any point is a hard blocker.**
If it occurs during A.1 validation, the audit missed a state path.
Reopen the investigation — do not flip the default until the path is
identified and closed.

### Expected gain

Eliminates the O(nnz) matrix copy + `assessMatrix` work per trial point.
For typical cobre LPs (~500 rows, ~5000 nonzeros), `passLp` costs tens
to hundreds of microseconds per call. Per iteration,
`num_scenarios × num_stages` trial points × `passLp` is a significant
wall-clock contributor. Replacing it with `clearSolver` (flag resets +
small vector clears) is roughly an O(1) op per trial point.

The user has reported backward-pass growing to >90% of program
execution time on recent benchmarks, driven in part by load imbalance.
A.1 reduces the per-trial-point overhead proportionally; load
imbalance is addressed separately (not in scope here).

### Risk

Medium. Bit-identicality tests are the safeguard. The audit is
thorough but empirical validation is what closes the risk.

## Phase B.2 — Per-`(scenario, stage, opening)` basis storage (deferred, empirical)

**Goal (when pursued).** Replace the single per-`(scenario, stage)` basis
slot with per-`(scenario, stage, opening)` slots, so each opening's solve
warm-starts from its own previously-stored basis rather than chaining
from the previous opening's end state.

**Trade-off.** `setBasis` **always** invalidates the invert (verified
in Finding 7 — both alien and non-alien paths call `newHighsBasis()`
→ `invalidateBasis()`). Per-opening basis therefore replaces
today's 1 refactor per `(m, t)` (the opening-0 solve's factor) with K
refactors per `(m, t)`. Non-alien `setBasis` from B.1 makes each
individual factor cheaper, but does not avoid them.

Whether this is a net win depends on:

- How correlated the openings at a given `(m, t)` are. Highly
  correlated → chained warm-start is already near-optimal; per-opening
  basis loses.
- The factor-cost / pivot-cost ratio for cobre's LP dimensions.
  Typical rule of thumb: one factor ≈ 10-25 pivots.
- How much per-opening basis saves in pivot count vs. chained.

**Criteria for pursuing.** Profile A.1 under production workloads.
If per-opening pivot counts are consistently > 15-20 and openings are
weakly correlated (noise realizations with broad stochastic variation),
B.2 is a candidate. Otherwise skip.

**Scope (when pursued).**

1. Add a parallel `BackwardBasisStore` to `cobre-sddp/src/workspace.rs`,
   indexed `[scenario * num_stages * num_openings + stage *
num_openings + opening]`. Same `CapturedBasis` payload.
2. Memory projection: `num_scenarios × num_stages × num_openings ×
sizeof_basis_slot` logged at training start, warned above a threshold.
3. In the backward-pass worker loop, per opening: `clear_solver_state`
   - `set_*_bounds` + `solve_with_basis(&per_opening_basis)`. No more
     natural warm-start chain across openings.
4. Validate bit-identicality (same criteria as A.1).

**Risk.** Low-medium. Memory footprint is the main concern — bases
are small (5-15 KB) so the K× multiplier stays tractable at typical
cobre workloads (~1.5-9 GB), but must be projected per-case.

## Phase C — Stage-major forward pass (deferred indefinitely)

Unchanged from prior analysis. Flipping the forward pass to
stage-major would allow the same per-stage `load_model` amortization
as A.1, but requires inverting the loop nesting, transposing basis
storage, and re-auditing determinism. **Switching axes does not itself
add parallelism** — same M × W coarse granularity. Phase A.1 does not
apply to the forward pass because scenarios walk stages 1..T
sequentially; there is no natural per-stage boundary to hoist
`load_model` out of.

Defer until profiling post-A.1 shows forward-pass `passLp` dominates,
which is unlikely.

## Community Evidence: Upstream Will Not Deliver

- **HiGHS #1598** (open, 2024, Enhancement): users reported
  non-determinism under instance reuse. The symptom pattern matches
  the `bad_basis_change_` taboo accumulation path (Finding 7.2).
  Assigned to jajhall, no PR. The fix users need — explicit
  force-refactorization and state reset across reuses — already
  exists as `Highs_clearSolver` but the upstream issue hasn't been
  closed with that guidance.
- **HiGHS #1607** (open): tangential documentation issue.
- **HiGHS.jl #192** (closed, zero comments, zero PRs): @odow (JuMP)
  and @joaquimg requested basis reset — upstream identified it as a
  HiGHS-core change, never picked it up.

Translation: extending `highs_wrapper.{h,c}` ourselves remains the only
way forward. The extension surface is now **one function** (non-alien
`setBasis`). `clearSolver` is already wired. No blob APIs, no iterate
serialization, no internal-layout coupling.

## Roadmap Summary

| Phase | Scope                                                                | New FFI | Trait change | Perf delta      | Risk    | Depends on   |
| ----- | -------------------------------------------------------------------- | ------- | ------------ | --------------- | ------- | ------------ |
| B.1   | Non-alien `setBasis` inside `solve_with_basis`                       | 1 fn    | None         | Medium          | Low     | —            |
| A.1   | `clear_solver_state` trait + per-stage `load_model` in backward pass | None    | 1 method     | **High**        | Medium  | B.1          |
| —     | Profile                                                              | —       | —            | —               | —       | A.1          |
| B.2   | Per-`(scenario, stage, opening)` basis storage                       | None    | None         | Empirically tbd | Low-Med | A.1, profile |
| C     | Stage-major forward pass                                             | None    | None         | Low             | High    | (deferred)   |

**Headline.** One new FFI function, one new trait method, backward pass
hoists `load_model` to per-stage cadence. That is the entire shipping
surface of the optimization.

## Revised Immediate Next Action

1. **Phase B.1 PR** — non-alien `setBasis` wrapper + fallback +
   `basis_non_alien_rejections` counter. Self-contained.
2. **Phase A.1 PR** — `clear_solver_state` trait method + `HighsSolver`
   impl + backward-pass restructure. Gated behind
   `CanonicalStateStrategy::ClearSolver`. **Must produce byte-identical
   outputs across worker counts 1/2/4/8 and across MPI 2/4/8 ranks × 1/2/4
   threads on D01-D30 before default flips.** Rebaseline the D-case
   fixtures in the same PR if the new binary lands on a different-but-
   equivalent optimum than the pre-A.1 baseline — the cross-distribution
   invariant, not the pre-A.1 match, is what gates the flip.
3. **Profile** after A.1 lands. Confirm backward-pass wall time drops
   proportionally to eliminated `passLp` work. Identify the new top
   hotspot.
4. **Phase B.2** only if profiling shows opening-level pivot counts are
   high and openings are weakly correlated.
5. **No Phase C** without separate restructuring justification.
