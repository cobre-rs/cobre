# Solver Hot-Path Refactor: Dead Code & Obsolescence Analysis

**Date**: 2026-03-08
**Context**: After completing the 3-epic solver-hot-path-refactor plan, the cobre-solver
crate now has two parallel paths for basis management and solution extraction. This
report analyzes which code paths are genuinely dead, which are not yet adopted but will
be, and proposes a migration strategy that aligns with the SDDP training loop's need
for per-stage basis warm-starting.

---

## 1. Current Dual-Path Architecture

The refactor added zero-copy alternatives but kept the original allocating paths:

| Operation        | Allocating Path (Original)                          | Zero-Copy Path (New)                                       |
| ---------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| Solve            | `solve() -> LpSolution`                             | `solve_view() -> SolutionView<'_>`                         |
| Basis extraction | `get_basis() -> Basis`                              | `get_raw_basis(&mut RawBasis)`                             |
| Warm-start solve | `solve_with_basis(&Basis) -> LpSolution`            | `solve_with_raw_basis_view(&RawBasis) -> SolutionView<'_>` |
| Hybrid           | `solve_with_basis_view(&Basis) -> SolutionView<'_>` | `solve_with_raw_basis_view(&RawBasis) -> SolutionView<'_>` |

The allocating paths now delegate to the zero-copy paths internally (e.g.,
`solve() -> solve_view().map(|v| v.to_owned())`), making them thin wrappers.

---

## 2. Why the Training Loop Does Not Use Basis Warm-Starting Yet

The SDDP training loop (`cobre-sddp`) currently performs **cold starts only** for every
LP solve. This is not an oversight — it was a deliberate Phase 6 scope decision.

### What the specs say

The cobre-docs specifications **fully define** per-stage basis warm-starting:

- **`solver-workspaces.md` SS1.2-SS1.5**: Each thread owns a workspace containing a
  per-stage basis cache (`T` slots, one per stage). After each solve, the basis is
  extracted into the cache. On the next iteration at the same stage, the cached basis
  warm-starts the solver.

- **`solver-abstraction.md` SS9.2**: Defines how basis validity works when cuts are
  added. Dynamic constraints (cuts) are appended at the bottom of the LP matrix. The
  static portion's basis is reused directly; new rows are initialized as `Basic`
  (slack in basis). This is exactly what `solve_with_raw_basis_view` already implements
  (the dimension mismatch handler fills extra rows with `HIGHS_BASIS_STATUS_BASIC`).

- **`training-loop.md` SS4.3-SS4.4**: The forward/backward pass sequences include
  basis injection and extraction steps.

### Why it was deferred

Phase 6 was scoped for correctness: implementing the complete SDDP training loop with
forward/backward passes, cut management, convergence monitoring, and all trait variants.
Basis warm-starting is a **performance optimization** that doesn't affect convergence —
the algorithm produces the same cuts and lower bounds regardless. The solver infrastructure
(trait methods, HiGHS implementation, conformance tests) was built first so that the
training loop can adopt it when ready.

### Current cold-start solve sequence

Both forward and backward passes follow this pattern:

```rust
solver.load_model(&templates[t]);           // Load structural LP for stage t
solver.add_rows(&cut_batches[t]);           // Append active Benders cuts
solver.set_row_bounds(&indices, &lo, &hi);  // Patch scenario-dependent RHS
let view = solver.solve_view()?;            // COLD START — no basis
// Extract primal/dual values, drop view
```

### What warm-starting would look like

```rust
solver.load_model(&templates[t]);
solver.add_rows(&cut_batches[t]);
solver.set_row_bounds(&indices, &lo, &hi);

let view = if basis_cache.has_basis_for(t) {
    solver.solve_with_raw_basis_view(basis_cache.get(t))?   // WARM START
} else {
    solver.solve_view()?                                     // Cold start (first iteration)
};

// After extracting solution values, store basis for next iteration
solver.get_raw_basis(basis_cache.get_mut(t));
```

### Where warm-starting applies

| Pass     | Context                           | Warm-Start Source                                                     |
| -------- | --------------------------------- | --------------------------------------------------------------------- |
| Forward  | Stage `t`, iteration `i+1`        | Basis from stage `t`, iteration `i` (same stage, prior iteration)     |
| Backward | Stage `t+1`, opening `omega_2..N` | Basis from stage `t+1`, opening `omega_1` (same stage, first opening) |
| Backward | Stage `t+1`, iteration `i+1`      | Basis from stage `t+1`, iteration `i` (same stage, prior iteration)   |

### Per-stage basis cache design (from specs)

Each thread would own a workspace containing:

| Component             | Type                      | Size (60-stage, HiGHS) |
| --------------------- | ------------------------- | ---------------------- |
| Solver instance       | `HighsSolver`             | ~15 MB                 |
| Per-stage basis cache | `Vec<RawBasis>` (T slots) | ~20 MB                 |
| RHS patch buffer      | `PatchBuffer`             | ~54 KB                 |
| Solution buffers      | primal/dual/rc scratch    | ~815 KB                |

### Why `RawBasis` is the right choice for production warm-starting

| Property             | `Basis` (enum path)                                | `RawBasis` (raw path)                     |
| -------------------- | -------------------------------------------------- | ----------------------------------------- |
| Storage              | `Vec<BasisStatus>` (1 byte enum, but Vec overhead) | `Vec<i32>` (native HiGHS format)          |
| Extraction cost      | Read i32 from FFI + per-element enum conversion    | Direct memcpy from FFI into caller buffer |
| Injection cost       | Per-element enum-to-i32 conversion + FFI write     | memcpy into solver buffer + FFI write     |
| Hot-path suitability | Poor — O(n) enum translation per solve             | Optimal — O(1) memcpy per solve           |

At production scale (60 stages x 200 scenarios x 1000 iterations = 12M solves), the
per-element enum conversion in `Basis` would add ~10s of overhead. `RawBasis` eliminates
this entirely.

---

## 3. Classification: Dead vs Deferred vs Active

### 3.1 Genuinely Dead Code (zero callers, no future need)

These items exist only because the enum-based `Basis`/`BasisStatus` path was the
original design. Now that `RawBasis` exists and will be the production path, the enum
path has no callers and no future use case.

| Item                                   | Location             | Dependent Code                         | Dead Because                                |
| -------------------------------------- | -------------------- | -------------------------------------- | ------------------------------------------- |
| `BasisStatus` enum                     | types.rs:18-29       | `Basis`, conversion functions          | `RawBasis` stores native i32 codes directly |
| `Basis` struct                         | types.rs:31-48       | `get_basis()`, `solve_with_basis*()`   | `RawBasis` replaces it for warm-starting    |
| `get_basis()` trait method             | trait_def.rs:161-167 | HiGHS impl, 6 mock stubs               | Replaced by `get_raw_basis()`               |
| `solve_with_basis_view()` trait method | trait_def.rs:141-151 | HiGHS impl, 6 mock stubs               | Replaced by `solve_with_raw_basis_view()`   |
| `solve_with_basis()` default method    | trait_def.rs:137-139 | Delegates to `solve_with_basis_view()` | Wrapper of dead method                      |
| `basis_status_to_highs()`              | highs.rs:540-549     | Used by `solve_with_basis_view()`      | Internal to dead method                     |
| `highs_to_basis_status()`              | highs.rs:566-578     | Used by `get_basis()`                  | Internal to dead method                     |
| `HighsSolver::get_basis()`             | highs.rs:1073-1114   | None in production                     | Replaced by `get_raw_basis()`               |
| `HighsSolver::solve_with_basis_view()` | highs.rs:980-1052    | None in production                     | Replaced by `solve_with_raw_basis_view()`   |

### 3.2 Allocating Wrappers (low value, may remove)

These default methods wrap the zero-copy methods with `.to_owned()`. They have zero
production callers. They add API surface without value.

| Item                             | Location             | Status                                                             |
| -------------------------------- | -------------------- | ------------------------------------------------------------------ |
| `solve()` default                | trait_def.rs:108-110 | Zero callers — `solve_view()` used everywhere                      |
| `solve_with_raw_basis()` default | trait_def.rs:213-215 | Zero callers — `solve_with_raw_basis_view()` will be used directly |

### 3.3 Active / Will Be Adopted (NOT dead)

| Item                          | Location             | Current Callers                         | Future Role                                 |
| ----------------------------- | -------------------- | --------------------------------------- | ------------------------------------------- |
| `solve_view()`                | trait_def.rs:112-123 | forward.rs, backward.rs, lower_bound.rs | Primary cold-start solve (stays)            |
| `get_raw_basis()`             | trait_def.rs:169-181 | Tests only                              | Per-stage basis extraction in training loop |
| `solve_with_raw_basis_view()` | trait_def.rs:183-198 | Tests only                              | Per-stage warm-start in training loop       |
| `RawBasis`                    | types.rs:50-86       | Tests only                              | Per-stage basis cache storage               |
| `SolutionView<'a>`            | types.rs:84-133      | Production (3 call sites)               | Stays — zero-copy solution access           |
| `LpSolution`                  | types.rs:88-120      | `SolverError` variants                  | Stays — owned data in error payloads        |
| `SolutionView::to_owned()`    | types.rs:155-171     | Default method impls                    | Stays — error payload materialization       |
| `SolverStatistics`            | types.rs             | Production                              | Stays — solve metrics                       |

### 3.4 Dead Conformance Tests

These tests exercise exclusively the enum-based basis path:

| Test                                               | File                | What It Tests                    |
| -------------------------------------------------- | ------------------- | -------------------------------- |
| `test_solver_highs_solve_with_basis_warm_start`    | conformance.rs:559  | Enum basis warm-start            |
| `test_solver_highs_solve_with_basis_cut_extension` | conformance.rs:607  | Enum basis + cuts                |
| `test_solver_highs_get_basis_dimensions`           | conformance.rs:696  | Enum basis dimensions            |
| `test_solver_highs_get_basis_roundtrip`            | conformance.rs:720  | Enum basis round-trip            |
| `test_solver_highs_get_basis_with_cuts`            | conformance.rs:754  | Enum basis after cuts            |
| `test_solver_highs_get_basis_preserves_status`     | highs.rs:1618       | BasisStatus conversion           |
| `test_basis_status_roundtrip_all_variants`         | highs.rs:1726       | BasisStatus<->i32 round-trip     |
| `test_highs_to_basis_status_all_codes`             | highs.rs:1754       | All HiGHS i32->BasisStatus       |
| `test_basis_status_clone_copy`                     | types.rs:498        | BasisStatus derives              |
| `solve_with_basis_view_equals_solve_with_basis`    | conformance.rs:1669 | View vs allocating equivalence   |
| `raw_basis_roundtrip_equals_enum_basis`            | conformance.rs:1793 | Raw vs enum equivalence (bridge) |

### 3.5 Dead Mock Stubs (6 files x 2 methods = 12 stubs)

| File                              | Dead Methods                             |
| --------------------------------- | ---------------------------------------- |
| `cobre-sddp/src/forward.rs`       | `get_basis()`, `solve_with_basis_view()` |
| `cobre-sddp/src/backward.rs`      | `get_basis()`, `solve_with_basis_view()` |
| `cobre-sddp/src/lower_bound.rs`   | `get_basis()`, `solve_with_basis_view()` |
| `cobre-sddp/src/training.rs`      | `get_basis()`, `solve_with_basis_view()` |
| `cobre-sddp/tests/conformance.rs` | `get_basis()`, `solve_with_basis_view()` |
| `cobre-sddp/tests/integration.rs` | `get_basis()`, `solve_with_basis_view()` |

---

## 4. Items That MUST Stay

Even after full cleanup, these items survive:

| Item                                       | Why It Stays                                                               |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| `solve_view()`                             | Primary solve method, used by all production code and as delegation target |
| `solve_with_raw_basis_view()`              | Production warm-start path for training loop                               |
| `get_raw_basis()`                          | Production basis extraction for per-stage caching                          |
| `RawBasis`                                 | Per-stage basis cache storage type                                         |
| `SolutionView<'a>`                         | Zero-copy solution access (active in production)                           |
| `LpSolution`                               | Owned data in `SolverError` variant fields                                 |
| `SolutionView::to_owned()`                 | Error payload materialization                                              |
| `basis_col_i32` / `basis_row_i32` in HiGHS | Internal buffers used by `solve_with_raw_basis_view()`                     |
| `SolverStatistics`                         | Solve metrics (active in production)                                       |

---

## 5. `LpSolution` Dependency Analysis

`LpSolution` is embedded in `SolverError` variants beyond its role as a solve return type:

1. **`SolverError::Infeasible { ray: Option<LpSolution> }`** — infeasibility ray
2. **`SolverError::Unbounded { .. }`** — partial solution
3. **`SolverError::NumericalInstability { partial_solution: Option<LpSolution> }`**
4. **`SolutionView::to_owned() -> LpSolution`** — materialization for error payloads

`LpSolution` cannot be removed without restructuring the error types. It stays as the
owned counterpart of `SolutionView`, used only for error payloads and materialization.
The hot path never constructs an `LpSolution` — it works exclusively with `SolutionView`.

---

## 6. cobre-docs Spec Impact

| Spec Section                        | Content                                                        | Action Needed                                              |
| ----------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------- |
| `architecture/solver-trait.md`      | Defines `solve()`, `solve_with_basis()`, `get_basis()`         | Remove enum-based methods, document `RawBasis` as standard |
| `architecture/solver-testing.md`    | SS1.8, SS1.10 test basis warm-start with `Basis`               | Replace with `RawBasis` tests                              |
| `architecture/solver-workspaces.md` | SS1.2-SS1.5 per-stage basis cache (references generic "basis") | Specify `RawBasis` as the concrete type                    |
| `architecture/training-loop.md`     | SS4.3-SS4.4 forward/backward solve sequences                   | Update to show `solve_with_raw_basis_view()` path          |
| `math/lp-structure.md`              | References `BasisStatus` for warm-starting                     | Update to raw i32 codes                                    |
| `overview/decision-log.md`          | No `RawBasis`-related DEC entry exists                         | Add DEC-018 for enum->raw migration                        |

---

## 7. Proposed Migration Strategy

### Phase A: Adopt `RawBasis` warm-starting in SDDP training loop

**Scope**: `cobre-sddp` only (forward.rs, backward.rs, training.rs)
**Prerequisites**: None — solver infrastructure is complete.

**What changes**:

1. Create a per-stage basis cache: `Vec<RawBasis>` with `T` slots (one per stage),
   pre-allocated with `RawBasis::new(num_cols, num_rows)`.

2. Modify the forward pass solve sequence:
   - After `solve_view()` or `solve_with_raw_basis_view()`, call
     `solver.get_raw_basis(&mut basis_cache[t])` to store the basis.
   - On subsequent iterations at the same stage, call
     `solver.solve_with_raw_basis_view(&basis_cache[t])` instead of `solver.solve_view()`.
   - First iteration at each stage always cold-starts (no cached basis yet).

3. Modify the backward pass solve sequence:
   - Same pattern: check for cached basis, warm-start if available.
   - For multiple openings at the same stage: first opening cold-starts (or uses
     forward-pass basis), subsequent openings reuse the first opening's basis.

4. Handle basis invalidation:
   - On `SolverError`, clear the basis slot for that stage (don't reuse a basis that
     led to a failure).
   - When cut selection removes cuts (changing LP row count), the existing dimension
     mismatch handler in `solve_with_raw_basis_view` already handles this correctly
     (fills new rows with BASIC status, truncates extra rows).

**Expected impact**: 80-95% reduction in simplex iterations per solve (from spec estimates).
At 12M solves in a production run, this is substantial.

**After this phase**:

- `RawBasis`, `get_raw_basis()`, `solve_with_raw_basis_view()` have production callers.
- `Basis`, `BasisStatus`, and enum-based methods still have zero production callers.
- The enum path is confirmed dead.

### Phase B: Remove enum basis path + allocating wrappers

**Scope**: `cobre-solver` (types, trait, HiGHS), `cobre-sddp` (mocks), `cobre-docs` (specs)
**Prerequisites**: Phase A is complete and validated.

**What changes**:

From `cobre-solver`:

- Remove `BasisStatus` enum (types.rs:18-29)
- Remove `Basis` struct (types.rs:31-48)
- Remove `get_basis()` from `SolverInterface` trait
- Remove `solve_with_basis_view()` from `SolverInterface` trait
- Remove `solve_with_basis()` default method
- Remove `solve()` default method
- Remove `solve_with_raw_basis()` default method (callers use the view version directly)
- Remove `basis_status_to_highs()` and `highs_to_basis_status()` from HiGHS
- Remove `HighsSolver::get_basis()` implementation (42 lines)
- Remove `HighsSolver::solve_with_basis_view()` implementation (73 lines)
- Remove `BasisStatus` and `Basis` from `lib.rs` re-exports
- Remove ~11 conformance tests that exercise the enum path
- Remove ~3 unit tests for BasisStatus/conversion functions
- Convert `raw_basis_roundtrip_equals_enum_basis` bridge test to standalone raw test

From `cobre-sddp`:

- Remove `get_basis()` and `solve_with_basis_view()` from all 6 mock solvers (12 stubs)
- Remove `Basis` from mock solver imports

From `cobre-docs`:

- Update `solver-trait.md` to remove `Basis`/`BasisStatus` method signatures
- Update `solver-testing.md` SS1.8/SS1.10 to reference `RawBasis` tests
- Update `solver-workspaces.md` SS1.2 to specify `Vec<RawBasis>` as cache type
- Update `training-loop.md` SS4.3-SS4.4 to show `solve_with_raw_basis_view()` sequence
- Update `lp-structure.md` to remove `BasisStatus` references
- Add DEC-018: "Raw i32 basis path (`RawBasis`) is the standard for basis management.
  The enum-based `Basis`/`BasisStatus` path is removed."

**Risk**: If a future solver backend (CLP, CPLEX) needs `BasisStatus` for diagnostics,
a standalone decoding utility can be added (not on the trait interface). The i32 codes
are backend-specific, so a trait-level `BasisStatus` was always a leaky abstraction.

### Phase C: Naming normalization (remove "Raw"/"View" qualifiers)

**Scope**: `cobre-solver` (types, trait, HiGHS), `cobre-sddp` (mocks + production),
`cobre-docs` (specs)
**Prerequisites**: Phase B is complete (enum path removed, `RawBasis` is the only basis
type; `solve_view()` is the only solve method).

After Phase B removes the enum-based alternatives, several types and methods retain
qualifier prefixes/suffixes that were only needed to disambiguate from the (now-deleted)
original versions. These names become misleading — "Raw" implies a less safe alternative
exists, "View" implies an allocating counterpart exists.

**What stays unchanged**: `SolutionView` keeps its name because `LpSolution` permanently
exists in `SolverError` variant fields. The "View" suffix correctly communicates the
borrowing relationship between `SolutionView<'a>` (borrows solver buffers) and
`LpSolution` (owns data). This distinction remains meaningful.

**Renames**:

| Current Name                  | New Name             | Rationale                                             | Call Sites |
| ----------------------------- | -------------------- | ----------------------------------------------------- | ---------- |
| `RawBasis`                    | `Basis`              | Only basis type remaining; "Raw" implies alternative  | 21         |
| `solve_view()`                | `solve()`            | Only solve method remaining; "View" implies allocator | 12         |
| `get_raw_basis()`             | `get_basis()`        | Only basis extraction method; "Raw" misleading        | 12         |
| `solve_with_raw_basis_view()` | `solve_with_basis()` | Only warm-start method; double qualifier              | 15         |

Note: `solve_with_raw_basis()` (the allocating default wrapper) is already removed in
Phase B. The rename of `solve_with_raw_basis_view()` → `solve_with_basis()` reclaims the
cleaner name.

**Call site inventory**:

`RawBasis` (21 sites):

- `cobre-solver`: `types.rs` (definition + tests), `trait_def.rs` (trait methods),
  `highs.rs` (implementation + tests), `lib.rs` (re-export), `conformance.rs` (tests)
- `cobre-sddp`: `forward.rs`, `backward.rs`, `lower_bound.rs`, `training.rs` (mocks),
  `tests/conformance.rs`, `tests/integration.rs` (mock stubs)

`solve_view()` (12 sites):

- `cobre-solver`: `trait_def.rs` (definition + default method delegation), `highs.rs`
  (implementation), `conformance.rs` (tests)
- `cobre-sddp`: `forward.rs`, `backward.rs`, `lower_bound.rs` (production callers),
  `training.rs`, `tests/conformance.rs`, `tests/integration.rs` (mock stubs)

`get_raw_basis()` (12 sites):

- `cobre-solver`: `trait_def.rs` (definition), `highs.rs` (implementation + tests),
  `conformance.rs` (tests)
- `cobre-sddp`: 6 mock files (stubs)

`solve_with_raw_basis_view()` (15 sites):

- `cobre-solver`: `trait_def.rs` (definition + default method), `highs.rs`
  (implementation + tests), `conformance.rs` (tests)
- `cobre-sddp`: 6 mock files (stubs delegating to `solve()`)

**Mechanical execution**: Each rename is a global search-and-replace (no semantic
changes). Rust's compiler will catch any missed sites at build time. The renames should
be done in a single commit per rename (or one commit for all four) to keep the diff
atomic and reviewable.

**cobre-docs impact**: Same spec files listed in Phase B — method signatures and type
names must be updated to match. DEC-018 should note the renames.

### Phase D: Audit `LpSolution` usage in error variants

**Scope**: `cobre-solver`, potentially `cobre-sddp`
**What changes**: Determine if `ray` and `partial_solution` fields in `SolverError`
variants are ever inspected by callers. If not, simplify error variants. If they are
needed, keep `LpSolution` as the owned materialization type.

---

## 8. Quantitative Impact

### After Phase A (adoption only, no removal)

| Metric                           | Before          | After                           |
| -------------------------------- | --------------- | ------------------------------- |
| Production callers of `RawBasis` | 0               | 3 (forward, backward, training) |
| Simplex iterations per solve     | Full cold-start | ~5-20% of cold-start            |
| Dead code in cobre-solver        | ~160 lines      | ~160 lines (unchanged)          |

### After Phase B (full cleanup)

| Metric                       | Before                      | After                      |
| ---------------------------- | --------------------------- | -------------------------- |
| Public types in cobre-solver | 9                           | 7 (-BasisStatus, -Basis)   |
| Trait methods                | 13 (7 required + 6 default) | 7 (7 required + 0 default) |
| HiGHS basis management lines | ~180                        | ~80 (raw path only)        |
| Conformance tests            | 47                          | ~33 (~14 removed)          |
| Mock stubs per file          | 11 methods                  | 9 methods                  |
| Conversion functions         | 2 (23 lines)                | 0                          |
| Total lines removed          | —                           | ~300                       |

### After Phase C (naming normalization)

| Metric                      | Before (Phase B done)                  | After                         |
| --------------------------- | -------------------------------------- | ----------------------------- |
| Types with misleading names | 1 (`RawBasis`)                         | 0                             |
| Methods with qualifiers     | 3 (`solve_view`, `get_raw_basis`, ...) | 0                             |
| Total renames               | —                                      | 4 (60 call sites)             |
| API surface                 | Correct but confusing naming           | Clean, self-documenting names |

---

## 9. Recommendation

**Phase A should be the next implementation target** after the current branch is merged.
It is the payoff for the entire solver-hot-path-refactor: the training loop gains
warm-starting with zero-copy basis management, which is the single largest performance
win available without algorithmic changes.

**Phase B should follow immediately**, while the context is fresh. Once Phase A validates
that `RawBasis` works correctly in the training loop, there is no reason to keep the
enum-based path.

**Phase C should be bundled with Phase B** or executed as the very next commit after it.
The renames are purely mechanical (search-and-replace, no semantic changes) and the
compiler catches any missed sites. Doing the renames while the Phase B changes are fresh
avoids a future "why is this called RawBasis when there's no other Basis?" confusion.
The combined B+C effort is still small (~300 lines removed + 60 call sites renamed).

**Phase D can be deferred** indefinitely — `LpSolution` is small and the error variants
may gain callers in simulation/output phases.

**DEC-018** should be created in cobre-docs before starting Phase A to formalize the
decision: "Raw i32 basis path is the standard for basis management. After enum path
removal, types and methods are renamed to their canonical forms (e.g., `RawBasis` →
`Basis`, `solve_view()` → `solve()`)."

---

## 10. Files Affected Summary

### Phase A (adoption)

- `crates/cobre-sddp/src/forward.rs` — add basis cache parameter, warm-start logic
- `crates/cobre-sddp/src/backward.rs` — add basis cache parameter, warm-start logic
- `crates/cobre-sddp/src/training.rs` — allocate per-stage `Vec<RawBasis>`, pass to passes

### Phase B (cleanup)

Production code:

- `crates/cobre-solver/src/types.rs` — remove BasisStatus, Basis, their tests
- `crates/cobre-solver/src/trait_def.rs` — remove 6 trait methods + NoopSolver stubs
- `crates/cobre-solver/src/highs.rs` — remove 2 impl methods + 2 conversion functions + 3 tests
- `crates/cobre-solver/src/lib.rs` — remove BasisStatus, Basis from re-exports
- `crates/cobre-solver/tests/conformance.rs` — remove ~14 enum-path tests

Mock stubs:

- `crates/cobre-sddp/src/forward.rs` — remove 2 stubs
- `crates/cobre-sddp/src/backward.rs` — remove 2 stubs
- `crates/cobre-sddp/src/lower_bound.rs` — remove 2 stubs
- `crates/cobre-sddp/src/training.rs` — remove 2 stubs
- `crates/cobre-sddp/tests/conformance.rs` — remove 2 stubs
- `crates/cobre-sddp/tests/integration.rs` — remove 2 stubs

Specs (cobre-docs):

- `src/specs/architecture/solver-trait.md` — update method inventory
- `src/specs/architecture/solver-testing.md` — update test specifications
- `src/specs/architecture/solver-workspaces.md` — specify `RawBasis` as cache type
- `src/specs/architecture/training-loop.md` — update solve sequences
- `src/specs/math/lp-structure.md` — update basis management section
- `src/specs/overview/decision-log.md` — add DEC-018

### Phase C (naming normalization)

Same files as Phase B — the renames are applied to whatever remains after cleanup:

- `crates/cobre-solver/src/types.rs` — rename `RawBasis` → `Basis`
- `crates/cobre-solver/src/trait_def.rs` — rename 3 methods + NoopSolver stubs
- `crates/cobre-solver/src/highs.rs` — rename method implementations + tests
- `crates/cobre-solver/src/lib.rs` — rename re-export
- `crates/cobre-solver/tests/conformance.rs` — rename in test bodies
- `crates/cobre-sddp/src/forward.rs` — rename in production calls + stubs
- `crates/cobre-sddp/src/backward.rs` — rename in production calls + stubs
- `crates/cobre-sddp/src/lower_bound.rs` — rename in production calls + stubs
- `crates/cobre-sddp/src/training.rs` — rename in production calls + stubs
- `crates/cobre-sddp/tests/conformance.rs` — rename in stubs
- `crates/cobre-sddp/tests/integration.rs` — rename in stubs
- cobre-docs specs: same 6 files as Phase B (update to final names)
