# Accumulated Learnings: Epics 01-03 (i32 CSC Indices + SolutionView + RawBasis)

**Plan**: solver-hot-path-refactor
**Last updated**: 2026-03-08
**Epics covered**: epic-01-i32-csc-indices, epic-02-solution-view, epic-03-raw-basis

---

## Patterns and Conventions (inherited from epics 01-02)

- `Vec<i32>` literals use `_i32` suffix on the first element to drive type inference
- Per-element `usize as i32` casts require both `clippy::cast_possible_truncation` AND
  `clippy::cast_possible_wrap` allows together
- SAFETY comments in `highs.rs` must name the actual owner of each pointer argument
- `solve_view(&mut self)` reads FFI scalars BEFORE `extract_solution_view` shared borrow
- Mock solvers use `buf_primal/buf_dual/buf_reduced_costs: Vec<f64>` buffer fields

## Epic 03 Patterns

- `RawBasis` stores `Vec<i32>` (solver-native status codes); no `PartialEq` (same as
  `SolutionView`); `Debug + Clone` only
- `get_raw_basis` writes directly to caller-owned `out.col_status`/`out.row_status`
  buffers via FFI, bypassing internal `basis_col_i32`/`basis_row_i32`. Resizes `out`
  to match LP dimensions before the FFI call.
- `solve_with_raw_basis_view` uses `copy_from_slice` (memcpy) into internal
  `basis_col_i32`/`basis_row_i32` buffers, then delegates to `solve_view()`
- Row dimension mismatch handling: `copy_len = min(basis_rows, lp_rows)`, fill
  remaining with `HIGHS_BASIS_STATUS_BASIC` via `.fill()` (not explicit loop)
- `solve_with_basis_view` and `solve_with_raw_basis_view` are independent methods;
  no shared logic extracted between them (learned from Epic 02 `extract_solution` issue)
- `solve_with_raw_basis` is a default method: `solve_with_raw_basis_view().map(|v| v.to_owned())`

## Epic 03 Spec Quality Lessons

- When a trait method is added as required (not default), ALL implementors must compile.
  Ticket-008 (define type + trait methods) forced ticket-009 (implement in HiGHS) work
  to happen simultaneously. For future plans: either use default methods with `todo!()`
  stubs, or include all implementor files in the same ticket's scope.
- The initial `get_raw_basis` implementation by ticket-008 specialist used internal
  buffers + copy instead of direct FFI write — the ticket-009 spec caught this and
  required the fix. Lesson: spec the exact data flow path, not just the interface.
- Guardian caught a missing `basis_rejections == 0` assertion (C4). Test completeness
  must be verified per-assertion, not per-test-function.

## Mock Solver Maintenance

- When new required trait methods are added to `SolverInterface`, all 6 mock solvers in
  cobre-sddp must be updated: `forward.rs`, `backward.rs`, `lower_bound.rs`,
  `training.rs`, `tests/conformance.rs`, `tests/integration.rs`
- Standard mock stubs: `get_raw_basis(&mut self, _out: &mut RawBasis) {}` (no-op) and
  `solve_with_raw_basis_view` delegates to `self.solve_view()`

## Infrastructure Crate Genericity

- `grep -riE 'sddp' crates/cobre-solver/src/` returns 0 matches post-epic-03
- All three new types (`SolutionView`, `RawBasis`, trait methods) use generic names
