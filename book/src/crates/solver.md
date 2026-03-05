# cobre-solver

<span class="status-experimental">experimental</span>

`cobre-solver` is the LP solver abstraction layer for the Cobre ecosystem. It
defines a backend-agnostic interface for constructing, solving, and querying
linear programs, with a production-grade [HiGHS](https://highs.dev) backend as
the default implementation.

The crate has no dependency on any other Cobre crate. It is infrastructure
that optimization algorithm crates consume through a generic type parameter,
not a shared registry or runtime-selected component. Every solver method call
compiles directly to the concrete backend implementation — there is no virtual
dispatch overhead on the hot path where iterative LP solving occurs.

## Module overview

| Module      | Purpose                                                                                                                    |
| ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| `ffi`       | Raw `unsafe` FFI bindings to the `cobre_highs_*` C wrapper functions                                                       |
| `types`     | Canonical data types: `StageTemplate`, `RowBatch`, `Basis`, `LpSolution`, `SolverError`, `SolverStatistics`, `BasisStatus` |
| `trait_def` | `SolverInterface` trait definition with all 10 method contracts                                                            |
| `highs`     | `HighsSolver` — the HiGHS backend implementing `SolverInterface`                                                           |
| (root)      | Re-exports: `SolverInterface`, `HighsSolver`, and all public types                                                         |

The `ffi` and `highs` modules are compiled only when the `highs` feature is
enabled (the default). The `trait_def` and `types` modules are always compiled,
making it possible to write algorithm code against `SolverInterface` without
depending on any particular backend.

## Architecture

### Compile-time monomorphization (DEC-002)

`SolverInterface` is resolved as a **generic type parameter at compile time**,
not as `Box<dyn SolverInterface>` or any other form of dynamic dispatch. An
optimization algorithm crate parameterizes its entry point as:

```rust
fn run<S: SolverInterface>(solver_factory: impl Fn() -> S, ...) { ... }
```

The compiler generates one concrete implementation per backend. The HiGHS
backend is the only active backend in a standard build; the binary contains
no solver-selection branch. This is specified in DEC-002 and implemented in
ADR-003.

### Custom FFI — not `highs-sys`

`cobre-solver` does not use any third-party `highs-sys` crate. Instead it
ships a thin C wrapper (`csrc/highs_wrapper.c`) that exposes the 20-odd HiGHS
C API functions needed by the backend as `cobre_highs_*` symbols. This approach:

- Controls exactly which HiGHS API surface is exposed.
- Allows the wrapper to enforce Cobre-specific invariants before delegating to
  the underlying `Highs_*` calls.
- Avoids a build-time dependency on any external Rust crate for FFI bindings.

The `ffi` module declares `extern "C"` signatures for each `cobre_highs_*`
function. All FFI calls are `unsafe`; safe wrappers live in `highs.rs`.

### Vendored HiGHS build

HiGHS is compiled from source at build time via the `cmake` crate. The source
lives in `vendor/HiGHS/` as a git submodule. The build script
(`crates/cobre-solver/build.rs`) invokes cmake with a fixed Release
configuration and links the resulting static library. HiGHS is always built in
Release mode regardless of the Cargo profile, because a debug HiGHS build is
roughly 10x slower and would produce misleading performance results.

### Per-crate `unsafe` override

The workspace lint configuration forbids `unsafe` code at the workspace level.
`cobre-solver` overrides this lint to `allow` in its own `Cargo.toml` because
the HiGHS FFI layer genuinely requires `unsafe` blocks. All other workspace
lints (`missing_docs`, `unwrap_used`, clippy pedantic) remain active. Every
`unsafe` block carries a `// SAFETY:` comment explaining the invariants that
justify it.

## `SolverInterface` trait

```rust
pub trait SolverInterface: Send { ... }
```

The trait defines 10 methods that together constitute the full LP lifecycle for
one solver instance. Implementations must satisfy the pre- and post-condition
contracts documented in each method's rustdoc. See the
[`trait_def` rustdoc](../api/cobre_solver/trait.SolverInterface.html) for the
complete contracts.

### Method summary

| Method             | `&self` / `&mut self` | Returns                           | Description                                                                 |
| ------------------ | --------------------- | --------------------------------- | --------------------------------------------------------------------------- |
| `load_model`       | `&mut self`           | `()`                              | Bulk-loads a structural LP from a `StageTemplate`; replaces any prior model |
| `add_rows`         | `&mut self`           | `()`                              | Appends a `RowBatch` of constraint rows to the dynamic region               |
| `set_row_bounds`   | `&mut self`           | `()`                              | Updates row lower/upper bounds at indexed positions                         |
| `set_col_bounds`   | `&mut self`           | `()`                              | Updates column lower/upper bounds at indexed positions                      |
| `solve`            | `&mut self`           | `Result<LpSolution, SolverError>` | Solves the current LP; encapsulates internal retry logic                    |
| `solve_with_basis` | `&mut self`           | `Result<LpSolution, SolverError>` | Sets a cached basis, then solves (warm-start path)                          |
| `reset`            | `&mut self`           | `()`                              | Clears solver state for error recovery or model switch                      |
| `get_basis`        | `&mut self`           | `Basis`                           | Extracts the current simplex basis after a successful solve                 |
| `statistics`       | `&self`               | `SolverStatistics`                | Returns accumulated monotonic solve counters                                |
| `name`             | `&self`               | `&'static str`                    | Returns a static string identifying the backend                             |

### Mutability convention

Methods that mutate solver state — loading a model, adding constraints, patching
bounds, solving, resetting, and extracting a basis — take `&mut self`. `get_basis`
requires `&mut self` because it writes to internal scratch buffers during
extraction. Methods that only read accumulated state (`statistics`, `name`) take
`&self`. This
convention makes data-race hazards visible at the type level: the borrow checker
prevents concurrent mutation without locks.

### Error recovery contract

When `solve` or `solve_with_basis` returns `Err`, the solver's internal state is
unspecified. The **caller** is responsible for calling `reset()` before reusing
the instance. Failing to reset after a terminal error may produce incorrect
results or panics on the next `load_model` call.

### Thread safety

`SolverInterface` requires `Send` but not `Sync`. `Send` allows a solver
instance to be transferred to a worker thread at startup. The absence of `Sync`
prevents concurrent access from multiple threads, which matches the reality of
C-library solver handles: they maintain mutable factorization workspaces that
are not thread-safe. Each worker thread owns exactly one solver instance.

## Public types

### `StageTemplate`

Pre-assembled structural LP for one stage, in CSC (column-major) form. Built
once at initialization from resolved internal structures and shared read-only
across all threads. Passed to `load_model` to bulk-load the LP. Fields include
the CSC matrix arrays (`col_starts`, `row_indices`, `values`), bounds, objective
coefficients, and layout metadata (`n_state`, `n_transfer`, `n_dual_relevant`,
`n_hydro`, `max_par_order`) used by the calling algorithm for state transfer and
cut extraction. See the [`StageTemplate` rustdoc](../api/cobre_solver/struct.StageTemplate.html).

### `RowBatch`

Batch of constraint rows for addition to a loaded LP, in CSR (row-major) form.
Assembled from an active constraint pool before each LP rebuild and passed to
`add_rows` in a single call. Appended rows occupy the dynamic constraint region
of the LP matrix. See the [`RowBatch` rustdoc](../api/cobre_solver/struct.RowBatch.html).

### `Basis`

Simplex basis for warm-starting subsequent solves. Contains one `BasisStatus`
per column and one per row, stored in the original (unpresolved) problem space
for portability across solver versions and presolve strategies. When the LP gains
new dynamic constraint rows after a basis was saved, `solve_with_basis` handles
the dimension mismatch by initializing the new rows as `BasisStatus::Basic`. See
the [`Basis` rustdoc](../api/cobre_solver/struct.Basis.html).

### `BasisStatus`

Enum with five variants: `AtLower`, `Basic`, `AtUpper`, `Free`, `Fixed`. Maps
to solver-specific integer codes internally; the public API always uses the
canonical Cobre representation. See the [`BasisStatus`
rustdoc](../api/cobre_solver/enum.BasisStatus.html).

### `LpSolution`

Complete solution from a successful LP solve: `objective` (f64, minimization
sense), `primal` (Vec of column values), `dual` (Vec of row dual multipliers,
normalized to the canonical sign convention), `reduced_costs`, `iterations`,
and `solve_time_seconds`. Dual values are normalized before the struct is
returned — HiGHS row duals are already in the canonical convention and require
no negation. See the [`LpSolution` rustdoc](../api/cobre_solver/struct.LpSolution.html).

### `SolverError`

Terminal LP solve error returned after all retry attempts are exhausted. Six
variants correspond to six failure categories:

| Variant               | Hard stop? | Carries partial solution?  |
| --------------------- | ---------- | -------------------------- |
| `Infeasible`          | Yes        | No (infeasibility ray)     |
| `Unbounded`           | Yes        | No (direction certificate) |
| `NumericalDifficulty` | No         | Optional                   |
| `TimeLimitExceeded`   | No         | Optional                   |
| `IterationLimit`      | No         | Optional                   |
| `InternalError`       | Yes        | No                         |

`Infeasible`, `Unbounded`, and `InternalError` indicate data or modeling errors
and require a hard stop. `NumericalDifficulty`, `TimeLimitExceeded`, and
`IterationLimit` may carry a partial solution that the calling algorithm can
inspect before deciding how to proceed. See the [`SolverError`
rustdoc](../api/cobre_solver/enum.SolverError.html).

### `SolverStatistics`

Accumulated solve metrics for one solver instance: `solve_count`,
`success_count`, `failure_count`, `total_iterations`, `retry_count`,
`total_solve_time_seconds`, and `basis_rejections`. All counters grow
monotonically from zero. `reset()` does not zero them — statistics persist for
the lifetime of the solver instance and are aggregated across threads after
iterative solving completes. See the [`SolverStatistics`
rustdoc](../api/cobre_solver/struct.SolverStatistics.html).

## HiGHS backend (`HighsSolver`)

### Construction

```rust
pub fn new() -> Result<Self, SolverError>
```

`HighsSolver::new()` allocates a HiGHS handle via `cobre_highs_create()` and
applies seven performance-tuned default options before returning:

| Option                         | Value       | Rationale                                          |
| ------------------------------ | ----------- | -------------------------------------------------- |
| `solver`                       | `"simplex"` | Simplex is faster than IPM for warm-started LPs    |
| `simplex_strategy`             | `4`         | Dual simplex; performs well on LP sequences        |
| `presolve`                     | `"off"`     | Avoid presolve overhead on repeated small LPs      |
| `parallel`                     | `"off"`     | Each thread owns one solver; no internal threads   |
| `output_flag`                  | `false`     | Suppress HiGHS console output                      |
| `primal_feasibility_tolerance` | `1e-7`      | Tighter than HiGHS default for numerical stability |
| `dual_feasibility_tolerance`   | `1e-7`      | Same                                               |

If HiGHS handle creation or any option call fails, the handle is destroyed
before returning `Err(SolverError::InternalError { .. })`.

### 5-level retry escalation

When HiGHS returns `SOLVE_ERROR` or `UNKNOWN` (not a definitive terminal
status), `HighsSolver::solve` escalates through five retry levels before giving
up:

| Level | Action                                                       |
| ----- | ------------------------------------------------------------ |
| 0     | Clear the cached basis and factorization (`clear_solver`)    |
| 1     | Enable presolve (`presolve = "on"`)                          |
| 2     | Switch to primal simplex (`simplex_strategy = 1`)            |
| 3     | Relax feasibility tolerances (`primal` and `dual` to `1e-6`) |
| 4     | Switch to interior point method (`solver = "ipm"`)           |

The first level that returns `OPTIMAL` exits the loop. If a definitive terminal
status (`INFEASIBLE`, `UNBOUNDED`, `TIME_LIMIT`, `ITERATION_LIMIT`) is reached
during a retry level, the loop exits immediately with the corresponding
`SolverError` variant. If all five levels are exhausted without a result, the
method returns `SolverError::NumericalDifficulty`. Default settings are restored
unconditionally after the retry loop, regardless of outcome, so subsequent calls
see the standard configuration.

The retry sequence is entirely internal — the caller of `solve` never sees
intermediate failures, only the final `Ok(LpSolution)` or `Err(SolverError)`.

### Dual normalization

HiGHS row duals are already in the canonical Cobre sign convention: a positive
dual on a `<=` constraint means increasing the RHS increases the objective.
`HighsSolver::extract_solution` copies `row_dual` directly into `LpSolution.dual`
without negation. The `col_dual` from HiGHS is the reduced cost vector and is
placed in `LpSolution.reduced_costs`.

### Warm-start basis management

`solve_with_basis` translates the canonical `Basis` into HiGHS-specific `i32`
status codes using pre-allocated buffers (`basis_col_i32`, `basis_row_i32`)
sized at `load_model` time and grown (but never shrunk) by `add_rows`. When the
saved basis has fewer rows than the current LP (because new dynamic constraint
rows were added since the basis was extracted), the extra rows are initialized
as `HIGHS_BASIS_STATUS_BASIC`. When the saved basis has more rows than the
current LP (a cut was removed), the extra saved entries are silently truncated.
If HiGHS rejects the basis (returns `HIGHS_STATUS_ERROR` from `Highs_setBasis`),
the method falls back to a cold-start solve and increments
`SolverStatistics.basis_rejections`. After setting the basis, `solve_with_basis`
delegates to `solve()`, which handles the retry escalation sequence.

## SoA bound patching (DEC-019)

The `set_row_bounds` and `set_col_bounds` methods take three separate slices:

```rust
fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);
fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]);
```

This is a Structure of Arrays (SoA) signature. The alternative — a single slice
of `(usize, f64, f64)` tuples (Array of Structures, AoS) — would require the
caller to convert from its natural SoA representation before the call, and the
HiGHS C API (`Highs_changeRowsBoundsBySet`) would then expect SoA again,
producing a double conversion on the hottest solver path.

DEC-019 documents the rationale: the calling algorithm naturally holds separate
index, lower-bound, and upper-bound arrays; the C API expects separate arrays;
so the trait signature matches both, eliminating any intermediate conversion.
The performance impact is meaningful because bound patching happens at every
scenario realization, which occurs on the innermost loop of iterative LP solving.

## Usage example

The following shows the complete LP rebuild sequence for one stage: load the
structural model, append active constraint rows, patch scenario-specific row
bounds, solve, and extract the basis for the next iteration.

```rust,no_run
use cobre_solver::{
    Basis, BasisStatus, HighsSolver, LpSolution, RowBatch, SolverError,
    SolverInterface, StageTemplate,
};

fn solve_stage(
    solver: &mut HighsSolver,
    template: &StageTemplate,
    cuts: &RowBatch,
    row_indices: &[usize],
    lower: &[f64],
    upper: &[f64],
    cached_basis: Option<&Basis>,
) -> Result<(LpSolution, Basis), SolverError> {
    // Step 1: load structural LP (replaces any prior model).
    solver.load_model(template);

    // Step 2: append active constraint rows.
    solver.add_rows(cuts);

    // Step 3: patch row bounds for this scenario realization.
    solver.set_row_bounds(row_indices, lower, upper);

    // Step 4: solve, optionally warm-starting from a cached basis.
    let solution = match cached_basis {
        Some(basis) => solver.solve_with_basis(basis)?,
        None => solver.solve()?,
    };

    // Step 5: extract basis for warm-starting the next iteration.
    let basis = solver.get_basis();

    Ok((solution, basis))
}

fn main() -> Result<(), SolverError> {
    let mut solver = HighsSolver::new()?;
    assert_eq!(solver.name(), "HiGHS");

    // Print cumulative statistics after a run.
    let stats = solver.statistics();
    println!(
        "solves={} successes={} retries={}",
        stats.solve_count, stats.success_count, stats.retry_count
    );

    Ok(())
}
```

## Build requirements

### Git submodule

HiGHS is vendored as a git submodule at `vendor/HiGHS/`. Before building
`cobre-solver` for the first time (or after a fresh clone), initialize the
submodule:

```
git submodule update --init --recursive
```

The build script checks for `vendor/HiGHS/CMakeLists.txt` and panics with a
clear error message if the submodule is not initialized.

### System dependencies

| Dependency   | Minimum version | Notes                                           |
| ------------ | --------------- | ----------------------------------------------- |
| cmake        | 3.15            | Required by the HiGHS build system              |
| C compiler   | C11             | gcc or clang; HiGHS and the C wrapper are C/C++ |
| C++ compiler | C++17           | Required by HiGHS internals                     |
| zlib         | any             | Required by HiGHS MPS file reader               |

### Feature flags

| Feature | Default | Description                                    |
| ------- | ------- | ---------------------------------------------- |
| `highs` | yes     | Enables the HiGHS backend and the build script |

Without the `highs` feature, only `SolverInterface`, the type definitions, and
the `ffi` module stubs are compiled. The `HighsSolver` struct is not available.
Additional solver backends (CLP, commercial solvers) are planned behind their
own feature flags but are not yet implemented.

## Testing

### Running the test suite

```
cargo test -p cobre-solver --features highs
```

This requires cmake, a C/C++ compiler, and an initialized `vendor/HiGHS/`
submodule (see [Build requirements](#build-requirements)).

### Conformance suite (`tests/conformance.rs`)

The integration test file `tests/conformance.rs` implements the backend-agnostic
conformance contract from the Solver Interface Testing spec. It verifies the
`SolverInterface` contract using only the public API against the `HighsSolver`
concrete type. The fixture LP is a 3-variable, 2-constraint minimization problem
(the SS1.1 fixture) with known optimal solution `(x0=6, x1=0, x2=2, obj=100.0)`.

The conformance suite covers:

- `load_model` loads a structural LP and produces the expected objective and
  primal values on `solve`.
- `load_model` fully replaces a previous model when called a second time.
- `add_rows` appends constraint rows without altering structural rows.
- `set_row_bounds` patches bounds and the re-solve reflects the new bounds.
- `solve_with_basis` warm-starts successfully and returns the correct optimal
  solution.
- `get_basis` returns a basis with the correct column and row count after a
  successful solve.
- `statistics` counters increment correctly across solve calls.
- `reset` clears model state, allowing `load_model` to be called again cleanly.

### Unit tests

`src/highs.rs` and `src/types.rs` carry `#[cfg(test)]` unit tests covering
individual methods in isolation, including the `NoopSolver` in `src/trait_def.rs`
that verifies `SolverInterface` compiles as a generic bound and satisfies the
`Send` requirement.
