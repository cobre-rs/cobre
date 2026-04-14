# cobre-solver

<span class="status-alpha">alpha</span>

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

| Module      | Purpose                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| `ffi`       | Raw `unsafe` FFI bindings to the `cobre_highs_*` C wrapper functions                                                        |
| `types`     | Canonical data types: `StageTemplate`, `RowBatch`, `Basis`, `LpSolution`, `SolutionView`, `SolverError`, `SolverStatistics` |
| `trait_def` | `SolverInterface` trait definition with all 10 method contracts                                                             |
| `highs`     | `HighsSolver` — the HiGHS backend implementing `SolverInterface`                                                            |
| (root)      | Re-exports: `SolverInterface`, `HighsSolver`, and all public types                                                          |

The `ffi` and `highs` modules are compiled only when the `highs` feature is
enabled (the default). The `trait_def` and `types` modules are always compiled,
making it possible to write algorithm code against `SolverInterface` without
depending on any particular backend.

## Architecture

### Compile-time monomorphization

`SolverInterface` is resolved as a **generic type parameter at compile time**,
not as `Box<dyn SolverInterface>` or any other form of dynamic dispatch. An
optimization algorithm crate parameterizes its entry point as:

```rust
fn run<S: SolverInterface>(solver_factory: impl Fn() -> S, ...) { ... }
```

The compiler generates one concrete implementation per backend. The HiGHS
backend is the only active backend in a standard build; the binary contains
no solver-selection branch. This pattern uses compile-time monomorphization.

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
lives in `crates/cobre-solver/vendor/HiGHS/` as a git submodule. The build script
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
[`trait_def` rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/trait.SolverInterface.html) for the
complete contracts.

### Method summary

| Method             | `&self` / `&mut self` | Returns                                 | Description                                                                 |
| ------------------ | --------------------- | --------------------------------------- | --------------------------------------------------------------------------- |
| `load_model`       | `&mut self`           | `()`                                    | Bulk-loads a structural LP from a `StageTemplate`; replaces any prior model |
| `add_rows`         | `&mut self`           | `()`                                    | Appends a `RowBatch` of constraint rows to the dynamic region               |
| `set_row_bounds`   | `&mut self`           | `()`                                    | Updates row lower/upper bounds at indexed positions                         |
| `set_col_bounds`   | `&mut self`           | `()`                                    | Updates column lower/upper bounds at indexed positions                      |
| `solve`            | `&mut self`           | `Result<SolutionView<'_>, SolverError>` | Solves the current LP; encapsulates internal retry logic                    |
| `solve_with_basis` | `&mut self`           | `Result<SolutionView<'_>, SolverError>` | Sets a cached basis, then solves (warm-start path)                          |
| `reset`            | `&mut self`           | `()`                                    | Clears solver state for error recovery or model switch                      |
| `get_basis`        | `&mut self`           | `()`                                    | Writes basis status codes into a caller-owned `&mut Basis`                  |
| `statistics`           | `&self`               | `SolverStatistics`                      | Returns accumulated monotonic solve counters                                |
| `name`                 | `&self`               | `&'static str`                          | Returns a static string identifying the backend                             |
| `solver_name_version`  | `&self`               | `String`                                | Returns `"name vX.Y.Z"` (e.g. `"HiGHS v1.8.1"`) for metadata output       |

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
cut extraction. See the [`StageTemplate` rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/struct.StageTemplate.html).

### `RowBatch`

Batch of constraint rows for addition to a loaded LP, in CSR (row-major) form.
Assembled from an active constraint pool before each LP rebuild and passed to
`add_rows` in a single call. Appended rows occupy the dynamic constraint region
of the LP matrix. See the [`RowBatch` rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/struct.RowBatch.html).

### `Basis`

Raw simplex basis stored as solver-native `i32` status codes — one per column
and one per row. The codes are opaque to the calling algorithm; they are
extracted from one solve via `get_basis` and passed back to the next via
`solve_with_basis` for warm-starting. Stored in the original (unpresolved)
problem space for portability across solver versions and presolve strategies.
When the LP gains new dynamic constraint rows after a basis was saved,
`solve_with_basis` handles the dimension mismatch by filling new row slots
with the solver-native "Basic" code. See the
[`Basis` rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/struct.Basis.html).

### `SolutionView<'a>`

Zero-copy borrowed view over solver-internal buffers, returned by `solve` and
`solve_with_basis`. Provides `objective()`, `primal()`, `dual()`,
`reduced_costs()`, `iterations()`, and `solve_time_seconds()` as slice
references into the solver's internal arrays. The view borrows the solver and
is valid until the next `&mut self` call. Call `to_owned()` to copy the data
into an `LpSolution` when the solution must outlive the borrow. See the
[`SolutionView` rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/struct.SolutionView.html).

### `LpSolution`

Owned solution produced by `SolutionView::to_owned()`: `objective` (f64,
minimization sense), `primal` (Vec of column values), `dual` (Vec of row dual
multipliers, normalized to the canonical sign convention), `reduced_costs`,
`iterations`, and `solve_time_seconds`. Dual values are normalized before the
struct is returned — HiGHS row duals are already in the canonical convention
and require no negation. See the [`LpSolution`
rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/struct.LpSolution.html).

### `SolverError`

Terminal LP solve error returned after all retry attempts are exhausted. Six
variants correspond to six failure categories:

| Variant               | Hard stop? | Diagnostic |
| --------------------- | ---------- | ---------- |
| `Infeasible`          | Yes        | No         |
| `Unbounded`           | Yes        | No         |
| `NumericalDifficulty` | No         | Yes        |
| `TimeLimitExceeded`   | No         | Yes        |
| `IterationLimit`      | No         | Yes        |
| `InternalError`       | Yes        | No         |

`Infeasible` and `Unbounded` are unit variants (no fields). `NumericalDifficulty`
carries a `message`, `TimeLimitExceeded` carries `elapsed_seconds`, and
`IterationLimit` carries `iterations`. `InternalError` carries `message` and
an optional `error_code`. See the [`SolverError`
rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/enum.SolverError.html).

### `SolverStatistics`

Accumulated solve metrics for one solver instance. All counters grow
monotonically from zero. `reset()` does not zero them — statistics persist for
the lifetime of the solver instance and are aggregated across threads after
iterative solving completes.

| Field                          | Type       | Description                                                                                         |
| ------------------------------ | ---------- | --------------------------------------------------------------------------------------------------- |
| `solve_count`                  | `u64`      | Total `solve` and `solve_with_basis` calls.                                                         |
| `success_count`                | `u64`      | Solves that returned optimal.                                                                       |
| `failure_count`                | `u64`      | Solves that returned terminal error after retries.                                                   |
| `total_iterations`             | `u64`      | Total simplex iterations across all solves.                                                          |
| `retry_count`                  | `u64`      | Total retry attempts across all solves.                                                              |
| `total_solve_time_seconds`     | `f64`      | Cumulative wall-clock solve time.                                                                    |
| `basis_rejections`             | `u64`      | Times `solve_with_basis` fell back to cold-start.                                                    |
| `first_try_successes`          | `u64`      | Solves optimal on first attempt. Enables: `first_try_rate = first_try_successes / solve_count`.      |
| `basis_offered`                | `u64`      | Total `solve_with_basis` calls. Enables: `basis_hit_rate = 1 - basis_rejections / basis_offered`.    |
| `load_model_count`             | `u64`      | Total `load_model` calls.                                                                            |
| `add_rows_count`               | `u64`      | Total `add_rows` calls.                                                                              |
| `total_load_model_time_seconds`| `f64`      | Cumulative time in `load_model`.                                                                     |
| `total_add_rows_time_seconds`  | `f64`      | Cumulative time in `add_rows`.                                                                       |
| `total_set_bounds_time_seconds`| `f64`      | Cumulative time in `set_row_bounds` / `set_col_bounds`.                                              |
| `total_basis_set_time_seconds` | `f64`      | Cumulative time in basis installation (`solve_with_basis`).                                          |
| `basis_padding_tight`          | `u64`      | Cut rows assigned `NONBASIC_LOWER` by basis-aware padding (Strategy S3). Set by the calling algorithm. |
| `basis_padding_slack`          | `u64`      | Cut rows assigned `BASIC` by basis-aware padding. Set by the calling algorithm.                      |
| `retry_level_histogram`        | `Vec<u64>` | Per-level retry success counts (length 12 for HiGHS). Sum = `success_count - first_try_successes`.  |

See the [`SolverStatistics`
rustdoc](https://docs.rs/cobre-solver/latest/cobre_solver/struct.SolverStatistics.html).

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

### 12-level retry escalation

When HiGHS returns `SOLVE_ERROR` or `UNKNOWN` (not a definitive terminal
status), `HighsSolver::solve` escalates through twelve retry levels organised
in two phases, with wall-clock budgets per level and an overall budget:

**Phase 1 (levels 0--4): core cumulative sequence**

| Level | Action                                                       |
| ----- | ------------------------------------------------------------ |
| 0     | Clear the cached basis and factorization (`clear_solver`)    |
| 1     | Enable presolve (`presolve = "on"`)                          |
| 2     | Switch to dual simplex (`simplex_strategy = 1`)              |
| 3     | Relax feasibility tolerances (`primal` and `dual` to `1e-6`) |
| 4     | Switch to interior point method (`solver = "ipm"`)           |

**Phase 2 (levels 5--11): extended strategies with scaling**

Each level starts from restored defaults with presolve and iteration limits,
then applies level-specific scaling, tolerance, and solver options.

| Level | Action                                                                      |
| ----- | --------------------------------------------------------------------------- |
| 5     | Presolve + scale strategy 3                                                 |
| 6     | Presolve + primal simplex + scale strategy 4                                |
| 7     | Presolve + scale strategy 3 + relaxed tolerances (`1e-6`)                   |
| 8     | Presolve + objective scale (`-10`)                                          |
| 9     | Presolve + primal simplex + objective scale (`-10`) + bound scale (`-5`)    |
| 10    | Presolve + objective scale (`-13`) + bound scale (`-8`) + relaxed tol       |
| 11    | Presolve + IPM + objective scale (`-10`) + bound scale (`-5`) + relaxed tol |

The first level that returns `OPTIMAL` exits the loop. If a definitive terminal
status (`INFEASIBLE`, `UNBOUNDED`, `TIME_LIMIT`, `ITERATION_LIMIT`) is reached
during a retry level, the loop exits immediately with the corresponding
`SolverError` variant. If all twelve levels are exhausted or the overall
wall-clock budget expires, the method returns
`SolverError::NumericalDifficulty`. Default settings are restored
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

`solve_with_basis` loads the `Basis` status codes directly into HiGHS via
`Highs_setBasis`. When the saved basis has fewer rows than the current LP
(because new dynamic constraint rows were added since the basis was extracted),
the extra rows are filled with the HiGHS "Basic" status code (1). When the
saved basis has more rows than the current LP, the extra entries are truncated.
If HiGHS rejects the basis (returns `HIGHS_STATUS_ERROR` from `Highs_setBasis`),
the method falls back to a cold-start solve and increments
`SolverStatistics.basis_rejections`. After setting the basis, `solve_with_basis`
delegates to `solve()`, which handles the retry escalation sequence.

## SoA bound patching

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

The calling algorithm naturally holds separate index, lower-bound, and
upper-bound arrays; the C API expects separate arrays; so the trait signature
matches both, eliminating any intermediate conversion. The performance impact
is meaningful because bound patching happens at every scenario realization,
which occurs on the innermost loop of iterative LP solving.

## Usage example

The following shows the complete LP rebuild sequence for one stage: load the
structural model, append active constraint rows, patch scenario-specific row
bounds, solve, and extract the basis for the next iteration.

```rust,no_run
use cobre_solver::{
    Basis, HighsSolver, LpSolution, RowBatch, SolverError,
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
    basis_buf: &mut Basis,
) -> Result<LpSolution, SolverError> {
    // Step 1: load structural LP (replaces any prior model).
    solver.load_model(template);

    // Step 2: append active constraint rows.
    solver.add_rows(cuts);

    // Step 3: patch row bounds for this scenario realization.
    solver.set_row_bounds(row_indices, lower, upper);

    // Step 4: solve, optionally warm-starting from a cached basis.
    let view = match cached_basis {
        Some(basis) => solver.solve_with_basis(basis)?,
        None => solver.solve()?,
    };

    // Step 5: copy the zero-copy view into an owned solution.
    let solution = view.to_owned();

    // Step 6: extract basis into the caller-owned buffer for warm-starting.
    solver.get_basis(basis_buf);

    Ok(solution)
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

HiGHS is vendored as a git submodule at `crates/cobre-solver/vendor/HiGHS/`. Before building
`cobre-solver` for the first time (or after a fresh clone), initialize the
submodule:

```
git submodule update --init --recursive
```

The build script checks for `crates/cobre-solver/vendor/HiGHS/CMakeLists.txt` and panics with a
clear error message if the submodule is not initialized.

### System dependencies

| Dependency   | Minimum version | Notes                                                       |
| ------------ | --------------- | ----------------------------------------------------------- |
| cmake        | 3.15            | Required by the HiGHS build system                          |
| C compiler   | C11             | gcc or clang; HiGHS and the C wrapper are C/C++             |
| C++ compiler | C++17           | Required by HiGHS internals                                 |
| ~~zlib~~     | ~~any~~         | Not needed — disabled via `CMAKE_DISABLE_FIND_PACKAGE_ZLIB` |

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

This requires cmake, a C/C++ compiler, and an initialized `crates/cobre-solver/vendor/HiGHS/`
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
