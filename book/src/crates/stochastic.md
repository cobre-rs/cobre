# cobre-stochastic

<span class="status-experimental">experimental</span>

`cobre-stochastic` provides the stochastic process models for the Cobre power
systems ecosystem. It builds probabilistic representations of hydro inflow
time series — using Periodic Autoregressive (PAR(p)) models — and generates
correlated noise scenarios for use by iterative scenario-based optimization
algorithms. The crate is solver-agnostic: it supplies fully-initialized
stochastic infrastructure components that any scenario-based iterative
optimization algorithm can consume read-only, with no dependency on any
particular solver vertical.

The crate has no dependency on `cobre-solver` or `cobre-comm`. It depends only
on `cobre-core` for entity types and on a small set of RNG and hashing crates
for deterministic noise generation.

## Module overview

| Module        | Purpose                                                                                                   |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `par`         | PAR(p) coefficient preprocessing: validation, original-unit conversion, and the `PrecomputedParLp` cache  |
| `noise`       | Deterministic noise generation: SipHash-1-3 seed derivation (`seed`) and `Pcg64` RNG construction (`rng`) |
| `correlation` | Cholesky-based spatial correlation: decomposition (`cholesky`) and profile resolution (`resolve`)         |
| `tree`        | Opening scenario tree: flat storage structure (`opening_tree`) and tree generation (`generate`)           |
| `sampling`    | InSample scenario selection: `sample_forward` for picking an opening for a given iteration/scenario/stage |
| `context`     | `StochasticContext` integration type and `build_stochastic_context` pipeline entry point                  |
| `error`       | `StochasticError` with five variants covering all failure domains of the stochastic layer                 |

## Architecture

### PAR(p) preprocessing and flat array layout

PAR(p) (Periodic Autoregressive) models describe the seasonal autocorrelation
structure of hydro inflow time series. Each hydro plant at each stage has an
`InflowModel` with a mean (`mean_m3s`), a standard deviation (`std_m3s`), and
a vector of AR coefficients in standardized form (`ar_coefficients`).

`PrecomputedParLp` is built once at initialization from raw `InflowModel`
parameters. It converts AR coefficients from **standardized form** (ψ\*,
direct Yule-Walker output) to **original-unit form** at build time:

```text
ψ_{m,ℓ} = ψ*_{m,ℓ} · s_m / s_{m-ℓ}
```

where `s_m` is `std_m3s` for the current stage's season and `s_{m-ℓ}` is
`std_m3s` for the season ℓ stages prior. The converted coefficients and their
derived intercepts (`base`) are stored in **stage-major flat arrays**:

```text
array[stage * n_hydros + hydro]          (2-D: means, stds, base terms)
psi[stage * n_hydros * max_order + hydro * max_order + lag]  (3-D: AR coefficients)
```

This layout ensures that all per-stage data for every hydro plant is contiguous
in memory, maximizing cache utilization during sequential stage iteration within
a scenario trajectory.

All hot-path arrays use `Box<[f64]>` (via `Vec::into_boxed_slice()`) rather
than `Vec<f64>`. The boxed-slice type communicates the no-resize invariant and
eliminates the capacity word from each allocation.

### Deterministic noise via SipHash-1-3 seed derivation (DEC-017)

Each scenario realization in an iterative optimization run requires a draw
from the noise distribution. Rather than broadcasting seeds across compute
nodes — which would require communication — each node independently derives
its own seed from a small tuple using SipHash-1-3 (DEC-017).

Two derivation functions are provided:

- **`derive_forward_seed(base_seed, iteration, scenario, stage) -> u64`**:
  hashes a 20-byte little-endian wire format
  `base_seed (8B) ++ iteration (4B) ++ scenario (4B) ++ stage (4B)`.
- **`derive_opening_seed(base_seed, opening_index, stage) -> u64`**:
  hashes a 16-byte wire format
  `base_seed (8B) ++ opening_index (4B) ++ stage (4B)`.

The different wire lengths provide domain separation without explicit prefixes,
preventing hash collisions between forward-pass seeds and opening-tree seeds.
`stage` in both functions is always `stage.id` (the domain identifier), never
`stage.index` (the array position), because array positions shift under stage
filtering while IDs are stable.

From the derived seed, a `Pcg64` RNG is constructed via `rng_from_seed`. The
PCG family provides good statistical quality with fast generation, suitable
for producing large numbers of standard-normal samples via the `StandardNormal`
distribution.

### Cholesky-based spatial correlation

Hydro inflow series at neighboring plants are spatially correlated. `cobre-stochastic`
applies a Cholesky transformation to convert independent standard-normal samples
into correlated samples.

The Cholesky decomposition is hand-rolled using the Cholesky-Banachiewicz
algorithm (~150 lines). No external linear algebra crate is added to the
dependency tree. The lower-triangular factor `L` (such that `Sigma = L * L^T`)
is stored in **packed lower-triangular format**: element `(i, j)` with `j <=
i` is at index `i*(i+1)/2 + j`. This eliminates the zero upper-triangle
entries and halves memory usage.

Correlation profiles can be defined per-season. `DecomposedCorrelation` holds
all profiles in a `BTreeMap<String, Vec<GroupFactor>>` — the `BTreeMap`
guarantees deterministic iteration order, which is required for
declaration-order invariance.

Before entering the hot optimization loop, callers must invoke
`DecomposedCorrelation::resolve_positions(&mut self, entity_order: &[EntityId])`
once. This pre-computes the positions of each group's entities within the
canonical entity order and stores them on each `GroupFactor` as
`Option<Box<[usize]>>`. With positions pre-computed, `apply_correlation`
avoids a per-call O(n) linear scan and heap allocation on the hot path.

If a correlation group's entity IDs are only partially present in
`entity_order`, the Cholesky transform is skipped for that group entirely.
Entities not in any group retain their independent noise values unchanged.

### Opening tree structure

The opening scenario tree pre-generates all noise realizations used during
the backward pass of the optimization algorithm, before the iterative loop
begins. This avoids per-iteration recomputation and ensures the backward pass
always operates on a fixed, reproducible set of scenarios.

`OpeningTree` stores all noise values in a single flat contiguous array with
stage-major ordering:

```text
data[stage_offsets[stage] + opening_idx * dim .. + dim]
```

The `stage_offsets` array has length `n_stages + 1`. The sentinel entry
`stage_offsets[n_stages]` equals `data.len()`, making bounds checks exact
without special-casing the last stage. This sentinel pattern is used
consistently in `PrecomputedParLp`, `OpeningTree`, and throughout
`StochasticContext`.

Pre-study stages (those with negative `stage.id`) are excluded from the
opening tree but remain in `inflow_models` for PAR lag initialization.

### `StochasticContext` as the integration entry point

`StochasticContext` bundles the three independently-built components into a
single ready-to-use value:

1. `PrecomputedParLp` — PAR coefficient cache for LP RHS patching.
2. `DecomposedCorrelation` — pre-decomposed Cholesky factors for all profiles.
3. `OpeningTree` — pre-generated noise realizations for the backward pass.

`build_stochastic_context(&system, base_seed)` runs the full preprocessing
pipeline in a fixed order: validate PAR parameters, build the coefficient
cache, decompose correlation matrices, generate the opening tree. After
construction, all fields are immutable. `StochasticContext` is `Send + Sync`,
verified by a compile-time assertion and a unit test.

### `sample_forward` for InSample scenario selection

`sample_forward` implements the InSample scenario selection strategy: for each
`(iteration, scenario, stage)` triple, it deterministically selects one opening
from the tree by deriving a seed via `derive_forward_seed` and sampling a
`Pcg64` RNG. The selected opening index and its noise slice are returned together,
so the caller can both log which opening was chosen and immediately use the
noise values.

## Public types

### `StochasticContext`

Owns all three preprocessing pipeline outputs: `PrecomputedParLp`,
`DecomposedCorrelation`, and `OpeningTree`. Constructed by
`build_stochastic_context` and then consumed read-only. Accessors:
`par_lp()`, `correlation()`, `opening_tree()`, `tree_view()`, `base_seed()`,
`dim()`, `n_stages()`. Both `Send` and `Sync`.

### `PrecomputedParLp`

Cache-friendly PAR(p) model data for LP RHS patching. Stores means, standard
deviations, original-unit AR coefficients (ψ), and intercept terms (base) in
stage-major flat arrays (`Box<[f64]>`). Built via `PrecomputedParLp::build`.
Accessors: `n_hydros()`, `n_stages()`, `max_order()`, `mean()`, `std()`,
`base()`, `psi()`.

### `DecomposedCorrelation`

Holds Cholesky-decomposed correlation factors for all profiles, keyed by
profile name in a `BTreeMap`. Built via `DecomposedCorrelation::build`, which
validates and decomposes all profiles eagerly — errors surface at initialization,
not at per-stage lookup time. Call `resolve_positions` once with the canonical
entity order before entering the optimization loop.

### `OpeningTree`

Fixed opening scenario tree holding pre-generated noise realizations. All noise
values are in a flat `Box<[f64]>` with stage-major ordering and a sentinel
offset array of length `n_stages + 1`. Provides `opening(stage_idx, opening_idx)
-> &[f64]` for element access and `view() -> OpeningTreeView<'_>` for a
zero-copy borrowed view.

### `OpeningTreeView<'a>`

A zero-copy borrowed view over an `OpeningTree`, with the same accessor API:
`opening(stage_idx, opening_idx)`, `n_stages()`, `n_openings(stage_idx)`,
`dim()`. Passed to `sample_forward` to avoid cloning the tree data.

### `StochasticError`

Returned by all fallible APIs. Five variants:

| Variant                       | When it occurs                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------- |
| `InvalidParParameters`        | AR order > 0 with zero standard deviation, or ill-conditioned coefficients        |
| `CholeskyDecompositionFailed` | Correlation matrix is not positive-definite                                       |
| `InvalidCorrelation`          | Missing default profile, ambiguous profile set, or out-of-range correlation entry |
| `InsufficientData`            | Fewer historical records than the PAR order requires                              |
| `SeedDerivationError`         | Hash computation produces an invalid result during seed derivation                |

Implements `std::error::Error`, `Send`, and `Sync`.

### `ParValidationReport`

Return type of `validate_par_parameters`. Contains a list of `ParWarning`
values for non-fatal issues (e.g., high AR coefficients that may indicate
numerical instability) that the caller can inspect or log before proceeding
to `PrecomputedParLp::build`.

### `ParWarning`

A non-fatal PAR parameter warning. Carries the hydro ID, stage ID, and a
human-readable description of the potential issue.

### `GroupFactor`

A single correlation group's Cholesky factor with its associated entity ID
mapping. Fields: `factor: CholeskyFactor`, `entity_ids: Vec<EntityId>`, and
pre-computed `positions: Option<Box<[usize]>>` (filled by `resolve_positions`).

### `CholeskyFactor`

The lower-triangular Cholesky factor `L` of a correlation matrix, stored in
packed row-major form. Element `(i, j)` with `j <= i` is at index
`i*(i+1)/2 + j`. Constructed via `CholeskyFactor::decompose(&matrix)` and
applied via `transform(&input, &mut output)`.

## Usage example

The following shows how to construct a stochastic context from a loaded system
and use it to sample a forward-pass scenario.

```rust,no_run
use cobre_stochastic::{
    build_stochastic_context,
    sampling::insample::sample_forward,
};

// `system` is a `cobre_core::System` produced by `cobre_io::load_case`.
// `base_seed` comes from the study configuration (application layer handles
// the Option<i64> -> u64 conversion and OS-entropy fallback).
let ctx = build_stochastic_context(&system, base_seed)?;

println!(
    "stochastic context: {} hydros, {} study stages",
    ctx.dim(),
    ctx.n_stages(),
);

// Obtain a borrowed view over the opening tree (zero-copy).
let tree_view = ctx.tree_view();

// In the iterative optimization loop, select a forward scenario for each
// (iteration, scenario, stage) triple.
let iteration: u32 = 0;
let scenario: u32 = 0;

for (stage_idx, stage) in study_stages.iter().enumerate() {
    // stage.id is the domain identifier; stage_idx is the array position.
    let (opening_idx, noise_slice) = sample_forward(
        &tree_view,
        ctx.base_seed(),
        iteration,
        scenario,
        stage.id as u32,
        stage_idx,
    );

    // `noise_slice` has length `ctx.dim()` (one value per hydro plant).
    // Pass to LP RHS patching together with `ctx.par_lp()`.
    let _ = (opening_idx, noise_slice);
}
# Ok::<(), cobre_stochastic::StochasticError>(())
```

## Performance notes

`cobre-stochastic` is designed so that all performance-critical preprocessing
happens once at initialization. The iterative optimization loop consumes
already-materialized data through slice indexing, with no re-allocation on the
hot path.

### Pre-computed entity positions (`resolve_positions`)

`DecomposedCorrelation::resolve_positions` must be called once before entering
the optimization loop. It pre-computes the mapping from each correlation group's
entity IDs to their positions in the canonical `entity_order` slice and stores
the result as `Option<Box<[usize]>>` on each `GroupFactor`. Without this
pre-computation, `apply_correlation` would perform an O(n) linear scan and a
`Vec` allocation for every noise draw.

### Stack-allocated buffers for small groups (`MAX_STACK_DIM = 64`)

Inside `apply_correlation`, intermediate working buffers for correlation groups
with at most 64 entities are stack-allocated (using `arrayvec` or a fixed-size
array on the stack). Groups larger than this threshold fall back to
heap-allocated `Vec`. The fast path covers the overwhelming majority of
practical correlation groups, eliminating heap allocation from the inner loop
for typical study configurations.

### Incremental `row_base` in Cholesky transform

The packed lower-triangular storage index for element `(i, j)` is
`i*(i+1)/2 + j`. Rather than recomputing the triangular index from scratch for
each row, the `transform` method maintains an incremental `row_base` variable
that is incremented by `i+1` at the end of each row. This eliminates a
multiplication per row iteration on the hot path of the Cholesky forward
substitution.

### `Box<[f64]>` for the no-resize invariant

All fixed-size hot-path arrays in `PrecomputedParLp`, `OpeningTree`, and
`CholeskyFactor` use `Box<[f64]>` rather than `Vec<f64>`. The boxed-slice
type communicates that these arrays are immutable after construction, eliminates
the capacity word from each allocation, and allows the optimizer to treat the
length as a compile-time-stable bound.

## Feature flags

`cobre-stochastic` has no optional feature flags. All dependencies are always
compiled. No external system libraries are required (HiGHS, MPI, etc.).

```toml
# Cargo.toml
cobre-stochastic = { version = "0.1" }
```

## Testing

### Running the test suite

```
cargo test -p cobre-stochastic
```

No external dependencies or system libraries are required. All dependencies
(siphasher, rand, rand_pcg, rand_distr, thiserror) are Cargo-managed. The
`--all-features` flag is not needed — there are no feature flags.

### Test suite overview

The crate has 125 tests total: 105 unit tests, 5 conformance integration tests,
4 reproducibility integration tests, and 11 doc-tests.

### Conformance suite (`tests/conformance.rs`)

The conformance test suite verifies the PAR(p) preprocessing pipeline against
hand-computed fixtures with known exact outputs.

Two fixtures are used:

- **AR(0) fixture**: a zero-order AR model (pure noise, no lagged terms). The
  precomputed `psi` array must be all-zeros and the `base` values must equal
  the raw means. Tolerance: 1e-10.
- **AR(1) fixture**: a first-order AR model with a pre-study stage (negative
  `stage.id`) that supplies the lag mean and standard deviation for coefficient
  unit conversion. The conversion formula `ψ = ψ* · s_m / s_lag` is tested
  against a hand-computed value. Tolerance: 1e-10.

### Reproducibility suite (`tests/reproducibility.rs`)

Four tests verify the determinism and invariance properties that are required
for correct behavior in a distributed, multi-run setting:

- **Seed determinism**: calling `derive_forward_seed` and `derive_opening_seed`
  with the same inputs always returns bitwise-identical seeds. Golden-value
  regression pins the exact hash output for a known `(base_seed, ...)` tuple.
- **Opening tree seed sensitivity**: different `base_seed` values produce
  different opening trees (verified by checking that at least one noise value
  differs across the full tree). Uses `any()` over all tree entries rather than
  `assert_ne!` on the whole tree, to handle the astronomically unlikely case
  where two seeds produce one identical value.
- **Declaration-order invariance**: inserting hydros in reversed order into a
  `SystemBuilder` (which sorts by `EntityId` internally) produces a
  `StochasticContext` with bitwise-identical PAR arrays, opening tree, and
  Cholesky transform output. This verifies the canonical-order invariant across
  the full preprocessing pipeline.
- **Infrastructure genericity gate**: a grep audit confirms that no algorithm-specific
  references appear anywhere in the crate source tree. The gate is encoded as a
  `#[test]` using `std::process::Command` so it runs automatically in CI.

## Design notes

### Communication-free noise generation (DEC-017)

The original design considered broadcasting a seed from the root rank to all
workers before each iteration. DEC-017 rejected this approach because it adds
an MPI collective on the hot path and creates a serialization point as the
number of ranks grows.

The alternative — deriving each rank's seeds independently from a common
`base_seed` plus a context tuple — requires no communication and produces
identical results regardless of the number of ranks. SipHash-1-3 was chosen
because it is non-cryptographic (fast), produces high-quality 64-bit hashes
suitable for seeding a CSPRNG, and is available in the `siphasher` crate with
no system dependencies.

The two wire formats (20 bytes for forward seeds, 16 bytes for opening seeds)
use length-based domain separation rather than an explicit prefix byte, which
is slightly more efficient and equally correct given that the two sets of
input tuples have different shapes and lengths.
