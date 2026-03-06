# Accumulated Learnings: Phase 5 cobre-stochastic (through Epic 04)

**Last updated**: 2026-03-06
**Epics covered**: epic-01-crate-scaffold-and-par-types, epic-02-noise-generation-and-opening-tree, epic-03-insample-sampling-and-integration, epic-04-conformance-tests

---

## Flat Array Layout Convention

- 2-D arrays: `array[stage * n_hydros + hydro]` — stage-major layout matches sequential stage iteration (`crates/cobre-stochastic/src/par/precompute.rs`)
- 3-D psi array: `psi[stage * n_hydros * max_order + hydro * max_order + lag]` — stage-major, hydro-minor, lag-innermost
- Opening tree: `data[stage_offsets[stage] + opening_idx * dim .. + dim]` — stage-major with pre-computed sentinel offset array (`crates/cobre-stochastic/src/tree/opening_tree.rs`)
- All fixed-size hot-path arrays use `Box<[f64]>` via `Vec::into_boxed_slice()` (eliminates capacity word, communicates no-resize invariant)

## Sentinel Offset Arrays

- Stage offset arrays have length `n_stages + 1`; the sentinel entry equals `data.len()`, making bounds checks exact without special-casing the last stage
- Confirmed in `PrecomputedParLp` (epic-01), `OpeningTree` (epic-02), and referenced in `StochasticContext` (epic-03) — treat as canonical for all stage-indexed flat arrays

## Deterministic Seed Derivation (DEC-017)

- `derive_forward_seed(base_seed, iteration, scenario, stage) -> u64`: 20-byte SipHash-1-3 wire format
- `derive_opening_seed(base_seed, opening_index, stage) -> u64`: 16-byte SipHash-1-3 wire format
- Different wire lengths provide domain separation without explicit prefixes
- Always use `stage.id` (domain identifier), never `stage.index` (array position) — index shifts under stage filtering; id is stable
- `stage` (u32) = domain id for seed derivation; `stage_idx` (usize) = array position for tree lookup — intentionally distinct parameters (`crates/cobre-stochastic/src/sampling/insample.rs`)
- Golden value regression tests pin exact output for known inputs (`crates/cobre-stochastic/src/noise/seed.rs`)

## rand 0.10 API (IMPORTANT for all noise tickets)

- In rand 0.10: use `rng.random::<T>()` (not `rng.gen::<T>()`); requires `use rand::RngExt`
- Distribution sampling: `rng.sample(StandardNormal)` is unchanged between 0.9 and 0.10
- RNG construction: `Pcg64::seed_from_u64(seed)` handles 64-to-256 bit expansion automatically
- Ticket specs referencing rand 0.9 patterns are stale; verify against `Cargo.toml` before dispatching any ticket using RNG (`crates/cobre-stochastic/Cargo.toml`)

## Cholesky Decomposition Pattern

- Hand-rolled Cholesky-Banachiewicz in ~150 lines; no external linear algebra crate added
- Packed lower-triangular storage: element (i, j) with j <= i at index `i*(i+1)/2 + j`
- `transform` writes directly into a pre-allocated output slice — no intermediate allocation; inner loop uses incremental `row_base` to avoid recomputing triangular index per row (`crates/cobre-stochastic/src/correlation/cholesky.rs`)
- Symmetry validated within tolerance 1e-10 before decomposition; non-PD detection via diagonal <= 0
- `BTreeMap<String, Vec<GroupFactor>>` for profile storage — deterministic iteration order required (`crates/cobre-stochastic/src/correlation/cholesky.rs`, `resolve.rs`)

## `resolve_positions` Pre-computation (Hot Path Optimization)

- `DecomposedCorrelation::resolve_positions(&mut self, entity_order: &[EntityId])` must be called once before entering the hot loop that calls `apply_correlation`
- Stores pre-computed positions as `Option<Box<[usize]>>` on each `GroupFactor`; eliminates per-call O(n) linear scan and `Vec` allocation inside `apply_correlation`
- `generate_opening_tree` signature changed to `&mut DecomposedCorrelation` to enable this; all call sites updated (`crates/cobre-stochastic/src/tree/generate.rs`, `src/context.rs`)
- Fast path uses stack-allocated buffers for groups ≤ 64 entities (`MAX_STACK_DIM`); larger groups use heap (`crates/cobre-stochastic/src/correlation/resolve.rs`)

## Clippy Allow Conventions

- `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]` at integration test module level (not per-test)
- `#![allow(clippy::float_cmp)]` in reproducibility tests for intentional bitwise f64 equality checks
- `#[allow(clippy::cast_precision_loss)]` on individual stat tests that cast `usize` counts to `f64`
- `#![cfg_attr(test, allow(...))]` in `lib.rs` — blanket test relaxation at crate level

## Validation Architecture

- `validate_par_parameters` and `PrecomputedParLp::build` are separate call sites; `build_stochastic_context` invokes validation first, then build
- `DecomposedCorrelation::build` validates and decomposes profiles eagerly — errors surface at initialization, not at per-stage lookup time
- Single-profile models use the single profile as implicit default even if not named "default" (`crates/cobre-stochastic/src/par/validation.rs`, `context.rs`)

## Pipeline Integration Pattern

- Integration entry point is a free function returning an owned struct: `build_stochastic_context(&system, base_seed) -> Result<StochasticContext, StochasticError>`
- `base_seed: u64` is always explicit; OS entropy handling belongs in the application layer, not the infrastructure crate
- Pre-study stages (negative `stage.id`) are excluded from the opening tree by filter `s.id >= 0`; they remain in `inflow_models` for PAR lag initialization (`crates/cobre-stochastic/src/context.rs`)

## `pub mod` + Flat `pub use` Re-export Layout

- All submodules are `pub mod`; the flat `pub use` block at crate root re-exports primary types alphabetically
- Each submodule has a corresponding `fn <name>_module_is_accessible()` test in `lib.rs` to guard against regressions (`crates/cobre-stochastic/src/lib.rs`)

## Compile-Time `Send + Sync` Assertion Pattern

- `const _: () = { const fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<T>(); }` for compile-time assertions not stripped by `--test`
- Keep a redundant `#[test]` version too for documentation visibility (`crates/cobre-stochastic/src/context.rs`)

## Partial Match Invariant for Correlation

- If a correlation group's entity IDs are partially absent from `entity_order`, the Cholesky transform is skipped entirely for that group
- Entities not in any group retain their independent noise values unchanged (`crates/cobre-stochastic/src/correlation/resolve.rs`)

## Conformance Fixture Pattern (Epic 04)

- AR(1) fixture requires a pre-study stage (`id=-1`) to supply lag mean/std for coefficient unit conversion; AR(0) fixtures can omit the pre-study stage entirely
- Pre-study stage uses `ar_coefficients: vec![]` and `residual_std_ratio: 1.0`; study stages use the target AR parameters
- Hand-computed PAR values: `psi = psi_star * s_m / s_lag`, `base = mu - psi * mu_lag`; tolerance `1e-10`
- Use AR(1) fixture only when coefficient conversion is the test objective; use AR(0) for reproducibility/invariance tests

## Infrastructure Genericity Test Pattern (Epic 04)

- Encode the `grep -riE 'sddp' crates/<crate>/src/` check as a `#[test]` using `std::process::Command`; exit code 1 = no matches (passing), code 0 = matches found (failing)
- Use `env!("CARGO_MANIFEST_DIR")` to construct the absolute `src/` path; never use relative paths
- Reusable pattern from `crates/cobre-stochastic/tests/reproducibility.rs::infrastructure_genericity_no_sddp_references`

## Seed Sensitivity Test Convention

- Never use `assert_ne!` on full trees; use `any()` over all entries — two different seeds could theoretically produce one identical value
- Pattern: `let any_differ = (0..n_stages).any(|s| (0..n_openings).any(|o| tree_a.opening(s,o).iter().zip(...).any(|(a,b)| a != b)));`

## Declaration-Order Invariance via `SystemBuilder` Sorting

- `SystemBuilder` sorts hydros by `EntityId` internally; inserting hydros in reversed order produces the same canonical internal order
- Integration test confirms PAR preprocessing, seed derivation, Cholesky entity-position mapping, and tree generation all respect this ordering (`crates/cobre-stochastic/tests/reproducibility.rs::declaration_order_invariance`)

## Scope and Dependency Discipline

- No new external dependencies added in epics 01-04; Cholesky is hand-rolled
- Scaffold tickets score `scope_adherence = 0.0` because workspace-level files are not declared in "Key Files to Modify" — list workspace manifest changes explicitly
- When a ticket produces a struct owning types from prior tickets, declare all transitively-modified files in "Key Files to Modify" to avoid scope-adherence penalties

## Statistical Test Bounds

- N(0,1) distribution: `|mean| < 0.15` and `|std - 1| < 0.15` over 500 samples
- Sample correlation: within ±0.10 of target over 2000 samples
- Bounds set for ~6-sigma reliability while still catching algorithmic failures (`crates/cobre-stochastic/src/tree/generate.rs`)
