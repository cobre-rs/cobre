# ADR: Noise Method Dispatch and ForwardSampler Design

Date: 2026-04-03
Status: Accepted
Scope: cobre-stochastic noise generation and forward-pass sampling

---

## Context

Cobre's SDDP solver requires stochastic noise for two separate subsystems:

1. **Opening tree generation** — a fixed scenario tree built once per training
   iteration, used by the backward pass. Each `(stage, opening)` pair receives
   correlated N(0,1) noise drawn at construction time.
2. **Forward-pass scenario sampling** — noise drawn per `(iteration, scenario,
stage)` triple during the forward pass and simulation. Under `InSample` the
   forward pass reads directly from the pre-built tree; under `OutOfSample` it
   generates fresh noise on-the-fly.

Before this work, tree generation used a single loop order (opening-major) and
always drew noise i.i.d. via SAA. The forward pass had only one code path
(`insample::sample_forward`). Extending to LHS, QMC-Sobol, and QMC-Halton
methods required rethinking both subsystems simultaneously.

The design had to satisfy the project's hard constraints:

- No `unsafe` code anywhere.
- No `Box<dyn Trait>` for closed variant sets — enum dispatch only.
- No per-iteration heap allocation on the hot path.
- Declaration-order invariance: results must be bitwise identical regardless of
  entity insertion order.
- Communication-free MPI reproducibility: each rank derives its seeds
  independently from `(base_seed, iteration, scenario, stage)`.
- No external numeric libraries: all mathematical primitives are hand-rolled.

---

## Design Decisions

| #   | Decision                                                                                                                     | Alternatives Considered                                                                                             | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Stage-major loop order in `generate_opening_tree`**                                                                        | Keep opening-major iteration: each opening generates noise independently before moving to the next.                 | LHS stratification requires knowledge of N (the total opening count) before any single sample can be placed. Sobol Gray-code recurrence processes points sequentially within a stage. Both batch methods require all openings of a stage to be available simultaneously. Stage-major iteration (outer loop: stages; inner: openings) satisfies this. SAA is indifferent to loop order; the stage-major refactor preserves it bitwise, verified by the golden-value regression test (`tests/saa_golden_value.rs`).                                                                                                                                                                                                                 |
| 2   | **Dual entry points: batch functions for tree generation, point-wise functions for forward pass**                            | A single unified interface for both tree generation and forward-pass sampling.                                      | Batch methods have inter-opening dependencies that make parallelizing across openings incorrect: LHS stratification requires knowledge of N and produces one sample per stratum; Sobol Gray-code recurrence is a sequential recurrence. The forward pass runs workers in parallel across scenarios with no inter-worker coordination. Point-wise functions (`sample_lhs_point`, `scrambled_sobol_point`, `scrambled_halton_point`) derive per-dimension permutations and scrambling seeds from a shared base seed, then sample the scenario's stratum or sequence index independently. Both batch and point-wise functions are implemented in the same module (`tree/lhs.rs`, `tree/qmc_sobol/mod.rs`, `tree/qmc_halton/mod.rs`). |
| 3   | **Enum dispatch for `ForwardSampler`, `ForwardNoise`, and `NoiseMethod`**                                                    | `Box<dyn Sampler>` trait object dispatched at runtime.                                                              | Project convention bans `Box<dyn Trait>` for closed variant sets. Exhaustive `match` arms with no wildcard ensure the compiler reports missing variants when a new method is added. Monomorphization eliminates virtual dispatch overhead on the forward-pass hot path. `ForwardSampler` is constructed once before the parallel region and shared via `&sampler` across all worker threads; `ForwardNoise<'a, 'b>` encodes whether noise was drawn from the tree (`TreeSlice`) or freshly generated (`FreshNoise`) through its type, not a flag field.                                                                                                                                                                           |
| 4   | **Hand-rolled numeric algorithms with no external library dependencies**                                                     | Use `statrs` for the inverse normal CDF, `nalgebra` for Cholesky decomposition, `sobol_burley` for Sobol sequences. | The project minimises external crates. Each algorithm is small and self-contained: Beasley-Springer-Moro inverse normal CDF is approximately 60 lines (`noise/quantile.rs`); Cholesky-Banachiewicz in packed lower-triangular format is approximately 150 lines; Sobol with Owen-scrambled Gray-code recurrence is approximately 200 lines; Halton with radical inverse and Owen scrambling is approximately 100 lines; prime sieve for Halton bases is approximately 30 lines. Hand-rolling eliminates transitive dependency trees, allows packed-format storage for the Cholesky factor, and permits incremental `row_base` computation that avoids re-reading the full direction number table on each call.                    |
| 5   | **Dual-seed architecture: `training.tree_seed` controls the opening tree; `scenario_source.seed` controls the forward pass** | A single seed for everything, or deriving the forward seed deterministically from the tree seed.                    | Deriving the forward seed from the tree seed creates a coupling: changing the branching factor or tree structure would change the out-of-sample forward trajectories even when the user wants stable simulation results. Separate seeds allow independent control. Domain separation uses SipHash wire length: `derive_stage_seed` hashes 12 bytes, `derive_opening_seed` hashes 16 bytes, `derive_forward_seed` hashes 20 bytes. Length alone separates the three domains with no explicit tag bytes needed. `forward_seed: Option<u64>` is stored on `StochasticContext`; `None` means the field is unconfigured, not "fall back to tree seed".                                                                                 |
| 6   | **`SampleRequest<'b>` and `FreshNoiseSpec` for parameter bundling**                                                          | Pass all seven or more arguments directly to `sample()` and `sample_fresh()`.                                       | The project bans `#[allow(clippy::too_many_arguments)]`; the required resolution is always a context struct. `SampleRequest<'b>` bundles the seven per-call dispatch arguments for `ForwardSampler::sample()`. `FreshNoiseSpec` (7 fields, `Copy`) bundles seed and dimension arguments for `sample_fresh`, reducing its visible signature to five parameters. Both types are stack-allocated structs with no heap indirection.                                                                                                                                                                                                                                                                                                   |
| 7   | **Worker-local noise buffers in the forward-pass thread closure**                                                            | Use `ScratchBuffers.raw_noise_buf` and `ScratchBuffers.perm_scratch` as originally planned.                         | `run_forward_stage(ws, ...)` borrows `&mut SolverWorkspace`. A live `&mut ws.scratch.raw_noise_buf` borrow and a call to a function that takes `&mut SolverWorkspace` would violate the single-mutable-reference rule. `SolverWorkspace` does not expose field-level two-phase borrowing. The resolution is `let mut raw_noise_buf = vec![0.0_f64; noise_dim];` inside the worker closure, allocated once before the scenario loop and reused across all stages. `ScratchBuffers.raw_noise_buf` and `ScratchBuffers.perm_scratch` remain on the struct for potential future use but are not wired into the noise path. See `forward.rs` worker closure for the canonical pattern.                                                 |

---

## Key Invariants

The design maintains four invariants that must not be broken by future changes:

- **Bitwise SAA compatibility**: the stage-major loop restructure produces
  bit-for-bit identical noise to the original opening-major implementation for
  `NoiseMethod::Saa`. Verified by golden-value regression test at
  `crates/cobre-stochastic/tests/saa_golden_value.rs` using 6 exact f64
  constants and bitwise `assert_eq!`.

- **Declaration-order invariance**: simulation results are bitwise identical
  regardless of the order in which entities (hydros, thermals, buses) are
  inserted. `entity_order: Box<[EntityId]>` is computed before any stochastic
  operations and passed consistently to all correlation and noise functions.
  Verified by integration test at
  `crates/cobre-stochastic/tests/reproducibility.rs`.

- **Communication-free MPI reproducibility**: each MPI rank derives all seeds
  independently from `(base_seed, iteration, scenario, stage)` using the
  SipHash-based `derive_*_seed` family. No inter-rank seed broadcast is
  required. Ranks produce identical noise for the same `(iteration, scenario,
stage)` triple regardless of the rank count or work partitioning.

- **No hot-path allocation**: all batch buffers (`Vec<f64>` for the flat tree
  data, per-stage slice views) are pre-allocated before the stage loop.
  Point-wise functions write into caller-supplied slices. Worker-local
  `raw_noise_buf` and `perm_scratch` are allocated once per thread before the
  scenario loop, not once per `(scenario, stage)` call. `ForwardSampler` is
  constructed once per run and shared as `&sampler` across the parallel region.

---

## Consequences

**Adding a new noise method** requires four changes in `cobre-stochastic`:

1. Add a variant to `NoiseMethod` in `cobre-core/src/temporal/mod.rs`.
2. Implement batch and point-wise functions in a new `tree::` submodule,
   following the structure of `tree/lhs.rs` or `tree/qmc_sobol/mod.rs`.
3. Add a match arm in `generate_opening_tree` (`tree/generate.rs`) for the
   batch path.
4. Add a match arm in `sample_fresh` (`sampling/out_of_sample.rs`) for the
   point-wise path. If the new method has special factory requirements, also add
   a match arm in `build_forward_sampler` (`sampling/mod.rs`).

The exhaustive `match` arms on `NoiseMethod` will cause a compile error on any
missed location, making the extension safe by construction.

**Unimplemented variants**: `NoiseMethod::Selective` is defined and parsed from
JSON but returns `Err(StochasticError::UnsupportedNoiseMethod)` from
`generate_opening_tree`. In the forward pass it falls back to SAA with a
`tracing::warn!`. Full implementation requires clustering infrastructure to
select representative scenarios, which is out of scope for this design.

**Stub sampling schemes**: `SamplingScheme::Historical` and
`SamplingScheme::External` are defined variants in `cobre-core`. The
`build_forward_sampler` factory returns `Err(StochasticError::MissingScenarioSource)`
for both. These schemes do not use the `NoiseMethod` dispatch path at all:
`Historical` replays a historical inflow file and `External` reads a
pre-generated scenario library. Implementing either requires a separate data
source abstraction, not changes to the noise generation subsystem.
