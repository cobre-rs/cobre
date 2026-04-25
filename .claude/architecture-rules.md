# Cobre Architecture Rules — Hot Path & Context Structs

This file is re-injected into Claude Code's context after compaction and should
be read before modifying any hot-path code. It describes the required shape of
hot-path drivers, context structs, and function signatures.

---

## The Context Struct Pattern

When the SDDP training loop needs new data threaded through the
forward/backward/simulate call chain, **add a field to an existing context
struct** instead of adding a function parameter.

Available context structs:

| Struct                 | File                                           | Purpose                                         | Mutability             |
| ---------------------- | ---------------------------------------------- | ----------------------------------------------- | ---------------------- |
| `StageContext`         | `cobre-sddp/src/context.rs`                    | Per-stage templates, base rows, layout          | Immutable (`&`)        |
| `TrainingContext`      | `cobre-sddp/src/context.rs`                    | Horizon, indexer, stochastic, initial state     | Immutable (`&`)        |
| `ScratchBuffers`       | `cobre-sddp/src/workspace.rs`                  | Per-worker noise/patch scratch space            | Mutable (`&mut`)       |
| `SolverWorkspace`      | `cobre-sddp/src/workspace.rs`                  | Solver + scratch + patch buffer                 | Mutable (`&mut`)       |
| `TrainingConfig`       | `cobre-sddp/src/config.rs`                     | Forward passes, iteration limit, seed           | Owned (moved in)       |
| `SimulationConfig`     | `cobre-sddp/src/simulation/config.rs`          | Scenario count, channel capacity, basis window  | Immutable (`&`)        |
| `ForwardPassBatch`     | `cobre-sddp/src/forward.rs`                    | Local pass count, iteration, offset             | Immutable (`&`)        |
| `LbEvalSpec`           | `cobre-sddp/src/lower_bound.rs`                | Template, noise scale, opening tree             | Immutable (`&`)        |
| `TrainingSession`      | `cobre-sddp/src/training_session/mod.rs`       | Owns solver, pools, sub-state structs, scratch  | Owned driver           |
| `BackwardPassState`    | `cobre-sddp/src/backward_pass_state.rs`        | Owned scratch for backward-pass helpers         | Mutable (`&mut self`)  |
| `ForwardPassState`     | `cobre-sddp/src/forward_pass_state.rs`         | Owned scratch for forward-pass workers          | Mutable (`&mut self`)  |
| `SimulationState`      | `cobre-sddp/src/simulation/state.rs`           | Owned scratch for simulation workers            | Mutable (`&mut self`)  |
| `BackwardPassInputs`   | `cobre-sddp/src/backward_pass_state.rs`        | Borrowed inputs to `BackwardPassState::run`     | Mutable bundle (`&mut`) |
| `ForwardPassInputs`    | `cobre-sddp/src/forward_pass_state.rs`         | Borrowed inputs to `ForwardPassState::run`      | Mutable bundle (`&mut`) |
| `SimulationInputs`     | `cobre-sddp/src/simulation/state.rs`           | Borrowed inputs to `SimulationState::run`       | Mutable bundle (`&mut`) |
| `ForwardWorkerParams`  | `cobre-sddp/src/forward_pass_state.rs`         | Read-only captures for rayon workers            | Immutable bundle (`&`) |
| `ForwardWorkerResult`  | `cobre-sddp/src/forward_pass_state.rs`         | Return bundle from per-worker forward execution | Owned (moved out)      |
| `OpeningTreeInputs`    | `cobre-stochastic/src/tree/generate.rs`        | Optional inputs to `generate_opening_tree`      | Immutable bundle (`&`) |
| `LbEvalScratch`        | `cobre-sddp/src/lower_bound.rs`                | 10 f64/usize scratch vectors reused across `evaluate_lower_bound` phases | Mutable (`&mut`)       |
| `LbEvalScratchBundle`  | `cobre-sddp/src/lower_bound.rs`                | Bundles `patch_buf`, `lb_cut_batch`, `lb_cut_row_map`, `lb_scratch` for `evaluate_lower_bound` | Mutable bundle (`&mut`) |
| `RiskMeasureScratch`   | `cobre-sddp/src/risk_measure.rs`               | CVaR weight-computation scratch (`upper_bounds`, `order`, `mu`) | Mutable (`&mut`)       |

**Decision tree when adding new data to the hot path:**

1. Per-stage, read-only, shared across workers → `StageContext`.
2. Study-level, read-only → `TrainingContext`.
3. Per-worker mutable scratch → `ScratchBuffers`.
4. Per-solve transient state → `SolverWorkspace`.
5. Backward-pass scratch reused across iterations → field on `BackwardPassState`.
6. Forward-pass scratch reused across iterations → field on `ForwardPassState`.
7. Simulation scratch reused across scenarios → field on `SimulationState`.
8. Per-call input to backward/forward/simulate → field on the matching `*Inputs` bundle.
9. Lower-bound-specific → `LbEvalSpec`.
10. None of the above → create a new spec or bundle struct. Do NOT add a bare parameter.

---

## StudySetup Sub-Structs

`StudySetup` owns all pre-computed study state. It holds cohesive sub-structs
plus a small number of bare residuals. Context constructors (`stage_ctx`,
`training_ctx`, `simulation_ctx`) borrow directly from the sub-structs.

### Cohesive sub-structs

| Struct                | File                                             | Purpose                                                                  | Visibility    | Storage form             |
| --------------------- | ------------------------------------------------ | ------------------------------------------------------------------------ | ------------- | ------------------------ |
| `StageData`           | `cobre-sddp/src/setup/stage_data.rs`             | All stage-indexed data: templates, indexer, stages, entity counts, blocks, lag transitions, noise groups, scaling report | `pub`         | Aggregated sub-struct    |
| `ScenarioLibraries`   | `cobre-sddp/src/setup/scenario_library_set.rs`   | Training + simulation `PhaseLibraries` pair                              | `pub`         | Aggregated sub-struct    |
| `PhaseLibraries`      | `cobre-sddp/src/setup/scenario_library_set.rs`   | Sampling schemes and optional libraries for one phase                    | `pub`         | Aggregated sub-struct    |
| `MethodologyConfig`   | `cobre-sddp/src/setup/methodology_config.rs`     | `horizon` + `inflow_method` — stochastic numerical methodology            | `pub(crate)`  | Aggregated sub-struct    |
| `LoopParams`          | `cobre-sddp/src/config.rs`                       | Pure-data projection of `LoopConfig` (excludes runtime-derived fields)    | `pub`         | Projection of `LoopConfig` |
| `SimulationConfig`    | `cobre-sddp/src/simulation/config.rs`            | `n_scenarios`, `io_channel_capacity`, `basis_activity_window`             | `pub`         | Literal reuse            |
| `CutManagementConfig` | `cobre-sddp/src/config.rs`                       | Cut selection, budget cap, activity tolerance, basis window, warm-start cuts, per-stage risk measures | `pub(crate)` | Literal reuse            |
| `EventParams`         | `cobre-sddp/src/config.rs`                       | Output-side event flags; excludes runtime handles                         | `pub(crate)`  | Projection of `EventConfig` |

### Literal reuse vs projection

- **Literal reuse**: store the config type verbatim. Use when the config type
  owns exactly the right fields with no runtime handles and no per-invocation
  values (example: `SimulationConfig`, `CutManagementConfig`). Access path:
  `setup.simulation_config.field`.
- **Projection**: create a `*Params` sibling that drops runtime-bound or
  per-invocation fields. Use when 1–3 fields must be excluded (example:
  `LoopParams` drops `n_fwd_threads`; `EventParams` drops runtime handles).
- **New sub-struct**: introduce a dedicated type when no existing type
  cohesively covers the grouping (example: `StageData`, `ScenarioLibraries`,
  `MethodologyConfig`).

### Accessor policy

`StudySetup` exposes a small impl surface: context builders (`stage_ctx`,
`training_ctx`, `simulation_ctx`) plus targeted mutation setters
(`replace_fcf`, `set_start_iteration`, `set_export_states`, `set_budget`) and
one typed read accessor (`simulation_config`). Every other access uses direct
field paths (`setup.sub_struct.field`). Do not add accessor methods for plain
field reads — prefer the direct path.

---

## State Struct Pattern

Hot-path drivers with long preludes and many captures follow a two-part shape:

1. **State struct** (`TrainingSession`, `BackwardPassState`, `ForwardPassState`,
   `SimulationState`) — owns scratch buffers allocated once and reused via
   `clear()` / `resize()` / `extend()` across every iteration. Constructed
   once by `TrainingSession::new` and stored as a field on `TrainingSession`.
   No allocation on the hot path.
2. **Inputs bundle** (`BackwardPassInputs`, `ForwardPassInputs`,
   `SimulationInputs`) — holds all borrowed per-call inputs (contexts, FCF,
   comm, baked templates, iteration counters). Constructed fresh at each
   `run` call via a `from_session_fields(...)` factory that borrows from
   `&mut TrainingSession` fields disjointly under NLL rules.

Every driver's `run` method signature is uniformly
`fn run(&mut self, inputs: &mut *Inputs) -> Result<..., SddpError>`.
No bare per-call parameters — everything rides on either `self` or `inputs`.

**Use this pattern when:**

- A function has a 50+ line prelude of buffer allocation or scratch init.
- A function would otherwise exceed the 9-argument budget.
- A helper has 10+ captures and should be extracted as a free function with
  explicit parameters (use a `*Params` bundle for the captures — see
  `run_forward_worker` + `ForwardWorkerParams`).

**Do not use this pattern when:**

- A function has fewer than 30 lines of setup — a plain local binding chain
  is clearer.
- A function is called once per training run with no per-iteration state —
  adding a state struct would be ceremony. The `train` entry point is a
  thin shim that constructs `TrainingSession` and drives its `run_iteration`
  loop.
- A helper takes 4 or fewer cohesive arguments — named parameters are clearer
  than a single-field bundle.

**Naming conventions for bundle structs:**

- `*State` — owns mutable scratch, method receiver (`&mut self`).
- `*Inputs` — borrowed inputs to a driver's `run` method (mutable bundle
  for drivers that need mutable access to the pool; immutable otherwise).
- `*Params` — read-only captures for parallel workers, shared across rayon
  workers via `&` reference.
- `*Result` — return bundle for multi-value returns. Prefer a named struct
  over a tuple as soon as the arity reaches 3 — this avoids
  `#[allow(clippy::type_complexity)]`.

---

## Function Signature Budgets

| Function                          | Max args | Location                                | Notes                                       |
| --------------------------------- | -------- | --------------------------------------- | ------------------------------------------- |
| `train`                           | 10       | `training.rs`                           | Public entry point; keep at or below target |
| `TrainingSession::run_iteration`  | 2        | `training_session/mod.rs`               | `&mut self` + iteration counter             |
| `BackwardPassState::run`          | 2        | `backward_pass_state.rs`                | `&mut self` + `&mut BackwardPassInputs`     |
| `ForwardPassState::run`           | 2        | `forward_pass_state.rs`                 | `&mut self` + `&mut ForwardPassInputs`      |
| `SimulationState::run`            | 2        | `simulation/state.rs`                   | `&mut self` + `&mut SimulationInputs`       |
| `evaluate_lower_bound`            | 9        | `lower_bound.rs`                        | At the workspace budget ceiling             |
| `build_row_lower_unscaled`        | 8        | `simulation/pipeline.rs`                |                                             |
| `run_forward_worker`              | 6        | `forward_pass_state.rs`                 | Free function; accepts `&ForwardWorkerParams` |
| `generate_opening_tree`           | 7        | `cobre-stochastic/src/tree/generate.rs` | Optional inputs go through `OpeningTreeInputs` |

All four hot-path drivers (train/backward/forward/simulate) must take exactly
`&mut self` + `&mut *Inputs` at their public `run` boundary. Any function that
would exceed the 9-argument budget must be refactored (bundle captures, move
state onto a `*State` struct, or extract a helper) rather than suppressed with
`#[allow(clippy::too_many_arguments)]`. The workspace clippy thresholds live
in `clippy.toml`; keep production functions within the current thresholds and
do not raise them to accommodate new code. When a suppression is unavoidable
(public-API shims, rayon-closure adapters with structural arity), write a
`// Rationale:` comment on the line above documenting why refactoring is not
appropriate.

---

## Common Mistakes to Avoid

**Adding `cut_batches: &mut [RowBatch]` as a parameter.** Per-stage workspace
state belongs in `SolverWorkspace` or a `TrainingWorkspace` struct.

**Adding `stage_bases: &[Option<Basis>]` as a parameter to `simulate`.** Study-
level read-only data produced by training belongs on `StudySetup` or in a
`SimulationSpec` struct.

**Adding LP scaling buffers as parameters.** Row scale factors, column scale
factors, and unscale buffers are per-stage data. They belong in `StageContext`
or attached to the stage template.

**Threading generic constraint data through 4 levels of calls.** Generic
constraint metadata is per-stage and read-only. It belongs in `StageContext`.

**Suppressing `clippy::too_many_arguments` instead of refactoring.** If a new
function exceeds the budget, bundle its captures into a `*Inputs` or `*Params`
struct. Suppression is reserved for cases with a documented architectural
reason that refactoring would not fix (public API stability, rayon iterator
shape).

**Tuple returns of arity ≥ 3.** Introduce a `*Result` struct. Suppressing
`clippy::type_complexity` is not acceptable for new code.

---

## Python Parity Checklist

When adding a new output file in the CLI (`write_outputs` in `run.rs`):

1. Does `run_inner()` in `cobre-python/src/run.rs` write the same file?
2. If not, add it. The Python path should call the same `cobre_io` write function.

Current gaps (to be fixed):

- None — all CLI output writes are mirrored in Python.
