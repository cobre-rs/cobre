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

| Struct                | File                                             | Purpose                                                                                                                            | Mutability              |
| --------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| `StageContext`        | `cobre-sddp/src/workspace/context.rs`            | Per-stage templates, layout                                                                                             | Immutable (`&`)         |
| `TrainingContext`     | `cobre-sddp/src/workspace/context.rs`            | Horizon, indexer, stochastic, initial state, the runtime node graph                                                                | Immutable (`&`)         |
| `ScratchBuffers`      | `cobre-sddp/src/workspace/workspace.rs`          | Per-worker noise/patch scratch space                                                                                               | Mutable (`&mut`)        |
| `SolverWorkspace`     | `cobre-sddp/src/workspace/workspace.rs`          | Solver + scratch + patch buffer                                                                                                    | Mutable (`&mut`)        |
| `TrainingConfig`      | `cobre-sddp/src/config.rs`                       | Forward passes, iteration limit, seed                                                                                              | Owned (moved in)        |
| `SimulationConfig`    | `cobre-sddp/src/simulation/config.rs`            | Scenario count, channel capacity                                                                                                   | Immutable (`&`)         |
| `ForwardPassBatch`    | `cobre-sddp/src/training/forward/mod.rs`         | Local pass count, iteration, offset                                                                                                | Immutable (`&`)         |
| `TrainingSession`     | `cobre-sddp/src/training/session/mod.rs`         | Owns solver, pools, sub-state structs, scratch                                                                                     | Owned driver            |
| `BackwardPassState`   | `cobre-sddp/src/training/backward_pass_state.rs` | Owned scratch for backward-pass helpers                                                                                            | Mutable (`&mut self`)   |
| `ForwardPassState`    | `cobre-sddp/src/training/forward_pass_state.rs`  | Owned scratch for forward-pass workers                                                                                             | Mutable (`&mut self`)   |
| `SimulationState`     | `cobre-sddp/src/simulation/state.rs`             | Owned scratch for simulation workers                                                                                               | Mutable (`&mut self`)   |
| `BackwardPassInputs`  | `cobre-sddp/src/training/backward_pass_state.rs` | Borrowed inputs to `BackwardPassState::run`                                                                                        | Mutable bundle (`&mut`) |
| `ForwardPassInputs`   | `cobre-sddp/src/training/forward_pass_state.rs`  | Borrowed inputs to `ForwardPassState::run`                                                                                         | Mutable bundle (`&mut`) |
| `SimulationInputs`    | `cobre-sddp/src/simulation/state.rs`             | Borrowed inputs to `SimulationState::run`                                                                                          | Mutable bundle (`&mut`) |
| `ForwardWorkerParams` | `cobre-sddp/src/training/forward_pass_state.rs`  | Read-only captures for rayon workers                                                                                               | Immutable bundle (`&`)  |
| `ForwardWorkerResult` | `cobre-sddp/src/training/forward_pass_state.rs`  | Return bundle from per-worker forward execution                                                                                    | Owned (moved out)       |
| `OpeningTreeInputs`   | `cobre-stochastic/src/tree/generate.rs`          | Optional inputs to `generate_opening_tree`                                                                                         | Immutable bundle (`&`)  |
| `LbEvalScratch`       | `cobre-sddp/src/training/lower_bound.rs`         | Rank-0 risk-measure aggregation scratch (`objectives_buf`, `weights_buf`) with no `ScratchBuffers` counterpart                     | Mutable (`&mut`)        |
| `LbEvalScratchBundle` | `cobre-sddp/src/training/lower_bound.rs`         | Bundles `patch_buf`, `lb_cut_batch`, `lb_cut_row_map`, `noise_scratch` (`ScratchBuffers`), `lb_scratch` for `evaluate_lower_bound` | Mutable bundle (`&mut`) |
| `RiskMeasureScratch`  | `cobre-sddp/src/convergence/risk_measure.rs`     | CVaR weight-computation scratch (`upper_bounds`, `order`, `mu`)                                                                    | Mutable (`&mut`)        |

**Decision tree when adding new data to the hot path:**

1. Per-stage, read-only, shared across workers → `StageContext`.
2. Study-level, read-only → `TrainingContext`.
3. Per-worker mutable scratch → `ScratchBuffers`.
4. Per-solve transient state → `SolverWorkspace`.
5. Backward-pass scratch reused across iterations → field on `BackwardPassState`.
6. Forward-pass scratch reused across iterations → field on `ForwardPassState`.
7. Simulation scratch reused across scenarios → field on `SimulationState`.
8. Per-call input to backward/forward/simulate → field on the matching `*Inputs` bundle.
9. Lower-bound-specific aggregation scratch (no `ScratchBuffers` counterpart) → `LbEvalScratch`.
10. None of the above → create a new spec or bundle struct. Do NOT add a bare parameter.

### F7 — the runtime node graph's named home

The node-native engine's runtime node graph (node identity/canonical order,
the `node → pool` map, per-node Ω views/out-edges — `NodeGraph` in
`cobre-sddp/src/setup/node_graph.rs`) is **study-level, read-only** data
(rule 2 above), so it is a field on `TrainingContext`
(`node_graph: &'a NodeGraph`) rather than a new dedicated context struct — the
same shape as `stochastic`, `study_dims`, and `cut_state_layouts`, which are
already single struct-typed `TrainingContext` fields for cohesive,
study-level data. A new `NodeContext` struct was the other option the
decision tree offered; it was not taken because it would add a second
top-level context type threaded through every hot-path signature budget for a
single field, when `TrainingContext` already carries exactly this shape of
data. `SolveInputs` owns the graph as `pub node_graph: NodeGraph` (built in
`from_broadcast_params`, immediately after `build_scenario_libraries` — an
`External`-bound node's Ω addresses the standardized library's raw scenario
axis, so binding earlier would race the library's own standardization);
`stage_ctx`/`training_ctx`/`simulation_ctx` borrow it exactly like every other
sub-struct. Downstream tickets (backward cut aggregation, the lower bound,
forward traversal, simulation, the exact upper bound) consume
`training_ctx().node_graph` unchanged — no second access route, no loose
parameter threading. Discount is deliberately **not** part of `NodeGraph`; it
stays per-stage on `StageContext::cumulative_discount_factors`, reached
through a node's own `stage` field.

---

## StudySetup Sub-Structs

`StudySetup` owns all pre-computed study state. It holds `SolveInputs` — the
resolved inputs shared by the stage, training, and simulation contexts,
disjoint from `fcf` — plus `fcf` and a small number of bare residuals. Context
constructors (`stage_ctx`, `training_ctx`, `simulation_ctx`) are `SolveInputs`
methods; `StudySetup` delegates to them unchanged. `StageData`'s templates are
built by threading `crate::setup::resolve_lp_build_inputs`'s single resolution
of the LP builder's study inputs (`LpBuildInputs`) into `build_stage_templates`,
rather than the builder re-deriving them per stage. `resolve_stage_data`
(`setup/mod.rs`) resolves the bucket topology and role-(a) state layout
together through `resolve_state_and_topology`, then hands the resolved
`LpBuildInputs` to `build_postprocessed_templates`, which builds the stage
templates and runs the scaling/state-box postprocess in one step — the same
`resolve_state_and_topology` step the test-support wrapper
(`build_stage_templates_resolving_layout`) delegates through, so neither
duplicates the other's resolution sequence.

### Cohesive sub-structs

| Struct                | File                                            | Purpose                                                                                                                              | Visibility   | Storage form                |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------ | --------------------------- |
| `SolveInputs`          | `cobre-sddp/src/setup/solve_inputs.rs`          | Every input `stage_ctx`/`training_ctx`/`simulation_ctx` borrow from: `stage_data`, `stochastic`, `scenario_libraries`, `node_graph`, `initial`, `ncs`, `study_stage_ids`, `horizon`, `cut_management`, `cut_state_layouts` | `pub`        | Aggregated sub-struct       |
| `StageData`            | `cobre-sddp/src/setup/stage_data.rs`            | All stage-indexed data: templates, time value, indexer, stages, entity counts, blocks, lag transitions, noise groups, scaling report | `pub`        | Aggregated sub-struct       |
| `ScenarioLibraries`    | `cobre-sddp/src/setup/scenario_library_set.rs`  | Training + simulation `PhaseLibraries` pair                                                                                          | `pub`        | Aggregated sub-struct       |
| `PhaseLibraries`       | `cobre-sddp/src/setup/scenario_library_set.rs`  | Sampling schemes and optional libraries for one phase                                                                                | `pub`        | Aggregated sub-struct       |
| `InitialConditions`    | `cobre-sddp/src/setup/mod.rs`                   | Initial state vector + derived per-hydro PAR lag-slot/accumulator seeds                                                              | `pub(crate)` | Aggregated sub-struct       |
| `NcsEntityData`        | `cobre-sddp/src/setup/mod.rs`                   | Per-stage and per-slot NCS entity data: dense column map, commissioning windows, max gen, curtailment                                | `pub(crate)` | Aggregated sub-struct       |
| `NodeGraph`            | `cobre-sddp/src/setup/node_graph.rs`            | Runtime node graph: node identity/order, `node → pool` map, per-node Ω views/out-edges                                               | `pub`        | Aggregated sub-struct       |
| `LoopParams`           | `cobre-sddp/src/config.rs`                      | Pure-data projection of `LoopConfig` (excludes runtime-derived fields)                                                               | `pub`        | Projection of `LoopConfig`  |
| `SimulationConfig`     | `cobre-sddp/src/simulation/config.rs`           | `n_scenarios`, `io_channel_capacity`                                                                                                 | `pub`        | Literal reuse               |
| `CutManagementConfig`  | `cobre-sddp/src/config.rs`                      | Cut selection, budget cap, activity tolerance, warm-start cuts, per-stage risk measures                                              | `pub(crate)` | Literal reuse               |
| `EventParams`          | `cobre-sddp/src/config.rs`                      | Output-side event flags; excludes runtime handles                                                                                    | `pub(crate)` | Projection of `EventConfig` |

### Literal reuse vs projection

- **Literal reuse**: store the config type verbatim. Use when the config type
  owns exactly the right fields with no runtime handles and no per-invocation
  values (example: `SimulationConfig`, `CutManagementConfig`). Access path:
  `setup.simulation_config.field`.
- **Projection**: create a `*Params` sibling that drops runtime-bound or
  per-invocation fields. Use when 1–3 fields must be excluded (example:
  `LoopParams` drops `n_fwd_threads`; `EventParams` drops runtime handles).
- **New sub-struct**: introduce a dedicated type when no existing type
  cohesively covers the grouping (example: `SolveInputs`, `StageData`,
  `ScenarioLibraries`, `NodeGraph`).

### Accessor policy

`StudySetup` exposes a small impl surface: context builders (`stage_ctx`,
`training_ctx`, `simulation_ctx`) are `SolveInputs` methods `StudySetup`
delegates to unchanged, plus targeted mutation setters (`replace_fcf`,
`set_start_iteration`, `set_export_states`) and one typed read accessor
(`simulation_config`). Every other access uses direct field paths
(`setup.inputs.sub_struct.field` for a `SolveInputs`-held sub-struct,
`setup.sub_struct.field` otherwise). Do not add accessor methods for plain
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
   comm, frozen templates, iteration counters). Constructed fresh at each
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

| Function                         | Max args | Location                                | Notes                                             |
| -------------------------------- | -------- | --------------------------------------- | ------------------------------------------------- |
| `train`                          | 10       | `training/training.rs`                  | Public entry point; keep at or below target       |
| `TrainingSession::run_iteration` | 2        | `training/session/mod.rs`               | `&mut self` + iteration counter                   |
| `BackwardPassState::run`         | 2        | `training/backward_pass_state.rs`       | `&mut self` + `&mut BackwardPassInputs`           |
| `ForwardPassState::run`          | 2        | `training/forward_pass_state.rs`        | `&mut self` + `&mut ForwardPassInputs`            |
| `SimulationState::run`           | 2        | `simulation/state.rs`                   | `&mut self` + `&mut SimulationInputs`             |
| `evaluate_lower_bound`           | 7        | `training/lower_bound.rs`               | Takes `&StageContext`/`&TrainingContext` directly |
| `build_row_lower_unscaled`       | 8        | `simulation/pipeline.rs`                |                                                   |
| `run_forward_worker`             | 6        | `training/forward_pass_state.rs`        | Free function; accepts `&ForwardWorkerParams`     |
| `generate_opening_tree`          | 7        | `cobre-stochastic/src/tree/generate.rs` | Optional inputs go through `OpeningTreeInputs`    |

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

Every output file the CLI writes must also be written by the Python bindings —
the invariant CLAUDE.md's Python-parity hard rule owns (state the rule, not a
"currently none missing" snapshot). When adding a new output:

1. The CLI writes it via `write_training_outputs` / `write_simulation_outputs`
   in `crates/cobre-cli/src/commands/run/outputs.rs`.
2. Wire the same `cobre_io` write into the Python path — `run_via_study` /
   `run_training_phase_py` in `crates/cobre-python/src/run.rs` — so both surfaces
   emit the file. `cobre-python` is excluded from the Cargo workspace, so
   `cargo test --workspace` does not catch a missing mirror; check both paths by
   hand.
