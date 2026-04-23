# Cobre Architecture Rules — Hot Path & Context Structs

This file is re-injected into Claude Code's context after compaction and should
be read before modifying any hot-path code. These rules exist because of a
real regression: v0.1.1 had 25-argument functions that were painfully refactored
down to 7 in v0.1.3. The rules below prevent that from happening again.

---

## The Context Struct Pattern

When the SDDP training loop needs new data threaded through the
forward/backward/simulate call chain, **add a field to an existing context
struct** instead of adding a function parameter.

Available context structs:

| Struct             | File                                 | Purpose                                     | Mutable?         |
| ------------------ | ------------------------------------ | ------------------------------------------- | ---------------- |
| `StageContext`     | `cobre-sddp/src/context.rs`          | Per-stage templates, base rows, layout      | Immutable (`&`)  |
| `TrainingContext`  | `cobre-sddp/src/context.rs`          | Horizon, indexer, stochastic, initial state | Immutable (`&`)  |
| `ScratchBuffers`   | `cobre-sddp/src/workspace.rs`        | Per-worker noise/patch scratch space        | Mutable (`&mut`) |
| `SolverWorkspace`  | `cobre-sddp/src/workspace.rs`        | Solver + scratch + patch buffer             | Mutable (`&mut`) |
| `TrainingConfig`   | `cobre-sddp/src/training.rs`         | Forward passes, iteration limit, seed       | Owned (moved in) |
| `SimulationConfig` | `cobre-sddp/src/simulation/config.rs` | Scenario count, channel capacity, basis window | Immutable (`&`)  |
| `BackwardPassSpec` | `cobre-sddp/src/backward.rs`         | Risk measures, opening tree, cut selection  | Mutable (`&mut`) |
| `ForwardPassBatch` | `cobre-sddp/src/forward.rs`          | Local pass count, iteration, offset         | Immutable (`&`)  |
| `LbEvalSpec`       | `cobre-sddp/src/lower_bound.rs`      | Template, noise scale, opening tree         | Immutable (`&`)  |

**Decision tree when adding new data to the hot path:**

1. Is it per-stage, read-only, and shared across workers? → `StageContext`
2. Is it study-level, read-only? → `TrainingContext`
3. Is it per-worker mutable scratch? → `ScratchBuffers`
4. Is it per-solve transient state? → `SolverWorkspace`
5. Is it backward-pass-specific? → `BackwardPassSpec`
6. Is it lower-bound-specific? → `LbEvalSpec`
7. Does none of the above fit? → Create a new spec struct. Do NOT add a bare parameter.

---

## StudySetup Sub-Structs (Epic 02)

`StudySetup` owns all pre-computed study state. After Epic 02 decomposition it
has **16 top-level fields**: 7 cohesive sub-structs + 9 bare residuals.
Context constructors (`stage_ctx`, `training_ctx`, `simulation_ctx`) borrow
directly from these sub-structs.

### Cohesive sub-structs

| Struct               | File                                         | Purpose                                                                  | Visibility     | Reuse/Projection        |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------ | -------------- | ----------------------- |
| `StageData`          | `cobre-sddp/src/setup/stage_data.rs`         | All stage-indexed data: templates, indexer, stages, entity counts, blocks, lag transitions, noise groups, scaling report | `pub`          | New sub-struct          |
| `ScenarioLibraries`  | `cobre-sddp/src/setup/scenario_library_set.rs` | Training + simulation `PhaseLibraries` pair; eliminates 14 flat `sim_`-prefixed fields | `pub`      | New sub-struct          |
| `PhaseLibraries`     | `cobre-sddp/src/setup/scenario_library_set.rs` | Sampling schemes and optional libraries for one phase (training or simulation) | `pub`     | New sub-struct          |
| `MethodologyConfig`  | `cobre-sddp/src/setup/methodology_config.rs` | `horizon` + `inflow_method` — stochastic numerical methodology parameters | `pub(crate)`   | New sub-struct          |
| `LoopParams`         | `cobre-sddp/src/config.rs`                   | Five pure-data fields projected from `LoopConfig` (`seed`, `forward_passes`, `max_iterations`, `start_iteration`, `max_blocks`, `stopping_rules`); excludes `n_fwd_threads` (runtime-derived) | `pub` | Projection of `LoopConfig` |
| `SimulationConfig`   | `cobre-sddp/src/simulation/config.rs`        | `n_scenarios`, `io_channel_capacity`, `basis_activity_window` — stored **verbatim** (literal reuse) | `pub` | Literal reuse           |
| `CutManagementConfig` | `cobre-sddp/src/config.rs`                  | Cut selection, budget cap, activity tolerance, basis window, warm-start cuts, per-stage risk measures — stored **verbatim** (literal reuse) | `pub(crate)` | Literal reuse           |
| `EventParams`        | `cobre-sddp/src/config.rs`                   | `export_states` flag (output-side only); runtime handles excluded        | `pub(crate)`   | Projection of `EventConfig` |

### Literal-reuse vs projection distinction

- **Literal reuse**: `SimulationConfig` and `CutManagementConfig` are stored on
  `StudySetup` without modification. Access is `setup.simulation_config.field`
  or `setup.cut_management.field`.
- **Projection**: `LoopParams` drops `n_fwd_threads` from `LoopConfig` (runtime
  arg). `EventParams` drops runtime handles from `EventConfig`. These are
  data-only siblings of their config counterparts.

### Accessor collapse (Epic 02 result)

`StudySetup` shrank from 41 accessor methods to **8** after Epic 02:
`replace_fcf`, `set_start_iteration`, `set_export_states`, `set_budget`,
`simulation_config` (read), `stage_ctx`, `training_ctx`, `simulation_ctx`.
All other access uses direct field paths (`setup.sub_struct.field`).

---

## Function Signature Budgets

| Function                   | Max args | Current | Location                 | Status    |
| -------------------------- | -------- | ------- | ------------------------ | --------- |
| `run_forward_pass`         | 8        | 8       | `forward.rs`             | at budget |
| `run_backward_pass`        | 8        | 8       | `backward.rs`            | at budget |
| `simulate`                 | 8        | 8       | `simulation/pipeline.rs` | at budget |
| `train`                    | 10       | 10      | `training.rs`            | at target |
| `evaluate_lower_bound`     | 9        | 9       | `lower_bound.rs`         | at budget |
| `build_row_lower_unscaled` | 8        | 8       | `simulation/pipeline.rs` | at budget |

If a function exceeds its budget, the correct response is to refactor, not to
add `#[allow(clippy::too_many_arguments)]`.

---

## Common Mistakes to Avoid

**Adding `cut_batches: &mut [RowBatch]` as a parameter.** This is per-stage
workspace state. It belongs in `SolverWorkspace` or a `TrainingWorkspace` struct.

**Adding `stage_bases: &[Option<Basis>]` as a parameter to simulate.** This is
study-level read-only data produced by training. It belongs on `StudySetup` or
in a `SimulationSpec` struct.

**Adding LP scaling buffers as parameters.** Row scale factors, column scale
factors, and unscale buffers are per-stage data. They belong in `StageContext`
or attached to the stage template.

**Threading generic constraint data through 4 levels of calls.** Generic
constraint metadata is per-stage and read-only. It belongs in `StageContext`.

---

## Python Parity Checklist

When adding a new output file in the CLI (`write_outputs` in `run.rs`):

1. Does `run_inner()` in `cobre-python/src/run.rs` write the same file?
2. If not, add it. The Python path should call the same `cobre_io` write function.

Current gaps (to be fixed):

- None — all CLI output writes are mirrored in Python.
