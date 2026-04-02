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
| `SimulationConfig` | `cobre-sddp/src/simulation/types.rs` | Scenario count, channel capacity            | Immutable (`&`)  |
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

## Clippy Suppression Policy

**`#[allow(clippy::too_many_arguments)]` is a signal, not a solution.**

When clippy fires `too_many_arguments` on production code (code before
`#[cfg(test)]`), the correct action is:

1. Identify which parameters belong together
2. Find the appropriate context struct from the table above
3. Add the parameters as fields on that struct
4. If no existing struct fits, create a new named struct
5. Only suppress the lint if the function genuinely needs all parameters
   independently (e.g., generic constructors with unrelated fields)

**Never suppress the lint on hot-path functions** (`forward.rs`, `backward.rs`,
`training.rs`, `simulation/pipeline.rs`, `lower_bound.rs`). These functions are
called millions of times; their signatures are the public contract of the
training loop.

Run `python3 scripts/check_suppressions.py --check too_many_arguments --max 10`
before committing. The count must never increase. The target is 0.

---

## Function Length Suppressions

Current baseline: **57** production `too_many_lines` suppressions.
Target: reduce through function splitting. The count must never increase.

Run `python3 scripts/check_suppressions.py --check too_many_lines --max 57`
before committing changes to `crates/`.

Combined check (both lints at once):

```
python3 scripts/check_suppressions.py --check too_many_arguments --check too_many_lines --max 74
```

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
