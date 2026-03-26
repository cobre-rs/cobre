# Design: Simulation Robustness and Output Resilience

## Context

Running a large production case (158 hydros, 118 stages, 500 scenarios, multiple
MPI ranks) exposed a fatal error during simulation result gathering:

```
error: simulation result gather error: invalid buffer size for 'allgatherv':
       expected 2147483647 elements, got 6651345374
```

Training completed successfully (50 iterations, 1h34m). Simulation completed
successfully on all ranks (500 scenarios solved). But the post-simulation
`gather_simulation_results` call serializes every rank's full
`Vec<SimulationScenarioResult>` into bytes and exchanges them via a single MPI
`allgatherv`. The total serialized payload (6.65 GB) exceeds MPI's `i32::MAX`
(2.15 GB) limit for message counts, causing `to_i32_vec` in
`cobre-comm/src/ferrompi.rs:359` to reject the conversion.

Because this error propagates with `?` from line 537 of `run.rs`, `write_outputs`
at line 601 is never reached. All training outputs (policy checkpoint, convergence
records, solver stats, cut selection) are lost despite being fully computed and
available in memory.

Three independent problems are exposed:

1. **Architectural flaw in simulation output**: the `gather_simulation_results`
   pattern centralises all detailed per-scenario results on rank 0 via MPI,
   which is both unnecessary (Parquet uses Hive partitioning by scenario) and
   unscalable (memory and i32 overflow).

2. **All-or-nothing output pipeline**: a late-stage failure (simulation gather,
   Parquet write) loses all training outputs. Training outputs have no
   dependency on simulation results and should be written independently.

3. **Training failure loses partial results**: if an LP solve fails mid-training
   (e.g., infeasibility at iteration 35 of 50), the `train()` function returns
   `Err` immediately and the CLI never writes the checkpoint for the 34
   completed iterations.

## Decision

### 1. Distributed Parquet Writing (eliminate the gather)

#### Current flow (broken)

```
simulate() on each rank
    → local_results: Vec<SimulationScenarioResult>
    → gather_simulation_results(&comm, &local_results)   ← allgatherv of ALL bytes
        → rank 0 gets all_results: Vec<SimulationScenarioResult>
    → rank 0: SimulationParquetWriter writes all scenarios
    → rank 0: write_outputs()
```

All detailed results are serialised, gathered to every rank (allgatherv, not
even gatherv), and then only rank 0 writes them. This is O(ranks × scenarios ×
stages × entities) in both memory and bandwidth. For the production case, the
serialised payload exceeds 6 GB.

#### New flow

```
simulate() on each rank
    → drain thread collects local Vec<SimulationScenarioResult>
    → each rank creates SimulationParquetWriter(output_dir, system, config)
    → each rank writes its own scenarios via write_scenario()
    → each rank calls finalize() → local SimulationOutput

aggregate_simulation(&local_costs, &config, &comm)   ← scalar costs only (~24 KB)
    → SimulationSummary (mean, std, CVaR, category stats)

merge_simulation_outputs(&comm, local_sim_output)     ← small metadata only
    → rank 0 gets merged SimulationOutput
```

**Why this works without coordination:**

- Hive partitioning layout is `simulation/{entity}/scenario_id=XXXX/data.parquet`
- Scenarios are pre-assigned to ranks by `assign_scenarios()` — no overlapping
  IDs across ranks
- Each rank writes to disjoint partition directories on the shared filesystem
- `SimulationParquetWriter::write_scenario` creates the partition directory and
  writes atomically — concurrent calls for different scenario IDs are safe
- POSIX guarantees that `mkdir` + `create` on different paths within the same
  parent directory are safe from concurrent processes (no lock needed)

**Metadata merge:**

After all ranks finish writing, each rank has a local `SimulationOutput` with:

- `n_scenarios`: count of scenarios this rank wrote
- `completed`: count successfully written
- `failed`: count of write failures
- `total_time_ms`: local wall-clock time
- `partitions_written`: Vec of relative paths this rank wrote

Rank 0 needs the merged view for the `write_results` summary. This requires:

- `allreduce(Sum)` on `n_scenarios`, `completed`, `failed` (3 × u32 → small)
- `allreduce(Max)` on `total_time_ms` (wall-clock = max across ranks)
- `allgatherv` on partition path strings — each rank contributes its local paths

The partition paths are the largest piece, but for 500 scenarios × ~10 entity
types = 5,000 short strings (~200 KB total). No overflow risk.

**Alternative considered:** only rank 0 tracks partition paths. Non-root ranks
don't need them. However, maintaining the current allgatherv approach for this
small metadata keeps the code symmetric and simple. If the partition path list
becomes problematic at extreme scale (millions of scenarios), rank 0 can
reconstruct it from the filesystem by scanning the output directory.

#### Changes required

| Crate          | File                              | Change                                                                                                                                                                              |
| -------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-cli`    | `src/commands/run.rs`             | Remove `gather_simulation_results()`. Each rank creates its own `SimulationParquetWriter`, writes local results, calls `finalize()`. Add `merge_simulation_outputs()` for metadata. |
| `cobre-cli`    | `src/commands/run.rs`             | The existing `drain_handle` thread that collects `Vec<SimulationScenarioResult>` feeds directly into the local writer instead of being serialised for MPI.                          |
| `cobre-python` | `src/run.rs`                      | No change needed — already writes locally with `LocalBackend`.                                                                                                                      |
| `cobre-sddp`   | `src/simulation/aggregation.rs`   | No change — `aggregate_simulation` only gathers scalar costs, well within i32 limits.                                                                                               |
| `cobre-io`     | `src/output/simulation_writer.rs` | No structural change. `SimulationParquetWriter` already supports being instantiated independently. Verify thread safety documentation.                                              |
| `cobre-io`     | `src/output/mod.rs`               | `SimulationOutput` may need a `merge()` or `From<Vec<SimulationOutput>>` method for combining per-rank outputs.                                                                     |

#### Output directory ownership

With distributed writing, the simulation subdirectories (`simulation/hydros/`,
etc.) must exist before any rank writes. Two strategies:

**A. Rank 0 pre-creates directories, barrier, then all ranks write.**

This matches the current `SimulationParquetWriter::new()` which creates the
entity-type directories. Rank 0 calls `new()` first, barrier, then other ranks
call `new()` (which becomes a no-op for directory creation since they already
exist).

**B. Every rank calls `create_dir_all` independently.**

`create_dir_all` is idempotent — races on the same directory are harmless. This
is simpler and avoids the barrier. The only risk is a race where rank N tries
to create `simulation/hydros/scenario_id=0042/` before rank 0 has created
`simulation/hydros/`, but `create_dir_all` handles intermediate directories
atomically.

**Decision:** use strategy B (every rank calls `create_dir_all` independently).
It is simpler, avoids a synchronisation point, and `create_dir_all` is
idempotent by contract. The only requirement is that `output_dir` itself exists,
which is already guaranteed by the CLI before entering `execute()`.

### 2. Training Resilience (partial checkpoint on failure)

#### Current flow (broken)

```rust
let training_result = setup.train(...)? ;   // Err → immediate return, no outputs
let training_output = setup.build_training_output(&training_result, &events);
// ... write_outputs() never reached on error
```

The `train()` function in `training.rs` uses `?` on every LP error, returning
`Err(SddpError)` immediately. The accumulated state (convergence records,
solver stats, basis cache) from completed iterations is discarded.

#### New flow

Introduce a `TrainingOutcome` type:

```rust
/// Result of a training run that always carries partial results.
///
/// When training completes normally, `error` is `None` and `result` contains
/// the full training statistics. When training fails mid-iteration, `error`
/// carries the failure cause and `result` contains statistics from all
/// fully completed iterations (the failing iteration is excluded).
pub struct TrainingOutcome {
    /// Training result from completed iterations. Always populated, even when
    /// `error` is `Some` — in that case, `result.iterations` reflects only
    /// the iterations that completed without error.
    pub result: TrainingResult,

    /// If training was interrupted by an error, the cause. `None` when
    /// training completed normally (convergence, iteration limit, or
    /// time limit).
    pub error: Option<SddpError>,
}
```

The training loop changes from:

```rust
let fwd_result = run_forward_pass(...)?;  // error → return Err immediately
```

to:

```rust
let fwd_result = match run_forward_pass(...) {
    Ok(r) => r,
    Err(e) => {
        // Build partial TrainingResult from completed iterations
        return Ok(TrainingOutcome {
            result: build_partial_result(completed_state),
            error: Some(e),
        });
    }
};
```

The same pattern applies to `run_backward_pass`, `evaluate_lower_bound`, and
any other fallible call within the iteration loop.

#### CLI handling

```rust
let outcome = setup.train(...);
let training_output = setup.build_training_output(&outcome.result, &events);

// Always write training outputs — they reflect completed iterations.
write_training_outputs(&WriteTrainingArgs { ... })?;

if let Some(training_error) = &outcome.error {
    // Log the error and report it to the user.
    tracing::error!("training failed after {} iterations: {training_error}", outcome.result.iterations);
    // Print what was saved so the user knows outputs are available.
    print_partial_training_notice(stderr, &output_dir, outcome.result.iterations);
    // Skip simulation — cannot simulate without full training.
    return Err(CliError::from(training_error));
}

// Proceed to simulation only if training completed fully.
```

#### Scope of `TrainingOutcome`

The `TrainingOutcome` struct lives in `cobre-sddp` alongside `TrainingResult`.
The `train()` function signature changes from:

```rust
pub fn train(...) -> Result<TrainingResult, SddpError>
```

to:

```rust
pub fn train(...) -> Result<TrainingOutcome, SddpError>
```

The outer `Result` is reserved for truly unrecoverable errors that prevent
even building a partial result (e.g., MPI communicator failure during
initialisation, before any iteration starts). Mid-iteration LP failures
return `Ok(TrainingOutcome { error: Some(...) })`.

#### Changes required

| Crate          | File                  | Change                                                                                                                                                   |
| -------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-sddp`   | `src/training.rs`     | Add `TrainingOutcome`. Refactor iteration loop to catch errors and return partial result. Snapshot convergence state at end of each completed iteration. |
| `cobre-sddp`   | `src/setup.rs`        | Update `train()` delegation to return `TrainingOutcome`.                                                                                                 |
| `cobre-sddp`   | `src/lib.rs`          | Export `TrainingOutcome`.                                                                                                                                |
| `cobre-cli`    | `src/commands/run.rs` | Handle `TrainingOutcome.error`. Always write training outputs.                                                                                           |
| `cobre-python` | `src/run.rs`          | Handle `TrainingOutcome.error`. Always write training outputs. Report partial completion to Python caller.                                               |

### 3. Output Ordering (training outputs first)

#### Current flow

```
train() → simulate() → gather_simulation_results() → write_outputs(training + simulation)
```

Training and simulation outputs are written together in a single `write_outputs`
call. If simulation gathering fails, training outputs are never written.

#### New flow

```
train()
  → write_training_outputs()         ← policy, convergence, solver stats, cut selection
  → (training outputs are now safe on disk)
simulate()
  → each rank writes simulation Parquet locally
  → aggregate_simulation()           ← scalar cost summary
  → merge_simulation_outputs()       ← metadata merge
  → write_simulation_summary()       ← simulation summary JSON, results.json update
```

#### Split `write_outputs` into two phases

**Phase 1 — `write_training_outputs()`** (runs immediately after training, before simulation):

- Policy checkpoint (`write_checkpoint`)
- Training convergence records (Parquet via `TrainingParquetWriter`)
- Training solver stats (`write_solver_stats`)
- Cut selection records (`write_cut_selection_records`)
- Entity dictionaries (`write_dictionaries`)
- Timing summary

**Phase 2 — `write_simulation_summary()`** (runs after distributed Parquet write + metadata merge):

- Update `results.json` with simulation summary (already exists from Phase 1, append simulation section)
- Simulation solver stats (`write_simulation_solver_stats`)

The Parquet files for individual simulation scenarios are already written by
each rank during Phase 2's distributed write — they don't go through this
function.

#### `write_results` refactoring

The current `write_results` function in `cobre-io/src/output/mod.rs` writes
both training and simulation data in one call. It needs to be split or made
tolerant of being called twice:

**Option A:** Split into `write_training_results()` and `write_simulation_results()`.

**Option B:** Keep `write_results` but make the simulation section idempotent —
call it once with `sim_output: None` after training, call it again with
`sim_output: Some(...)` after simulation to update the simulation section.

**Decision:** Option A. Clean separation, no idempotency reasoning required.
The `results.json` file can be written in Phase 1 with training-only data, then
updated in Phase 2 to add the simulation section. The `write_training_results`
function writes the initial version; `write_simulation_results` reads the
existing file and patches in the simulation data (or creates a new one if it
doesn't exist).

#### Changes required

| Crate          | File                  | Change                                                                                                                                                                                 |
| -------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cobre-io`     | `src/output/mod.rs`   | Split `write_results` into `write_training_results()` and `write_simulation_results()`. Keep `write_results` as a convenience wrapper that calls both.                                 |
| `cobre-cli`    | `src/commands/run.rs` | Split `write_outputs` into `write_training_outputs` (called after training) and `write_simulation_outputs` (called after simulation). Remove the `gather_simulation_results` function. |
| `cobre-python` | `src/run.rs`          | Mirror the split: write training outputs immediately after training, write simulation outputs after simulation.                                                                        |

## Implementation Order

The three changes have dependencies:

```
                    ┌───────────────────────────┐
                    │ 2. TrainingOutcome         │
                    │    (cobre-sddp/training.rs)│
                    └─────────┬─────────────────┘
                              │ uses
                              ▼
┌──────────────────────────────────────────────────┐
│ 3. Output ordering                                │
│    Split write_outputs, write training first       │
│    (cobre-io/mod.rs, cobre-cli/run.rs)            │
└─────────┬────────────────────────────────────────┘
          │ enables
          ▼
┌──────────────────────────────────────────────────┐
│ 1. Distributed Parquet writing                    │
│    Remove gather, each rank writes locally         │
│    (cobre-cli/run.rs, cobre-io/mod.rs)            │
└──────────────────────────────────────────────────┘
```

**Phase A** — `TrainingOutcome` in `cobre-sddp`:

- Define `TrainingOutcome` struct.
- Refactor `train()` iteration loop to catch mid-iteration errors.
- Snapshot convergence state at end of each completed iteration.
- Update `setup.rs` delegation.
- Update exports.
- Tests: simulate LP failure at iteration N, verify partial result has N-1
  iterations of data.

**Phase B** — Output split in `cobre-io` and `cobre-cli`:

- Split `write_results` → `write_training_results` + `write_simulation_results`.
- Split `write_outputs` → `write_training_outputs` + `write_simulation_outputs`.
- Move training output writing to immediately after training.
- Handle `TrainingOutcome.error` in CLI and Python.
- Tests: verify training outputs exist after simulated training failure.

**Phase C** — Distributed simulation writing in `cobre-cli`:

- Remove `gather_simulation_results()` entirely.
- Each rank creates `SimulationParquetWriter` and writes local scenarios.
- Add `merge_simulation_outputs()` for metadata (allreduce on counts,
  allgatherv on partition paths).
- Wire up `write_simulation_summary()` on rank 0 after merge.
- Tests: verify multi-rank scenario writing produces correct Hive partitions
  (unit test with `LocalBackend` simulating single-rank; integration test
  with actual MPI if CI supports it).

**Phase D** — Python parity:

- Update `cobre-python/src/run.rs` to match the split output flow.
- Write training outputs before simulation.
- Handle `TrainingOutcome.error` and surface partial results to Python.

## Testing Strategy

### Unit tests (no MPI)

- **`TrainingOutcome` partial result**: mock a training loop that fails at
  iteration N; verify `outcome.result.iterations == N-1`, solver stats log
  has N-1 entries, convergence records cover N-1 iterations.

- **`SimulationParquetWriter` concurrent safety**: spawn two writers pointing
  at the same output directory with disjoint scenario IDs; verify both produce
  correct Hive partitions without error.

- **`write_training_results` / `write_simulation_results` independence**: call
  `write_training_results` alone; verify training outputs are complete. Then
  call `write_simulation_results`; verify simulation data is appended without
  corrupting training data.

- **`merge_simulation_outputs`**: merge two `SimulationOutput` structs; verify
  counts are summed, time is max, partition paths are concatenated.

### Integration tests (with MPI, if CI supports)

- **Distributed Parquet write**: 2+ ranks each write scenarios; verify the
  output directory contains all expected Hive partitions with correct data.

- **End-to-end resilience**: training succeeds, simulation Parquet write
  partially fails on one rank; verify training outputs and successful
  simulation partitions survive.

### Regression tests

- **D01–D16 deterministic cases**: must remain bit-for-bit identical. The
  output layout changes (training outputs written earlier), but content must
  not change.

## Risk Assessment

| Risk                                                                                    | Likelihood | Impact         | Mitigation                                                                                                                                                                           |
| --------------------------------------------------------------------------------------- | ---------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Filesystem race in concurrent `create_dir_all`                                          | Low        | Low            | POSIX guarantees idempotency; no data corruption possible                                                                                                                            |
| NFS/Lustre metadata contention from many ranks creating directories simultaneously      | Medium     | Medium         | `SimulationParquetWriter::new()` creates entity-type dirs once; only `scenario_id=XXXX` leaf dirs are created per-scenario. Leaf creation is spread over time as scenarios complete. |
| `results.json` update race if Phase 2 runs before Phase 1 completes on slow filesystems | Low        | Low            | Phase 2 only runs after Phase 1 returns on rank 0; barrier ensures ordering.                                                                                                         |
| `TrainingOutcome` changes the `train()` return type, breaking downstream                | Certain    | Low            | Mechanical refactor — all callers are in-tree (`cobre-cli`, `cobre-python`, tests).                                                                                                  |
| Partial training checkpoint produces a valid but incomplete policy                      | Certain    | None (feature) | The policy is valid for the iterations completed. User can resume training from it. Document this in output.                                                                         |
