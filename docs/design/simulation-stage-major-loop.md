# Design: Simulation Stage-Major Loop

**Related**: This design applies the same stage-major pattern described in
[Work-Stealing Parallelism](work-stealing-parallelism.md), Options A and B
(Forward Pass appendix). The training forward pass already uses stage-major
ordering. The simulation pipeline does not, creating redundant LP setup calls.
Unlike the training forward pass, simulation does not require `UnsafeCell` or
`SlottedArray` because results are dispatched through a channel rather than
written to shared record arrays, and `BasisStore` writes are read-only
(per-stage warm-start from training).

## Context

The simulation pipeline currently uses a scenario-major loop structure: for each
scenario, iterate over all stages, calling `load_model + add_rows` at every
`(scenario, stage)` pair. The training forward pass, by contrast, uses a
stage-major structure: each worker loads the LP once per stage, then processes
all assigned scenarios by patching bounds only.

With `S` scenarios per worker and `T` stages, the simulation makes `S * T`
LP setup calls (`load_model + add_rows`), while training makes only `T`. For
typical production runs (400 scenarios, 120 stages), simulation performs 48,000
LP setups versus 120 in training. Since the cut batches are stage-constant
(pre-built once in `simulate` before the loop), the per-scenario rebuild is
entirely redundant.

### Current Code

In `crates/cobre-sddp/src/simulation/pipeline.rs`:

- `simulate` distributes scenarios across workspaces using rayon `par_iter_mut`.
  Each worker calls `process_scenario_stages` per scenario.
- `process_scenario_stages` iterates over all stages for one scenario. At each
  stage, it calls `solve_simulation_stage`.
- `solve_simulation_stage` calls `ws.solver.load_model(&ctx.templates[t])` and
  `ws.solver.add_rows(cut_batch)` for every `(scenario, stage)` pair.

The cut batch per stage is pre-built once:

```rust
let cut_batches: Vec<RowBatch> = (0..num_stages)
    .map(|t| build_cut_row_batch(fcf, t, indexer, &ctx.templates[t].col_scale))
    .collect();
```

These batches are immutable during simulation.

## Proposed Change

Refactor the simulation loop from scenario-major to stage-major:

```
Current (scenario-major):          Proposed (stage-major):
for scenario in 0..S:              for stage in 0..T:
    for stage in 0..T:                 load_model(stage)        // once
        load_model(stage)              add_rows(cut_batch)      // once
        add_rows(cut_batch)            for scenario in 0..S:
        patch_bounds(scenario)             patch_bounds(scenario)
        solve()                            solve()
```

### LP Setup Reduction

| Metric                        | Scenario-Major | Stage-Major         |
| ----------------------------- | -------------- | ------------------- |
| `load_model` calls per worker | `S * T`        | `T`                 |
| `add_rows` calls per worker   | `S * T`        | `T`                 |
| Bound patches per worker      | `S * T`        | `S * T` (unchanged) |
| LP solves per worker          | `S * T`        | `S * T` (unchanged) |

For 400 scenarios and 120 stages per worker: **47,880 fewer LP setup calls**.

### Design Details

1. **Outer loop**: stages `t = 0..num_stages`. Each worker loads the LP once
   per stage (`load_model + add_rows`).

2. **Inner loop**: scenarios within each worker. Only bounds are patched per
   scenario (same as training forward pass).

3. **Scenario assignment**: static partitioning via `partition()`, identical at
   every stage. Each worker processes the same scenario set throughout.

4. **Per-scenario state**: each workspace maintains a `Vec<Vec<f64>>` of
   per-scenario state vectors (indexed by local scenario offset). At each stage,
   the worker restores a scenario's state, patches bounds, solves, and stores
   the updated state back. This adds memory proportional to
   `scenarios_per_worker * n_state * 8` bytes.

5. **Noise sampling**: must occur per `(scenario, stage)` in the same order as
   the current implementation. The `ForwardSampler` is stateless and
   thread-safe.

6. **Warm-start basis**: applied per `(worker, stage)`, not per scenario.

7. **Result dispatch**: scenarios are dispatched as complete
   `SimulationScenarioResult` after all stages finish, not after each stage.
   Progress events are emitted after the final stage for each scenario.

### Invariants Preserved

- Bit-identical per-scenario costs and stage results.
- Identical scenario-to-worker assignment.
- Identical noise sampling sequence.
- Identical warm-start basis application.

### Risks and Considerations

- **Per-scenario state management**: the main complexity. Each workspace must
  track `current_state` and `lag_state` per scenario across stages. Off-by-one
  errors in state indexing would produce silent numerical divergence (caught by
  deterministic regression tests).

- **Solver statistics attribution**: with scenario-major, per-scenario stats are
  captured by snapshotting before/after each scenario's full stage loop. With
  stage-major, the snapshot boundaries change. Per-scenario stats would need
  per-scenario solve counters or accept approximate attribution.

- **Error propagation**: if a scenario fails at stage `t`, subsequent stages
  must skip that scenario. A per-scenario error flag (`Vec<Option<Error>>`)
  tracks this.

- **Memory**: per-scenario state buffers add `O(scenarios_per_worker * n_state)`
  memory. For 400 scenarios and 50-dimensional state, this is ~160KB per
  worker -- negligible.

## Decision

**Status**: Proposed (not yet implemented).

The stage-major refactor is a clear win for LP setup cost reduction. The
per-scenario state management adds complexity but follows the same pattern as
the training forward pass. Implementation should target
`crates/cobre-sddp/src/simulation/pipeline.rs` only.

### Prerequisites

- All deterministic regression tests (D01-D26) must pass with bit-identical
  results before and after the refactor.
- The `solve_simulation_stage` function should be split: LP setup moves to the
  stage-level outer loop, and a new `patch_and_solve` helper handles the
  per-scenario inner loop.
