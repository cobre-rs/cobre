# Dominated Cut Selection — Implementation Plan

## Context

The `CutSelectionStrategy::Dominated` variant has been wired since v0.1.2 but
its `select_for_stage` method is a stub returning an empty `DeactivationSet`.
This document specifies the full implementation: a visited states archive,
the domination algorithm, rayon parallelisation across stages, and FlatBuffers
serialization for checkpoint/resume.

**Mathematical definition** (from `cobre-docs/specs/math/cut-management.md` SS7.3):

Cut _k_ at stage _t_ is dominated iff at every visited forward-pass state
x_hat, some other active cut achieves a strictly higher value (within
tolerance epsilon):

```
  max_{j != k} { alpha_j + pi_j' * x_hat } - (alpha_k + pi_k' * x_hat) > epsilon
    for ALL x_hat in visited_states
```

Equivalently: if cut _k_ achieves the global maximum at _any_ visited state
(within epsilon), it is NOT dominated and is retained.

---

## Design Decisions

| #   | Decision                                                   | Rationale                                                                                                                            |
| --- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **Accumulate ALL visited states** (unbounded)              | Consistent with how cuts are stored — active + inactive persist for output. Windowed forgetting is a future optimisation.            |
| 2   | **Break `select_for_stage` signature**                     | Dominated is the most complex strategy; the API will stabilise after this change.                                                    |
| 3   | **Per-state evaluation with early-exit** (not full matrix) | O(cuts x states x n_state) with candidate-set pruning. Full matrix approach (O(cuts x states) memory) is ~800 MB per stage at scale. |
| 4   | **Rayon across stages** (not MPI distribution)             | 8-16x speedup for ~10 lines of change. MPI distribution (spec SS2.2a) deferred.                                                      |
| 5   | **Treat states like cuts** for serialization               | Same FlatBuffers pattern, same `policy/` subdirectory structure (`states/stage_NNN.bin`), same checkpoint/resume lifecycle.          |

---

## Algorithm

Per stage (executed in parallel across stages via rayon):

```
Input: pool (CutPool), visited_states (flat &[f64]), threshold (f64)
Output: DeactivationSet

populated    = pool.populated_count
n_state      = pool.state_dimension
n_candidates = pool.active_count()

is_candidate = pool.active[..populated].clone()   -- bitwise copy
scratch      = vec![0.0; populated]                -- reused per state

for each state x_hat in visited_states (chunks of n_state):

    -- Step 1: compute all active cut values, track global max
    max_val = -inf
    for k in 0..populated where pool.active[k]:
        val = pool.intercepts[k] + dot(pool.coefficients[k], x_hat)
        scratch[k] = val
        max_val = max(max_val, val)

    -- Step 2: remove non-dominated cuts from candidates
    cutoff = max_val - threshold
    for k in 0..populated where is_candidate[k]:
        if scratch[k] >= cutoff:
            is_candidate[k] = false
            n_candidates -= 1

    -- Step 3: early exit
    if n_candidates == 0: break

-- Collect remaining candidates
indices = [k as u32 for k in 0..populated where is_candidate[k]]
return DeactivationSet { stage_index, indices }
```

**Complexity:**

- Per state: O(active_cuts x n_state) for dot products + O(populated) for candidate check
- Total worst case: O(active_cuts x visited_states x n_state) per stage
- Early exit: terminates when all candidates are cleared (typically after a small fraction of states)

**Memory per stage:** O(populated) for `scratch` + `is_candidate`. Allocated once per rayon worker via `map_init`.

---

## Components

### 1. VisitedStatesArchive (`crates/cobre-sddp/src/visited_states.rs`)

New module. Per-stage flat buffer storing all forward-pass trial points
accumulated across iterations.

```rust
/// Single-stage visited states buffer.
pub struct StageStates {
    /// Flat buffer: data[i * state_dimension .. (i+1) * state_dimension]
    data: Vec<f64>,
    /// Number of states stored.
    count: usize,
    /// State dimension.
    state_dimension: usize,
}

/// Multi-stage archive, one StageStates per stage.
pub struct VisitedStatesArchive {
    stages: Vec<StageStates>,
}
```

**Pre-allocation**: `capacity = max_iterations * total_forward_passes` per stage.
Memory: `capacity * n_state * 8 bytes` per stage. At production scale
(50 iter x 200 fwd x 2080 x 8 = 166 MB/stage x 60 stages = ~10 GB).
Comparable to the cut pool memory.

**Methods**:

- `new(num_stages, n_state, max_iterations, total_forward_passes)` — pre-allocate
- `archive_gathered_states(stage, gathered: &[f64], total_fwd)` — append one
  iteration's states for a stage from the exchange buffer
- `states_for_stage(stage) -> &[f64]` — flat slice of all accumulated states
- `count(stage) -> usize` — number of states at a stage
- `from_deserialized(stage_results)` — load from policy checkpoint
- `new_with_preloaded(stage_results, max_iterations, total_fwd)` — warm-start

**Integration point**: Inside `run_backward_pass`, after each per-stage
`exchange.exchange()` call (backward.rs ~line 642). The gathered buffer
(`exchange.gathered_states()`) contains all ranks' states for stage _t_.
Copy into `archive.stages[t]`.

**Conditional creation**: Only created when `CutSelectionStrategy::Dominated`
is active. Passed as `Option<&mut VisitedStatesArchive>` to the backward pass.

### 2. Signature Change for `select_for_stage`

**Before:**

```rust
pub fn select_for_stage(
    &self,
    metadata: &[CutMetadata],
    active: &[bool],
    current_iteration: u64,
    stage_index: u32,
) -> DeactivationSet
```

**After:**

```rust
pub fn select_for_stage(
    &self,
    pool: &CutPool,
    visited_states: &[f64],   // flat, len = count * pool.state_dimension
    current_iteration: u64,
    stage_index: u32,
) -> DeactivationSet
```

`CutPool` already contains `metadata`, `active`, `coefficients`, `intercepts`,
`populated_count`, and `state_dimension`. Level1/LML1 access only
`pool.metadata` and `pool.active`; they ignore `visited_states`.
Dominated accesses all fields.

The caller passes `&[]` for `visited_states` when using Level1/LML1, or
the actual states slice from the archive when using Dominated.

The convenience `select` method (used in doc-tests) is updated to accept
`&CutPool` + `&[f64]` as well.

### 3. Rayon Parallelisation

**Current** (training.rs ~line 598): serial loop over stages 1..num_sel_stages.

**New**: Replace with `rayon::par_iter` + `map_init` for per-thread scratch buffers.

```rust
use rayon::prelude::*;

let deactivation_sets: Vec<DeactivationSet> = (1..num_sel_stages)
    .into_par_iter()
    .map_init(
        || vec![0.0f64; max_pool_capacity],
        |scratch, stage| {
            let pool = &fcf.pools[stage];
            let states = archive.states_for_stage(stage);
            strategy.select_for_stage_with_scratch(
                pool, states, iteration, stage as u32, scratch,
            )
        },
    )
    .collect();

// Apply deactivations sequentially (needs &mut fcf)
for deact in &deactivation_sets {
    let stage = deact.stage_index as usize;
    fcf.pools[stage].deactivate(&deact.indices);
    // ... emit per-stage StageSelectionRecord
}
```

Stage 0 remains exempt (no binding activity tracked). The scratch buffer
(`Vec<f64>`) is allocated once per rayon worker thread via `map_init`.

Note: for Level1/LML1, the scratch buffer is unused but the allocation is
harmless (~80 KB at populated_count = 10,000) and `map_init` only creates
one per thread.

**Alternative**: An internal `select_for_stage` method that allocates its own
scratch buffer (simpler signature, tiny overhead given check_frequency
amortisation). We can choose during implementation based on which reads
cleaner.

### 4. FlatBuffers Serialization for Visited States

Follow the established cuts pattern in `crates/cobre-io/src/output/policy.rs`.

**Schema (StageStates table):**

```
table StageStates:
  stage_id:        uint32    (field offset 4)
  state_dimension: uint32    (field offset 6)
  count:           uint32    (field offset 8)
  data:            [double]  (field offset 10)  -- flat, len = count * state_dimension
```

**Directory layout:**

```
policy/
  metadata.json          (written LAST)
  cuts/
    stage_000.bin
    ...
  basis/
    stage_000.bin
    ...
  states/                <-- NEW
    stage_000.bin
    stage_001.bin
    ...
```

**Functions to add** (in `cobre-io/src/output/policy.rs`):

- `serialize_stage_states(stage_id, state_dimension, data) -> Vec<u8>`
- `deserialize_stage_states(buf) -> Result<StageStatesReadResult, OutputError>`

**Functions to update**:

- `write_policy_checkpoint` — create `states/` dir, write per-stage files
- `read_policy_checkpoint` — read `states/stage_NNN.bin` files, return in `PolicyCheckpoint`
- `PolicyCheckpoint` struct — add `states: Vec<StageStatesReadResult>` field
- `PolicyCheckpointMetadata` — add `total_visited_states: u64` field

**Backward compatibility**: `read_policy_checkpoint` handles missing `states/`
directory gracefully (older checkpoints have no states). `VisitedStatesArchive`
starts empty when loading a checkpoint without states.

### 5. Policy Export Bridge (`crates/cobre-sddp/src/policy_export.rs`)

New function:

```rust
pub fn build_stage_states_payloads(archive: &VisitedStatesArchive) -> Vec<StageStatesPayload>
```

Borrows the flat data from each stage in the archive (no copy), packages
into payload structs for the serializer.

### 6. Training Loop Integration (`crates/cobre-sddp/src/training.rs`)

**At initialisation** (~line 291):

```rust
let mut visited_archive = match &cut_selection {
    Some(CutSelectionStrategy::Dominated { .. }) =>
        Some(VisitedStatesArchive::new(num_stages, n_state, max_iterations, total_forward_passes)),
    _ => None,
};
```

**Backward pass call** (~line 507): pass `visited_archive.as_mut()`.

**Cut selection block** (~line 574): replace serial loop with rayon parallel
dispatch. Pass `&archive` (immutable borrow) to each `select_for_stage` call
in the parallel section.

### 7. Backward Pass Integration (`crates/cobre-sddp/src/backward.rs`)

**Signature change**: Add `visited_archive: Option<&mut VisitedStatesArchive>`
to `run_backward_pass` (or to `BackwardPassSpec`).

**Per-stage archiving** (~line 642, after exchange):

```rust
if let Some(archive) = visited_archive.as_deref_mut() {
    let gathered = spec.exchange.gathered_states();
    let total_fwd = spec.exchange.total_scenarios();
    archive.archive_gathered_states(t, gathered, total_fwd);
}
```

### 8. CLI and Python Parity

Both the CLI (`crates/cobre-cli/src/commands/run.rs`) and Python bindings
(`crates/cobre-python/src/run.rs`) must write visited states when the
policy checkpoint is written.

- Pass the `VisitedStatesArchive` (or its payloads) to
  `write_policy_checkpoint`
- Gate on `exports.states` config flag (already exists, defaults to `true`)
- Loading: `read_policy_checkpoint` returns states alongside cuts and basis;
  warm-start path feeds them into `VisitedStatesArchive::new_with_preloaded`

---

## File Changes Summary

| File                                      | Change                                                                                                                |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `crates/cobre-sddp/src/visited_states.rs` | **NEW** — `StageStates`, `VisitedStatesArchive`                                                                       |
| `crates/cobre-sddp/src/lib.rs`            | Add `mod visited_states;` + re-export                                                                                 |
| `crates/cobre-sddp/src/cut_selection.rs`  | New signature for `select_for_stage`, implement Dominated algorithm, update `select` wrapper, update tests            |
| `crates/cobre-sddp/src/training.rs`       | Create archive, pass to backward, rayon-parallel cut selection                                                        |
| `crates/cobre-sddp/src/backward.rs`       | Accept archive in `BackwardPassSpec`, archive states after exchange                                                   |
| `crates/cobre-io/src/output/policy.rs`    | FlatBuffers schema + serialize/deserialize for `StageStates`, update write/read checkpoint, update `PolicyCheckpoint` |
| `crates/cobre-sddp/src/policy_export.rs`  | `build_stage_states_payloads`                                                                                         |
| `crates/cobre-cli/src/commands/run.rs`    | Wire states to policy write                                                                                           |
| `crates/cobre-python/src/run.rs`          | Wire states to policy write                                                                                           |

---

## Implementation Order

1. **VisitedStatesArchive** struct and unit tests (standalone, no integration)
2. **`select_for_stage` signature change** — update Level1/LML1 to use `&CutPool`, fix all call sites and tests
3. **Dominated algorithm** — implement in `select_for_stage`, unit tests from conformance spec (SS1.3)
4. **Backward pass integration** — archive states after exchange, pass-through in `BackwardPassSpec`
5. **Training loop integration** — conditional archive creation, rayon-parallel cut selection
6. **FlatBuffers serialization** — `serialize_stage_states` / `deserialize_stage_states`, update checkpoint
7. **CLI + Python parity** — wire states in both write paths
8. **Integration test** — deterministic case with `"domination"` method, verify deactivation counts

Steps 1-3 are the algorithmic core. Steps 4-5 are the training integration.
Steps 6-7 are the persistence layer. Step 8 is end-to-end validation.

---

## Testing

### Unit tests (cut_selection.rs)

From `cobre-docs/specs/architecture/cut-selection-testing.md` SS1.3:

| Test                                           | Fixture                                                   | Expected                    |
| ---------------------------------------------- | --------------------------------------------------------- | --------------------------- |
| `dominated_select_deactivate_dominated`        | 5 cuts, 3 states (1D). Cuts 0,3,4 dominated at all states | DeactivationSet = {0, 3, 4} |
| `dominated_select_partial_domination_retained` | Cut 0 dominated at 2/3 states, tied at 1                  | Cut 0 NOT deactivated       |
| `dominated_select_none_dominated`              | 3 cuts, each achieves max at >= 1 state                   | Only cut 2 deactivated      |

### Aggressiveness ordering test (SS2)

| Level1                     | LML1                         | Dominated                    |
| -------------------------- | ---------------------------- | ---------------------------- |
| deactivates {1,4} (2 cuts) | deactivates {1,2,4} (3 cuts) | deactivates {0,3,4} (3 cuts) |

Verify `|Level1| <= |LML1| <= |Dominated|`.

### VisitedStatesArchive unit tests

- Construction and pre-allocation
- `archive_gathered_states` accumulates correctly across iterations
- `states_for_stage` returns correct flat slice
- `count` tracks per-stage state count
- Round-trip through FlatBuffers serialization

### Integration test

Add a deterministic regression case (e.g., D26) that runs with
`"cut_selection": { "enabled": true, "method": "domination", "threshold": 0, "check_frequency": 5 }`
and asserts specific deactivation counts and convergence behaviour.

---

## Verification

```bash
# 1. All existing tests pass (signature change doesn't break anything)
cargo test --workspace --all-features

# 2. Clippy clean
cargo clippy --workspace --all-targets --all-features -- -D warnings

# 3. Format
cargo fmt --all

# 4. Pre-commit check (suppression count)
python3 scripts/check_suppressions.py --max 10

# 5. Run an example with domination enabled
cargo run --release -p cobre-cli -- run examples/1dtoy \
    --forward-passes 5 --max-iterations 20 \
    --cut-selection-method domination --cut-selection-frequency 5

# 6. Verify policy/states/ directory exists with per-stage .bin files
ls examples/1dtoy/output/policy/states/

# 7. Warm-start from the checkpoint (round-trip test)
cargo run --release -p cobre-cli -- run examples/1dtoy \
    --forward-passes 5 --max-iterations 10 \
    --warm-start examples/1dtoy/output/policy \
    --cut-selection-method domination --cut-selection-frequency 5
```
