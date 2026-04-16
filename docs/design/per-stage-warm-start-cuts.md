# Design: Per-Stage Warm-Start Counts and Terminal-Stage Boundary Conditions

## Implementation Status

**IMPLEMENTED** — v0.4.4, branch `feat/tier1-tier2-correctness-and-performance`

| Component                                                                                                | Status |
| -------------------------------------------------------------------------------------------------------- | ------ |
| API: `warm_start_counts: &[u32]` replaces scalar `warm_start_count`                                      | DONE   |
| Checkpoint field: `warm_start_counts: Vec<u32>` with backward-compat fallback (empty vector → all zeros) | DONE   |
| Terminal theta variable when `warm_start_counts[T-1] > 0`                                                | DONE   |
| Warm-start cut sentinel: `iteration = u64::MAX` (WARM_START_ITERATION)                                   | DONE   |
| Horizon mode constraint: boundary cuts only valid for `HorizonMode::Finite`                              | DONE   |

---

## Context

`FutureCostFunction::new` currently accepts a single `warm_start_count: u32` parameter
and allocates every stage's `CutPool` with the same capacity:

```text
capacity = warm_start_count + max_iterations * forward_passes
```

This uniform allocation is incorrect for checkpoint resume. A FlatBuffers policy
checkpoint stores cuts per stage but does not record how many warm-start cuts each
stage contains. Early stages accumulate more cuts than later stages during training
because the backward pass visits each stage the same number of times but stage 1 has
had more updates applied. When training resumes from a checkpoint, every stage pool
must be pre-allocated with its actual warm-start count, not a shared estimate.

A second problem is structural: `HorizonMode::Finite` treats the terminal stage `T`
as having no theta variable. The training loop skips cut generation at stage `T`, and
the LP at stage `T` contains no future cost approximation. This is correct for the
standard finite-horizon SDDP formulation. However, DECOMP-style formulations can
represent the "end-of-world" cost beyond the planning horizon with boundary cuts that
act as a static linear approximation of the recourse beyond stage `T`. These boundary
cuts are loaded from a checkpoint, not generated during training. Without a theta
variable at stage `T`, the LP cannot incorporate them.

Both problems share the same fix: generalize `FutureCostFunction::new` to accept a
per-stage warm-start count and allocate each pool independently.

## Decision

### 1. API change to `FutureCostFunction::new`

Replace `warm_start_count: u32` with `warm_start_counts: &[u32]`, a slice of length
`num_stages`. Each stage's `CutPool` is allocated with its own capacity:

```text
capacity[stage] = warm_start_counts[stage] + max_iterations * forward_passes
```

The uniform case is expressed as `vec![count; num_stages]` at the call site. The
`CutPool` constructor signature is unchanged; the change is in how `FutureCostFunction`
passes the per-stage count to each pool.

A `debug_assert!` at construction time verifies that `warm_start_counts.len() ==
num_stages`.

### 2. FlatBuffers schema addition for per-stage cut counts

The policy checkpoint format (FlatBuffers) is extended with a new header field:

```
warm_start_counts: [uint32]  // per-stage count; length == num_stages
```

This field has a FlatBuffers default of an empty vector. Old checkpoint files that
omit the field are read as if every stage has `warm_start_count = 0`, which is the
current behavior. The extension is therefore backward-compatible: new code reading
old files produces the same result as old code reading old files.

When writing a checkpoint, the serializer populates `warm_start_counts[stage]` with
the active cut count of that stage's pool at the time of writing. On resume, the
checkpoint reader extracts this vector and passes it directly to
`FutureCostFunction::new`, so each pool is sized for exactly the existing cuts plus
the new cuts that training will generate.

### 3. Terminal-stage theta variable for boundary conditions

When `warm_start_counts[T-1] > 0` (the terminal stage has boundary cuts), the LP
construction at stage `T` includes a theta variable with the boundary cuts as rows,
identical in structure to the theta variable added at non-terminal stages. The LP
at stage `T` then has a future cost contribution from the boundary cuts.

The backward pass continues to skip cut generation at stage `T` during training.
Boundary cuts are static: they are loaded once from the checkpoint and never
updated. This preserves the standard SDDP property that the terminal stage generates
no new cuts, while allowing a pre-loaded cost-to-go approximation to be present.

When `warm_start_counts[T-1] == 0` (the default case), stage `T` behaves exactly as
it does today: no theta variable, no LP rows for cuts.

### 4. Warm-start cut identification

Warm-start cuts (including boundary cuts) are inserted with `iteration = u64::MAX`.
This sentinel distinguishes them from training-generated cuts, which use 0-based
iteration counters. Cut selection strategies can query the iteration counter to
identify warm-start cuts and apply a separate pruning policy. Whether warm-start cuts
are exempt from pruning is a configuration option of the cut selection strategy; it
is not hardcoded in `FutureCostFunction` or `CutPool`.

### 5. Horizon mode constraint

The boundary condition concept (static cuts at the terminal stage) applies only to
`HorizonMode::Finite`. The deferred `Cyclic` variant has different terminal stage
semantics (no unique terminal stage exists; the horizon wraps). If boundary cuts are
requested with `HorizonMode::Cyclic`, the behavior is undefined and the caller is
responsible for not providing a non-zero `warm_start_counts[T-1]` in that context.
This constraint is documented in the `FutureCostFunction::new` API docs when the
change is implemented.

## Consequences

### Benefits

- **Accurate checkpoint resume**: each stage's pool is sized exactly for the cuts it
  will actually hold, avoiding over-allocation at early stages and potential capacity
  exhaustion if uniform sizing is too small.
- **DECOMP boundary condition modeling**: the terminal stage can carry a static cost
  approximation loaded from an external source, enabling interoperability with
  DECOMP-style decompositions that pre-compute end-of-world costs.
- **Backward-compatible schema evolution**: old checkpoint files continue to work
  without any migration step; new files carry the per-stage counts explicitly.
- **Cut selection flexibility**: the `u64::MAX` sentinel for warm-start cuts exposes
  policy control to cut selection strategies without coupling the decision to the
  pool's internal allocation logic.

### Costs and trade-offs

- **API complexity**: callers must now construct and pass a `Vec<u32>` of length
  `num_stages` rather than a single scalar. The uniform case adds one call site
  allocation (`vec![count; num_stages]`) that did not previously exist.
- **Schema migration responsibility**: the checkpoint writer must be updated to
  populate `warm_start_counts` before the accurate resume guarantee holds. Existing
  checkpoints written by v0.1.1 code will resume with all warm-start counts set to 0
  (capacity undersized by the actual warm-start count), which is tolerable for
  correctness but causes a runtime capacity assertion failure in debug builds if a
  warm-start cut is inserted beyond the computed capacity.
- **Terminal theta complexity**: LP construction at the terminal stage now has a
  conditional path (with/without theta). The condition is a simple integer comparison
  but it must be exercised in tests to avoid the zero-default case masking bugs.
- **Horizon mode coupling**: the boundary condition feature silently does nothing for
  `HorizonMode::Cyclic`. This is acceptable for v0.1.2 because `Cyclic` is deferred,
  but the coupling must be revisited when `Cyclic` is implemented.
