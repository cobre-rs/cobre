# ADR-010: Complete Tree Work Distribution

**Status:** Accepted
**Date:** 2026-03-13
**Spec reference:** DEC-009 (per-stage allgatherv for cut synchronization)

## Context

The current SDDP forward pass uses a "full-trajectory" execution model: the `M`
forward passes are statically partitioned across MPI ranks (each rank gets
`M / n_ranks` scenarios), and each scenario traverses all `T` stages sequentially
before the next scenario begins. This model works well for the InSample sampling
scheme because every trajectory is independent — the outcome at stage `t` of one
scenario does not affect any other scenario.

Complete tree execution is a different execution mode in which all scenario paths
through the opening tree are enumerated rather than sampled. At each stage `t`, the
tree has `B^t` nodes (where `B` is the branching factor). Stage 1 has `B` nodes,
stage 2 has `B^2`, and so on. The full tree has `sum(B^t for t in 1..=T)` nodes
total. Complete tree enumeration produces an exact representation of the stochastic
program for small problems, making it valuable for three use cases:

1. **Exact solution** — for problems small enough that the full tree fits in memory,
   complete tree forward passes remove the sampling approximation error entirely.
2. **Verification baseline** — against which sampling-based results can be validated;
   if InSample converges to the same policy as the complete tree solution, the sample
   size is sufficient.
3. **Academic benchmarking** — published SDDP results sometimes use complete tree
   enumeration on toy problems; this mode enables reproduction.

The existing full-trajectory model cannot be applied to complete tree execution
without modification. In full-trajectory mode, the assignment of scenarios to ranks
is fixed for the entire forward pass. In complete tree mode, the number of active
nodes at stage `t` is `B^t`, which varies across stages. A rank assigned scenarios
`[0, M/n_ranks)` in full-trajectory terms would need to know which stage-`t` nodes
fall within its partition — but the per-stage node count changes every stage, so
a single static partition cannot cover the whole tree.

The `OpeningTree` data structure already supports variable `openings_per_stage`,
meaning the branching factor can differ across stages. The work distribution
design must handle this general case.

## Decision

Complete tree execution uses a stage-by-stage execution model. The forward pass
iterates over stages `t = 1, 2, ..., T` in order. At each stage, all `N_t = B^t`
nodes for that stage are distributed across ranks and solved. After all nodes at
stage `t` are solved, ranks synchronize and exchange outgoing states before
proceeding to stage `t+1`.

**Work distribution formula.** At stage `t` with `N_t` nodes, node `i`
(zero-indexed) is assigned to rank `i % n_ranks`. Each rank solves at most
`ceil(N_t / n_ranks)` nodes. Load imbalance per stage is bounded by `1 / n_ranks`
(at most one extra node on the largest rank relative to the smallest).

**Within-rank parallelism.** Within each rank at each stage, the assigned nodes are
distributed across Rayon workers exactly as scenarios are distributed in the existing
forward pass — via static partitioning across `SolverWorkspace` instances from the
`WorkspacePool`. The per-stage parallel region replaces the per-trajectory parallel
region; the warm-start basis indexing scheme is adapted to use `(node_index, stage)`
rather than `(scenario, stage)`.

**State exchange.** After solving all nodes at stage `t`, each rank holds the
outgoing states for its assigned subset of nodes. An `allgatherv` (as established
by DEC-009 for cut synchronization) collects all outgoing states so that every rank
has the complete state vector for stage `t`. This is required because node `j` at
stage `t+1` derives its incoming state from its parent node at stage `t`, and any
rank may be assigned any node at stage `t+1`. The state vector transmitted per
node is the `n_state`-dimensional outgoing state (storage levels and inflow lags).
Total bytes exchanged at stage `t`:

```
state_buffer_bytes(t) = n_state × N_t × 8
```

where `n_state` is the number of state variables and `8` is the byte width of
`f64`. With `B = 5`, `T = 10`, and `n_state = 20`, stage 10 requires:

```
20 × 5^10 × 8 = 20 × 9,765,625 × 8 ≈ 1.5 GB
```

Stage 5 requires `20 × 5^5 × 8 = 20 × 3,125 × 8 ≈ 500 KB`, which is manageable.
The memory footprint grows by a factor of `B` each stage.

**Maximum tree size cap.** Because `N_t` grows exponentially, complete tree mode
includes a `max_tree_nodes` configuration parameter (integer, default `1_000_000`)
that caps the total number of enumerated nodes across all stages. Before the forward
pass begins, the runtime computes `sum(N_t for t in 1..=T)` and returns an error if
it exceeds `max_tree_nodes`. Complete tree mode is intended for small problems
(`B <= 5`, `T <= 10`) or short verification runs; for `B = 20` and `T = 120` the
full tree has `20^120` nodes, which is physically impossible to enumerate and must
be rejected early.

**Configuration.** Complete tree mode is selected via `execution_mode: "complete_tree"`
in `config.json`. When not set, the default `execution_mode: "sampling"` retains
the existing full-trajectory InSample model. The backward pass is not affected by
this setting — it always uses the opening tree as defined.

**Interaction with cut selection.** In complete tree mode, every leaf node
contributes one cut per iteration, giving `B^T` cuts at the leaf stage. This can
produce a much larger cut pool per iteration than the sampling-based forward pass
(which contributes `M` cuts). The existing `CutSelectionStrategy` is applied
unchanged; the interaction with large per-iteration cut volumes is noted here but
a strategy-specific response (if needed) is deferred to a future ADR.

## Consequences

**Benefits:**

- Complete tree forward passes produce an exact representation of the stochastic
  program for small problems, eliminating sampling approximation error entirely.
- The exact solution provides a deterministic verification baseline: if InSample
  forward passes converge to the same policy as the complete tree, the sample size
  is demonstrably sufficient for that problem.
- The stage-by-stage execution model with `allgatherv` state exchange is a natural
  extension of the existing DEC-009 communication pattern, reusing the same
  `allgatherv` infrastructure already present for cut synchronization.
- Load imbalance per stage is bounded by `1 / n_ranks` regardless of stage depth,
  which is better balance than a naive static trajectory partition for uneven
  branching factors.

**Costs:**

- The `allgatherv` state buffer at stage `t` is `n_state × B^t × 8` bytes and
  grows by a factor of `B` per stage. For problems with `B = 5` and `T = 10`,
  peak stage-10 memory reaches approximately 1.5 GB per rank. Complete tree mode
  is not viable for production-scale problems.
- Communication volume per iteration is `sum(n_state × B^t × 8 for t in 1..=T)`,
  which dominates the per-stage barrier synchronization cost. For deep trees, the
  MPI collective at late stages transfers gigabytes per rank per iteration.
- The per-iteration cut count is `B^T` at the leaf stage, potentially orders of
  magnitude larger than the `M` cuts produced per iteration by InSample forward
  passes. Cut pool management overhead grows accordingly.
- Implementation complexity is higher than the sampling-based forward pass:
  the basis store indexing, trajectory record layout, and upper bound summation
  logic must all be adapted for stage-by-stage execution with a variable number
  of active nodes per stage.
