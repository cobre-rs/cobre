# Cut Selection Strategy Trait

## Purpose

This spec defines the `CutSelectionStrategy` abstraction -- the enum-based trait through which the SDDP training loop decides which cuts to deactivate from the per-stage cut pools during training. Cut selection controls the growth of the piecewise-linear outer approximation $\underline{V}_t$ by periodically deactivating cuts that are no longer contributing to the approximation quality at visited states, thereby bounding LP solve time and memory consumption. The three supported strategies -- Level-1, Limited Memory Level-1 (LML1), and Dominated -- correspond to the three selection methods defined in [Cut Management SS7](../math/cut-management.md) and implemented in [Cut Management Implementation SS2](./cut-management-impl.md). Because the cut selection strategy is a global property of the training run (one strategy for all stages), the abstraction uses enum dispatch, consistent with the pattern established in [HorizonMode SS4](./horizon-mode-trait.md) and [SamplingScheme SS4](./sampling-scheme-trait.md).

> **Convention: Rust traits as specification guidelines.** The Rust trait definitions, method signatures, and struct declarations throughout this specification corpus serve as _guidelines for implementation_, not as absolute source-of-truth contracts that must be reproduced verbatim. Their purpose is twofold: (1) to express behavioral contracts, preconditions, postconditions, and type-level invariants more precisely than prose alone, and (2) to anchor conformance test suites that verify backend interchangeability (see [Backend Testing §1](../hpc/backend-testing.md)). Implementation may diverge in naming, parameter ordering, error representation, or internal organization when practical considerations demand it -- provided the behavioral contracts and conformance tests continue to pass. When a trait signature and a prose description conflict, the prose description (which captures the domain intent) takes precedence; the conflict should be resolved by updating the trait signature. This convention applies to all trait-bearing specification documents in `src/specs/`.

## 1. Trait Definition

The cut selection strategy is modeled as a flat enum with three variants, matching the three strategies supported by Cobre ([Cut Management SS7](../math/cut-management.md)):

```rust
/// Cut selection strategy for controlling cut pool growth during SDDP training.
///
/// A single `CutSelectionStrategy` value is resolved from the `cut_selection`
/// field in `config.json` during configuration loading (see Extension Points SS6).
/// The strategy is global to the training run -- all stages use the same
/// selection method. Selection runs periodically (every `check_frequency`
/// iterations) and only deactivates cuts -- it never deletes them, preserving
/// slot indices for reproducibility.
#[derive(Debug, Clone)]
pub enum CutSelectionStrategy {
    /// Level-1 selection: retain any cut that was ever binding.
    ///
    /// A cut is deactivated only if its cumulative active count is zero --
    /// i.e., it has never been binding at any visited state during the entire
    /// algorithm execution. This is the least aggressive strategy.
    ///
    /// See [Cut Management SS7.1](../math/cut-management.md).
    Level1 {
        /// Activity count threshold. A cut is deactivated when
        /// `active_count <= threshold`. Typical value: 0 (deactivate only
        /// cuts that have never been binding).
        threshold: u64,

        /// Number of iterations between selection runs.
        check_frequency: u64,
    },

    /// Limited Memory Level-1 (LML1): retain cuts active within a recent window.
    ///
    /// Each cut is timestamped with the most recent iteration at which it was
    /// binding. Cuts whose timestamp is older than `memory_window` iterations
    /// are deactivated. More aggressive than Level-1 because cuts that were
    /// active early but are now permanently dominated will eventually be removed.
    ///
    /// See [Cut Management SS7.2](../math/cut-management.md).
    Lml1 {
        /// Number of iterations to retain inactive cuts before deactivation.
        /// A cut whose `last_active_iter` is older than
        /// `current_iteration - memory_window` is deactivated.
        memory_window: u64,

        /// Number of iterations between selection runs.
        check_frequency: u64,
    },

    /// Dominated cut detection: remove cuts dominated at all visited states.
    ///
    /// A cut k is dominated if at every visited state x_hat, some other active
    /// cut achieves a higher (or equal within threshold) value. This is the
    /// most aggressive strategy and the most computationally expensive:
    /// O(|active cuts| x |visited states|) per stage per check.
    ///
    /// See [Cut Management SS7.3](../math/cut-management.md).
    Dominated {
        /// Activity threshold epsilon for near-binding tolerance.
        threshold: f64,

        /// Number of iterations between selection runs.
        check_frequency: u64,
    },
}
```

## 2. Method Contracts

### 2.1 should_run

`should_run` determines whether cut selection should execute at the current iteration. Selection runs periodically based on `check_frequency` to amortize the cost of scanning the cut pool. The method is called once per iteration, after the backward pass completes and before the next forward pass begins.

```rust
impl CutSelectionStrategy {
    /// Determine whether cut selection should run at this iteration.
    ///
    /// Returns `true` if the current iteration is a multiple of the
    /// variant's `check_frequency`, indicating that the cut pool should
    /// be scanned for deactivation candidates.
    pub fn should_run(&self, iteration: u64) -> bool {
        let freq = match self {
            Self::Level1 { check_frequency, .. } => *check_frequency,
            Self::Lml1 { check_frequency, .. } => *check_frequency,
            Self::Dominated { check_frequency, .. } => *check_frequency,
        };
        iteration > 0 && iteration % freq == 0
    }
}
```

**Preconditions:**

| Condition        | Description                     |
| ---------------- | ------------------------------- |
| `iteration >= 0` | Valid iteration index (0-based) |

**Postconditions:**

| Condition                                                              | Description                                |
| ---------------------------------------------------------------------- | ------------------------------------------ |
| Returns `false` when `iteration == 0`                                  | Never run selection before any cuts exist  |
| Returns `true` iff `iteration > 0 && iteration % check_frequency == 0` | Periodic triggering at configured interval |

**Infallibility:** This method does not return `Result`. The `check_frequency` value is validated at configuration load time to be positive (see SS5).

### 2.2 select

`select` is the primary method that scans the cut pool for a single stage and returns the set of cut indices to deactivate. The method operates on a single stage and is invoked in parallel across all stages: stages are distributed across MPI ranks via static contiguous block assignment (the same formula used for forward pass trajectory distribution — see [Work Distribution §3.1](../hpc/work-distribution.md)), and threads work-steal stages within each rank's block. After all ranks complete their assigned stages, an `allgatherv` gathers the per-stage `DeactivationSet` results so that every rank has the complete deactivation picture. The leader rank then applies the deactivations to the SharedRegion StageLpCache. See SS2.2a for the full work distribution model.

```rust
impl CutSelectionStrategy {
    /// Scan the cut pool for a single stage and identify cuts to deactivate.
    ///
    /// Convenience wrapper that delegates to `select_for_stage` with
    /// `stage_index = 0`. Prefer `select_for_stage` in production code
    /// where the stage index must be propagated into the DeactivationSet.
    pub fn select(
        &self,
        pool: &CutPool,
        visited_states: &[f64],
        current_iteration: u64,
    ) -> DeactivationSet {
        self.select_for_stage(pool, visited_states, current_iteration, 0)
    }

    /// Scan the cut pool for a specific stage and identify cuts to deactivate.
    ///
    /// Returns the indices of cuts that should be deactivated according to
    /// the configured selection strategy, wrapped in a `DeactivationSet`
    /// tagged with `stage_index`. The caller is responsible for applying
    /// the deactivation to the activity bitmap.
    ///
    /// The method does NOT modify the cut pool -- it is a pure query that
    /// returns a deactivation set. This separation allows the training loop
    /// to log deactivation counts before applying them.
    ///
    /// `visited_states` is a flat `&[f64]` of visited forward-pass state
    /// vectors (row-major, one state per `pool.state_dimension` elements).
    /// Pass `&[]` when using Level1 or Lml1 (they read only metadata).
    pub fn select_for_stage(
        &self,
        pool: &CutPool,
        visited_states: &[f64],
        current_iteration: u64,
        stage_index: u32,
    ) -> DeactivationSet {
        // Dispatch on variant:
        // - Level1: deactivate cuts with active_count <= threshold
        // - Lml1: deactivate cuts with current_iteration - last_active_iter > memory_window
        // - Dominated: call select_dominated (see SS6.4)
        todo!()
    }
}
```

**Preconditions (all variants):**

| Condition                               | Description                                                                               |
| --------------------------------------- | ----------------------------------------------------------------------------------------- |
| `cut_pool.populated_count() > 0`        | At least one cut exists in the pool                                                       |
| `should_run(iteration)` returned `true` | Selection is only invoked when the periodic check fires                                   |
| Activity tracking data is up-to-date    | `update_activity` has been called for all binding cuts from the most recent backward pass |

**Postconditions (all variants):**

| Condition                                                     | Description                                                          |
| ------------------------------------------------------------- | -------------------------------------------------------------------- |
| Returned indices are a subset of currently active cut indices | Never deactivates already-inactive cuts                              |
| Returned indices do not include cuts added in this iteration  | Newly generated cuts are never candidates for immediate deactivation |
| Returned set may be empty                                     | If no cuts meet the deactivation criteria, an empty set is returned  |

**Variant-specific behavioral contracts:**

**Level1:**

| Postcondition                                      | Description                                                                                          |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Deactivates cuts where `active_count <= threshold` | A cut whose cumulative binding count is at or below the threshold is deactivated                     |
| Retains all cuts with `active_count > threshold`   | Cuts with sufficient binding activity are preserved, regardless of how long ago they were last bound |

**Lml1:**

| Postcondition                                                                 | Description                                                     |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Deactivates cuts where `current_iteration - last_active_iter > memory_window` | Cuts not active within the recent memory window are deactivated |
| Retains cuts where `current_iteration - last_active_iter <= memory_window`    | Recently active cuts are preserved                              |
| When `memory_window` is very large, behavior approaches Level1                | A sufficiently large window retains all ever-active cuts        |

**Dominated:**

| Postcondition                                                                                                          | Description                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Deactivates cut $k$ only if dominated at **all** visited states                                                        | $\forall \hat{x} \in \text{visited states}: \max_{j \neq k} \{ \alpha_j + \pi_j^\top \hat{x} \} - (\alpha_k + \pi_k^\top \hat{x}) > \epsilon$ |
| A cut dominating at only some visited states is retained                                                               | Partial domination is not sufficient for deactivation                                                                                         |
| The domination check uses only currently active cuts                                                                   | Inactive cuts do not participate in the domination comparison                                                                                 |
| Computational cost is $\mathcal{O}(\lvert\text{active cuts}\rvert \times \lvert\text{visited states}\rvert)$ per stage | The most expensive strategy; cost is amortized by `check_frequency`                                                                           |

**Infallibility:** This method does not return `Result`. The cut pool is guaranteed to be in a valid state because it is initialized at startup and modified only through the deterministic slot assignment protocol ([Cut Management Implementation SS1.2](./cut-management-impl.md)). Visited states are guaranteed to exist because `should_run` returns `false` at iteration 0 (before any forward pass has produced trial points).

### 2.2a Work Distribution for Cut Selection

> **Decision [DEC-016](../overview/decision-log.md#dec-016) (active):** Cut selection uses deferred parallel execution — stages distributed across ranks and threads, with DeactivationSet allgatherv and leader-only SharedRegion write.

This subsection documents the parallel calling convention for `select`. Cut selection is embarrassingly parallel across stages because each stage's selection decision depends only on that stage's cut pool metadata and visited states — data that is already synchronized across all ranks by the backward pass per-stage `allgatherv` ([Synchronization §1.4](../hpc/synchronization.md)).

**Why inputs are already synchronized.** The backward pass synchronizes cuts at every stage boundary via `allgatherv` ([Cut Management Implementation SS4.1](./cut-management-impl.md)). After the backward pass completes, every rank holds an identical copy of each stage's cut pool (same coefficients, same metadata, same activity bitmap). The `update_activity` calls during the backward pass update the metadata that `select` reads. Because all ranks process all received cuts identically, the metadata is consistent across ranks at the point where cut selection runs.

**Stage partitioning formula.** Stages $\{2, 3, \ldots, T\}$ are distributed across $R$ ranks using static contiguous block assignment:

| Parameter                           | Formula                                                        |
| ----------------------------------- | -------------------------------------------------------------- |
| Total stages eligible for selection | $S = T - 1$ (stages 2 through $T$; stage 1 has no FCF)         |
| Block size                          | $B = \lceil S / R \rceil$                                      |
| Rank $r$'s first stage              | $s_{\text{first}}(r) = 2 + r \cdot B$                          |
| Rank $r$'s last stage               | $s_{\text{last}}(r) = \min(s_{\text{first}}(r) + B - 1, \; T)$ |

This mirrors the forward pass trajectory distribution formula from [Work Distribution §3.1](../hpc/work-distribution.md), applied to stages instead of trajectories.

**Within-rank threading model.** Each rank distributes its assigned stages across threads using Rayon's work-stealing pool. The `select` call for each stage is independent, so no synchronization is needed between threads within a rank. Each thread reads the stage's cut pool metadata (shared, read-only) and produces a `DeactivationSet` (thread-local output).

**Result gathering.** After all ranks complete their assigned stages, an `allgatherv` ([Communicator Trait SS2.1](../hpc/communicator-trait.md)) gathers the per-stage `DeactivationSet` payloads so that every rank has the full set of deactivations across all stages. The wire format for `DeactivationSet` is specified in [Synchronization §1.4a](../hpc/synchronization.md). The leader rank then applies all deactivations to the SharedRegion StageLpCache (see [Cut Management Implementation SS7.1b](./cut-management-impl.md)).

**Conditional execution.** The entire parallel selection phase — partitioning, parallel `select`, `allgatherv`, and StageLpCache update — only runs when `should_run(iteration)` returns `true` (SS2.1). On non-selection iterations, no stage distribution or communication occurs, and the iteration proceeds directly from backward pass to convergence check.

### 2.3 update_activity

`update_activity` records that a specific cut was binding at the current LP solution. This method is called during the backward pass, after each LP solve, for every cut whose dual multiplier is positive (indicating the cut is binding). The method updates the per-cut tracking metadata that `select` later reads to make deactivation decisions.

```rust
impl CutSelectionStrategy {
    /// Update tracking metadata for a cut after an LP solution.
    ///
    /// Called during the backward pass for every cut whose dual multiplier
    /// is inspected. When `is_binding` is `true` (dual exceeds the solver
    /// tolerance), the metadata is updated according to the active strategy.
    /// When `is_binding` is `false`, the metadata is not modified.
    pub fn update_activity(
        &self,
        metadata: &mut CutMetadata,
        is_binding: bool,
        current_iteration: u64,
    ) {
        if !is_binding {
            return;
        }

        match self {
            Self::Level1 { .. } => {
                metadata.active_count += 1;
            }
            Self::Lml1 { .. } => {
                metadata.last_active_iter = current_iteration;
            }
            Self::Dominated { .. } => {
                metadata.domination_count = 0;
            }
        }
    }
}
```

**Preconditions:**

| Condition                                             | Description                                                                |
| ----------------------------------------------------- | -------------------------------------------------------------------------- |
| `is_binding` reflects the dual multiplier comparison  | `true` when the cut's dual exceeds the solver tolerance, `false` otherwise |
| `current_iteration` is the current training iteration | Timestamp correctness for Lml1                                             |

**Postconditions:**

| Condition                                                | Description                                                                             |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| When `is_binding == false`, metadata is unchanged        | The method returns immediately without modifying any field                              |
| Level1, `is_binding == true`: `active_count` incremented | Monotonically increasing counter; once positive, the cut is never deactivated by Level1 |
| Lml1, `is_binding == true`: `last_active_iter` updated   | Timestamp refreshed to `current_iteration`; the cut's retention window restarts         |
| Dominated, `is_binding == true`: `domination_count` = 0  | The cut is not dominated at this state; reset counter                                   |

**Thread safety:** `update_activity` is called on per-thread solver workspaces during the backward pass. Each thread processes its own trajectories and updates the metadata for the cuts it evaluated. No locking is needed because the per-stage synchronization barrier ensures that all threads have finished their updates before `select` reads the metadata. See [Cut Management Implementation SS6.2](./cut-management-impl.md).

**Infallibility:** This method does not return `Result`. It performs simple counter/timestamp updates on validated, pre-allocated metadata structures.

## 3. Supporting Types

### 3.1 CutSelectionConfig

The `CutSelectionConfig` struct represents the deserialized form of the `cut_selection` field in `config.json` ([Configuration Reference](../configuration/configuration-reference.md)):

```rust
/// Configuration representation of the cut selection strategy, matching
/// the `training.cut_selection` field in `config.json`.
///
/// Deserialized from the JSON object with fields `enabled`, `method`,
/// `threshold`, `check_frequency`, and optionally `memory_window`.
#[derive(Debug, Clone, Deserialize)]
pub struct CutSelectionConfig {
    /// Whether cut selection is enabled. When `false`, no selection runs
    /// and all cuts remain active for the entire training run.
    pub enabled: bool,

    /// Selection method: `"level1"`, `"lml1"`, or `"domination"`.
    pub method: String,

    /// Activity threshold. For Level1, this is the `u64` activity count
    /// threshold (recommended: 0). For Dominated, this is the `f64`
    /// near-binding tolerance epsilon.
    pub threshold: f64,

    /// Number of iterations between selection runs.
    pub check_frequency: u64,

    /// Number of iterations to retain inactive cuts (Lml1 only).
    /// Ignored for Level1 and Dominated.
    pub memory_window: Option<u64>,
}
```

**Conversion:** `CutSelectionConfig` is validated and converted to `Option<CutSelectionStrategy>` during configuration loading. When `enabled` is `false`, the conversion produces `None` and the training loop skips all selection logic. When `enabled` is `true`, the validated config produces a `Some(CutSelectionStrategy)` value. Validation rules are specified in SS5.

### 3.2 Per-Cut Tracking Metadata

Each cut slot in the pre-allocated cut pool carries metadata used by the selection strategies. The metadata fields are written during `update_activity` (SS2.3) and read during `select` (SS2.2):

```rust
/// Per-cut metadata for cut selection tracking.
///
/// Stored alongside the cut coefficients and intercept in the pre-allocated
/// cut pool (see Cut Management Implementation SS1.1). All fields are
/// initialized to their zero/default values when the cut is first written.
pub struct CutMetadata {
    /// Iteration at which this cut was generated (1-based).
    /// Used to prevent deactivation of cuts generated in the current iteration.
    pub iteration_generated: u64,

    /// Forward pass index that generated this cut.
    /// Combined with `iteration_generated`, uniquely identifies the cut's
    /// deterministic slot (see Cut Management Implementation SS1.2).
    pub forward_pass_index: u32,

    /// Cumulative number of times this cut was binding at an LP solution.
    /// Used by Level1: deactivate if `active_count <= threshold`.
    /// Initialized to 0; incremented by `update_activity` for Level1 variant.
    pub active_count: u64,

    /// Most recent iteration at which this cut was binding.
    /// Used by Lml1: deactivate if `current_iteration - last_active_iter > memory_window`.
    /// Initialized to `iteration_generated`; updated by `update_activity` for Lml1.
    pub last_active_iter: u64,

    /// Number of visited states at which this cut is dominated by other cuts.
    /// Used by Dominated: deactivate if dominated at ALL visited states.
    /// Reset to 0 by `update_activity` when the cut is binding at a state.
    pub domination_count: u64,
}
```

**Initialization semantics:** When a new cut is written to its deterministic slot, the metadata fields are initialized as follows:

| Field                 | Initial Value         | Rationale                                                   |
| --------------------- | --------------------- | ----------------------------------------------------------- |
| `iteration_generated` | Current iteration     | Identifies when the cut was created                         |
| `forward_pass_index`  | Forward pass index    | Identifies which forward pass trajectory generated the cut  |
| `active_count`        | 0                     | The cut has not yet been evaluated in a subsequent LP solve |
| `last_active_iter`    | `iteration_generated` | Prevents immediate deactivation by Lml1 on the first check  |
| `domination_count`    | 0                     | Not yet evaluated for domination                            |

### 3.3 DeactivationSet

```rust
/// Set of cut indices to deactivate at a single stage.
///
/// Returned by `select` / `select_for_stage` and consumed by the FCF
/// manager to update the activity bitmap. The indices are slot positions
/// in the pre-allocated cut pool.
pub struct DeactivationSet {
    /// Stage index (0-based) that this deactivation set belongs to.
    pub stage_index: u32,

    /// Cut slot indices to deactivate.
    pub indices: Vec<u32>,
}
```

The `DeactivationSet` is a lightweight transfer type. The `stage_index` field identifies which stage the deactivation applies to, enabling the caller to route the result to the correct per-stage cut pool after the parallel selection phase gathers results from all ranks. The caller applies each index to the activity bitmap by clearing the corresponding bit and decrementing the active count.

## 4. Dispatch Mechanism

The cut selection strategy uses **enum dispatch** -- a `match` on the `CutSelectionStrategy` variant at each call site. This is the same pattern used by the three sibling trait specs: [RiskMeasure SS4](./risk-measure-trait.md), [HorizonMode SS4](./horizon-mode-trait.md), and [SamplingScheme SS4](./sampling-scheme-trait.md).

**Why enum dispatch is the natural choice:** The cut selection strategy is a global setting (one strategy for the entire run, applied identically to all stages), so compile-time monomorphization would also work. However, enum dispatch is preferred for consistency with the other algorithm abstraction points and because the variant set is small and closed (three strategies, with no additional variants planned). The `match` cost is negligible: `select` executes at most once per stage per `check_frequency` iterations, which at production scale (60 stages, `check_frequency=10`) amounts to at most 6 dispatches per iteration -- amortized over the dominant LP solve cost.

**Why not trait objects:** The variant set is closed (Level1, Lml1, and Dominated only). `Box<dyn CutSelectionStrategy>` would add heap allocation and virtual dispatch overhead without the extensibility benefit. The enum approach keeps the strategy value on the stack and allows the compiler to inline the variant-specific logic.

**Why not compile-time monomorphization:** While the strategy is global (unlike the per-stage risk measure), monomorphization would require propagating a generic type parameter `S: CutSelectionStrategyTrait` through the training loop and FCF manager signatures. The marginal performance benefit is zero (the `match` executes on a cold path), while the compile-time complexity would increase. Enum dispatch avoids this cost-free complexity.

## 5. Validation Rules

The following validation rules apply to `CutSelectionConfig` during configuration loading:

| Rule | Condition                                                            | Error                                                                                                      |
| ---- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| C1   | `method` must be one of `"level1"`, `"lml1"`, `"domination"`         | Unrecognized cut selection method                                                                          |
| C2   | `threshold` must be $\geq 0$                                         | Negative threshold is meaningless                                                                          |
| C3   | `check_frequency` must be $> 0$                                      | Zero frequency would mean selection runs every iteration (use 1 for that) and zero causes division-by-zero |
| C4   | When `method` is `"lml1"`, `memory_window` must be present and $> 0$ | Lml1 requires a finite positive memory window                                                              |
| C5   | When `method` is not `"lml1"`, `memory_window` is ignored if present | No error, silently ignored (logged as warning)                                                             |

Validation is performed once during the variant selection pipeline ([Extension Points SS6](./extension-points.md), step 5). After validation, the `CutSelectionStrategy` enum value is guaranteed to satisfy these constraints for the entire training run. This is why `should_run`, `select`, and `update_activity` are infallible -- they operate on validated inputs.

## 6. Interaction with Cut Pool

### 6.1 Activity Bitmap

The per-stage cut pool maintains an activity bitmap that tracks which slots contain active cuts ([Cut Management Implementation SS1.1](./cut-management-impl.md)). The bitmap has one bit per pre-allocated slot. The cut selection strategy interacts with this bitmap through a strict protocol:

| Operation      | Who performs it                | Bitmap effect                                   |
| -------------- | ------------------------------ | ----------------------------------------------- |
| Add new cut    | Training loop                  | Sets the slot's bit to 1 (active)               |
| Deactivate cut | FCF manager (on select result) | Clears the slot's bit to 0 (inactive)           |
| Reactivate cut | Never                          | Deactivation is permanent within a training run |

**Deactivation semantics:** When a cut is deactivated, its activity bitmap bit is cleared and the stage's active count is decremented. The cut's coefficient data, intercept, and metadata remain in the slot -- they are never overwritten or zeroed. This is essential for:

1. **Checkpoint/resume correctness** -- the full cut history (active and inactive) is serialized to FlatBuffers on checkpoint ([Cut Management Implementation SS3](./cut-management-impl.md))
2. **Deterministic slot assignment** -- slot indices are computed from `(iteration, forward_pass_index)` and must remain stable regardless of deactivation ([Cut Management Implementation SS1.2](./cut-management-impl.md))
3. **Post-hoc analysis** -- inactive cuts can be inspected for debugging and diagnostics

### 6.2 Deactivation vs. Deletion

Cuts are **never deleted** from the pool. The deactivation mechanism relaxes the cut's contribution to the LP by ensuring that deactivated cuts are not loaded into the solver workspace for subsequent LP solves. Concretely:

- **Active cuts** are included in the CSR assembly for the LP constraint matrix ([Solver Abstraction SS5](./solver-abstraction.md))
- **Inactive cuts** are skipped during CSR assembly -- they have no effect on the LP solution

This is functionally equivalent to setting the cut's bound to $-\infty$ (as described in [Cut Management SS5](../math/cut-management.md)), but implemented more efficiently by excluding inactive cuts from the solver entirely rather than adding a redundant row.

### 6.3 Populated Count and Active Count

The cut pool tracks two counts per stage:

| Counter           | Description                                                                      | Updated by                                    |
| ----------------- | -------------------------------------------------------------------------------- | --------------------------------------------- |
| `populated_count` | Number of slots containing valid cuts (active or inactive). Grows monotonically. | Training loop, on each new cut addition       |
| `active_count`    | Number of currently active cuts. May decrease after selection runs.              | FCF manager, after applying `DeactivationSet` |

The `select` method reads `populated_count` to know the range of slots to scan (slots `0..populated_count`), and reads the activity bitmap to identify which of those slots are currently active. The relationship `active_count <= populated_count <= capacity` always holds.

### 6.4 Visited States and Dominated Cut Detection

#### 6.4.1 VisitedStatesArchive

The `VisitedStatesArchive` stores all forward-pass trial-point state vectors accumulated across training iterations, organized as one `StageStates` buffer per stage. Each `StageStates` stores its state vectors in a single flat `Vec<f64>` for cache-friendly iteration during the domination sweep.

```rust
/// Single-stage visited-states buffer.
///
/// Stores forward-pass trial points as a flat contiguous `Vec<f64>`.
/// Entry `i * state_dimension .. (i + 1) * state_dimension` holds state `i`.
pub struct StageStates {
    data: Vec<f64>,
    count: usize,
    state_dimension: usize,
}

/// Multi-stage archive of visited forward-pass states.
///
/// One `StageStates` per stage.
pub struct VisitedStatesArchive {
    stages: Vec<StageStates>,
}
```

**Key methods:**

| Method                                                  | Description                                                               |
| ------------------------------------------------------- | ------------------------------------------------------------------------- |
| `new(num_stages, state_dim, max_iterations, total_fwd)` | Pre-allocates each stage buffer for `max_iterations * total_fwd` states   |
| `archive_gathered_states(stage, gathered, total_fwd)`   | Appends one iteration's gathered states into the specified stage's buffer |
| `states_for_stage(stage) -> &[f64]`                     | Returns the flat slice of all accumulated states for `select_for_stage`   |
| `count(stage) -> usize`                                 | Returns the number of states accumulated at a given stage                 |

**Allocation policy:** The archive is **always** allocated at training start, regardless of which cut selection strategy is active (or whether cut selection is enabled at all). This ensures forward-pass trial points are recorded for export and post-hoc analysis. The `Dominated` variant reads from this archive at pruning time; `Level1` and `Lml1` ignore it (they pass `&[]` as `visited_states`).

**Archival timing:** States are archived in the backward pass, after the per-stage `allgatherv` exchange produces the gathered state buffer for each stage $t$. The call `archive.archive_gathered_states(t, gathered, total_fwd)` appends `total_fwd` state vectors (each of length `state_dimension`) from the exchange buffer into the stage's flat storage.

**Lifecycle:** The archive is returned in `TrainingResult.visited_archive` at training completion. The caller decides whether to persist it to the policy checkpoint directory (as `states/stage_NNN.bin` FlatBuffers files) based on configuration. See [Binary Formats SS3.1](../data-model/binary-formats.md) for the persistence schema.

#### 6.4.2 Dominated Cut Detection Algorithm

The `select_dominated` function is the core algorithm for the `Dominated` variant. It is a **stateless** function that uses a local `is_candidate` boolean vector -- it does **not** read or write `CutMetadata.domination_count`. The `domination_count` field is updated only by `update_activity` (SS2.3) when a cut is found binding during the backward pass; it is not used by the selection algorithm itself.

**Algorithm (`select_dominated`):**

```
Input:
  pool: &CutPool          -- per-stage cut pool (coefficients, intercepts, metadata, active bitmap)
  visited_states: &[f64]  -- flat slice of accumulated state vectors (row-major)
  threshold: f64          -- near-binding tolerance epsilon
  current_iteration: u64  -- current training iteration

Output: Vec<u32>          -- slot indices of cuts to deactivate

1. Early exit if visited_states is empty, state_dimension == 0, or active_count < 2
2. Initialize is_candidate[k] = active[k] && iteration_generated[k] < current_iteration
   (active cuts not from the current iteration are candidates for deactivation)
3. For each visited state x_hat in visited_states (chunked by state_dimension):
   a. Compute val_k = intercept_k + coefficients_k . x_hat for ALL active cuts
   b. Find max_val = max over all active cuts of val_k
   c. Compute cutoff = max_val - threshold
   d. For each candidate k: if val_k >= cutoff, mark is_candidate[k] = false
      (this cut is NOT dominated at this state -- it achieves near-max value)
   e. If no candidates remain, break early
4. Return all indices k where is_candidate[k] is still true
   (these cuts are dominated at ALL visited states)
```

**Complexity:** $\mathcal{O}(\lvert\text{active cuts}\rvert \times \lvert\text{visited states}\rvert \times \text{state\_dimension})$ per stage per check. The inner loop computes a dot product of length `state_dimension` for each (cut, state) pair.

**Early termination:** The algorithm breaks out of the state loop as soon as no candidates remain, which can significantly reduce cost when most cuts are not dominated (the common case).

#### 6.4.3 Production-Scale Cost

At production scale (15,000 cut capacity, 192 forward passes, 50 iterations between checks), the visited state set at the first check contains $192 \times 10 = 1{,}920$ states, and the active cut count is approximately $192 \times 10 = 1{,}920$. The domination check cost is $1{,}920 \times 1{,}920 \approx 3.7\text{M}$ cut evaluations per stage, each requiring a dot product of dimension `state_dimension` (2,080 at production scale). This is significant -- approximately $3.7\text{M} \times 2{,}080 \approx 7.7\text{G}$ floating-point operations per stage per check -- which is why `check_frequency` amortizes the cost across multiple iterations.

#### 6.4.4 Stage 0 Exemption and Parallel Execution

**Stage 0 exemption:** Stage 0 is exempt from cut selection. Its cuts are never the "successor" in the backward pass (there is no stage $-1$ to generate cuts for stage 0's FCF), so their binding activity metadata is never updated by `update_activity`. Deactivating them based on stale metadata would weaken the lower bound approximation. The training loop skips stage 0 and processes only stages $1 \ldots T-1$.

**Parallel execution:** The training loop distributes the eligible stages ($1 \ldots T-1$) across threads within each rank using Rayon's `into_par_iter()`. Each thread calls `select_for_stage(pool, states, iteration, stage_index)` on its assigned stages independently. The selection calls are embarrassingly parallel -- each reads only the stage's own cut pool metadata and visited states (both read-only at this point), and produces an independent `DeactivationSet`. After the parallel phase, deactivations are applied sequentially because `deactivate` requires `&mut` access to the cut pool.

## 7. Convergence Guarantee

**Theorem** (Bandarra & Guigues, 2021): Under Level-1 or LML1 cut selection, SDDP with finitely many scenarios converges to the optimal value function with probability 1.

This theorem is stated and cited in [Cut Management SS8](../math/cut-management.md). The key insight is that removing cuts that are never active at any visited state does not affect the outer approximation quality at those states. As the set of visited states becomes dense over iterations, the approximation converges.

**Applicability to variants:**

| Variant   | Convergence guarantee                                                                                                                                                                                                                    |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Level1    | Preserved. Only cuts that have _never_ been binding are removed; all ever-useful cuts remain.                                                                                                                                            |
| Lml1      | Preserved. The memory window ensures that periodically revisited cuts are retained. The theorem's proof covers the Lml1 case explicitly.                                                                                                 |
| Dominated | Not formally covered by the Bandarra-Guigues theorem. Dominated cut detection is a heuristic that may remove cuts which would be active at unvisited states. In practice, the visited state set grows dense and the heuristic converges. |

**Implication for production use:** Level1 and Lml1 are the recommended strategies when convergence guarantees are required. The Dominated strategy offers more aggressive pruning at the cost of a weaker theoretical guarantee, making it suitable for cases where empirical convergence monitoring (via [Convergence Monitoring](./convergence-monitoring.md)) provides sufficient confidence.

## Cross-References

- [Cut Management (Math)](../math/cut-management.md) -- Cut activity definition (SS6), three selection strategies (SS7.1 Level-1, SS7.2 LML1, SS7.3 Dominated), convergence guarantee theorem (SS8), selection parameters (SS9)
- [Cut Management Implementation](./cut-management-impl.md) -- Cut pool structure and activity bitmap (SS1.1), deterministic slot assignment (SS1.2), selection strategy implementation (SS2), cut serialization for checkpoint (SS3), cross-rank synchronization (SS4), activity tracking and binding detection (SS6)
- [Extension Points](./extension-points.md) -- Variant architecture overview (SS1), dispatch mechanism analysis (SS7), variant selection pipeline (SS6), variant composition validation (SS8)
- [Training Loop](./training-loop.md) -- Cut metadata and active count tracking (SS7.3), backward pass where `update_activity` is invoked
- [Solver Abstraction](./solver-abstraction.md) -- Cut pool design (SS5), how active cuts enter the solver LP via CSR assembly (SS5.4)
- [Solver Workspaces](./solver-workspaces.md) -- Stage solve workflow (SS1.4) where binding detection occurs and activity counters are updated
- [Configuration Reference](../configuration/configuration-reference.md) -- `training.cut_selection` JSON schema: `enabled`, `method`, `threshold`, `check_frequency`, `memory_window`
- [Binary Formats](../data-model/binary-formats.md) -- FlatBuffers schema for cut metadata persistence (SS3.1), visited states for dominated detection (SS3.4)
- [Convergence Monitoring](./convergence-monitoring.md) -- Uses FCF statistics (lower bound from stage 1 cuts) that depend on cut pool quality
- [Risk Measure Trait](./risk-measure-trait.md) -- Sibling trait specification following the same enum dispatch pattern (SS4)
- [Horizon Mode Trait](./horizon-mode-trait.md) -- Sibling trait specification following the same enum dispatch pattern (SS4)
- [Sampling Scheme Trait](./sampling-scheme-trait.md) -- Sibling trait specification following the same enum dispatch pattern (SS4)
- [Work Distribution](../hpc/work-distribution.md) -- Contiguous block assignment formula (§3.1) applied to stage partitioning for parallel selection (SS2.2a)
- [Synchronization](../hpc/synchronization.md) -- Per-stage backward pass barrier (§1.4) ensuring inputs are synchronized; DeactivationSet allgatherv wire format (§1.4a)
- [Shared Memory Aggregation](../hpc/shared-memory-aggregation.md) -- SharedRegion StageLpCache write model (§1.2) for leader-only deactivation application
- [Communicator Trait](../hpc/communicator-trait.md) -- Reference pattern for trait specification structure and convention blockquote; allgatherv contract (SS2.1) used for DeactivationSet gathering
