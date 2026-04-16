# Design: LP Scalability — Adaptive Cut Management

## Implementation Status

**IMPLEMENTED** — v0.4.4, branch `feat/tier1-tier2-correctness-and-performance`

> **IMPORTANT DESIGN CHANGE — S1 Angular Pruning**: The original design
> ("cluster by cosine similarity, keep tightest at a reference point") was
> found to violate Assumption (H2) from Guigues 2017 — near-parallel cuts
> with different intercepts create crossing hyperplanes where dominance is
> state-dependent. The implemented version uses angular clustering as a
> computational accelerator for pointwise dominance verification (same
> criterion as the existing `select_dominated` function, but restricted to
> within-cluster pairs). This preserves (H2) and finite convergence.
> See `angular-prunning-research-report.md` (root) and
> `docs/design/angular-pruning-mathematical-background.md` for the
> mathematical analysis.

| Strategy                                                                             | Status | Config key                                                       | Default  |
| ------------------------------------------------------------------------------------ | ------ | ---------------------------------------------------------------- | -------- |
| S1: Angular-accelerated dominance (`angular_pruning.rs`, `select_angular_dominated`) | DONE   | `angular_pruning.enabled`, `cosine_threshold`, `check_frequency` | disabled |
| S2: Active cut budget (`CutPool::enforce_budget`)                                    | DONE   | `max_active_per_stage: Option<u32>`                              | disabled |
| S3: Basis-aware warm-start padding (`basis_padding.rs`, `pad_basis_for_cuts`)        | DONE   | `basis_padding: Option<bool>`                                    | disabled |

References:

- `angular-prunning-research-report.md` (repo root) — mathematical analysis of pruning safety
- `docs/design/angular-pruning-mathematical-background.md` — formal propositions

---

## Problem

When `forward_passes` is large, active cuts accumulate faster than existing
selection strategies can prune them. Each SDDP iteration generates
`forward_passes × world_size` new cuts per stage. Cut selection methods
(Level1, LML1, Dominated) control _which_ cuts to deactivate, but the
pruning rate is insufficient when the generation rate is high.

Empirical evidence (118-stage hydrothermal case, 2 ranks × 3 threads):

| Config       | Iters | Active cuts | Avg active/stage | Avg ms/solve (bwd) | Parallel eff. |
| ------------ | ----- | ----------- | ---------------- | ------------------ | ------------- |
| 5 fw × 50 it | 50    | 3,125       | 26.7             | 7.74               | 78%           |
| 50 fw × 5 it | 5     | 14,621      | 125.0            | 13.34              | 91%           |

The 50-forward-pass case has **higher parallel efficiency** (91% vs 78%)
but **worse wall-clock time** because each simplex solve is 72% more
expensive — each pivot costs 81 μs vs 46 μs due to the denser constraint
matrix. The LP size grows linearly with `iterations × forward_passes` and
the simplex cost per pivot is O(rows), giving superlinear total cost.

### Root Cause Analysis

The LML1 deactivation criterion is "not binding in the last W iterations,"
where binding is determined by a positive dual multiplier during the
backward pass. With F=50 forward passes per iteration, each cut has 50
chances to be binding at any trial point. A cut that binds at 2% of states
(1 out of 50) survives the LML1 window indefinitely — but a cut binding
at 2% of states contributes negligibly to the policy.

Additionally, many forward passes from nearby states produce cuts with
nearly identical normal vectors. With F=50, clusters of 10–20 near-parallel
cuts often coexist, of which only 1–2 are non-redundant. Existing selection
methods don't detect geometric redundancy.

### Why Not HiGHS Presolve

HiGHS presolve can detect redundant rows (including dominated cuts and
phantom rows), but it:

1. Triggers internal HiGHS behavior that empirically hurts performance
   in most cases (observed in prior testing)
2. Interferes with basis warm-start (presolve may discard the provided
   basis)
3. Is a blunt instrument — it analyzes the entire LP, not just cuts
4. May select a different optimal vertex, changing baseline results in
   a way that can't be controlled from our side

All useful redundancy detection can be done more efficiently in Rust,
targeted specifically at the cut structure, where we know the geometry.

### Why Not Incremental LP (Model Persistence)

Skipping `load_model` between iterations would save the LP assembly cost
(O(all_cuts × nnz_per_cut) per stage per worker). However, it breaks the
bit-for-bit determinism guarantee:

- Without `load_model`, HiGHS retains internal state (edge weights,
  factorization, refactorization counters) from the previous iteration's
  solves on that worker
- Deactivated cuts become phantom rows (bounds zeroed but still in the LP),
  changing the LP dimensions relative to a fresh rebuild
- Phantom rows change the basis factorization structure, which in
  degenerate cases (common in cut-heavy SDDP LPs) steers the simplex to
  a different optimal vertex
- The current `load_model + add_rows + solve_with_basis` pattern guarantees
  that each solve's result is a pure function of `(LP, basis, bounds)`,
  independent of which worker solved which scenarios previously

The determinism cost of incremental LP is unacceptable. The correct
approach is to reduce the number of active cuts that enter the LP.

## Decision

Implement three complementary Rust-side strategies that reduce the number
of active cuts before they enter the LP. All strategies are
determinism-safe: they produce the same cut set regardless of MPI rank
count or thread assignment. They compose in a pipeline:

```
Backward pass generates cuts
    → MPI sync (allgatherv + allreduce for binding metadata)
    → Strategy-based selection (LML1 / Level1 / Dominated)       [existing]
    → Angular diversity pruning (remove near-duplicate normals)   [NEW: S1]
    → Budget enforcement (hard cap on active count)               [NEW: S2]
    → Basis-aware warm-start padding                              [NEW: S3]
    → Forward/backward pass uses the pruned cut set
```

S1 and S2 reduce LP size (fewer rows = cheaper pivots). S3 reduces
simplex iterations per solve (better starting basis for new cut rows).

### Strategy S1: Angular Diversity Pruning

#### Motivation

Benders cuts are hyperplanes `θ ≥ α + π'x`. Two cuts with nearly parallel
normal vectors (π₁ ≈ c·π₂) are geometrically redundant — only the outermost
one (largest intercept along that direction) contributes to the polyhedral
approximation of the cost-to-go function.

With F=50 forward passes, nearby trial-point states produce cuts with
similar normals. Clusters of 10–20 near-parallel cuts are common. Pruning
all but the outermost cut in each cluster can reduce active cuts by 30–50%
without measurable policy quality loss.

#### Algorithm

For each stage independently:

1. **Normalize coefficient vectors**: For each active cut `k`, compute
   the unit normal `n̂ₖ = πₖ / ‖πₖ‖₂`. Store the L2 norms.

2. **Build clusters by angular proximity**: Use a greedy single-linkage
   approach:
   - Iterate active cuts in slot order (deterministic)
   - For each unassigned cut, start a new cluster
   - Assign all unassigned cuts within angular distance `cos(θ) > τ`
     to this cluster, where `τ` is the cosine similarity threshold
     (e.g., `τ = 0.999` ≈ 1.8° separation)
   - The greedy scan is O(active² × n_state) in the worst case, but
     with early exit on norm mismatch it's typically much faster

3. **Select representative per cluster**: Within each cluster, evaluate all
   cuts at a reference state (the centroid of recent visited states for that
   stage) and keep only the cut with the highest value
   `αₖ + πₖ'x_ref`. All others are marked for deactivation.

4. **Protection**: Cuts from the current iteration are excluded from
   clustering (same as all existing selection strategies). Single-member
   clusters (no near-duplicate) are never deactivated.

#### Determinism

- Cluster formation iterates in slot order (deterministic)
- Cosine similarity is computed from cut coefficients stored identically
  across all ranks (post-allgatherv sync)
- The reference state is the elementwise mean of the most recent
  iteration's gathered forward-pass states, which is identical across
  ranks (computed from the allgatherv output)
- Representative selection within each cluster is a deterministic argmax
  with slot-index tiebreaker

#### Configuration

```jsonc
// config.json → training.cut_selection
{
  "enabled": true,
  "method": "lml1",
  "memory_window": 3,
  "check_frequency": 1,
  "angular_pruning": {
    "enabled": true,
    "cosine_threshold": 0.999, // cos(θ) similarity threshold
    "check_frequency": 1, // run every N iterations (null = same as parent)
  },
}
```

- `cosine_threshold`: cosine similarity above which two cuts are
  considered near-parallel. Default 0.999 (≈ 2.6° separation). Higher
  values are more conservative (fewer cuts pruned). Range: (0.0, 1.0].
- `check_frequency`: how often angular pruning runs (in iterations).
  Independent of the parent strategy's `check_frequency`. Defaults to the
  parent's frequency if null.

#### Complexity

- **Per-stage cost**: O(A² × S) where A = active cuts, S = state dimension
- **Total cost**: O(T × A² × S) across all stages, parallelized via rayon
- **Practical cost**: With A=200, S=100, T=117: ~470M FP ops, ~2–5ms on
  modern hardware (SIMD-friendly dot products). Negligible vs LP solve time.

#### Edge Cases

- **Zero-norm cuts**: Cuts with ‖π‖₂ < ε (pure intercept, no state
  dependence) are placed in a special cluster together. The one with the
  largest intercept survives. This handles degenerate cuts that can appear
  at terminal stages.
- **Sparse state indices**: When `nonzero_state_indices` is non-empty,
  compute cosine similarity using only the nonzero indices. This preserves
  the sparse-path optimization.
- **Single active cut**: No clustering needed; skip entirely.

### Strategy S2: Active Cut Budget (Per-Stage Hard Cap)

#### Motivation

Angular pruning reduces geometric redundancy, but the active count can
still exceed what the LP solver handles efficiently. A hard budget provides
a guaranteed upper bound on LP size regardless of generation rate.

#### Design

This strategy is fully specified in `docs/design/cut-budget-per-stage.md`.
The design is ready for implementation with the following summary:

- **Config**: `max_active_per_stage: Option<u32>` in `CutSelectionConfig`
- **Enforcement**: Every iteration, after strategy selection + angular
  pruning. Force-evicts least-valuable cuts (oldest `last_active_iter`,
  tiebreak by lowest `active_count`) until active count ≤ budget.
- **Protection**: Cuts from current iteration are exempt from eviction.
- **Bound**: LP size per stage is ≤ `budget + forward_passes × world_size`.
- **Timing**: Runs every iteration, not gated by `check_frequency`.

One refinement to the existing design: the budget enforcement runs
**after** angular pruning (S1), so S1 reduces the pool first and the
budget only triggers when S1 is insufficient. This makes the budget a
true safety net rather than the primary pruning mechanism.

#### Integration with S1

```
Cut selection pipeline per iteration:
  1. Strategy-based selection (LML1 / Level1 / Dominated)  [existing]
  2. Angular diversity pruning (S1)                          [NEW]
  3. Budget enforcement (S2)                                 [NEW]
```

The budget counts the total active cuts after steps 1 and 2. If still
above the cap, step 3 evicts the excess. The budget should be set above
the expected natural active count after S1 pruning (e.g., 2–3× the
typical surviving count).

### Strategy S3: Basis-Aware Warm-Start Padding

#### Motivation

When the LP grows between iterations (new cuts added), the stored basis
from the previous forward pass has fewer row entries than the current LP.
The current code pads new cut rows with `BASIC` status (slack variable in
basis = cut not tight). This is correct but suboptimal: HiGHS starts with
a basis where all new cut slacks are basic and must discover through
simplex pivots which new cuts are actually binding.

With 50 new cuts per iteration (F=50, single rank), each forward-pass
solve may need up to 50 "discovery" pivots. At O(total_rows) per pivot,
this adds meaningful cost.

#### Algorithm

When preparing the basis for `solve_with_basis`, for each new cut row
(rows beyond the stored basis dimension):

1. Identify which cut slot corresponds to this LP row. After
   `build_cut_row_batch_into`, cuts are appended to the LP in the order
   returned by `active_cuts()` iterator (slot order). The mapping is:
   LP row `template_num_rows + k` corresponds to the k-th active cut.

2. Evaluate the cut at the warm-start state `x`:

   ```
   cut_value = α + π'x
   theta_hat = x[theta_index]   // stored value from previous forward pass
   slack = theta_hat - cut_value
   ```

3. Set the basis status:
   - If `slack > dual_feasibility_tolerance` (cut is comfortably slack):
     keep `BASIC` (current behavior — slack in basis)
   - If `slack ≤ dual_feasibility_tolerance` (cut is tight or violated):
     set `NONBASIC_LOWER` (constraint at lower bound — cut is binding)

This gives HiGHS a starting basis that already reflects which new cuts
are likely binding, reducing the number of discovery pivots.

#### Where to Implement

The basis padding currently happens in `HighsSolver::solve_with_basis`
(`crates/cobre-solver/src/highs.rs:1188–1197`). However, the solver
doesn't have access to cut coefficients or the warm-start state — it
only sees raw basis status vectors.

The padding logic should move to the call site in `forward.rs` and
`backward.rs`, where the caller has access to both the basis and the
cut information. Specifically:

**Forward pass** (`forward.rs`, `run_forward_stage`):

- Before calling `solve_with_basis`, compute cut slack for each new
  cut row using the current state `ws.current_state` and the cut batch
- Modify the basis row statuses for the new rows
- Pass the modified basis to `solve_with_basis`

**Backward pass** (`backward.rs`, `process_trial_point_backward`):

- Opening 0 uses `solve_with_basis(basis_from_forward)`. The same
  padding applies using trial-point state `x_hat`.
- Openings 1+ use `solve()` (internal hot-start) — no padding needed.

#### Determinism

The padding decision depends on:

- Cut coefficients (identical across ranks — post-allgatherv)
- Warm-start state (deterministic per scenario/trial-point)
- `dual_feasibility_tolerance` (global solver config, constant)

The padding is a pure function of `(cut, state, tolerance)`. No
dependence on worker assignment or prior solve history. Determinism
is preserved.

#### Basis Consistency

For a valid simplex basis, the number of BASIC variables must equal the
number of rows. When we change a row from BASIC to NONBASIC_LOWER, we
remove one basic variable but add nothing. This makes the basis singular.

To maintain consistency, when setting a new cut row to NONBASIC_LOWER,
we must ensure another variable becomes BASIC. In practice, the most
natural swap is: the theta variable (or a dispatch variable that was
NONBASIC) becomes BASIC when the cut slack becomes NONBASIC. However,
computing the correct swap requires solving a system — which defeats
the purpose.

**Practical approach**: Set the basis status as described and let HiGHS
handle the basis repair during its crash procedure. HiGHS accepts
inconsistent bases and repairs them internally before starting simplex.
The repair is typically O(1) per inconsistency. This is cheaper than
the 50 discovery pivots it replaces.

**Alternative**: Only change rows to NONBASIC*LOWER when the cut is
\_very* tight (`slack < -tolerance`, i.e., violated). Violated cuts must
be NONBASIC_LOWER to be dual-feasible. Nearly-tight cuts can remain
BASIC with less basis disruption. This is more conservative but safer.

#### Configuration

No user-facing configuration. The padding logic activates automatically
whenever `solve_with_basis` is called with a basis that has fewer rows
than the LP. The tolerance used is the existing
`dual_feasibility_tolerance` from the solver config.

#### Estimated Impact

With F=50 and 200 active cuts, each forward solve has ~50 new cut rows.
If 90% are slack (typical), 5 are near-tight, and 45 remain BASIC:

- **Before**: ~50 discovery pivots × O(250 rows) per pivot = ~12,500 ops
- **After**: ~5 discovery pivots + basis repair = ~1,500 ops
- **Savings**: ~80% reduction in discovery work per solve

The total impact depends on how many discovery pivots the solver actually
performs (may be fewer than the theoretical maximum due to HiGHS's
pricing strategy). Instrumentation should measure `total_iterations` per
solve before and after to quantify the real improvement.

## Implementation Plan

### Phase 1: Angular Diversity Pruning (S1)

#### 1.1 Config — `cobre-io/src/config.rs`

Add nested config struct:

```rust
/// Angular diversity pruning for near-duplicate cuts.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct AngularPruningConfig {
    /// Enable angular diversity pruning.
    pub enabled: Option<bool>,

    /// Cosine similarity threshold. Cuts with cos(θ) > threshold are
    /// considered near-parallel. Default: 0.999.
    pub cosine_threshold: Option<f64>,

    /// Iterations between angular pruning runs. Default: same as parent
    /// check_frequency.
    pub check_frequency: Option<u32>,
}

impl Default for AngularPruningConfig {
    fn default() -> Self {
        Self {
            enabled: None,
            cosine_threshold: None,
            check_frequency: None,
        }
    }
}
```

Add to `CutSelectionConfig`:

```rust
pub struct CutSelectionConfig {
    // ... existing fields ...

    /// Angular diversity pruning for near-duplicate cut removal.
    #[serde(default)]
    pub angular_pruning: Option<AngularPruningConfig>,

    /// Hard cap on active cuts per stage (from cut-budget design doc).
    #[serde(default)]
    pub max_active_per_stage: Option<u32>,
}
```

#### 1.2 Angular pruning module — `cobre-sddp/src/angular_pruning.rs`

New module with the clustering algorithm:

```rust
/// Configuration for angular diversity pruning (parsed from input config).
#[derive(Debug, Clone, Copy)]
pub struct AngularPruningParams {
    pub cosine_threshold: f64,
    pub check_frequency: u64,
}

/// Result of angular pruning for one stage.
pub struct AngularPruningResult {
    /// Slot indices of cuts to deactivate (near-duplicates).
    pub deactivate: Vec<u32>,
    /// Number of clusters formed.
    pub clusters_formed: usize,
}

/// Identify near-duplicate cuts for deactivation at a single stage.
///
/// Returns slot indices of cuts whose normal vectors are near-parallel to
/// another active cut that provides a tighter bound at the reference state.
///
/// # Arguments
///
/// - `pool`: the stage's cut pool (coefficients, intercepts, active flags)
/// - `reference_state`: state at which to evaluate cut values for
///   representative selection. Typically the centroid of recent visited
///   states.
/// - `cosine_threshold`: similarity threshold in (0, 1].
/// - `current_iteration`: cuts from this iteration are excluded.
///
/// # Determinism
///
/// Clustering iterates in ascending slot order. Within each cluster, the
/// representative is chosen by argmax of cut value at `reference_state`,
/// with slot index as tiebreaker. The result is independent of thread
/// assignment or MPI rank.
pub fn select_angular_duplicates(
    pool: &CutPool,
    reference_state: &[f64],
    cosine_threshold: f64,
    current_iteration: u64,
) -> AngularPruningResult {
    // ... see algorithm description above ...
}
```

**Internal algorithm sketch:**

```rust
// Step 1: Compute unit normals and norms for all active candidates.
let mut candidates: Vec<CandidateInfo> = Vec::new();
for i in 0..pool.populated_count {
    if !pool.active[i] || pool.metadata[i].iteration_generated >= current_iteration {
        continue;
    }
    let coeffs = &pool.coefficients[i * n_state..(i + 1) * n_state];
    let norm = l2_norm(coeffs);
    if norm < f64::EPSILON {
        // Zero-norm cut: special handling (pure intercept).
        // Group with other zero-norm cuts; keep largest intercept.
        continue; // handled in zero-norm pass below
    }
    candidates.push(CandidateInfo {
        slot: i,
        norm,
        // Unit normal stored in scratch buffer (avoid per-cut allocation).
    });
}

// Step 2: Greedy single-linkage clustering.
// Pre-allocate scratch for unit normals: flat buffer of n_candidates × n_state.
let mut unit_normals: Vec<f64> = Vec::with_capacity(candidates.len() * n_state);
for c in &candidates {
    let coeffs = &pool.coefficients[c.slot * n_state..(c.slot + 1) * n_state];
    let inv_norm = 1.0 / c.norm;
    for &v in coeffs {
        unit_normals.push(v * inv_norm);
    }
}

let mut cluster_id: Vec<Option<usize>> = vec![None; candidates.len()];
let mut n_clusters = 0usize;

for i in 0..candidates.len() {
    if cluster_id[i].is_some() {
        continue; // already assigned
    }
    let cid = n_clusters;
    cluster_id[i] = Some(cid);
    n_clusters += 1;

    let ni = &unit_normals[i * n_state..(i + 1) * n_state];
    for j in (i + 1)..candidates.len() {
        if cluster_id[j].is_some() {
            continue;
        }
        let nj = &unit_normals[j * n_state..(j + 1) * n_state];
        let cos_sim = dot(ni, nj); // both unit vectors, so dot = cosine
        if cos_sim > cosine_threshold {
            cluster_id[j] = Some(cid);
        }
    }
}

// Step 3: Within each cluster, keep the cut with highest value at
// reference_state. Deactivate the rest.
// Group slots by cluster_id, compute values, select representative.
let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
for (idx, cid) in cluster_id.iter().enumerate() {
    if let Some(c) = cid {
        cluster_members[*c].push(idx);
    }
}

let mut deactivate = Vec::new();
for members in &cluster_members {
    if members.len() <= 1 {
        continue; // singleton cluster, nothing to prune
    }
    // Evaluate each member at reference_state.
    let mut best_idx = members[0];
    let mut best_val = evaluate_cut(pool, candidates[members[0]].slot, reference_state);
    for &m in &members[1..] {
        let val = evaluate_cut(pool, candidates[m].slot, reference_state);
        if val > best_val || (val == best_val && candidates[m].slot < candidates[best_idx].slot) {
            best_val = val;
            best_idx = m;
        }
    }
    // Deactivate non-representatives.
    for &m in members {
        if m != best_idx {
            deactivate.push(candidates[m].slot as u32);
        }
    }
}
```

**Key implementation note**: The scratch buffers (unit normals, cluster
assignments) must be pre-allocated and reused across stages to avoid
per-stage allocation on the hot path. Since angular pruning runs in
rayon parallel across stages, each thread needs its own scratch. Use
thread-local or per-worker buffers passed via the parallel closure.

#### 1.3 Reference state computation — `cobre-sddp/src/training.rs`

The reference state for representative selection is the elementwise mean
of the most recent iteration's gathered forward-pass states. This is
already available from the exchange buffers:

```rust
// After forward pass exchange, before cut selection:
// Compute per-stage reference state for angular pruning.
let mut angular_ref_states: Vec<Vec<f64>> = Vec::new();
if angular_pruning.is_some() {
    for stage in 0..num_stages {
        let mut centroid = vec![0.0_f64; n_state];
        let total = exchange_bufs.real_total_scenarios();
        for m in 0..total {
            let state = exchange_bufs.state_at_stage(stage, m);
            for (c, &s) in centroid.iter_mut().zip(state) {
                *c += s;
            }
        }
        let inv_n = 1.0 / total as f64;
        for c in &mut centroid {
            *c *= inv_n;
        }
        angular_ref_states.push(centroid);
    }
}
```

**Determinism**: The centroid is computed from the allgatherv output
(identical across ranks) using deterministic arithmetic (sum in scenario
order, divide by count). The result is identical on every rank.

**Alternative**: Instead of computing the centroid after each iteration's
forward pass, reuse the visited states archive (if available) and compute
the centroid of all archived states for that stage. This gives a more
stable reference as iterations progress. However, it couples angular
pruning to the visited-states infrastructure. For simplicity, use the
current iteration's exchange buffers.

**Note on timing**: The centroid must be computed AFTER the backward
pass's state exchange (which runs per-stage inside `backward_pass`), but
the exchange buffers are reused across stages. The simplest approach:
compute the centroid for all stages from the LAST stage's exchange buffer
after the backward pass completes. But this only gives stage T-2's
states.

A better approach: compute and store per-stage centroids INSIDE the
backward pass loop, right after `exchange.exchange(records, t, ...)`.
At that point, the exchange buffer contains states for stage `t`. Store
the centroid in a pre-allocated `Vec<Vec<f64>>` indexed by stage.

#### 1.4 Integration in training loop — `cobre-sddp/src/training.rs`

After the existing cut selection block (line 833) and before LB rebuild:

```rust
// Step 5b: Angular diversity pruning (after strategy selection).
if let Some(angular) = &angular_pruning_params {
    if angular.should_run(iteration) {
        let ang_start = Instant::now();
        let num_sel_stages = num_stages.saturating_sub(1);

        let deactivations: Vec<(usize, Vec<u32>, usize)> = (1..num_sel_stages)
            .into_par_iter()
            .map(|stage| {
                let pool = &fcf.pools[stage];
                let ref_state = &angular_ref_states[stage];
                let result = angular_pruning::select_angular_duplicates(
                    pool,
                    ref_state,
                    angular.cosine_threshold,
                    iteration,
                );
                (stage, result.deactivate, result.clusters_formed)
            })
            .collect();

        let mut angular_deactivated = 0u32;
        for (stage, deact, _clusters) in deactivations {
            angular_deactivated += deact.len() as u32;
            fcf.pools[stage].deactivate(&deact);
        }

        // Emit event for observability.
        emit(event_sender.as_ref(), TrainingEvent::AngularPruningComplete {
            iteration,
            cuts_deactivated: angular_deactivated,
            selection_time_ms: ang_start.elapsed().as_millis() as u64,
        });
    }
}
```

#### 1.5 Validation and config parsing — `cobre-sddp/src/cut_selection.rs`

Add to `parse_cut_selection_config`:

```rust
let angular_pruning = config
    .angular_pruning
    .as_ref()
    .filter(|ap| ap.enabled.unwrap_or(false))
    .map(|ap| {
        let cosine = ap.cosine_threshold.unwrap_or(0.999);
        if !(0.0 < cosine && cosine <= 1.0) {
            return Err("angular_pruning.cosine_threshold must be in (0.0, 1.0]");
        }
        let freq = ap.check_frequency
            .map(u64::from)
            .unwrap_or(check_frequency);
        if freq == 0 {
            return Err("angular_pruning.check_frequency must be > 0");
        }
        Ok(AngularPruningParams { cosine_threshold: cosine, check_frequency: freq })
    })
    .transpose()?;
```

### Phase 2: Active Cut Budget (S2)

Implement `docs/design/cut-budget-per-stage.md` as designed. The only
change from that document: budget enforcement runs AFTER angular pruning
in the pipeline, not directly after strategy selection.

Updated training loop sequence:

```
Per iteration:
  1. Forward pass
  2. Backward pass (cuts generated, synced, metadata allreduced)
  3. Strategy selection (LML1 / Level1 / Dominated)    [existing, if should_run]
  4. Angular diversity pruning (S1)                     [NEW, if should_run]
  5. Budget enforcement (S2)                            [NEW, every iteration]
  6. Lower bound evaluation
  7. Convergence check
```

Steps 3 and 4 are gated by their respective `check_frequency`. Step 5
runs every iteration because the budget is a safety net.

### Phase 3: Basis-Aware Warm-Start Padding (S3)

#### 3.1 New struct — `cobre-sddp/src/basis_padding.rs`

```rust
/// Compute basis padding for new cut rows based on cut tightness.
///
/// For each new cut row (beyond `basis_rows`), evaluates the cut at the
/// given state and returns the appropriate basis status:
/// - `BASIC` (1) if the cut is comfortably slack
/// - `NONBASIC_LOWER` (0) if the cut is tight or violated
///
/// # Arguments
///
/// - `pool`: cut pool for the stage
/// - `active_cut_slots`: ordered slot indices of active cuts (same order
///   as rows in the LP after `add_rows`)
/// - `state`: current state vector (theta is the last element)
/// - `basis_num_rows`: number of rows in the stored basis
/// - `lp_num_rows`: number of rows in the current LP
/// - `template_num_rows`: number of structural rows in the LP template
/// - `tolerance`: dual feasibility tolerance
///
/// # Returns
///
/// Vec of i32 basis statuses for rows `basis_num_rows..lp_num_rows`.
pub fn compute_cut_basis_padding(
    pool: &CutPool,
    active_cut_slots: &[usize],
    state: &[f64],
    theta: f64,
    basis_num_rows: usize,
    lp_num_rows: usize,
    template_num_rows: usize,
    tolerance: f64,
) -> Vec<i32> {
    let n_new = lp_num_rows - basis_num_rows;
    let mut statuses = Vec::with_capacity(n_new);

    for row in basis_num_rows..lp_num_rows {
        let cut_index = row - template_num_rows;
        if cut_index >= active_cut_slots.len() {
            // Safety fallback: BASIC for rows beyond tracked cuts.
            statuses.push(HIGHS_BASIS_STATUS_BASIC);
            continue;
        }
        let slot = active_cut_slots[cut_index];
        let n_state = pool.state_dimension;
        let coeffs = &pool.coefficients[slot * n_state..(slot + 1) * n_state];
        let intercept = pool.intercepts[slot];

        // Cut constraint: θ ≥ α + π'x, i.e., -π'x + θ ≥ α.
        // Slack = θ - (α + π'x). Positive slack means cut is not tight.
        let cut_val: f64 = intercept
            + coeffs.iter().zip(state).map(|(c, x)| c * x).sum::<f64>();
        let slack = theta - cut_val;

        if slack <= tolerance {
            // Cut is tight or violated: start at lower bound.
            statuses.push(HIGHS_BASIS_STATUS_NONBASIC_LOWER);
        } else {
            // Cut is slack: slack variable is basic.
            statuses.push(HIGHS_BASIS_STATUS_BASIC);
        }
    }

    statuses
}
```

#### 3.2 Integration in forward pass — `cobre-sddp/src/forward.rs`

In `run_forward_stage`, before the `solve_with_basis` call:

```rust
// If basis has fewer rows than the LP (new cuts added since last
// iteration), compute informed padding instead of blanket BASIC.
let basis = basis_store.get(m, t);
if let Some(rb) = basis {
    if rb.row_status.len() < ws.solver.num_rows() {
        let padding = basis_padding::compute_cut_basis_padding(
            &fcf.pools[t],
            &active_cut_slots[t],  // pre-built ordered slot list
            &ws.current_state,
            ws.current_state[indexer.theta],  // theta value from state
            rb.row_status.len(),
            ws.solver.num_rows(),
            ctx.templates[t].num_rows,
            DUAL_FEASIBILITY_TOLERANCE,
        );
        // Extend the basis row_status with the computed padding.
        // This requires a mutable copy of the basis -- use the
        // per-worker scratch basis buffer.
        ws.scratch_basis.clone_from(rb);
        ws.scratch_basis.row_status.extend_from_slice(
            &padding.iter().map(|&s| s).collect::<Vec<_>>()
        );
        ws.solver.solve_with_basis(&ws.scratch_basis)
    } else {
        ws.solver.solve_with_basis(rb)
    }
}
```

**Note**: The active cut slots list must be built once per stage (when
building the cut batch) and stored for use during basis padding. This
is a minor bookkeeping change to `build_cut_row_batch_into` or a
parallel computation.

#### 3.3 Integration in backward pass — `cobre-sddp/src/backward.rs`

In `process_trial_point_backward`, opening 0:

```rust
let view = if omega == 0 {
    if let Some(rb) = succ.basis_store.get(m, s) {
        if rb.row_status.len() < ws.solver.num_rows() {
            let padding = basis_padding::compute_cut_basis_padding(
                // ... same pattern as forward pass ...
            );
            ws.scratch_basis.clone_from(rb);
            ws.scratch_basis.row_status.extend(padding);
            ws.solver.solve_with_basis(&ws.scratch_basis)
        } else {
            ws.solver.solve_with_basis(rb)
        }
    } else {
        ws.solver.solve()
    }
} else {
    ws.solver.solve()
};
```

#### 3.4 Solver interface — `cobre-solver/src/highs.rs`

No changes to `solve_with_basis` needed. The padding is applied on the
caller side before the call. The solver still receives a complete basis
vector (original + padding) and handles it as before.

The one change: remove the internal BASIC padding in `solve_with_basis`
for the case where `basis.row_status.len() < lp_rows`, since the caller
now handles this. Or better: keep the internal padding as a fallback
(for callers that don't use basis_padding) and skip it when
`basis.row_status.len() == lp_rows`.

## Output and Observability

### New Training Events

```rust
/// Angular pruning completed for all stages.
AngularPruningComplete {
    iteration: u64,
    cuts_deactivated: u32,
    selection_time_ms: u64,
}

/// Budget enforcement completed for all stages.
BudgetEnforcementComplete {
    iteration: u64,
    cuts_evicted: u32,
    stages_over_budget: u32,
}
```

### Extended Parquet Output

Add columns to `training/cut_selection/iterations.parquet`:

| Column                 | Type | Description                                       |
| ---------------------- | ---- | ------------------------------------------------- |
| `angular_deactivated`  | i32  | Cuts deactivated by angular pruning (per stage)   |
| `angular_clusters`     | i32  | Clusters formed by angular pruning (per stage)    |
| `budget_evicted`       | i32  | Cuts evicted by budget enforcement (per stage)    |
| `active_after_angular` | i32  | Active count after angular pruning, before budget |
| `active_after_budget`  | i32  | Active count after budget enforcement             |

### Solver Statistics Extensions

Add to `SolverStatistics`:

| Field                 | Type | Description                        |
| --------------------- | ---- | ---------------------------------- |
| `basis_padding_tight` | u64  | New cut rows set to NONBASIC_LOWER |
| `basis_padding_slack` | u64  | New cut rows set to BASIC          |

These allow measuring how often the basis padding logic changes behavior
vs the default all-BASIC padding.

## Validation

### Regression Tests

All 26 deterministic regression tests (D01–D26) must pass unchanged when:

- `angular_pruning` is absent or `enabled: false`
- `max_active_per_stage` is absent or null

The default configuration must produce bit-for-bit identical results.

### New Test Cases

#### S1: Angular Pruning

1. **Unit test**: Two cuts with cosine similarity 0.9999 and different
   intercepts → one deactivated (the inner one).
2. **Unit test**: Two cuts with cosine similarity 0.99 and threshold 0.999
   → neither deactivated (below threshold).
3. **Unit test**: Zero-norm cuts grouped and pruned to best intercept.
4. **Unit test**: Deterministic clustering order regardless of input
   shuffling.
5. **Integration test**: Full training run with angular pruning enabled,
   verify active count is lower than without.

#### S2: Budget

As specified in `docs/design/cut-budget-per-stage.md` validation section.

#### S3: Basis Padding

1. **Unit test**: Cut that is slack at state → BASIC status.
2. **Unit test**: Cut that is tight at state → NONBASIC_LOWER status.
3. **Unit test**: Cut with large negative slack (violated) →
   NONBASIC_LOWER status.
4. **Integration test**: Verify `basis_padding_tight` counter increases
   when many cuts are added.

### Performance Benchmarks

Use the 118-stage case from the problem statement:

| Metric                  | Baseline (LML1 only) | +S1     | +S1+S2  | +S1+S2+S3 |
| ----------------------- | -------------------- | ------- | ------- | --------- |
| Active cuts/stage       | measure              | measure | measure | measure   |
| Avg ms/solve (backward) | measure              | measure | measure | measure   |
| Total backward time     | measure              | measure | measure | measure   |
| Simplex iters/solve     | measure              | measure | measure | measure   |
| Total training time     | measure              | measure | measure | measure   |

Each strategy is added incrementally to isolate its individual impact.

## Consequences

### Benefits

- **Bounded LP size**: Budget guarantees O(budget + F × W) active cuts
  per stage regardless of iterations or forward passes.
- **Reduced redundancy**: Angular pruning removes near-duplicate cuts
  that waste LP capacity without improving the policy.
- **Faster warm-start convergence**: Basis-aware padding reduces the
  number of simplex pivots needed after cut additions.
- **Composable**: All three strategies are independent and compose in a
  pipeline. Each can be enabled/disabled individually.
- **Determinism-safe**: All strategies produce identical results
  regardless of MPI rank count or thread assignment.
- **Zero overhead when disabled**: Unconfigured strategies add no
  runtime cost. All new data structures are only allocated when the
  corresponding feature is enabled.

### Costs

- **Angular pruning**: O(A² × S) per stage — quadratic in active cuts.
  Acceptable for A ≤ 500, but may need optimization (e.g., LSH-based
  approximate nearest-neighbor) for very large pools. Monitor via the
  `selection_time_ms` output.
- **Budget eviction**: O(populated × log(populated)) per stage for
  sorting. Negligible.
- **Basis padding**: O(new_cuts × n_state) per solve for dot products.
  Negligible relative to simplex cost.
- **Policy quality**: Aggressive pruning or tight budgets may require
  more iterations to converge. The budget should be calibrated against
  steady-state active counts from long runs.

### Interaction Matrix

| Feature            | LML1        | Level1      | Dominated   | Budget      | Angular     |
| ------------------ | ----------- | ----------- | ----------- | ----------- | ----------- |
| LML1               | —           | N/A         | N/A         | Combines    | Combines    |
| Level1             | N/A         | —           | N/A         | Combines    | Combines    |
| Dominated          | N/A         | N/A         | —           | Combines    | Combines    |
| Budget             | Combines    | Combines    | Combines    | —           | Stacks      |
| Angular            | Combines    | Combines    | Combines    | Stacks      | —           |
| Basis Padding (S3) | Independent | Independent | Independent | Independent | Independent |

"Combines" = both run in sequence, complementary. "Stacks" = S1 runs
first, S2 caps what S1 didn't catch. "Independent" = S3 operates at
solve time, orthogonal to selection.

## Status

**IMPLEMENTED** (v0.4.4). Supersedes and incorporates
`docs/design/cut-budget-per-stage.md` (S2).
