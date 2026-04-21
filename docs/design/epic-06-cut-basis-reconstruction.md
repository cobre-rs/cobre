# Epic 06 — Cut-row basis reconstruction: activity-guided new-cut classification + slot-identity preservation

**Status:** Scope proposal for Epic 06 of `plans/backward-basis-cache`
**Depends on:** Epic 05 (unified per-scenario `BasisStore`) shipped and A/B #6 measured
**Target:** Reduce the residual +65% ω=0 iter≥2 pivot overhead that Epic 05 does not fully close
**Logged:** 2026-04-21 during Epic 05 refinement
**Replaces / subsumes:** `smarter-cut-initialization.md` (Option A path); `cepel-sc-adoption.md` Option C (identical mechanism, different framing)

---

## 1. Executive summary

Epic 03 A/B #3 measured that enabling the v0.5.0 ω=0-only backward basis cache
delivers **−13.5% total LP wall-time on `convertido_a`** (production scale: 50
forward passes × 5 iterations × 117 stages × 10 openings) **despite** a
**+65% increase in ω=0 iter≥2 mean pivot count** (2,623 with cache ON vs 1,588
with cache OFF). The win came from HiGHS's retained LU factorisation amortising
across ω≥1 and the forward pass; the pivot overhead is the measurable slack.

Epic 05 (unified per-scenario `BasisStore`) attacks the pivot overhead from one
angle: it replaces cross-scenario canonical broadcast with per-scenario state
matching and makes forward reads dimension-exact with respect to backward
writes. Epic 05's A/B #6 will measure how much of the +65% it closes.

Epic 06 attacks the **two remaining sources** of pivot overhead that Epic 05
does not touch, both rooted in how cut-row basis statuses are carried across
iterations:

- **Q1 source — new cut rows on the backward side.** Every iteration, `P`
  new cuts per stage (P = num forward passes, 50 on convertido) are appended
  to the LP via `add_rows`. Their basis statuses are filled with `BASIC`
  unconditionally by the HiGHS wrapper's dimension-extension path. The
  simplex then pivots to discover tight cuts. Epic 06 replaces this default
  with an **activity-counter-driven classifier** that guesses tight vs slack
  based on per-cut binding history.

- **Q2 source — cut-selection-induced positional misalignment.** When cut
  selection deactivates cuts at the end of an iteration, the baked template
  rebuilds with shifted positions. The stored basis copies into the new shape
  by position, not by slot identity. The slot-tracking machinery in
  `basis_reconstruct.rs` exists but is currently bypassed on the baked path
  (`std::iter::empty()` is passed as the delta iterator in `stage_solve.rs`).
  HiGHS repairs the misalignment via extra pivots. Epic 06 re-enables the
  **slot-identity reconciliation** by routing `cut_row_slots` through the
  baked path.

**Expected combined wall-time win: 5–15% additional LP-time reduction** over
Epic 05 HEAD, depending on which of Q1/Q2 dominates the residual overhead
(resolved by Epic 05's A/B #6 post-mortem). Composable with Epic 05's
unified-store win (separate mechanisms, different solve sites, no overlap).

Both changes preserve determinism, declaration-order invariance, and
cross-MPI bit-identity. Neither changes the LP selection semantics (no risk
of breaking the `bit-for-bit identical regardless of input ordering` hard
rule from `CLAUDE.md`).

---

## 2. Measurement context

### 2.1 Epic 03 A/B #3 baseline (convertido_a, 10 threads, release, 5 iterations)

| Metric                            | Cache OFF  | Cache ON (ω=0-only) | Delta    |
| --------------------------------- | ---------- | ------------------- | -------- |
| ω=0 iter≥2 mean pivots            | 1,588      | 2,623               | **+65%** |
| All-ω iter≥2 total backward LP ms | 10,112,036 | 8,769,619           | −13.3%   |
| Total LP solve ms (all phases)    | 12,570,416 | 10,877,415          | −13.5%   |
| Total training wall time ms       | 1,401,729  | 1,221,242           | −12.9%   |
| ω=0 cache-hit rate (iter≥2)       | N/A        | 1.000               | —        |
| Basis rejections                  | 0          | 0                   | —        |
| MPI broadcast cost fraction       | N/A        | 0.000%              | —        |

The cache-ON arm wins despite the pivot overhead because HiGHS retains its LU
factorisation across ω and across phases; a stable starting basis amortises
the factorisation across ~10× more solves than it costs in extra pivots.
Epic 06 asks whether we can capture the LU-amortisation win AND close the
pivot gap.

### 2.2 What Epic 05 closes (predicted, subject to A/B #6)

Epic 05's unified `BasisStore` eliminates two of the current design's
limitations:

1. **Dimension mismatch on forward reads.** Under v0.5.0, forward iter k+1 at
   (m, t) reads the iter-k forward capture (has pool size Y); the actual LP
   has Y+P cuts (iter-k backward added P). `reconstruct_basis` classifies P
   new cut rows via the slack heuristic. Under Epic 05, forward reads the
   iter-k **backward** capture at (m, t, ω=0) which was taken after the P
   cuts were added — dimensions match exactly, no reconstruction of cut
   rows needed on forward reads.
2. **Cross-scenario canonical on backward reads.** Under v0.5.0, every
   scenario's backward at (m, s, ω=0) reads the canonical rank-0 m=0 basis.
   Under Epic 05, each scenario reads its own (m, s) forward capture — a
   state-matched per-scenario basis.

What Epic 05 **does not** close:

- **Backward-side new-cut classification** — the P iter-(k+1) cuts added
  during iter k+1's backward sweep still hit the `add_rows` path and get
  `BASIC` statuses unconditionally.
- **Cut-selection-induced positional drift** — when cut selection deactivates
  cuts at end of iter k, the rebaked template's cut rows shift up; the
  stored basis's cut-row statuses copy positionally, not by slot identity.

Epic 06 targets exactly these two.

---

## 3. Problem decomposition

### 3.1 Q1 — New cut rows get `BASIC` unconditionally

**Code trace** (post-Epic-05):

1. Backward pass at step `t' = t-1` solving stage `s = t`:
   - `load_backward_lp` (`backward.rs:409-417`):
     ```rust
     ws.solver.load_model(succ.baked_template);
     if succ.cut_batch.num_rows > 0 {
         ws.solver.add_rows(succ.cut_batch);
     }
     ```
     Baked template has iter 1..k-1 cuts; `cut_batch` has P iter-k delta cuts.
     Final LP row count = `baked.num_rows + P`.
2. `run_stage_solve` calls `reconstruct_basis` with `std::iter::empty()` as
   the delta iterator (`stage_solve.rs:170-180`). The resulting basis has
   `baked.num_rows` row statuses — nothing for the P delta rows.
3. `ws.solver.solve(Some(&basis))` calls into the HiGHS wrapper. At
   `highs.rs:1144-1150`:
   ```rust
   let basis_rows = basis.row_status.len();     // baked.num_rows
   let lp_rows = self.num_rows;                 // baked.num_rows + P
   let copy_len = basis_rows.min(lp_rows);
   self.basis_row_i32[..copy_len].copy_from_slice(&basis.row_status[..copy_len]);
   if lp_rows > basis_rows {
       self.basis_row_i32[basis_rows..lp_rows].fill(HIGHS_BASIS_STATUS_BASIC);
   }
   ```
   The P new rows are filled with `BASIC`.
4. HiGHS accepts the dimensionally-consistent basis. Simplex pivots during the
   solve to identify which of the P new cuts are tight at the current
   primal/dual iterate.

**Why BASIC** (from `basis_reconstruct.rs:255-266`):

> New cut — classify as BASIC (slack) unconditionally to preserve
> `basic_count == num_row`. Each appended cut grows `num_row` by 1, so
> marking it BASIC grows `basic_count` by 1 and keeps the invariant exact.
> The simplex discovers tight cuts via pivots during the solve.

The invariant is required by HiGHS's `isBasisConsistent` check. BASIC is the
only status that preserves the count invariant while adding one row; any
fraction of tight guesses must be offset by demotion elsewhere in the basis.

**Why this is fixable** — if we knew ahead of time which of the P new cuts
are likely to be tight at the current state, we could mark those as
`NONBASIC_LOWER` and compensate by demoting an equal number of `BASIC` row
statuses from the carried-over baked rows (using a conservative demotion
heuristic that targets rows with low dual values). Net: fewer pivots to
discover tight cuts.

**Pivot savings estimate** — Epic 03 showed ω=1..9 sits at ~969 pivots
(cv=3.3%) while ω=0 iter≥2 sits at 2,623 (+65% vs cache-OFF baseline). The
excess ≈ 1,035 pivots per ω=0 iter≥2 solve. If activity-guided classification
closes half of that gap, wall-time savings ≈ 5%. If it closes all of it,
≈ 10%.

### 3.2 Q2 — Cut-selection-induced positional misalignment

**Code trace**:

1. **End of iteration k** — cut selection runs at `training.rs:1016-1083`:
   ```rust
   let deactivations: Vec<(usize, DeactivationSet, f64)> = (1..num_sel_stages)
       .into_par_iter()
       .map(|stage| strategy.select_for_stage(pool, states, iteration, stage))
       .collect();
   for (stage, deact, _) in deactivations {
       fcf.pools[stage].deactivate(&deact.indices);
   }
   ```
   `pool.deactivate` flips `active[i] = false` in the bitmap
   (`cut/pool.rs:384-393`). Coefficients, intercepts, metadata stay.
2. **Template rebake** (`training.rs:1157-1176`):
   ```rust
   for t in 0..num_stages {
       build_cut_row_batch_into(&mut bake_row_batches[t], fcf, t, ...);
       bake_rows_into_template(&templates[t], &bake_row_batches[t], &mut baked_templates[t]);
   }
   ```
   `build_cut_row_batch_into` iterates `fcf.active_cuts(stage)` at
   `cut/pool.rs:280-293` — slot-order filtered by `active[]`. Deactivated
   slots are skipped; later cuts **shift up** in row position.
3. **Iter k+1 forward at stage t** reads the stored basis. It was captured at
   iter k with `cut_row_slots = [slots 0..M where active at capture time]`.
   Stored `row_status` has M cut-row entries in slot-order (pre-deactivation).
4. `reconstruct_basis` (`basis_reconstruct.rs:206-214`):
   ```rust
   if stored.basis.row_status.len() >= target.base_row_count {
       out.row_status.extend_from_slice(&stored.basis.row_status[..target.base_row_count]);
   }
   ```
   **Positional copy** of the first `new_baked.num_rows` entries. If old slot
   5 was deactivated, new position 5 maps to old slot 6 — stored position 5
   was for old slot 5 (now dropped) — positions 5+ are all off by one.
5. `enforce_basic_count_invariant` (`basis_reconstruct.rs:278+`) fixes the
   total BASIC count by demoting trailing BASICs to LOWER, but does not fix
   per-position slot misalignment.
6. HiGHS accepts the dimensionally-consistent-but-positionally-misaligned
   basis. Simplex pivots to repair the resulting infeasibility/suboptimality.

**Why the slot-identity machinery is currently dormant** — the delta iterator
at `stage_solve.rs:170-180` is `std::iter::empty()`:

```rust
let recon_stats = reconstruct_basis(
    captured,
    ReconstructionTarget { base_row_count: baked.num_rows, ... },
    std::iter::empty(),        // <-- no slot-identity reconciliation for cut rows
    padding,
    ...
);
```

The `slot_lookup` scratch at `basis_reconstruct.rs:216-243` builds a slot→
position map from `stored.cut_row_slots`, but the consumer loop at
`basis_reconstruct.rs:247-268` only walks the `current_cut_rows` iterator,
which is empty. The preservation logic (`stats.preserved`) never fires on the
production baked path.

**The original design intent** — per the module docstring at
`basis_reconstruct.rs:1-19`:

> The legacy `pad_basis_for_cuts` helper used the LP row count as the only
> reconciliation key. If cut selection replaced one cut with another of equal
> count, the two bases were the same length but positionally misaligned —
> HiGHS received a basis with mismatched row statuses and crashed back to
> cold start (or, worse, warm-started with a corrupt basis).
>
> `reconstruct_basis` takes `CapturedBasis::cut_row_slots` as its key: each
> stored cut row carries the `CutPool` slot that generated it. The
> reconstruction walks the current LP's cut rows in order, looks each slot
> up in an O(1) scratch map, and either copies the stored status (slot found
> → preserved) or evaluates the cut at `padding_state` to assign tight/slack
> (slot not found → new).

The architecture shift to the **baked path** (cuts embedded as structural
rows in the baked template rather than appended via delta) made the delta
iterator naturally empty, and the slot-identity reconciliation stopped
firing — without a corresponding update to reroute slot identity through
the baked path.

**Why this is fixable** — route the active-cuts slot list through
`reconstruct_basis`. Instead of passing `std::iter::empty()`, pass an
iterator that yields `(slot, intercept, coefficients)` for each active cut
in slot-order. The existing `slot_lookup` machinery then correctly
preserves status by slot identity even when cut selection has shifted
positions.

**Pivot savings estimate** — cut selection's impact on pivot count depends
on how aggressively it runs. For `CutSelectionStrategy::Level1` with default
threshold, a ~5% deactivation rate per iteration is typical. On convertido's
~5,000-cut pool that's ~250 misaligned positions per stage per iteration
after selection runs. Repair cost per misaligned position ≈ 1–2 pivots.
Savings: ≈ 250–500 pivots per affected solve, perhaps 1/3 of the ω=0 iter≥2
excess → ~3% wall-time reduction.

### 3.3 Combined win estimate

Q1 and Q2 attack different sources with different magnitudes. Q1 fires on
every ω=0 iter≥2 backward solve (117 stages × 50 trial points × 4 iterations
per run on convertido = 23,400 solves). Q2 fires only after cut-selection
runs (every `check_frequency` iterations, typically 5-10). A lower bound on
combined savings is Q1 alone (~5–10%); an upper bound is Q1 + Q2 stacked
(~7–15%).

---

## 4. Proposed design

### 4.1 Part A — Activity-counter-driven new-cut classifier (attacks Q1)

**Data model — add `active_window: u32` to `CutMetadata`**

`CutMetadata` at `cut_selection.rs:57-85` currently carries `iteration_generated`,
`forward_pass_index`, `active_count: u64`, `last_active_iter: u64`. Add:

```rust
pub struct CutMetadata {
    pub iteration_generated: u64,
    pub forward_pass_index: u32,
    pub active_count: u64,
    pub last_active_iter: u64,
    /// Sliding-window bitmap: bit i set iff this cut was binding (dual > ε)
    /// at iteration `current_iter - i`. Bit 0 = current iteration.
    /// Initialised to 0; shifted left by 1 at end of each iteration;
    /// OR'd with 1 at bit 0 whenever the cut's dual exceeds
    /// `activity_epsilon` during a backward solve.
    pub active_window: u32,
}
```

**Memory cost** — 4 bytes per cut. On convertido's 35,100-generated-cut
pool: ~140 KB per rank. Negligible.

**Update rules**:

- **Bind event** — during `process_trial_point_backward`, when a cut row's
  dual at the current solve exceeds `activity_epsilon = 1e-6`, set bit 0 in
  its `active_window`. This is co-located with the existing
  `active_count += inc` update at `backward.rs:1331-1337`.
- **Shift event** — at end-of-iteration (after cut selection, before bake),
  perform a pool-wide sweep: for each `CutMetadata`, `active_window <<= 1`.
  Pre-allocated; no hot-path allocation.

**Cross-MPI reduction** — `active_window` must be OR-reduced across ranks
(union semantics: the union of ranks' observed activity). Extends the
existing metadata reduction at `backward.rs:1296-1336`. OR is commutative
and associative → determinism is preserved.

**Classifier rule in `reconstruct_basis`** — for cut rows not preserved by
slot identity (new cuts OR cuts whose slot was absent in the stored basis),
consult `active_window` on the cut's metadata:

```rust
const RECENT_WINDOW_BITS: u32 = (1u32 << 5) - 1;  // low 5 bits = last 5 iters

if metadata.active_window & RECENT_WINDOW_BITS != 0 {
    HIGHS_BASIS_STATUS_LOWER    // likely tight — give HiGHS the hint
} else {
    HIGHS_BASIS_STATUS_BASIC    // likely slack — original heuristic
}
```

**Count-invariance requirement** — marking a new cut `LOWER` reduces the
basis's total_basic by 1 relative to the BASIC-by-default behaviour.
HiGHS requires `total_basic == num_row` exactly. Compensation is required.

Two compensation schemes, in order of preference:

1. **Symmetric demotion in the baked tail** — for each new cut guessed as
   `LOWER`, demote one trailing `BASIC` cut-row in the baked tail to
   `LOWER` (tail order is slot-order; pick the cut with lowest
   `active_window` popcount to minimise false-negatives).
2. **Conservative guess budget** — cap the number of `LOWER` guesses at
   `budget = floor(remaining_basic / 10)` so that even if all guesses are
   wrong the repair pivots are bounded.

Scheme 1 is preferred (fine-grained, symmetric); Scheme 2 is a fallback if
Scheme 1's bookkeeping is intractable during refinement.

**Default window size** `k = 5` (low 5 bits). Configurable; matches CEPEL's
`k₂` calibration from the `cepel-sc-adoption.md` doc (the same parameter in
a different context).

### 4.2 Part B — Slot-identity preservation on the baked path (attacks Q2)

**Core change** — route `fcf.active_cuts(stage)` through `reconstruct_basis`
as a non-empty iterator, so the slot_lookup machinery fires on the baked
path.

**Code touch at `stage_solve.rs:170-180`**:

```rust
// BEFORE (current)
let recon_stats = reconstruct_basis(
    captured,
    ReconstructionTarget { base_row_count: baked.num_rows, ... },
    std::iter::empty(),              // <-- dormant slot-identity
    padding,
    &mut ws.scratch_basis,
    &mut ws.scratch.recon_slot_lookup,
);

// AFTER (Epic 06)
let recon_stats = reconstruct_basis(
    captured,
    ReconstructionTarget { base_row_count: template_num_rows, ... },  // <-- non-cut rows only
    inputs.pool.active_cuts(),       // <-- route slot identity
    padding,
    &mut ws.scratch_basis,
    &mut ws.scratch.recon_slot_lookup,
);
```

The `ReconstructionTarget.base_row_count` shrinks from `baked.num_rows`
(includes cuts as structural rows) to `template_num_rows` (non-cut rows
only). The cut rows at positions `[template_num_rows, baked.num_rows)` are
now handled by the iterator loop at `basis_reconstruct.rs:247-268`, which
uses slot identity.

**Complication — baked template row layout** — the baked template currently
places cuts as structural rows at the tail of the baked row block. The
iterator in `build_cut_row_batch_into` iterates `active_cuts(stage)` in
slot-order; the baked template's cut-row indices match this order. Routing
the same iterator through `reconstruct_basis` preserves the mapping.

**Complication — iter-k delta cuts on backward path** — during iter k's
backward at stage t, the baked template has iter 1..k-1 cuts; iter k cuts
are appended via `add_rows`. The appended delta rows are not in
`pool.active_cuts(stage)` at the moment of the solve (they haven't been
added to the pool yet at solve time — they're added after the solve via
`fcf.add_cut`). So the routed iterator correctly excludes them, and the
HiGHS-wrapper dimension-extension path continues to handle the delta rows
(now enhanced by Part A's classifier).

**Composability with Part A** — the routed iterator yields `(slot, intercept,
coefficients)` tuples. For each, the loop in `basis_reconstruct.rs:247-268`
either:

- Finds the slot in `slot_lookup` → copies the stored status (preserved).
- Does not find the slot → new cut; classify via **Part A's activity-window
  rule** (see §4.1) instead of the current unconditional BASIC.

### 4.3 Combined consumer loop

After both parts, the cut-row classification loop becomes:

```rust
for (target_slot, intercept, coefficients) in current_cut_rows {
    let metadata = pool.metadata[target_slot];
    let row_status_byte = if let Some(pos) = slot_lookup.get(target_slot).and_then(|o| *o) {
        // Slot preserved — copy stored status.
        let stored_row_idx = stored.base_row_count + pos as usize;
        stats.preserved += 1;
        stored.basis.row_status[stored_row_idx]
    } else if metadata.active_window & RECENT_WINDOW_BITS != 0 {
        // New cut with recent activity — guess tight.
        stats.new_tight += 1;
        HIGHS_BASIS_STATUS_LOWER
    } else {
        // New cut, no recent activity — default slack.
        stats.new_slack += 1;
        HIGHS_BASIS_STATUS_BASIC
    };
    out.row_status.push(row_status_byte);
}
```

---

## 5. Data model summary

| Field                                | Where                  | Purpose                           | Memory / scope               |
| ------------------------------------ | ---------------------- | --------------------------------- | ---------------------------- |
| `CutMetadata.active_window: u32`     | `cut_selection.rs`     | Sliding-window binding bitmap     | 4 B per cut × pool size      |
| `activity_epsilon: f64` (config)     | `config.rs`            | Dual threshold for bind detection | 8 B constant; default `1e-6` |
| `RECENT_WINDOW_BITS: u32` (constant) | `basis_reconstruct.rs` | Mask for "recent" classifier      | compile-time                 |

No new structs. No `CapturedBasis` changes. No MPI wire-format changes (the
broadcast carries `CapturedBasis`, not `CutMetadata`; `active_window` lives
in the pool which is MPI-reduced separately).

---

## 6. Implementation sketch (file-by-file)

**`crates/cobre-sddp/src/cut_selection.rs`**

- Add `active_window: u32` field to `CutMetadata` with a doc comment citing
  this design.
- Update `Default` / constructors to initialise to `0`.
- ~5 lines of mechanical change.

**`crates/cobre-sddp/src/backward.rs`**

- At the existing metadata-update site (~line 1331-1337, near
  `active_count += inc` and `last_active_iter = iteration`): extract the
  per-cut dual from the solve view, compare to `activity_epsilon`, and set
  bit 0 of `active_window` when binding.
- Extend the MPI metadata reduction (~line 1296-1336) to OR-reduce
  `active_window` across ranks alongside the existing `active_count` sum
  and `last_active_iter` max. OR is the correct reduction for a presence
  bitmap.
- ~15 lines.

**`crates/cobre-sddp/src/training.rs`**

- At end-of-iteration, after cut selection but before template rebake
  (~line 1155-1156): sweep each pool's metadata, `active_window <<= 1` per
  cut. This is pre-allocated scratch work, no hot-path allocation.
- ~8 lines.

**`crates/cobre-sddp/src/basis_reconstruct.rs`**

- Extend `reconstruct_basis` signature (or add a companion function) to
  accept a `&CutPool` parameter from which to read `metadata[slot].active_window`
  for new-cut classification.
- Update the classification branch in the consumer loop
  (`basis_reconstruct.rs:247-268`) per §4.3.
- Extend `ReconstructionStats` with a `new_tight_by_activity: u32` counter
  alongside the existing `new_tight`, `new_slack`, `preserved` fields.
- ~40 lines.

**`crates/cobre-sddp/src/stage_solve.rs`**

- Change the `current_cut_rows` argument from `std::iter::empty()` to
  `inputs.pool.active_cuts()` (line 176).
- Change `target.base_row_count` from `baked.num_rows` to
  `inputs.stage_context.templates[stage_index].num_rows` (the non-cut
  template row count at `~line 173`).
- Thread `inputs.pool` through to `reconstruct_basis` (the pool reference
  is already in `StageInputs`).
- Update `enforce_basic_count_invariant` to account for Part A's demotion
  scheme — either integrate the demotion directly or document why it is
  not needed (Scheme 1 maintains count invariance by construction).
- ~15 lines net.

**`crates/cobre-sddp/src/config.rs`**

- Add `activity_epsilon: f64` to the cut-management config; default `1e-6`.
- ~5 lines.

**Testing**

- Unit tests for the bitmap semantics: shift, OR-on-bind, MPI OR-reduction.
- Unit tests for the classifier: preserved / activity-hit (LOWER) /
  activity-miss + slack (BASIC) paths.
- Integration tests: extend `test_backward_cache_hit_rate` thresholds;
  confirm `reconstructed_basis_preserves_invariant_on_baked_truncation`
  still passes.
- D-suite byte-identity gate at iter=1 (all 27 cases); iter≥2 may drift on
  `solver_iterations.parquet` stats columns only (pivot counts change);
  extend `EXPECTED_DRIFTS.txt` with a 1-line reference to this epic.
- MPI parity at 1/2/4 ranks (hard gate).

Estimated total: ~90 lines of non-test code, ~150 lines of tests.

---

## 7. Determinism analysis

**Part A — activity-window update**

- `active_window |= 1` on bind and `active_window <<= 1` per iteration are
  deterministic for a given sequence of binding events.
- Binding events depend on per-solve dual values. For a fixed LP and fixed
  warm-start basis, HiGHS returns a deterministic optimal dual (non-
  degenerate) or a deterministic tie-broken dual (degenerate, via HiGHS's
  internal rule).
- MPI OR-reduction is commutative + associative → bit-identical across
  MPI configurations.

**Part B — slot-identity preservation**

- The routed iterator yields `(slot, intercept, coefficients)` in
  slot-order (stable under deactivation; cuts keep their slot index
  until the slot is overwritten by a new cut, which happens only after
  explicit pool maintenance).
- The `slot_lookup` scratch is pre-sized and deterministic.
- `reconstructed_basis` output is a deterministic function of
  `(stored_basis, current_cut_rows, padding_state)`, all of which are
  themselves deterministic.

**Net** — Epic 06 does not introduce any non-deterministic sources. The
`reconstructed_basis_preserves_invariant_on_baked_truncation` regression
test remains valid. The D-suite byte-identity gate is maintained at iter=1
(before any activity data accumulates); iter≥2 drift is expected only on
observability columns (pivot counts, basis statuses) — not on operational
outputs (hydro generation, thermal dispatch, costs).

---

## 8. Memory & performance cost

**Memory** — 4 bytes × pool size per rank. Convertido: ~140 KB per rank.
Negligible.

**Hot-path cost** — one `active_window |= 1` per binding cut per solve
(co-located with existing `active_count += inc`); one `active_window <<= 1`
per cut per iteration (pool-wide sweep, end-of-iteration); zero allocations.

**MPI reduction cost** — one extra `u32` per cut in the existing metadata
allreduce. Pool size × 4 bytes × num-stages per iteration. On convertido:
35,100 × 4 = 140 KB extra per allreduce call, amortised across the
existing per-stage sync. <0.1% bandwidth increase.

**Baked-path reconstruction cost** — the slot_lookup scratch is already
pre-allocated (`ScratchBuffers::recon_slot_lookup`). The consumer loop now
walks `pool.active_cuts(stage)` instead of an empty iterator; work per
solve is O(num_active_cuts), same as the prior positional copy. Net
compute cost is unchanged.

---

## 9. Composition with Epic 05

Epic 05 eliminates **pre-solve** dimension mismatch on forward reads by
using the dimension-exact backward capture. Epic 06 attacks **at-solve**
classification quality for cut rows that still need classification:

| Path                              | Epic 05 contribution               | Epic 06 contribution                                                                                  |
| --------------------------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Forward read (m, t) iter k+1      | Dimension match (0 new cut rows)   | Slot-identity preservation if cut-selection shifted positions                                         |
| Backward read (m, s) iter k+1 ω=0 | Per-scenario state match           | Both: activity-guided classification for P new iter-(k+1) delta cuts AND slot-identity for baked cuts |
| Backward read ω≥1                 | Unchanged (no stored_basis)        | Unchanged (no stored_basis)                                                                           |
| Simulation reads                  | Richer scenario-0 checkpoint basis | Same benefit as training paths                                                                        |

No overlap. Independent mechanisms. Wins stack.

---

## 10. Alternatives considered

### 10.1 Full SC (Seleção de Cortes, delayed constraint generation)

Proposed in the retired `cepel-sc-adoption.md` (Option A). Adds an inner
loop per backward solve that starts with a seed cut-set and iteratively
adds violated cuts until convergence. CEPEL measured 46–53% backward-pass
speedup on NEWAVE; discounted estimate for Cobre: 25–40%.

**Rejected for Epic 06** because:

1. Full SC produces the same LP optimum but different per-entity operational
   trajectories (CEPEL's §3.5.1: multiple-optima divergence). Breaks Cobre's
   `results must be bit-for-bit identical regardless of input entity
ordering` hard rule (`CLAUDE.md`).
2. Its benefit shrinks once Epic 05 + Epic 06 land (CEPEL's §4.3.3 argues
   their 60% toy-problem speedup became 46–53% on NEWAVE precisely because
   NEWAVE already had basis caches).
3. Complex inner-loop state machine; high implementation risk.

**Deferred to**: potential v0.6.0+ architectural discussion, conditional on
Epic 06 A/B #7 showing a ≥ 10% residual backward-pass cost that delayed-
constraint-generation could plausibly attack.

### 10.2 Option B of `cepel-sc-adoption.md` (activity-guided cut deactivation)

Uses `active_window` to drive deactivation priority (deactivate low-activity
cuts first). **Orthogonal to Epic 06's basis-classification target** — could
ship as a follow-up once `active_window` infrastructure is in place. Not in
Epic 06 scope; logged as future work.

### 10.3 Cross-opening basis aggregation (Scheme 3, median-opening)

Proposed in `cross-opening-basis-aggregation.md`. Selects the ω with
smallest Hamming distance to all other openings as the representative
capture. Targets the **column-status** dimension; Epic 06 targets the
**row-status** dimension. **Composable**; different attack surface.
Measured gain bounded by ω-to-ω variance (cv=3.3% on convertido), so
smaller magnitude. Deferred unless Epic 05 A/B #6 or Epic 06 A/B #7 shows
residual column-status-dominated overhead.

### 10.4 Remove `BasisStore` entirely (backward-basis-as-single-source)

Proposed in `backward-basis-as-single-source.md`. **Subsumed by Epic 05**
(unified per-scenario store) — Epic 05 already consolidates the two-store
architecture into one. Retained in the design-doc corpus as historical
context only.

### 10.5 Cut-row dual capture in `CapturedBasis` (Option C of `smarter-cut-initialization.md`)

Add per-row duals to the basis wire format; use prior duals to classify
new cuts on reconstruction. **Rejected** — the HiGHS `setBasis` C API is
statuses-only (2026-04-21 API spike confirmed); duals cannot be passed
back to the solver, so they would only serve as a heuristic input. The
activity-window counter achieves the same classification goal with:

- No wire-format extension (Epic 06 invariant preserved for free).
- Smaller per-cut memory footprint (u32 vs f64 per cut-row).
- Simpler removal path if the heuristic later turns out not to help.

---

## 11. Risks

- **R1 — Classification worse than BASIC default.** If the activity-window
  heuristic produces more `LOWER` guesses that turn out to be wrong (cuts
  not actually tight at the current state), the simplex pivots to repair
  from `LOWER` back to `BASIC` — potentially more pivots than the
  BASIC-default path. A/B #7 gates this. Mitigation: the new_tight-by-
  activity counter in `ReconstructionStats` provides per-solve telemetry
  to post-mortem a regression.
- **R2 — Slot-identity routing changes D-suite outputs.** Positional
  misalignment today already affects the warm-start basis, which affects
  pivot count and tie-breaking in degenerate LPs. Re-enabling slot identity
  may shift which LP tie HiGHS selects, producing different byte outputs
  on degenerate D-suite cases. Mitigation: extend `EXPECTED_DRIFTS.txt`
  with 1-line references for affected columns; aggregate operational
  outputs (hydro, thermal, cost) must remain byte-identical.
- **R3 — Count-invariance bookkeeping under activity classifier.** If Scheme
  1 (symmetric demotion) is intractable for a specific LP shape, fall back
  to Scheme 2 (bounded budget). Mitigation: A `debug_assert!` on the
  invariant after classification catches construction bugs; an integration
  test with a pathological cut-selection scenario (high deactivation rate
  - high new-cut rate) exercises the demotion edge cases.
- **R4 — Activity-window at iter=1 is empty**, making the classifier no
  better than the default on the first iteration. Acceptable — iter-1
  pivot count is small compared to steady-state iterations; the window
  fills quickly (within 5 iterations for k=5).
- **R5 — MPI OR-reduction correctness under degenerate ties**. If two ranks
  see different dual values for the same (stage, slot) pair due to
  degenerate-LP tie-breaking, the OR-reduction union is conservative
  (counts the cut as "bound on at least one rank"). Deterministic and safe;
  marginal overestimate of activity — not a correctness issue.
- **R6 — CI time for A/B #7**. The A/B requires a production-scale run on
  `convertido_backward_basis` at 10 threads, release mode. Same cost as
  Epic 05 A/B #6; no new cost.
- **R7 — Integration with a future Epic (activity-guided deactivation)**.
  Reusing `active_window` for both basis init and deactivation priority is
  clean, but if the two consumers want different window sizes, the single
  u32 field must accommodate both. Default `k = 5` is documented; deactivation
  can compute `popcount(active_window)` for finer ranking without touching
  the bit layout.

---

## 12. A/B #7 design

**Cases**: `convertido_backward_basis` (production scale) and D01 (edge
coverage).

**Arms**:

- A — Epic 05 HEAD (unified basis store shipped; no activity classifier;
  no slot-identity routing).
- B — Epic 06 HEAD (Parts A + B enabled).
- Optional C — Epic 06 with Part A only (isolates Q1 vs Q2 contribution).
- Optional D — Epic 06 with Part B only.

**Primary metrics**:

- ω=0 iter≥2 mean pivot count (target: strictly lower than Epic 05 HEAD).
- Total LP wall-time (target: ≥ 5% reduction vs Epic 05 HEAD).
- `ReconstructionStats.preserved` / `new_tight` / `new_slack` per solve
  (diagnostic, not a gate).

**Secondary metrics**:

- `basis_rejections` counter (must stay at 0 or within noise; elevated
  rejection would indicate a classification-invariance bug).
- Per-stage pivot distribution (catch regressions concentrated at specific
  stages).

**Gate criteria for go verdict**:

- Total LP wall-time reduction ≥ 5% vs Epic 05 HEAD on
  `convertido_backward_basis`.
- ω=0 iter≥2 mean pivots strictly lower than Epic 05 HEAD's measured value.
- D-suite byte-identity preserved on operational columns (hydro, thermal,
  cost); drift allowlist extended for solver-stats columns only.
- MPI parity at 1/2/4 ranks preserved.

**Null-result path** — if A/B #7 inside jitter (< 2% wall-time delta): revert
Parts A + B via `git revert`; retain `active_window` field only if it has
zero observable cost (set but unread). Close Epic 06 as not-pursued with a
decision record at
`docs/assessments/backward-basis-cache-epic-06-decision.md`. The same
null-result protocol that Epic 04 used.

---

## 13. Ordering & conditionality

Epic 06 **must wait for Epic 05** because:

1. Part A's "new cut rows" only exist on reads where Epic 05 hasn't already
   eliminated them. Epic 05's A/B #6 post-mortem quantifies the residual
   gap that Part A attacks.
2. Part B's slot-identity routing touches the same `reconstruct_basis` call
   site that Epic 05 modifies. Sequencing Part B after Epic 05 avoids merge
   conflicts and spares one round of D-suite reconciliation.

**Gating** — Epic 06 kicks off only if Epic 05's A/B #6 ships with a
measured residual overhead of > 5% vs the cache-OFF baseline (i.e., if
Epic 05 alone does not fully close the +65% gap). If Epic 05 closes the
gap to within 5%, Epic 06 is deferred to v0.6.0 discussion.

**Within Epic 06** — Parts A and B are independent. They can land in either
order or in parallel, but bundling them in the same A/B amortises the
D-suite reconciliation and MPI parity runs.

---

## 14. Open questions

1. **Window size `k`** — default 5 bits for the classifier's "recent" test.
   Larger `k` captures more history but includes older, potentially less-
   relevant activity. A/B #7 can evaluate `k ∈ {3, 5, 10}` as a sub-test.
2. **Demotion scheme** — Scheme 1 (symmetric demotion) is preferred; Scheme 2
   (budget cap) is the fallback. A/B #7 should report rejection rates and
   pivot counts for both to inform the production default.
3. **Config-gate or default-on** — the master plan's AD-5 spirit is "ship
   without a runtime flag". Epic 06 follows the same default unless A/B #7
   reveals a case-specific regression that motivates opt-in.
4. **Integration with `CutSelectionStrategy::Dominated`** — the dominated-cut
   selection strategy has richer per-cut information than Level1/Lml1. If
   dominated-cut information can feed the classifier, the activity-window
   may be complementary rather than primary. Deferred — Epic 06 ships the
   simpler activity-window first.
5. **Interaction with `enforce_basic_count_invariant`** — today's
   enforcement runs after `reconstruct_basis` and fixes excess-BASIC by
   demoting trailing rows. Epic 06's Scheme 1 demotes during
   reconstruction. The enforcement pass becomes a safety net (should fire
   zero demotions when Scheme 1 is correct). Confirm via
   `debug_assert!(enforcer_demoted == 0)` in tests.
6. **Activity epsilon calibration** — default `1e-6` matches the existing
   `cut_activity_tolerance` at `backward.rs:870`. Confirm stability across
   D-suite fixtures during refinement.

---

## 15. Related artifacts

- **v0.5.0 architectural baseline**: commit `7d67ea0` (Epic 03 ship);
  `plans/backward-basis-cache/00-master-plan.md` AD-1 through AD-6.
- **A/B #3 report (ω=0-only cache baseline)**:
  `docs/assessments/backward-basis-cache-ab3-convertido.md`.
- **Epic 03 go decision**:
  `docs/assessments/backward-basis-cache-decision.md`.
- **Epic 04 null-result precedent (rollback protocol reference)**:
  `docs/assessments/backward-basis-cache-all-omega-decision.md`.
- **Epic 05 unified-store scope** (prerequisite):
  `plans/backward-basis-cache/epic-05-unified-basis-store/00-epic-overview.md`.
- **Epic 05 A/B #6 (gate for Epic 06 kickoff; will exist after Epic 05
  ships)**: `docs/assessments/backward-basis-cache-unified-store-ab6.md`.
- **HiGHS `setBasis` API spike findings (2026-04-21)**: statuses-only;
  no dual hints accepted. Covered in §4.1 rationale and §10.5.
- **Cross-MPI parity script**:
  `scripts/test_per_opening_mpi_parity.sh`.
- **`basis_source` analyzer**: `scripts/analyze_basis_source.py`.
- **Key source files referenced in §6**:
  - `crates/cobre-sddp/src/basis_reconstruct.rs` (the hot-path entry point;
    Epic 06 re-enables its slot-identity machinery).
  - `crates/cobre-sddp/src/cut_selection.rs` (`CutMetadata`; Epic 06 adds
    `active_window`).
  - `crates/cobre-sddp/src/cut/pool.rs` (`CutPool.deactivate`,
    `CutPool.active_cuts`).
  - `crates/cobre-sddp/src/backward.rs` (metadata update + MPI reduction).
  - `crates/cobre-sddp/src/training.rs` (end-of-iteration sweep;
    template rebake).
  - `crates/cobre-sddp/src/stage_solve.rs` (the call site where Part B's
    iterator change lands).
  - `crates/cobre-solver/src/highs.rs` (the dimension-extension path that
    today fills new cut rows with BASIC; Epic 06 supersedes it for the
    baked-path cut rows, retains it for delta-cut appended rows).

---

## 16. Replaces / consolidates

This document is the authoritative scope reference for Epic 06. It subsumes
the following earlier proposals in `docs/design/`:

- **`smarter-cut-initialization.md`** — the Option A path (activity counter
  on `CutMetadata`) is incorporated as §4.1 of this document. Option B
  (state-bucketed activity) and Option C (per-row duals in `CapturedBasis`)
  are evaluated in §10 and rejected with rationale. The original file may
  be retained as historical context or retired.
- **`cepel-sc-adoption.md`** Option C (identical mechanism to this doc's
  §4.1). Option B (activity-guided deactivation) is a composable follow-up
  (§10.2). Option A (full SC) is deferred to v0.6.0+ (§10.1). The original
  file may be retained as the CEPEL technical reading reference, or retired
  once Epic 06 ships.

The other two `docs/design/` files address different attack surfaces and
remain relevant:

- **`backward-basis-as-single-source.md`** — subsumed by Epic 05 (unified
  per-scenario store). Retain as historical context or retire after Epic 05
  closes.
- **`cross-opening-basis-aggregation.md`** — attacks the column-status
  dimension (Epic 06 attacks row-status); orthogonal and composable. Retain
  as a future-work reference.
