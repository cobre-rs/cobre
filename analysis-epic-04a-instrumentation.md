# Analysis of Epic-04a Solver Instrumentation

**Case:** `~/git/cobre-bridge/example/convertido_arch`
**Data sources (10-iter rerun on 2026-04-19):**

- `training/solver/iterations.parquet` — 24,590 rows (per-opening solver stats)
- `training/solver/retry_histogram.parquet` — solver-retry events
- `training/cut_selection/iterations.parquet` — 1,170 rows (cut-selector behaviour)
- `training/timing/iterations.parquet` — 10 rows (per-iter training-loop wall breakdown)

**Shape:** 10 iterations × 118 stages × 20 openings × 10 trial points, 1 scenario group.
Reusable analyzer: `scripts/analyze_solver_iterations.py <case-output-dir>` — reproduces
every table in this document.
**Purpose:** Falsify or confirm the hypotheses in `discussion-best-basis-reuse-strategy.md`
before proceeding to epic-05, and decide where the next perf lever should land.

---

## TL;DR

The parquet confirms the **10× backward/forward wall-time ratio** (measured **8.36× overall, 8.66× in
steady state**) and refines where that cost lives. Two headline findings re-shape the roadmap:

1. **The ω=0 dominance hypothesis is falsified.** Opening 0 carries **9.7 % of backward pivots and
   12.2 % of backward time** — only ~2× the fair share of a uniform 1/20, not the ~30–40 % the
   discussion conjectured. Step 2 (cut-tightness classification) therefore has a **~6 % backward-time
   ceiling** on this case, not the 40 % the doc's hypothetical suggested.
2. **The real mass is in ω ≥ 1** — 87.8 % of backward time, sitting at a remarkably **flat ~164
   pivots/solve across ω=1..19**. The chain warm-start from ω=0 is working and reaches a stable
   cost by ω=1 (no decay across the chain). This means **Step 3's (per-opening basis) ceiling is
   bounded not by today's ω>0 cost but by how much lower than the "chain fixed point" a
   same-noise-previous-iter basis could push it.**

Both steps are worth doing, but their sizing is now inverted vs. the hypothesis: Step 3 is the larger
lever, but it has to beat an already-decent chain warm-start.

---

## Raw numbers

### Workload

| Phase       | Solves      | Pivots         | Total ms      | piv/solve | ms/solve |
| ----------- | ----------- | -------------- | ------------- | --------- | -------- |
| Forward     | 5,900       | 4,676,406      | 125,473       | 792.6     | 21.27    |
| Backward    | 117,000     | 20,240,876     | 1,048,490     | 173.0     | 8.96     |
| Lower bound | 100         | 27,477         | 1,132         | 274.8     | 11.32    |
| **Sum**     | **123,000** | **24,944,759** | **1,175,095** | —         | —        |

Backward/forward wall ratio: **8.36 ×** (8.66 × excluding warm-up iter 1).
Per-solve cost: **forward is 2.37× slower than backward** — the 8× ratio comes from **19.8× more
solves** on the backward side (1 per opening × 20 openings), not from individual solves being harder.

### Per-opening backward breakdown (aggregated across 5 iterations × 117 stages × 10 trial points)

| ω     | piv/solve | ms/solve  | % of backward time |
| ----- | --------- | --------- | ------------------ |
| **0** | **335.7** | **21.86** | **12.2 %**         |
| 1     | 155.6     | 8.90      | 5.0 %              |
| 2     | 165.6     | 9.22      | 5.1 %              |
| 3     | 170.4     | 9.39      | 5.2 %              |
| 4     | 160.9     | 8.70      | 4.9 %              |
| 5     | 162.7     | 8.72      | 4.9 %              |
| 6     | 172.6     | 9.08      | 5.1 %              |
| 7     | 163.2     | 8.46      | 4.7 %              |
| 8     | 169.3     | 8.70      | 4.9 %              |
| 9     | 163.5     | 8.28      | 4.6 %              |
| 10    | 161.3     | 8.22      | 4.6 %              |
| 11    | 162.2     | 8.09      | 4.5 %              |
| 12    | 167.2     | 8.10      | 4.5 %              |
| 13    | 172.3     | 8.20      | 4.6 %              |
| 14    | 154.2     | 7.40      | 4.1 %              |
| 15    | 164.2     | 7.73      | 4.3 %              |
| 16    | 164.3     | 7.57      | 4.2 %              |
| 17    | 165.5     | 7.63      | 4.3 %              |
| 18    | 165.6     | 7.63      | 4.3 %              |
| 19    | 163.5     | 7.37      | 4.1 %              |

**ω=0 = 2.04 × the mean of ω=1..19.** Every ω>0 row is within [0.92 ×, 1.05 ×] of the ω>0 mean —
**the chain warm-start reaches a stable fixed point at ω=1 with no decay along the chain**.

### Per-iteration dynamics

| iter | ω=0 piv/solve | ω>0 piv/solve | ω=0 ms | ω>0 ms  | ω>0 / total |
| ---- | ------------- | ------------- | ------ | ------- | ----------- |
| 1    | 461.0         | 138.2         | 31,908 | 145,636 | 82.0 %      |
| 2    | 312.1         | 166.1         | 23,071 | 179,430 | 88.6 %      |
| 3    | 318.4         | 167.7         | 25,598 | 194,602 | 88.4 %      |
| 4    | 304.4         | 171.6         | 24,116 | 190,186 | 88.7 %      |
| 5    | 282.7         | 178.5         | 23,186 | 210,755 | 90.1 %      |

**Two opposite trends:**

- ω=0 pivots **fall 39 %** over 5 iterations (461 → 283) as stored forward bases get better.
- ω>0 pivots **rise 29 %** (138 → 179) as the cut set grows and the chain's fixed point creeps up.

Extrapolating to a production-length run, ω=0 and ω>0 per-solve costs may converge. **Step 2's
relative value shrinks over time; Step 3's grows.**

### Basis-reconstruction counters (ω=0 only — ω>0 never calls `setBasis`)

| iter | offered | preserved | new_tight | new_slack | demotions | consistency_fails |
| ---- | ------- | --------- | --------- | --------- | --------- | ----------------- |
| 1    | 1,170   | 0         | 0         | 11,600    | 0         | 0                 |
| 2    | 1,170   | 11,600    | 0         | 11,600    | 0         | 0                 |
| 3    | 1,170   | 23,200    | 0         | 11,600    | 0         | 0                 |
| 4    | 1,170   | 26,200    | 0         | 11,600    | 0         | 0                 |
| 5    | 1,170   | 26,650    | 0         | 11,600    | 0         | 0                 |

- `basis_new_tight = 0` **everywhere** — confirms _every_ new-cut row is currently marked BASIC
  (slack) by default. This is exactly the Step-2 target.
- `basis_consistency_failures = 0` — `setBasis` with `alien=false` is never being rejected.
- `basis_demotions = 0` — the `enforce_basic_count_invariant` helper has nothing to do (today's
  all-BASIC classifier naturally keeps the BASIC count balanced; demotions would only appear if new
  cuts got classified as LOWER).
- Retries across 117 k backward solves: **2 level-0 retries**, **0 failures**. Solver retry budget
  is essentially idle — the safeguard pipeline is not masking any hidden pathology.

### Wall-time breakdown outside `solve_time_ms`

| Component            | ms        |
| -------------------- | --------- |
| `solve_time_ms`      | 1,048,490 |
| `load_model_time_ms` | 0         |
| `add_rows_time_ms`   | 0         |
| `set_bounds_time_ms` | 0         |
| `basis_set_time_ms`  | **44**    |

The baked-template path eliminates load/add/set_bounds cost entirely. `basis_set_time_ms = 44 ms`
is **0.004 %** of backward wall — the "setBasis is expensive" concern from older HiGHS behaviour is
not visible in these counters. **This removes one of the motivations for keeping basis-install
confined to ω=0**: the install itself isn't the cost; the pivot count after install is.

### Stage homogeneity

| Stage bucket   | ω=0 piv/solve | ω>0 piv/solve | total ms |
| -------------- | ------------- | ------------- | -------- |
| early (0–38)   | 348.1         | 174.0         | 354,866  |
| middle (39–78) | 331.1         | 156.4         | 346,015  |
| late (79–117)  | 328.4         | 163.4         | 347,608  |

Work is **evenly spread across stages**. No single stage bucket dominates. Worst individual stage
(71) is 2.7 × the best individual stage (65), but there's no structural hot-spot.

---

## Hypothesis-by-hypothesis verdict

| Hypothesis (from discussion doc)                          | Verdict       | What the data shows                                                                          |
| --------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------- |
| ω=0 carries ~30–40 % of backward pivots                   | **Falsified** | 9.7 % pivots, 12.2 % time                                                                    |
| ω=1..K-1 have a decay curve, rising for late openings     | **Falsified** | Flat at 164 ±5 piv/solve, σ < 5 %                                                            |
| Specific tail-quantile openings are outliers              | **Falsified** | No opening is > 1.05 × the ω>0 mean                                                          |
| Opening 0 alone may suffice for a per-opening basis store | **Refined**   | Wrong frame — ω=0 is not the dominant cost; ω>0 IS                                           |
| Step 2 could cut ω=0 from ~800 pivots to ~100–200         | **Mis-sized** | ω=0 is already 335, not 800. Ceiling ≈ 170 pivots saved per ω=0 solve = **~6 % of backward** |
| `basis_non_alien_rejections` stays at 0 on production     | **Confirmed** | 0/117,000 rejections                                                                         |
| `setBasis` itself is expensive (older concern)            | **Falsified** | `basis_set_time_ms` = 0.004 % of backward wall                                               |

---

## Savings sizing

Back-of-envelope ceilings on this case (1,048 s backward):

| Lever                                             | Target population | Ceiling assumption                                                    | Estimated savings                                 |
| ------------------------------------------------- | ----------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| **Step 2** — cut-tightness classification         | ω=0 (12.2 %)      | ω=0 pivots drop from 335 → 164 (match ω>0 chain-fixed-point)          | **65 s = 6.2 %**                                  |
| **Step 3-lite** — per-opening basis, ω=0 only     | ω=0 (12.2 %)      | ω=0 pivots drop 40 % (analogous to current iter-to-iter gain)         | **~50 s = 4.8 %**                                 |
| **Step 3-full** — per-opening basis, all ω        | ω>0 (87.8 %)      | ω>0 pivots drop 40 % below chain fixed-point (aspirational)           | **~368 s = 35.1 %**                               |
| **Step 3-full** — per-opening basis, all ω        | ω>0 (87.8 %)      | ω>0 pivots drop 20 % (more realistic — chain is already near-optimal) | **~184 s = 17.6 %**                               |
| **Step 4** — load-imbalance (not in this parquet) | backward wall     | Prior timing parquet showed ~29.6 % imbalance                         | **up to ~29 %** (separate instrumentation needed) |

The Step 3-full numbers are **uncertain bounds, not predictions**. The flat ω>0 curve at 164 pivots
(with no decay ω=1 → ω=19) tells us the chain warm-start has already found a stable basis
configuration within 1 opening. Per-opening basis can beat that only if the same-noise previous-iter
optimal basis is closer to the current iter's optimum than this-iter ω=0's post-pivot basis — which
is a real but modest improvement. A 20–40 % ω>0 reduction is the plausible band; 60 % would require
evidence we don't yet have.

---

## Recommendation for the next epic

**Do Step 2 cheap, gate Step 3 on a narrower experiment, park Step 4 for a focused plan.**

### 1. Ship Step 2 as a small patch (not an epic)

Cost: a single helper (`enforce_basic_count_deficit`), a tight-slack classifier invocation in
`reconstruct_basis`, a config flag, a D-suite byte-identity gate. The data confirms:

- `basis_consistency_failures = 0`: safety headroom is real.
- `basis_new_tight = 0` today: the opportunity is real.
- Demotions machinery is idle today: the "promote K_lower old cuts to BASIC" path is net-new and
  need to keep invariant-enforcement tight — but it's a scoped change.

Ceiling is ~6 % backward time (this case) and the value **decays with iteration count** because ω=0
is already learning from the stored forward basis. It's a "cheap, finite, worth having" win — not a
headliner. **Don't build an epic around it; make it a 1–3 ticket patch.**

### 2. Before committing to Step 3-full, run a focused probe

Before we spend an epic on `BackwardBasisStore[m × s × ω]` and 4.5 GB of memory on production, we
should falsify or confirm the **"per-opening basis beats chain fixed-point"** assumption. Two
cheap probes:

- **Probe A: break the chain.** At every ω (not just ω=0), re-install the ω=0-stored basis rather
  than letting simplex inherit post-pivot state. Measure ω>0 piv/solve vs. today. If chain-inherited
  is materially better (today's baseline), the chain is carrying most of the warm-start value and
  per-opening basis has limited headroom. If roughly equal, per-opening basis has real room.
- **Probe B: single-opening per-opening store** (the "ω=0 only" Step 3-lite). Add an iter-to-iter
  stored ω=0 basis path at the backward layer, alongside the forward-stored basis. Measure the
  same-noise iter-to-iter gain — this is the upper bound for what Step 3-full would deliver on
  ω>0 (which has the same noise across iterations).

Both probes reuse existing machinery. Neither needs a plan.

### 3. Sideline Step 3 behind the probe result

If Probe B shows < 15 % per-opening ω>0 reduction, Step 3-full is not worth the memory and
complexity — better ROI is elsewhere (Step 4, or forward-pass parallelism).

If Probe B shows ≥ 25 %, Step 3-full is a solid epic. **Memory must be sized before committing.**
Production: 192 × 118 × 20 × ~10 kB ≈ **4.5 GB per rank** — unaffordable on the typical MPI layout
unless we shard by scenario across ranks (which the epic would need to spec explicitly).

### 4. Load-imbalance (Step 4) deserves a separate instrumentation pass

`iterations.parquet` has no per-worker timing. The 29.6 % `bwd_load_imbalance_ms` figure in the
discussion came from a different timing parquet. Before designing a fix, instrument per-worker and
per-stage timing in the same parquet shape, then decide.

---

## Data quality / follow-ups

- `load_model_time_ms`, `add_rows_time_ms`, `set_bounds_time_ms` are **identically zero** across
  all 12,295 rows. This is consistent with the fully-baked simulation + baked backward path. **Epic
  05 (baked templates only)** should keep these at zero; any non-zero row is a regression signal
  worth adding as a CI check.
- `basis_offered = lp_solves` at ω=0, zero elsewhere: matches the single-install model. If Step
  3-full ever lands, this should become `lp_solves` at every ω.
- Iter 1 forward has 2,141 piv/solve (5 × iter-2's 505). Expected — cold start — but worth noting
  for the "per-iteration steady state" baseline any future benchmark uses.

---

## Appendix — hypothesis-reset cheat sheet

If writing the epic-05+ roadmap before new probe data, use these re-sized assumptions:

- ω=0 is **a modest 12 % slice** of backward, shrinking with iter count.
- ω>0 is **a flat 88 % slice**, each ω contributing uniformly ~4.5 %.
- Chain warm-start reaches fixed point at ω=1; no ω=19 tail.
- `setBasis` overhead is negligible (0.004 % of backward wall).
- `basis_non_alien_rejections` never fires — safety budget is untouched.
- Forward per-solve is 2.4 × backward per-solve; the backward/forward ratio is driven by
  **solve count**, not **per-solve cost**.

---

## 10-iteration re-run: scaling check (2026-04-19)

Re-ran the same case for **10 iterations** instead of 5 to test the extrapolation from the
first report. Every directional prediction held; every magnitude was off by < 15 %.

### Prediction vs. reality

| Claim from 5-iter report                     | 10-iter result                                       | Verdict |
| -------------------------------------------- | ---------------------------------------------------- | ------- |
| ω=0 piv/solve continues to drop iter-to-iter | 283 → 263 (−7 %) from iter 5 → iter 10               | ✓       |
| ω>0 piv/solve continues to rise iter-to-iter | 179 → 186 (+4 %) — flattening                        | ✓       |
| ω=0/ω>0 gap narrows                          | gap 104 (iter 5) → 77 (iter 10)                      | ✓       |
| bw/fw ratio grows toward 10×                 | 8.36× → 9.03× overall, 9.56× at iter 10              | ✓       |
| ω=0 share of backward shrinks                | 12.2 % (5-iter) → 10.6 % (10-iter), 9.3 % at iter 10 | ✓       |
| Flat ω>0 pivot curve (no decay across chain) | Still flat: σ < 5 % across ω=1..19 at iter 10        | ✓       |
| ω=0 is ~2× ω>0 mean                          | 1.74× at 10-iter aggregate (was 2.04× at 5-iter)     | ✓       |

### Dynamics (10 iterations)

| iter | ω=0 piv | ω>0 piv | gap | ω=0 ms share | bw/fw |
| ---- | ------- | ------- | --- | ------------ | ----- |
| 1    | 461     | 138     | 323 | 17.7 %       | 6.94× |
| 2    | 312     | 166     | 146 | 11.2 %       | 7.41× |
| 3    | 318     | 168     | 151 | 11.6 %       | 9.04× |
| 4    | 304     | 172     | 133 | 11.2 %       | 8.97× |
| 5    | 283     | 179     | 104 | 9.8 %        | 9.35× |
| 6    | 278     | 179     | 99  | 10.0 %       | 9.39× |
| 7    | 274     | 181     | 93  | 9.8 %        | 9.78× |
| 8    | 262     | 183     | 79  | 9.3 %        | 9.91× |
| 9    | 251     | 182     | 69  | 8.8 %        | 9.86× |
| 10   | 263     | 186     | 77  | 9.3 %        | 9.56× |

**ω>0 appears to asymptote around 185–190 piv/solve** — the rate of growth slows from +20 %
(iter 1→2) to +2 % (iter 9→10). This tightens the Step-3-full ceiling: per-opening basis must
reduce pivots below the chain's stable fixed point, which at iter 10 is ~186. Chain warm-start
has most of the value; per-opening basis fights for the last ~30–60 pivots/solve.

---

## New levers revealed by timing & cut-selection parquets

The analyzer now also ingests `training/timing/iterations.parquet` and
`training/cut_selection/iterations.parquet`. Two headline findings change the roadmap:

### 1. Backward load imbalance is a first-class lever (Step 4, confirmed)

Training-loop wall breakdown (totals across 10 iterations):

| Component                                | ms      | % of total wall |
| ---------------------------------------- | ------- | --------------- |
| `backward_wall_ms`                       | 324,657 | **90.4 %**      |
| &nbsp;&nbsp;`bwd_setup_ms`               | 39,122  | 10.9 %          |
| &nbsp;&nbsp;`bwd_load_imbalance_ms`      | 96,415  | **26.8 %**      |
| &nbsp;&nbsp;`bwd_scheduling_overhead_ms` | 27      | 0.0 %           |
| `forward_wall_ms`                        | 31,532  | 8.8 %           |
| &nbsp;&nbsp;`fwd_load_imbalance_ms`      | 6,202   | 1.7 %           |
| `cut_selection_ms`                       | 0       | 0.0 %           |
| `cut_sync_ms` / `mpi_allreduce_ms`       | 0       | 0.0 %           |
| `lower_bound_ms`                         | 2,595   | 0.7 %           |

**bwd_load_imbalance_ms is 29.7 % of backward_wall_ms** — exactly matching the 29.6 % figure
quoted in the discussion doc. This is _pure waiting_; no work done. Its ceiling is comparable to
Step 3-full's aspirational estimate:

| Lever                                        | Ceiling (this case)    | Comment                                          |
| -------------------------------------------- | ---------------------- | ------------------------------------------------ |
| Step 2 (cut classification, ω=0 only)        | 4.5 % of backward      | 100 s saved — shrinks further with iter count    |
| Step 3-full conservative (per-opening basis) | 17.9 % of backward     | 398 s saved — 20 % pivot reduction on ω>0        |
| Step 3-full aspirational (per-opening basis) | 35.8 % of backward     | 796 s saved — 40 % pivot reduction (unconfirmed) |
| **Step 4 (load-imbalance recovery)**         | **29.7 % of backward** | **96 s saved today; scales linearly with iters** |

Step 4 should **not** be deferred. It's roughly as large as the aspirational Step 3-full
estimate, but with none of the memory/complexity cost — and the measurement is concrete, not
hypothetical.

### 2. Cut-selector behaviour is healthy; its correlations with pivots are mild

Cut-selection per-iteration averages (across 117 stages):

| iter | populated | active_before | deactivated | active_after | total sel_ms |
| ---- | --------- | ------------- | ----------- | ------------ | ------------ |
| 1    | 20.0      | 10.0          | 0.0         | 10.0         | 0.020        |
| 2    | 30.0      | 20.0          | 0.0         | 20.0         | 0.053        |
| 3    | 40.0      | 30.0          | 7.4         | 22.6         | 0.079        |
| 5    | 60.0      | 33.1          | 9.0         | 24.2         | 0.091        |
| 10   | 110.0     | 39.0          | 7.9         | 31.1         | 0.101        |

- **Selector is fast** — total `selection_time_ms` across all stages × iterations is ~1 ms.
  Step-4 rewrites can ignore selector cost.
- **Active set grows slowly**: 10 → 31 active cuts per stage over 10 iterations despite 110
  cuts being populated. Selector is doing meaningful pruning.
- **`budget_evicted` and `active_after_budget` are NULL** — active-cut budget was never hit on
  this case. Production's budget behaviour will need its own re-analysis.

Pearson correlations (cut-set at iter _k_ ↔ solver behaviour at iter _k+1_, n=1,044):

| driver               | r(ω=0 piv) | r(ω>0 piv) | r(total piv) |
| -------------------- | ---------- | ---------- | ------------ |
| `cuts_populated`     | −0.231     | +0.130     | +0.104       |
| `cuts_active_before` | −0.154     | +0.224     | +0.203       |
| `cuts_deactivated`   | −0.158     | +0.019     | +0.003       |
| `cuts_active_after`  | −0.092     | +0.289     | +0.273       |
| `selection_time_ms`  | −0.078     | +0.018     | +0.010       |

Interpretation:

- **Active-cut count mildly drives ω>0 pivots up** (+0.29 for `cuts_active_after`). More cuts
  = bigger LP = more pivots. The effect is modest — not a 1:1 scaling.
- **Active-cut count mildly drives ω=0 pivots down** (−0.09 to −0.23). Likely a confound: stages
  with many active cuts are "mature", so their stored forward basis is closer to the current
  iter's optimum. This actively argues against interpreting ω=0 cost as "cut-classification is
  the bottleneck" — if it were, more cuts would mean more mis-classified pivots, not fewer.
- **Cut count explains < 10 % of pivot variance** (max r² ≈ 0.084). Most per-stage pivot
  variation comes from something else — likely stage LP topology + trajectory.

Bucket comparisons (p75 split):

| Condition                           | n   | ω=0 piv/solve | ω>0 piv/solve |
| ----------------------------------- | --- | ------------- | ------------- |
| HIGH deactivation (≥ p75 = 10 cuts) | 313 | 267.3         | 175.5         |
| LOW deactivation (< p75)            | 731 | 290.7         | 177.7         |
| LARGE active set (≥ p75 = 27 cuts)  | 268 | 272.9         | 198.8         |
| SMALL active set (< p75)            | 776 | 287.4         | 169.5         |

**Heavy cut pruning doesn't hurt solver performance** on either ω=0 or ω>0 — in fact it
correlates with slightly _fewer_ ω=0 pivots. This falsifies a subtle worry from the discussion
doc: "guessing a good basis for the cuts is hard … we usually consider the basis for the cuts
part as BASIC by default." The data shows cut-selection's effect on basis carry-over is small
even when ~10 cuts/stage are deactivated per iteration.

---

## Updated production-case projection (192 trial points × 50 iterations)

Using the 10-iter steady-state rates, and noting that per-iteration backward grows with cut
count until the active-cut budget binds:

- Per-iter backward solves at production: 192 × 117 × 20 = 449,280 (vs. 23,400 here) — **19.2×
  more solves per iter**.
- Per-iter backward pivots at iter 50 steady state: extrapolating ω>0 ~ 190–210 piv/solve,
  ω=0 ~ 220–250 (decaying) — total piv per iter ≈ 90M (vs. 4.2M here).
- 50 iterations → cumulative backward wall ≈ 20 × 5 × 246 s ≈ **24,600 s ≈ 6.8 h** single-rank.
  Production's actual MPI layout will divide this across ranks.

**Implications for each lever at production scale:**

| Lever                         | Projected production savings    | Confidence                  |
| ----------------------------- | ------------------------------- | --------------------------- |
| Step 2 (cut classification)   | ~3–5 % backward (≈ 10–20 min)   | High — linear in ω=0 share  |
| Step 3-lite (ω=0 basis store) | ~4–6 % backward                 | Medium — needs probe        |
| Step 3-full conservative      | ~15–20 % backward (≈ 1 h)       | Medium — ω>0 near asymptote |
| Step 3-full aspirational      | ~30–35 % backward (≈ 2 h)       | Low — optimistic target     |
| **Step 4 (load imbalance)**   | **~25–30 % backward (≈ 1.7 h)** | **High — 29.7 % confirmed** |

**Memory cost reminder for Step 3-full at production**: 192 × 118 × 20 × ~10 KB ≈ **4.5 GB per
rank**. Unaffordable on typical MPI layouts without sharding by scenario across ranks.

---

## Revised recommendation for post-epic-04a work

1. **Step 4 (load imbalance) is now the #1 lever.** The measurement is concrete (29.7 % of
   backward), the savings scale linearly with iter count, and there's no memory penalty. Make
   this an epic with dedicated per-worker instrumentation.
2. **Step 2 (cut classification) is a cheap 1-3 ticket patch**. Ship it. Ceiling ~4.5 %. Value
   decays with iter count, but the implementation is contained.
3. **Step 3-full stays gated** behind a probe run. The 10-iter ω>0 asymptote at 186 piv/solve
   tightens its ceiling — the realistic band is 15–20 % backward, not 35 %. The memory cost
   (4.5 GB/rank at production) demands a sharding design before committing.
4. **Watch the `bwd_setup_ms` slice** (10.9 % of wall). Epic 05 (baked templates only) likely
   addresses part of it. Instrument what remains post-05 to see if there's a separate lever.

---

## Script usage

Run against any case output dir:

```bash
python3 scripts/analyze_solver_iterations.py <case-output-dir>
# or
python3 scripts/analyze_solver_iterations.py --parquet path/to/iterations.parquet
```

The analyzer picks up companion parquets (`cut_selection/`, `timing/`, `retry_histogram`) when
they sit in the expected training-output layout.
