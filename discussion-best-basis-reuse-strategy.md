Here we will describe in more detail some insights regarding the relationship between sddp algorithm steps and basis reuse. You should read and add your knowledge of our solver interface trait and HiGHS internals to ensure we are in the right direction.

- The SDDP algorithm works by iteration in two main steps: the forward and backward passes.

- The forward pass simulates a decision process 'forward in time', sampling uncertainties (noises) and transfering states from stages (PAR lags, hydro storages). This pass uses the current Future Cost Function for making the decisions. The result is a set of sampled visited states that are worth to improve our decision on them via backward pass.

- The forward pass is highly parllelizable. So we sample many states in a single iteration. Current production cases use 192 forward passes, so after the forward pass step, we have 192 new sampled states by stages to process in the backward pass.

- The backward pass improves the Future Cost Function at the visited states. It does so by calculating new benders cuts, which are added as constraints to the LP. The actual Future Cost Function is just a set of constraints. We generate new benders cuts at each stage, and the cut generation process involves a sequential solving of the LPs of each stage

- To generate a new cut, you build the stage's LP at the same initial state that was visited in the forward step. Then, you use a pre-calculated set of noises (openings) to solve the LP under some different uncertainties. Currently we use 20 openings in the production case. Then, we have 20 LP results. We extract some properties of the results and make a weighted average based on the risk measure that was selected to generate what we call the "average cut". This is the 'single-cut' formulation of the SDDP algorithm

- We generate a new cut for every sampled state in the forward pass. So if we visited 192 states in the forward pass, we will have 192 new cuts, which are 192 new constraints to each stage's LP.

- This has an inherently serial processing cost. Also, we have to generate all the cuts and communicate them between all ranks before moving from stage 't+1' to stage 't' in the backward pass.

Now, some insights on this algorithm that might pass unnoticed. We might not implement the proper treatment for all of them, so you MUST compare with the current code before taking any conclusions

- For easier abstraction, we like to see the LP as made of two main parts: the 'structural' part, and the benders' cuts. The structural part describes the problem, balance equations, decision variables, etc. The cuts are constraints that are added on each iteration.

- When we are in the forward pass at some iteration 'k', we store the basis of each forward scenario and stage. This basis might be useful in the future. Basis reuse is a MUST to have in the sddp algorithm. HOWEVER, there are several caveats in here that could either improve or hurt performance.

- Our current implementation uses the basis we had in the previous iteration 'k-1' to warm-start the solution for each LP in iteration 'k'. It also uses the basis that we stored in the forward pass to warm-start the respective solution in the backward pass too. However, it isn't as simple as it seems.

- The final step of the backward pass is the cut selection. Is changes the set of benders' cuts that we have in our problem. Usually we have a good starting basis for the structural part of our LP, but guessing a good basis for the cuts is hard. Different cut constraints might be binding depending on the state variables that we have, which vary based on the past trajectory. Also, the sampled noise uncertainty might have an effect too. For this reason, we usually consider the basis for the benders' cut part as BASIC by default (inactive). This is one possible point for improvement.

- Because of this, we must update the initial basis that we pass to the solver before moving to the next forward iteration to properly reshape it to match the amount of benders' cuts that we have for each stage. This is done by editing the baked stage template.

- In a real produciton case, we generate 192 new cuts per iteration, and the cut selection might add or remove other hundreds of them, for each stage. So we must really think on how to properly initialize this step to ease the solver job.

- When we reach the backward step, currently we apply the same basis that we reached in the forward step. However, we only do it once, for the first opening. The others are solved sequentially, after the first. The code is like this because our only basis set method in the past triggered a highs behavior of considering an alien basis, and this killed performance.

- We know that each of the openings are not correlated with each other. Their sampled noises might vary a lot, so I don't know at which point it is worth to warm start solution for opening 1 from what we got to opening 0. We might have to instrument per-opening stats to we know how the solver is behaving on each opening.

- Despite setting the basis in opening 0, what happens after we finish processing stage 't+1' also matters. I think we currently don't update the baked template in the middle of the backward pass. This forces us to constantly 'manually' add the cuts that were generated on this iteration, and this might hurt performance. I'm afraid we are doing many repeated steps of padding BASIC values to the basis in many different places. We have to evaluate updating the template once in the middle of the backward stages and once more after the cut selection.

Remember that we have to make the HiGHS job as easy as we can. We know from evaluating other software (NEWAVE) that their backward step performance is better, when compared to ours. They run the same production case with the backward pass taking around 4x the forward time with 20 openings, and we are currently at a 10x ratio. Lets improve this by properly instrumenting, benchmarking, and solving the bottlenecks that we find.

---

## Concrete next steps

The plan below is ordered so each step's outcome informs the next. Ship steps 1 and 2 independently; gate 3 on what 1 reveals; 4 is a parallel workstream.

### Step 1 — Per-opening `SolverStatistics` instrumentation (BLOCKING)

Nothing else gets a go/no-go answer until we can see, for each individual opening, what its solve looks like — pivots, wall time, basis set time, retries, every counter we already track. Current `SolverStatistics` aggregates across all solves in a phase; we need the full per-opening breakdown so we can query any dimension later, not just a pre-chosen split.

**Why this is #1.** Every downstream choice — cut-tightness classification, B.2 per-opening basis storage, breaking vs keeping the chained warm-start, even "is opening 7 an outlier because of its noise vector" — depends on seeing the pivot pattern across the full opening grid. A pre-decided `{ω=0, ω>0}` split hides whether the chain degrades monotonically, whether there's a convergence pattern at late openings, or whether a specific opening is consistently bad across trial points.

**Design — per-opening accumulator at the SDDP layer, solver stays single-valued:**

- Keep `cobre_solver::SolverStatistics` single-valued as today. The solver crate has no notion of "opening" and shouldn't grow one.
- At the `cobre-sddp` layer, maintain a `Vec<SolverStatistics>` per (iteration, phase, stage), sized to `n_openings` for backward (and simulation), to 1 for forward/LB. Around each backward solve in `process_trial_point_backward`, snapshot `solver.statistics()` before and after, compute the delta, and accumulate into the slot for that `omega`.
- The existing stats-delta helper (`solver_stats.rs::subtract` or equivalent) is the building block — this step reuses it rather than adding new fields.
- Thread-safety: each worker owns its solver, so the snapshot/delta loop is worker-local. The merge across workers at the end of a stage uses the same `sum` pattern as today's phase aggregation, summing by `(stage, opening)` instead of `(stage,)`.

**Parquet schema — explode rows, don't widen columns:**

- `solver/iterations.parquet` today has one row per `(iteration, phase, stage)`.
- New shape: one row per `(iteration, phase, stage, opening)`. Add an `opening: Int32` column. For forward and lower_bound rows, `opening = -1` (or NULL — pick whichever is cleaner for downstream consumers, probably NULL since it carries no-data semantics).
- All existing columns (`lp_solves`, `simplex_iterations`, `solve_time_ms`, `basis_set_time_ms`, `basis_offered`, `basis_rejections`, etc.) stay as-is — every field of `SolverStatistics` is preserved per-opening.
- This keeps queries trivial: `GROUP BY iteration, phase, stage` recovers the old schema; `GROUP BY iteration, phase, opening` gives the per-opening view; full-resolution queries are one `WHERE` clause.

**MPI allreduce:**

- The solver-stats flat-buffer packer (`pack_stats_to_buffer` in `solver_stats.rs`) currently serializes one `SolverStatistics`. Extend to serialize `Vec<SolverStatistics>` with a length prefix. Only backward/simulation frames are larger; forward/LB frames stay at length 1.
- Cross-MPI parity test: same per-opening values after allreduce regardless of rank count.

**Memory:**

- 10 workers × 20 openings × ~100 bytes/SolverStatistics ≈ 20 KB per stage during accumulation; cleared between stages. Parquet output: `num_iters × num_stages × num_openings` rows ≈ 23,600 rows on `convertido_base`, ≈ 450k rows on production. Well under 100 MB of parquet.

**Instrumentation call site:**

- `process_trial_point_backward` already has `omega` in scope (`backward.rs:461`). The opening loop is the natural accumulator boundary — snapshot stats before, solve, snapshot after, delta into `per_opening_stats[omega]`.
- Forward and LB paths: accumulate into slot 0, same API, emit with `opening = NULL`.

**Acceptance:**

- New parquet schema present on `convertido_base` training run. Row count = `num_iters × num_phases × num_stages × num_openings` for backward; `num_iters × num_stages` for forward + lower_bound.
- Invariant test: `SUM(lp_solves) GROUP BY iteration, phase, stage` matches today's single-row values.
- D-suite byte-identical under the instrumentation (counters only, no behavior change).
- Cross-MPI parity: per-opening values identical across rank counts.
- Python parity: `cobre-python` produces identical per-opening rows.

**Expected readout (hypotheses, to be falsified or confirmed by the real data):**

- Opening 0 carries disproportionate pivots (hypothesis: ~30-40% of backward pivots from 1/20 of solves).
- Openings 1..K-1 have a decay curve — pivots high at ω=1, dropping as the chain finds a stable pivot pattern, possibly rising again for late openings with noise far from the chain's current vertex.
- Specific openings may be outliers (e.g. tail-quantile noise vectors).

The exact shape of that curve determines whether Step 2 (cut-tightness classification) or Step 3 (B.2 per-opening basis) is the bigger lever, and whether "basis for ω=0 only" is sufficient or whether full per-opening storage is needed.

---

### Step 2 — Cut-tightness classification for new cut rows

Restore a tight/slack classification for new delta cut rows during `reconstruct_basis`. Today every new cut is marked `BASIC` unconditionally (`basis_reconstruct.rs:288`); when the true optimum has ~K of the M delta cuts binding, simplex spends pivots discovering them one by one.

**Safety on the non-alien path — the finding:**

`Highs::setBasis` with `alien = false` runs `isBasisConsistent` which enforces exactly:

1. Size match: `|col_status| == num_col`, `|row_status| == num_row`.
2. Total BASIC count equals `num_row`.

It does NOT check non-singularity, dual feasibility, or bound compatibility. So marking new cuts LOWER is safe from the acceptance check IF AND ONLY IF the total basic count stays exact. Misclassification only degrades performance — HiGHS handles singular bases internally via `handleRankDeficiency`. The worst case is equivalent to today's all-BASIC default.

**Design:**

1. During `reconstruct_basis` on the legacy arm (and/or baked arm for delta cuts), evaluate each new cut's slack at `padding.state`:
   - slack = intercept + coeff·x − θ_value (already computed as `theta_value` in `stage_solve.rs:159`)
   - if `slack <= tolerance`: classify `LOWER` (tight)
   - else: classify `BASIC` (slack, cut inactive)
2. Enforce the invariant. Let `K_lower = count(new cuts classified LOWER)`. The naive reconstruction now has a deficit of `K_lower`. Two options:
   - **Option A (cheap, default):** promote `K_lower` old cuts from LOWER to BASIC. Pick the old cuts with the highest evaluated slack (least likely to be tight now), breaking ties by slot-id for determinism.
   - **Option B (expensive, reject for now):** re-evaluate every old cut at `padding.state` and reconcile the full tight/slack set. O((K_old+K_new) × n_state) per solve. Probably too much.
3. Add a new `enforce_basic_count_deficit` helper symmetric to the existing `enforce_basic_count_invariant` (which handles excess). Single scan; `K_lower` demotions in the opposite direction.
4. Re-enable the `new_tight` counter in `ReconstructionStats` (today it's always 0 per the comment at `basis_reconstruct.rs:283-285`). Bump a stat every time classification selects LOWER.

**Determinism:**

- Evaluation is a pure function of stored `(intercept, coefficients)`, `padding.state`, and `tolerance`. Same inputs → same outputs across workers. ✓
- Tie-breaking for which old cut to promote: slot-id is deterministic. ✓
- Add a D-suite byte-identity gate under the new classifier before flipping a default.

**Rollout:**

- Ship behind a config flag (`training.solver.cut_classification: {all_basic, tight_slack}`). Default `all_basic` initially.
- Add instrumentation (pivots/solve with each classifier) and flip default only after confirmed perf win on `convertido_base` and at least one production-scale case.

**Expected gain (hypothesis, gated on Step 1):**

If opening 0 dominates backward pivots and the dominant cost is binding-cut discovery, classification could cut opening-0 pivots from ~800 down to ~100–200. That alone would move backward wall from 306s → ~180s on the test case — a real dent in the 10× ratio.

**Acceptance:**

- Byte-identical LP optimum vs. `all_basic` baseline (same optimal x, obj, duals on D-suite).
- `basis_non_alien_rejections` stays at 0 on `convertido_base` and on a production-scale case.
- Per-opening pivot counts (from Step 1) show a measurable drop on ω=0.

---

### Step 3 — B.2: per-(scenario, stage, opening) basis storage (GATED on Step 1)

Extend the backward basis store from `(m, s)` → `(m, s, ω)`. Each opening's solve warm-starts from its own previous-iteration optimal basis. Since backward openings are a _fixed_ noise grid across iterations, the stored basis is optimal for exactly the same noise vector — only the cut set drifts, which `reconstruct_basis` already handles.

**Gating criterion (from Step 1 readout):**

- Pursue B.2 only if openings 1..K-1 pivot counts are consistently >50–75 per solve (meaning chained warm-start is not recovering the locality we want).
- If ω=0 alone dominates, a narrower "per-opening basis for ω=0 only" variant is a fraction of the cost for most of the benefit.

**Scope (when pursued):**

- New `BackwardBasisStore` parallel to the existing forward-pass store, indexed `[scenario × num_stages × num_openings + stage × num_openings + ω]`. Payload = existing `CapturedBasis`.
- Capture site: after each backward solve, `get_basis` into the opening-specific slot.
- Read site: in `process_trial_point_backward`, change `stored_basis` selection to `basis_store_bwd.get(m, s, omega)` with fallback to the forward-pass store (or None) for the first iteration when backward storage is empty.
- Clear the chain-warm-start assumption: under B.2 we install a stored basis at every opening, not just ω=0.
- Memory projection: log `num_scenarios × num_stages × num_openings × sizeof_basis_slot` at training start; warn if above a configurable threshold. For `convertido_base` (10×118×20 ≈ 23.6k slots × ~10 KB) = ~250 MB; for production (192×118×20 ≈ 450k) ≈ 4.5 GB.

**Determinism + reproducibility:**

- Already satisfied by epic-04 (`clear_solver_state` per trial point). Per-opening basis install adds no new non-determinism path.
- D-suite byte-identity gate mandatory.

**Acceptance:**

- D-suite byte-identical across worker counts (1/2/4/8) and MPI configurations (per the epic-04 contract).
- Per-opening pivot counts drop significantly vs. Step-1 baseline.
- Memory usage within projected envelope on `convertido_base` and production cases.

---

### Step 4 — Address the 30% backward-pass load imbalance (PARALLEL TO 1-3)

Timing parquet shows `bwd_load_imbalance_ms = 90.5s` of the 306s backward wall on `convertido_base` (29.6%). This is pure waiting, orthogonal to per-solve optimization.

**Investigation targets:**

- Is the imbalance concentrated at specific stages (e.g. first/last) or spread evenly?
- Does trial-point work stealing balance within a stage but get stuck at inter-stage barriers?
- Does per-stage `add_rows` serialize in a way that penalizes workers that finish early?

**Defer to a separate plan** once Step 1 ships — the instrumentation from Step 1 will also surface per-worker timing breakdowns that inform this.

---

## Out of scope

- **Mid-pass template rebake.** Investigated and deprioritized: the delta cut count per successor LP is constant (= M trial points) across the whole backward sweep, not accumulating. A mid-pass rebake would not shrink any LP's row count for the remaining stages, and new delta cuts don't match stored basis slots from prior iterations regardless of baking timing.
- **Phase C (stage-major forward).** Warm-start-observability doc claims this is needed to match NEWAVE's ratio. Tabling: if Steps 2+3 move the ratio from 10× → ~4× as hypothesized, Phase C isn't needed.

## Open questions requiring code reading before Step 2 design finalization

- Does `load_backward_lp` at `backward.rs:329-345` call `add_rows(cut_batch)` with cut rows ordered deterministically across workers? (Must be — but double-check slot→row_index mapping.)
- On the baked arm, where exactly does the delta-cut row status get set to BASIC by default? (Appears to happen via the auto-pad in `HighsSolver::solve_with_basis`, `highs.rs:1225-1226`. If so, Step 2's classification must happen _before_ that pad, i.e. during `reconstruct_basis` or via a subsequent override in `run_stage_solve`.)
- Do we want Step 2's classification to apply to ω=0 only (where reconstruction runs) or also to ω≥1 (currently no reconstruction, just chained warm-start)? Probably ω=0 only — ω≥1 inherits the updated-post-pivot basis from ω=0 which is structurally different.
