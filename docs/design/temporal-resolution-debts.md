# Temporal Resolution Debts & Design Gaps

**Date:** 2026-04-11 (revised 2026-04-12, structural revision 2026-04-12,
status update 2026-04-13, production-decomp update 2026-04-14, temporal-resolution-cd closure 2026-04-15)
**Status:** All patterns complete (v0.4.3 + production-decomp + temporal-resolution-cd)
**Context:** Cobre supports variable-length stages, but the stochastic pipeline
assumes uniform monthly resolution in several places. This document catalogs
the known debts and design gaps that must be addressed before non-monthly
studies (weekly, quarterly, mixed-resolution DECOMP-like) can be supported.

### Resolution history

| Debt                          | Status                 | Version | Commit    | Notes                                                                               |
| ----------------------------- | ---------------------- | ------- | --------- | ----------------------------------------------------------------------------------- |
| 1 (month0 bug)                | **Resolved**           | v0.4.3  | `316feab` | SeasonMap three-tier fallback replaces month0()                                     |
| 2 (observation alignment)     | **Resolved**           | v0.4.5  | `0c1aace3` | Layer 2 aggregation implemented; observation-to-season aggregation for Pattern D    |
| 3 (past_inflows metadata)     | **Resolved**           | v0.4.5  | `c149c738` | Optional season_ids field on HydroPastInflows for lag-resolution validation         |
| 4 (season_id validation)      | **Resolved**           | v0.4.3  | `9c37529` | Rules 27–30 (range, coverage, consistency, contiguity)                              |
| 5 (noise sharing)             | **Resolved**           | v0.4.5  | `7146930c` | Noise group precomputation + ForwardSampler + opening tree sharing                  |
| 6 (lag accumulation)          | **Resolved**           | v0.4.5  | `97f9655` | StageLagTransition precomputation + accumulate_and_shift hot-path function          |
| 7 (PAR behavior doc)          | **Resolved**           | v0.4.3  | `ae4403a` | mdBook page documents behavior and validation rules                                 |
| 8 (historical residuals)      | **Resolved**           | v0.4.3  | `d4a604d` | NoiseMethod::HistoricalResiduals with hash-based window selection                   |
| 9 (external standardization)  | **Resolved**           | v0.4.5  | `8f8cf73` | standardize_external_inflow uses StageLagTransition for frozen-lag + accumulation   |
| 10 (recent_observations)      | **Resolved**           | v0.4.5  | `08c1a32` | RecentObservation type + IO parser + accumulator seeding for mid-season starts      |
| 12 (terminal-stage FCF)       | **Resolved**           | v0.4.5  | `667c240` | BoundaryPolicy config + boundary cut loader + inject_boundary_cuts                  |
| 13 (multi-res PAR transition) | **Resolved**           | v0.4.5  | `eaa1dd27` | StageLagTransition downstream fields + ring buffer + transition rebuild              |

Additionally, a latent bug was discovered and fixed during implementation:
window year normalization (`1aa6ed6`) — `build_observation_sequence` now
normalizes year offsets so the first study stage gets `year_offset=0`.

---

## Executive summary

This document catalogs 13 debts and 2 opportunities across Cobre's temporal
resolution handling. The work falls into four layers:

1. **Production DECOMP critical path** (Debts ~~4~~→~~6~~→~~9~~→~~10~~→~~12~~):
   ~~lag accumulation~~, ~~external standardization~~, ~~mid-season starts~~,
   and ~~terminal-stage FCF~~. **All resolved** in the production-decomp plan.
2. **Mixed-res PAR correctness** (Debts ~~1~~, ~~5~~, ~~8~~): ~~noise sharing~~,
   ~~`month0` bug fix~~, and ~~historical residuals as opening tree noise~~.
3. **Defense-in-depth validation** (Debts ~~2~~, ~~4~~): ~~observation-to-season
   alignment~~ with multi-resolution aggregation, and ~~season_id consistency
   enforcement~~ across heterogeneous season spaces.
4. **Multi-resolution generalization** (Debt ~~13~~): ~~monthly→quarterly PAR
   transition with lag resolution conversion at the boundary.~~

**Pattern A (monthly uniform) is fully resolved as of v0.4.3.** Debts 1, 4,
7, and 8 are closed. Debt 2 Layer 1 validation is implemented; Layer 2
aggregation resolved in the temporal-resolution-cd plan.

**Pattern B (production DECOMP) is fully resolved as of the production-decomp
plan (2026-04-14).** Debts 6 (lag accumulation), 9 (external standardization),
10 (recent_observations), and 12 (terminal boundary cuts) are closed. D28
DECOMP integration test verifies end-to-end correctness with weekly+monthly
stages, External scenarios, recent_observations, and non-uniform scenario counts.

**Pattern C (generic mixed-resolution + PAR) is fully resolved as of the
temporal-resolution-cd plan (2026-04-15).** Debts 5 (noise sharing) and 3
(past_inflows metadata) are closed. D29 integration test verifies weekly
stages with PAR(1) noise sharing.

**Pattern D (monthly-to-quarterly multi-resolution) is fully resolved as of
the temporal-resolution-cd plan (2026-04-15).** Debts 2 Layer 2 (observation
aggregation), 13 (PAR lag transition), and 3 (past_inflows metadata) are
closed. D30 integration test verifies monthly-to-quarterly multi-resolution
with observation aggregation and lag transition.

All debts are resolved. All four study patterns — A, B, C, and D — are fully
functional as of v0.4.5.

All proposed hot-path changes are zero-allocation with O(n_hydros) per-stage
cost. The accumulator design degenerates to current behavior for uniform
monthly studies with negligible overhead (one multiply by 1.0).

~~Debt 7 is documentation-only.~~ Resolved in v0.4.3. Opportunity 11
(external initial state sampling) eliminates pre-study burn-in for long-term
planning.

---

## Study patterns and applicability matrix

This document serves three distinct study patterns. Not all debts apply to
all patterns — the matrix below shows which are critical, relevant, or not
applicable for each.

### Pattern A: Monthly uniform (NEWAVE-like)

All stages are monthly. One `season_id` per stage. No sub-monthly stages.
This is the current production pattern for Cobre.

### Pattern B: Production DECOMP (all-External scenarios)

Weekly stages (Saturday–Friday) for the current month with a **single
deterministic external forecast** per week, followed by a monthly stage
with an **external scenario tree** (e.g., generated by GEVAZP or a prior
Cobre run). Months beyond the horizon are captured by a **terminal-stage
external FCF** (boundary cuts from a monthly model). All inflows are
External — the PAR model provides the baseline for standardization but
does not generate noise during the forward pass.

Key characteristics:

- Weekly stages have `n_scenarios = 1` (deterministic forecast)
- Monthly stage has `n_scenarios = N` (external stochastic tree)
- `recent_observations` seeds the lag accumulator with observed partial-month data
- Rolling revisions (rv0 → rv1 → rv2) reduce weekly stages as weeks pass
- Terminal-stage boundary cuts from a monthly model (Cobre→Cobre coupling)

### Pattern C: Generic mixed-resolution (weekly stages + PAR)

Weekly SDDP stages using the monthly PAR model for scenario generation.
Users who need weekly cut boundaries (e.g., weekly storage arbitrage value)
but use PAR rather than external forecasts. Weekly stages may contain
parallel blocks for sub-weekly decision granularity.

Key characteristics:

- Noise sharing (Debt 5) is required — independent draws fabricate variability
- Frozen lags + accumulation (Debt 6) maintain monthly lag resolution
- Backward pass opening tree must share noise consistently with forward pass

### Pattern D: Long-term multi-resolution (monthly→quarterly)

Monthly stages for the near-term horizon (years 1–5) transitioning to
quarterly stages for the long-term horizon (years 6–10+). Each resolution
has its own PAR model fitted at the appropriate aggregation level. The
lag accumulator handles the monthly→quarterly boundary by aggregating
monthly inflows into quarterly lags at the transition point.

Key characteristics:

- Two PAR models: monthly (12 seasons) and quarterly (4 seasons)
- Disjoint `season_id` ranges (0–11 monthly, 12–15 quarterly) in a single
  `Custom` `SeasonMap`
- Lag resolution transition at the monthly→quarterly boundary (Debt 13)
- Observation aggregation: monthly history aggregated to quarterly for
  fitting the quarterly PAR model (Debt 2)

### Applicability matrix

| Debt                          | A: Monthly uniform         | B: Production DECOMP                 | C: Mixed-res + PAR  | D: Multi-res (mo→qtr) |
| ----------------------------- | -------------------------- | ------------------------------------ | ------------------- | --------------------- |
| 1 (month0 bug)                | ~~N/A~~ ✅ v0.4.3          | N/A (no Historical sampling)         | ✅ v0.4.3           | ✅ v0.4.3             |
| 2 (observation aggregation)   | ✅ L1 v0.4.3, L2 v0.4.5   | N/A (no PAR estimation from history) | ✅ v0.4.5           | ✅ v0.4.5             |
| 3 (past_inflows metadata)     | Low priority               | Low priority                         | ✅ v0.4.5           | ✅ v0.4.5             |
| 4 (season_id validation)      | ✅ v0.4.3                  | ✅ production-decomp                  | ✅ v0.4.3           | ✅ v0.4.3             |
| 5 (noise sharing)             | N/A (monthly stages)       | N/A (External scenarios)             | ✅ v0.4.5           | ✅ v0.4.5             |
| 6 (lag accumulation)          | N/A (monthly stages)       | ✅ production-decomp                 | ✅ production-decomp | ✅ production-decomp  |
| 7 (PAR behavior doc)          | ✅ v0.4.3                  | N/A                                  | ✅ v0.4.3           | ✅ v0.4.3             |
| 8 (historical residuals)      | ✅ v0.4.3                  | N/A                                  | ✅ v0.4.3           | ✅ v0.4.3             |
| 9 (external standardization)  | N/A (no External)          | ✅ production-decomp                 | If External used    | If External used      |
| 10 (recent_observations)      | N/A                        | ✅ production-decomp                 | If mid-season start | N/A                   |
| 12 (terminal-stage FCF)       | N/A                        | ✅ production-decomp                 | If coupling needed  | If coupling needed    |
| 13 (multi-res PAR transition) | N/A                        | N/A                                  | N/A                 | ✅ v0.4.5             |

**Critical paths by pattern:**

- **Pattern B (production DECOMP):** ✅ **Fully resolved.**
  ~~Debt 4 → Debt 6 → Debt 9 → Debt 10 → Debt 12~~ (all closed) +
  non-uniform scenario count support via V3.4 relaxation with padding.
- **Pattern C (mixed-res + PAR):** ✅ **Fully resolved.**
  ~~Debt 5 → Debt 6~~ (both closed). Noise group precomputation + ForwardSampler integration
  + opening tree sharing. D29 integration test verifies end-to-end.
- **Pattern D (multi-resolution):** ✅ **Fully resolved.**
  ~~Debt 2 → Debt 4 → Debt 6 → Debt 13~~ (all closed). Observation aggregation,
  lag accumulation, downstream ring buffer, and transition rebuild.
  D30 integration test verifies monthly-to-quarterly multi-resolution.

---

## Debt 1: `month0()` hardcoding in Historical sampling pipeline

> **Status: RESOLVED** in v0.4.3 (`316feab`). Both `discover_historical_windows`
> and `standardize_historical_windows` now use a three-tier SeasonMap fallback
> chain instead of `month0()`. The `season_map` parameter is threaded from
> `setup.rs` through both training and simulation call sites. Monthly studies
> produce bit-identical results; non-monthly SeasonMap configurations now map
> observations correctly.

**Severity:** Code defect — produces wrong results for any non-monthly study.

**Locations:**

- `cobre-stochastic/src/sampling/window.rs:146` — `discover_historical_windows`
- `cobre-stochastic/src/sampling/historical.rs:420` — `standardize_historical_windows`

**Problem:** Both functions map historical observation dates to season IDs via
`date.month0()` (chrono: 0=January, 11=December). This assumes season_id ==
calendar month, which is only true for monthly studies.

**Correct pattern exists:** The PAR estimation pipeline (`fitting.rs:199`) uses
a two-tier fallback: `find_season_for_date(&stage_index, date)` for in-study
dates, then `season_map.season_for_date(date)` for historical dates. This
works for any cycle type.

**Root cause:** `setup.rs` never passes `SeasonMap` to the Historical pipeline.
Neither `discover_historical_windows` nor `standardize_historical_windows`
accept a `SeasonMap` parameter.

**Fix:**

1. Add `season_map: Option<&SeasonMap>` parameter to both functions.
2. Replace `date.month0()` with the same fallback chain from `fitting.rs`.
3. Thread `system.policy_graph().season_map.as_ref()` from `setup.rs`.
4. Add non-monthly test cases (at minimum quarterly, 4-season).

**Impact if unfixed:** For non-monthly cycles, window discovery maps
observations to wrong seasons → valid windows rejected or invalid windows
accepted. Standardization writes eta values to wrong season slots → incorrect
PAR noise → wrong inflow scenarios. Particularly insidious because the PAR
estimation itself (which uses SeasonMap) would be correct — you'd get correct
coefficients applied to incorrect noise.

---

## Debt 2: Observation-to-season alignment and multi-resolution aggregation

> **Status: PARTIALLY RESOLVED** in v0.4.3 (`316feab`). Layer 1 validation
> implemented as rule 31: observation-to-season alignment with the same
> three-tier SeasonMap fallback chain used in Debt 1. Errors on mismatches
> between observation granularity and season resolution. Layer 2 (fine-to-coarse
> aggregation for Pattern D multi-resolution studies) remains deferred.

**Severity:** Silent data quality issue (current); required feature for
Pattern D (multi-resolution studies).

**Problem:** `inflow_history.parquet` rows are `(hydro_id, date, value_m3s)`.
The code assumes each observation maps 1:1 to a season — one observation per
(hydro, season, year). But nothing validates this:

- No check that observation granularity matches stage resolution
- No check that each (hydro, season, year) has exactly one observation
- No check that observation dates fall within defined season boundaries

**Example failure (silent corruption):** User has monthly history but
quarterly stages. Each quarter contains 3 monthly observations. The code
silently maps all three to the same season (via SeasonMap) and the last one
overwrites the first two in the observation table — or all three end up in
the PAR estimation as separate "observations" for the same season, inflating
the sample count.

**Example use case (multi-resolution):** User has monthly history and a
study with monthly stages (years 1–5) transitioning to quarterly stages
(years 6–10). The quarterly PAR model needs quarterly observations derived
from the monthly history. This is not a validation error — it's a legitimate
aggregation requirement.

### Fix: two-layer approach

**Layer 1 — Validation (error on ambiguity):**

At IO load time, cross-check observation granularity against
`season_definitions`:

- If observation resolution is **finer** than season resolution (e.g.,
  monthly observations, quarterly seasons): this is a valid aggregation
  case. Proceed to Layer 2.
- If observation resolution is **coarser** than season resolution (e.g.,
  quarterly observations, monthly seasons): **error**. Cannot disaggregate
  without assumptions.
- If observation resolution **matches** season resolution: current behavior
  (1:1 mapping). Validate uniqueness per (hydro, season, year).

Resolution comparison uses `SeasonMap` season boundaries: a season spanning
3 calendar months is quarterly; a season spanning 1 month is monthly.

**Layer 2 — Aggregation (fine→coarse):**

When observations are finer than the season, aggregate them into the
season's resolution before PAR fitting:

```
For each (hydro, season, year):
    obs_in_season = observations whose dates fall within the season's
                    calendar boundaries (from SeasonMap)
    aggregated_value = Σ(obs.value × obs.duration_hours) / season_total_hours
```

This produces one aggregated observation per (hydro, season, year) at the
PAR model's resolution. The existing PAR fitting pipeline (`fitting.rs`)
consumes these aggregated values unchanged.

**Concrete example (Pattern D):**

Monthly history with 12 seasons (ids 0–11) and quarterly seasons (ids
12–15). The quarterly season for Q1 (id 12) spans January through March.
Aggregation for hydro 0, Q1, 2024:

```
obs_jan = (2024-01-15, 450.0 m³/s, 744h)
obs_feb = (2024-02-15, 380.0 m³/s, 696h)
obs_mar = (2024-03-15, 520.0 m³/s, 744h)
q1_value = (450×744 + 380×696 + 520×744) / (744+696+744) = 451.8 m³/s
```

The monthly PAR model uses the raw monthly observations. The quarterly PAR
model uses the aggregated quarterly values. Both are fitted independently.

**Testing:** Add test cases for (1) monthly obs + quarterly seasons
(aggregation path), (2) quarterly obs + monthly seasons (error path),
(3) monthly obs + monthly seasons (identity path, regression).

---

## Debt 3: `past_inflows` has no temporal metadata

**Severity:** Design limitation — fragile convention.

**Problem:** `initial_conditions.json` `past_inflows` is a flat array:

```json
{ "hydro_id": 0, "values_m3s": [600.0, 500.0] }
```

Values are ordered from most recent (lag 1) to oldest (lag p). No dates, no
season association, no period duration. The code writes them into lag slots
by position (`setup.rs:1868`).

**Current state:** This works because:

- The PAR model treats lags as dimensionless positions
- The user is responsible for providing values at the correct temporal resolution
- For monthly studies, lag 1 = last month, which is unambiguous

**Where it becomes fragile:**

- For mixed-resolution studies (weekly + monthly), which stage's resolution do
  the past_inflows correspond to? If PAR kicks in at the first monthly stage
  (stage 4), the lags should be monthly. But the user might provide weekly values
  thinking they're for stage 0.
- No validation that past_inflows values are consistent with the PAR model's
  seasonal statistics (monthly means/stds vs weekly raw values).

**Not yet a code defect** — but will need design attention when per-stage-group
schemes (Debt 5) are implemented.

---

## Debt 4: `season_id` consistency enforcement across heterogeneous season spaces

> **Status: RESOLVED** in v0.4.3 (`9c37529`). Four validation sub-rules
> added to the semantic validation layer:
>
> - Rule 27 (V4.1): season_id range coverage — errors for undefined seasons
> - Rule 28 (V4.2): observation coverage — warns for seasons with zero observations
> - Rule 29 (V4.3): resolution consistency — errors for incompatible durations
> - Rule 30 (V4.4): contiguity — warns for defined but unreferenced seasons

**Severity:** Silent misconfiguration risk; becomes critical for Pattern D.

**Problem:** Each stage's `season_id` is set directly in `stages.json` by the
user. The code derives `n_seasons = max(season_id) + 1` and uses modular
arithmetic throughout. But nothing validates that:

- `season_id` values are consistent with `season_definitions` (e.g., a Monthly
  SeasonMap with 12 seasons but stages only use season_ids 0–3)
- `season_id` values cover a contiguous range (gaps produce empty seasons with
  no observations)
- Multiple stages sharing a `season_id` is intentional (same-season stages get
  identical stochastic parameters)

### Multi-resolution season spaces (Pattern D)

For studies that transition between resolutions (e.g., monthly→quarterly),
the `season_id` space must accommodate both resolutions using disjoint
ranges within a single `Custom` `SeasonMap`:

```json
{
  "cycle_type": "Custom",
  "seasons": [
    { "id": 0,  "label": "January",  "month_start": 1,  "month_end": 1  },
    { "id": 1,  "label": "February", "month_start": 2,  "month_end": 2  },
    ...
    { "id": 11, "label": "December", "month_start": 12, "month_end": 12 },
    { "id": 12, "label": "Q1",       "month_start": 1,  "month_end": 3  },
    { "id": 13, "label": "Q2",       "month_start": 4,  "month_end": 6  },
    { "id": 14, "label": "Q3",       "month_start": 7,  "month_end": 9  },
    { "id": 15, "label": "Q4",       "month_start": 10, "month_end": 12 }
  ]
}
```

Monthly stages use `season_id` 0–11. Quarterly stages use `season_id`
12–15. The PAR model has 16 parameter sets: 12 fitted on monthly
observations, 4 fitted on quarterly-aggregated observations (see Debt 2).

This design avoids introducing a "stage group" or "model group" concept.
The `season_id` is the single key into PAR parameters. The `SeasonMap`
defines what each season means. The validation layer ensures consistency.

**Note on `season_for_date()` ambiguity:** With overlapping date ranges
(January matches both season 0 and season 12/Q1), `season_for_date()`
returns the first match. This is acceptable because `season_for_date()` is
used for **historical observation mapping**, which always targets a specific
resolution (Debt 2 handles the aggregation). Stage-to-season mapping uses
the explicit `stage.season_id` field, not `season_for_date()`.

### Fix: multi-level validation

**V4.1 — Range coverage:** Every `season_id` referenced by a stage must
exist in `season_definitions`. Error if a stage references an undefined
season.

**V4.2 — Observation coverage:** For each season with PAR-generated noise,
at least `max_par_order + 1` observations must exist (after Debt 2
aggregation). Warn if a season has zero observations — this is valid only
when all stages using that season use External scenarios exclusively.

**V4.3 — Resolution consistency within season groups:** Stages sharing the
same `season_id` must have compatible durations. Four weekly stages with
`season_id = 3` (April) is valid (sub-season stages, Debt 5 applies).
A monthly stage and a quarterly stage sharing `season_id = 0` is an
error — they imply different PAR resolutions for the same season.

**V4.4 — Contiguity within resolution bands:** Within each resolution
band (monthly: 0–11, quarterly: 12–15), season_ids should be contiguous.
Gaps produce empty seasons with no PAR parameters. Warn (not error) since
this could be intentional for External-only seasons.

**Testing:** Add test cases for (1) valid monthly-only, (2) valid
monthly+quarterly with disjoint ranges, (3) invalid: stage references
undefined season, (4) invalid: mixed resolutions sharing a season_id.

---

## Debt 5: Noise sharing for same-season stages

**Severity:** Behavioral defect for mixed-resolution studies using PAR
scenario generation (Pattern C). Not applicable to production DECOMP
(Pattern B), which uses External scenarios exclusively.

**Problem:** When multiple stages share the same `season_id` (e.g., 4 weekly
stages all with `season_id = 0` for January), each stage currently gets an
**independent** noise draw ξ. This overstates independence — 4 independent
draws scaled by monthly σ fabricates weekly variability that isn't supported
by the monthly historical data.

**Correct behavior:** Stages that share the same `(season_id, year)` should
share the same noise realization. This is what SPARTACUS does (see below) and
is the honest representation of the data resolution.

**When this applies:** Noise sharing is needed when weekly SDDP stages use
the PAR model for scenario generation (Pattern C). Users who want weekly
cut boundaries (e.g., weekly storage arbitrage value) with PAR noise must
share ξ across same-`(season_id, year)` stages to avoid fabricating weekly
variability. These weekly stages may also contain parallel blocks for
sub-weekly decision granularity (peak vs off-peak dispatch within a week).

**When this does NOT apply:** Production DECOMP (Pattern B) uses External
scenarios for all stages. The PAR model provides the baseline for
standardization but does not generate noise during the forward pass. With
External scenarios, noise sharing is irrelevant — each stage's inflow is
determined by the external data, not by ξ.

**Why "stochastic period" is not a new concept:** The information needed is
already present. `season_id` groups stages by recurring season. The stage
dates identify which year each occurrence belongs to. Same `(season_id, year)`
= same noise. No new temporal entity is required — it's a policy change in
the `ForwardSampler`.

**Implementation:** The `ForwardSampler` would check whether stage `t` shares
a `(season_id, year)` group with stage `t-1`. If so, reuse the same ξ. If
not, draw fresh. The grouping is derivable from existing stage metadata.

**Backward pass consistency requirement:** If the forward pass shares noise
for same-`(season_id, year)` stages, the backward pass opening tree must do
the same. When generating the opening tree at stage W2 (week 2 of January),
each opening selects a ξ value for January. Stages W3 and W4 within the
same opening must use the same January ξ as W2. If the backward pass draws
independent noise per stage, cuts would be conditioned on noise scenarios
the forward pass can never realize, degrading cut quality.

Concretely, the opening tree at W2 looks like:

```
Opening 1: ξ_jan=0.3  → W2=0.3, W3=0.3, W4=0.3, Feb draws fresh
Opening 2: ξ_jan=-0.5 → W2=-0.5, W3=-0.5, W4=-0.5, Feb draws fresh
Opening 3: ξ_jan=1.1  → W2=1.1, W3=1.1, W4=1.1, Feb draws fresh
```

Different openings can have different ξ values for January (capturing
monthly uncertainty), but within each opening all January stages share ξ.

**Why weekly stages with noise sharing (not just chronological blocks):**
Chronological blocks are the simpler option when weekly cuts are not needed
(see recommended pattern below). However, weekly SDDP stages provide value
when:

1. **Weekly cut boundaries matter** — the value of water saved across weeks
   is captured by cuts, enabling better sequential decision-making
2. **Parallel blocks within weeks** — users may want sub-weekly decision
   granularity (peak/off-peak blocks) inside each weekly stage
3. **External coupling** — another model reads Cobre's weekly cuts

For these cases, noise sharing ensures the PAR model's monthly variance is
preserved. The lag accumulation mechanism (Debt 6) handles the state
transition correctly regardless of whether noise is shared or external.

**Lag aggregation at coupling boundaries:** Whether noise is shared (all
weeks get same rate) or external (each week has a different rate from
External scenarios), the lag resolution is determined by the **cuts**, not
the stages. The monthly lag at the coupling boundary requires aggregation:

```
monthly_inflow_lag = Σ(inflow_w × hours_w) / Σ(hours_w)
```

With noise sharing, all weekly inflows are identical, so the aggregation
is trivially the shared rate. With External scenarios providing different
weekly values, the aggregation produces the weighted average. Both cases
are handled by the accumulator mechanism in Debt 6.

---

## Debt 6: State transition assumes uniform stage resolution

> **Status: RESOLVED** in the production-decomp plan (`81776e1`, `97f9655`).
> `StageLagTransition` struct added to `cobre-core::temporal` with precomputation
> algorithm in `cobre-sddp::lag_transition`. The `accumulate_and_shift_lag_state`
> hot-path function replaces `shift_lag_state` in both `forward.rs` and
> `simulation/pipeline.rs`. Lag accumulator and weight accumulator fields added
> to `ScratchBuffers`. Monthly studies produce bit-identical results (degenerate
> identity: `accumulate_weight=1.0`, `finalize_period=true` every stage).
> All 32 deterministic regression tests pass unchanged. D28 DECOMP integration
> test verifies weekly+monthly lag accumulation end-to-end.

**Severity:** Architectural gap — blocks mixed-resolution and DECOMP coupling.

### Current mechanism (`noise.rs:shift_lag_state`)

The state vector layout is `[storage₀..N | lag0₀..N | lag1₀..N | ... | lagL₀..N]`
where lag 0 = most recent, lag L-1 = oldest. The state transition after solving
stage `t` is a simple shift register:

```rust
// noise.rs:170-180
for h in 0..n_h {
    let z_t_h = unscaled_primal[z_inflow.start + h];  // realized inflow from LP
    for lag in (1..l_max).rev() {
        state[lag_start + lag * n_h + h] = incoming_lags[(lag - 1) * n_h + h];
    }
    state[lag_start + h] = z_t_h;  // newest lag = this stage's realized inflow
}
```

This writes `z_inflow[h]` (the LP's realized inflow for hydro `h`) into lag
slot 0, and shifts all older lags back by one position. There is no
intermediate buffer, no aggregation, no awareness of temporal resolution.

### What `z_inflow[h]` represents

After solving a stage's LP, `z_inflow[h]` is the total natural inflow in
m³/s for hydro `h` at that stage. This value is determined by the PAR
formula: `z = μ + Σψ·(lag - μ_lag) + σ·ε`. The PAR parameters (μ, σ, ψ)
come from `PrecomputedPar` indexed by stage and hydro.

For a **monthly** stage, `z_inflow[h]` is the monthly average inflow rate.
For a **weekly** stage, `z_inflow[h]` is the weekly average inflow rate.
The shift register doesn't know or care about the duration.

### Where this breaks: 4 weekly stages → monthly coupling

```
State flow:  [W1] → [W2] → [W3] → [W4] → [M2] → boundary cuts

After W1: lag0 = z_W1 (weekly inflow of week 1)
After W2: lag0 = z_W2, lag1 = z_W1
After W3: lag0 = z_W3, lag1 = z_W2, lag2 = z_W1
After W4: lag0 = z_W4, lag1 = z_W3, lag2 = z_W2, lag3 = z_W1

Entering M2: lag0 = z_W4 (week 4 inflow!)
             lag1 = z_W3 (week 3 inflow!)
```

But boundary cuts from a monthly model expect:

```
lag0 = monthly_inflow_month1 (= weighted average of W1..W4)
lag1 = monthly_inflow_month0 (= the month before the study)
```

The shift register has pushed 4 weekly values through the lag slots. The
lag window is now "week 4, week 3, week 2, week 1" — not "month 1, month 0".

### The three problems

**P1: Resolution mismatch.** The lag slots contain weekly inflows where
the cuts expect monthly inflows. The cut coefficients (π_lag) were computed
against monthly values. Evaluating them with weekly values gives wrong
future costs.

**P2: Lag depth overflow.** 4 weekly stages push 4 values through the shift
register, consuming 4 lag slots. If `max_par_order = 2` (typical), lags 2
and 3 don't exist — the weekly inflows from W1 and W2 fall off the end of
the shift register and are lost. You can't recover the monthly average even
if you wanted to.

**P3: No aggregation pathway.** There is no code path that accumulates
inflow across stages. The shift register is a 1-in-1-out FIFO per hydro.
To produce a monthly lag from weekly stages, you'd need to either:

- Accumulate inflows across stages sharing a `(season_id, year)` group
- Or avoid the problem by using chronological blocks instead of stages

### How noise sharing (Debt 5) helps but doesn't fully solve it

If weekly stages share monthly noise (same ξ for all 4 weeks), then
`z_W1 = z_W2 = z_W3 = z_W4 = z_monthly` (all the same rate, since the PAR
parameters are monthly). After W4, `lag0 = z_monthly`. This is the correct
monthly lag value — by accident. The shift register pushed 4 identical values
through, and the most recent one happens to equal the monthly average.

But **P2 still applies**: the shift consumed 4 lag slots for one month. With
`max_par_order = 2`, the previous month's lag was overwritten by W2's value.
For PAR(1) this is fine (only lag 0 matters). For PAR(2)+, the deeper lags
are wrong.

### Solutions

**Solution A: Chronological blocks (recommended for DECOMP pattern).**
Use 1 monthly stage with 4 weekly chronological blocks instead of 4 weekly
SDDP stages. The LP handles weekly decisions internally. The shift register
sees one stage transition per month. `z_inflow[h]` is the monthly inflow
rate. Lags are naturally monthly. No aggregation needed. No lag depth
overflow. This completely sidesteps P1, P2, and P3.

**Solution A (chronological blocks)** remains useful when users DON'T need
weekly cuts, but **cannot be the only solution** — users who need weekly
future cost function approximations require weekly SDDP stages with Benders
cuts at each week boundary.

### The complication: stages spanning season boundaries

A week from Jan 28 to Feb 3 contributes 4 days to January and 3 days to
February. The lag accumulation requires fractional weights:

```
inflow_jan = Σ(z_stage × hours_in_jan) / total_jan_hours
inflow_feb = Σ(z_stage × hours_in_feb) / total_feb_hours
```

A single stage can finalize one lag period AND start accumulating into the
next. These weights are entirely determined by stage dates and lag period
boundaries — fully pre-computable at setup time.

The same pattern applies to monthly stages spanning quarter boundaries
(Pattern D): a monthly stage in March finalizes Q1 accumulation, a monthly
stage in April starts Q2. See Debt 13 for the multi-resolution transition.

### Proposed solution: precomputed lag accumulation (B+C hybrid)

The lag register's resolution is a property of the **stochastic model**,
not of individual stages. The lag period boundaries come from the
`SeasonMap` — month boundaries for monthly PAR, quarter boundaries for
quarterly PAR. The design has two parts:

**Part 1: Precomputed per-stage lag configuration (setup time)**

Computed once from stage dates, `season_id`, and lag period boundaries.
Stored as a flat array indexed by stage, consumed read-only on hot path.

```rust
/// Pre-computed lag transition config for one stage.
/// Built at setup, consumed read-only on the hot path.
struct StageLagTransition {
    /// Weight of this stage's z_inflow in the current lag period's
    /// weighted average. Equals hours_in_current_period / total_period_hours.
    /// Zero when the stage falls entirely outside the current lag period
    /// (shouldn't happen in well-formed studies).
    accumulate_weight: f64,

    /// If this stage straddles a lag period boundary, the fraction that
    /// belongs to the NEXT period. Zero for stages entirely within one
    /// period. Example: week Jan 28 – Feb 3 has spillover_weight = 3/7
    /// (3 days in February) normalized to the next period's total hours.
    spillover_weight: f64,

    /// True when this stage is the last one contributing to the current
    /// lag period. Triggers: finalize weighted average, shift lag register,
    /// start next period with spillover (if any).
    finalize_period: bool,
}
```

All fields are derivable from `stage.start_date`, `stage.end_date`, and the
lag period boundaries (month boundaries for monthly PAR). No runtime
computation needed — just table lookups.

**Part 2: Per-worker accumulator buffer (hot path)**

Added to `ScratchBuffers` in `workspace.rs`. Pre-allocated at workspace
creation alongside existing scratch buffers. Length: `n_hydros`.

```rust
// In ScratchBuffers:
lag_accumulator: Vec<f64>,       // length: n_hydros, zeroed at period start
lag_weight_accum: f64,           // total weight accumulated so far
```

**Part 3: Hot path logic (replaces shift_lag_state call)**

After each stage solve, at the same point where `shift_lag_state` is
currently called (`forward.rs:838`):

```
// 1. Accumulate this stage's inflow into the lag period buffer
for h in 0..n_h:
    scratch.lag_accumulator[h] += z_inflow[h] * stage_lag.accumulate_weight
scratch.lag_weight_accum += stage_lag.accumulate_weight

// 2. If this stage ends the current lag period: finalize and shift
if stage_lag.finalize_period:
    for h in 0..n_h:
        monthly_inflow[h] = scratch.lag_accumulator[h] / scratch.lag_weight_accum

    shift_lag_state(state, incoming_lags, monthly_inflow, indexer)
    //                                    ^^^^^^^^^^^^^^
    //                    uses accumulated monthly value, not raw z_inflow

    // Reset accumulator and seed with spillover into next period
    scratch.lag_accumulator.fill(0.0)
    scratch.lag_weight_accum = 0.0
    if stage_lag.spillover_weight > 0.0:
        for h in 0..n_h:
            scratch.lag_accumulator[h] = z_inflow[h] * stage_lag.spillover_weight
        scratch.lag_weight_accum = stage_lag.spillover_weight

// 3. If NOT finalizing: lags are frozen (no shift_lag_state call)
//    The state vector keeps the previous period's lag values.
//    The next stage's LP sees the same lags (monthly resolution).
```

**Cost analysis:**

- Per stage: `n_hydros` multiply-adds (same order as existing inflow RHS
  patching). Zero allocations — accumulator is pre-allocated in scratch.
- At period boundary: `n_hydros` divisions + one `shift_lag_state` call
  (same cost as today, just less frequent).
- For the common case (uniform monthly stages): `accumulate_weight = 1.0`,
  `finalize_period = true` every stage, `spillover_weight = 0.0`. The
  accumulator reduces to `monthly_inflow = z_inflow * 1.0 / 1.0 = z_inflow`.
  Identical to current behavior with negligible overhead (one multiply by 1.0).

### What this means for lag-fixing constraints and cuts

Between weekly stages where `finalize_period = false`, the lag register is
frozen. The LP's lag-fixing constraints fix lags to the **previous month's**
values. This is correct because:

1. The PAR formula `z = μ_m + Σψ·(lag - μ_lag) + σ·ε` uses monthly μ_lag,
   so lag values must be monthly averages.
2. Cut coefficients π_lag at weekly boundaries reference monthly lag values,
   making them consistent with boundary cuts from a monthly model.
3. All weekly stages within a month see the same AR contribution from the
   previous month — consistent with the data resolution.

### Backward pass considerations

The backward pass builds cuts using duals of the lag-fixing constraints.
When lags are frozen between weekly stages, the lag-fixing constraints
have the same RHS for all weeks in a month. The cut coefficients π_lag
are meaningful at monthly resolution. This is exactly what's needed for
DECOMP coupling — the cuts built at weekly boundaries have monthly lag
semantics, compatible with monthly boundary cuts.

**The backward pass does NOT need its own accumulator.** Standard SDDP
does a one-step lookahead: for each (trial point, opening), solve the LP
at stage `t+1` and extract duals. The trial point's lag state comes from
the forward pass, which already applied the frozen-lag + accumulation
logic. The backward pass consumes these lag values as-is — no state
propagation beyond one step, so no accumulation is needed. The future
cost approximation comes from cuts built in previous iterations, not from
multi-step state propagation within the backward pass.

### Opening tree clamping rule

The backward pass opening tree generator should adapt the effective number
of openings to the available noise pool for each stage:

```
effective_openings = min(configured_n_openings, available_noise_scenarios)
```

Where `available_noise_scenarios` depends on the noise source:

- **OutOfSample (SAA/LHS/QMC):** continuous distribution → unlimited →
  use configured value
- **External:** `library.n_scenarios()` → could be 1, 50, 2000
- **InSample (opening tree):** `tree.n_openings()` → the pre-generated count
- **Historical residuals (Debt 8):** `n_valid_windows` → typically 30–80

This handles the production DECOMP case naturally: weekly stages with
`n_scenarios = 1` get `effective_openings = 1`, avoiding redundant LP
solves that all produce the same cut. The monthly stage with N external
scenarios gets `min(configured, N)` openings. No special-casing needed.

This rule also applies to the Historical residuals case (Debt 8), where
the finite pool of valid windows may be smaller than the configured
opening count. Currently `window.rs:227-234` warns when `n_windows <
forward_passes`; the same pattern applies to the backward pass.

### What about the PAR model between weekly stages?

With frozen lags, all weekly stages within January compute:

```
z = μ_jan + ψ · (lag_dec - μ_dec) + σ_jan · ε
```

The AR contribution is the same for all weeks (conditioned on December).
With noise sharing (Debt 5), ε is also the same → all weeks get identical
inflow. This is the honest monthly-data behavior.

With independent weekly noise (e.g., if somehow ε differs per week), the
AR contribution is still from December, but each week gets different noise
realization. The accumulator then computes the weighted average of these
different weekly inflows for the monthly lag, which is the correct behavior
for Solution C.

### DECOMP with weekly External scenarios: end-to-end trace

**Scenario:** 4 weekly stages with External inflow scenarios providing
different target inflows per week (e.g., W1=500, W2=480, W3=520, W4=510
m³/s), followed by a monthly stage that evaluates boundary cuts.

**Step 1: Standardization at setup time (`standardize_external_inflow`)**

The reverse PAR formula computes eta for each (stage, scenario, hydro):

```
eta = (target - deterministic_value) / sigma
where: deterministic_value = det_base + Σψ · lag[l]
```

**Current code bug for this case:** `external.rs:335-347` builds lag values
by looking at the previous **stage's** raw external value. For stage 1 (W2),
`lag[0] = raw_values[W1]` — a weekly value. But the PAR's ψ and μ_lag are
monthly. The AR term `ψ · (weekly_value)` mixes resolutions.

**Correct behavior with frozen lags:** The lag values used for eta
computation should be **monthly**, matching the PAR model's resolution:

- Stages W1–W4: all use `lag[0] = past_inflows[0]` (December's monthly
  inflow from initial conditions), not the previous week's value.
- The reverse formula then gives different eta values per week because
  the targets differ, but the AR contribution is the same for all weeks.

This means `standardize_external_inflow` needs the same lag-freezing
logic as the runtime: when consecutive stages share `(season_id, year)`,
the lag buffer should NOT advance between them.

**Step 2: Runtime forward pass**

Each weekly stage's LP uses the eta from the External library:

```
W1: eta_W1 → z_inflow = det_base + ψ·lag_dec + σ·eta_W1 = 500 m³/s ✓
W2: eta_W2 → z_inflow = det_base + ψ·lag_dec + σ·eta_W2 = 480 m³/s ✓
W3: eta_W3 → z_inflow = det_base + ψ·lag_dec + σ·eta_W3 = 520 m³/s ✓
W4: eta_W4 → z_inflow = det_base + ψ·lag_dec + σ·eta_W4 = 510 m³/s ✓
```

All weeks use the same frozen lag (December). Different targets produce
different eta values. The LP realizes the target inflow for each week. ✓

**Step 3: Lag accumulation for the next month**

The accumulator collects weekly realized inflows:

```
accum += 500 × 168h  (W1)
accum += 480 × 168h  (W2)
accum += 520 × 168h  (W3)
accum += 510 × 192h  (W4, longer week)
monthly_lag = accum / (168+168+168+192) = 502.5 m³/s
```

At the W4→M2 boundary, `finalize_period = true`. The lag register shifts:
`lag[0] = 502.5` (January's weighted monthly average). ✓

**Step 4: Monthly stage M2 evaluates boundary cuts**

M2's LP sees `lag[0] = 502.5` (monthly resolution). Boundary cuts from the
monthly model evaluate with the correct monthly lag value. ✓

### Impact on `standardize_external_inflow` → see Debt 9

The external standardization pipeline (`external.rs:325-358`) advances
lags per-stage, not per-lag-period — the same resolution mismatch as the
runtime forward pass. The fix uses the same `StageLagTransition` config.
See Debt 9 for the full specification and corrected pseudocode.

### Edge case: first stage straddles backward into pre-study month

Study starts with a week Dec 29 – Jan 4. Stage 0 has `season_id = 0`
(January).

- `past_inflows` provides December, November, etc. (complete monthly lags)
- Lags are frozen at December values for all January stages
- `accumulate_weight` for stage 0 = `4_jan_days × 24h / total_jan_hours`
- The 3 December days don't contribute to any accumulation (December is
  pre-study, fully captured by `past_inflows`)
- The 450 m³/s target rate is attributed to just the 4 January days
- No special flag needed — `accumulate_weight` naturally excludes pre-study
  days by only counting hours within the current lag period

### Concrete example: PMO_APR_2026_rv0 (production DECOMP pattern)

Production DECOMP stages always begin on Saturday and end on Friday.
April 1, 2026 is a Wednesday, so the first Saturday-anchored week that
contains an April day is **2026-03-28 (Sat) to 2026-04-03 (Fri)**.

**Stage layout:**

```
Stage 0: 2026-03-28 → 2026-04-03  (W1)  season_id=3 (April)
Stage 1: 2026-04-04 → 2026-04-10  (W2)  season_id=3
Stage 2: 2026-04-11 → 2026-04-17  (W3)  season_id=3
Stage 3: 2026-04-18 → 2026-04-24  (W4)  season_id=3
Stage 4: 2026-04-25 → 2026-05-01  (W5)  season_id=3 (partial: 6 Apr days)
Stage 5: 2026-05-01 → 2026-05-31  (M2)  season_id=4 (May)
         ↳ terminal-stage FCF: boundary cuts from monthly model
```

**Scenarios:** W1–W5 each have a single external forecast (n_scenarios=1).
M2 has N external scenarios from GEVAZP (or a prior Cobre monthly run).

**StageLagTransition precomputation:**

April has 30 days = 720 hours total.

| Stage | accumulate_weight        | spillover_weight        | finalize_period |
| ----- | ------------------------ | ----------------------- | --------------- |
| W1    | 3d × 24h / 720h = 0.100  | 0.0 (no May days)       | false           |
| W2    | 7d × 24h / 720h = 0.233  | 0.0                     | false           |
| W3    | 7d × 24h / 720h = 0.233  | 0.0                     | false           |
| W4    | 7d × 24h / 720h = 0.233  | 0.0                     | false           |
| W5    | 6d × 24h / 720h = 0.200  | 1d × 24h / 744h = 0.032 | **true**        |
| M2    | 30d × 24h / 744h = 0.968 | 0.0                     | true            |

Note: W1 has only 3 April days (March 28–31 is pre-study, covered by
`past_inflows`). W5 straddles into May: 6 April days finalize April, and
the 1 May day seeds the May accumulator via `spillover_weight`. M2's
`accumulate_weight` is less than 1.0 because May has 31 days (744h) but
the May accumulator was seeded with W5's spillover 1 day.

**Accumulator trace (hydro h, single external forecast):**

```
Initial: past_inflows = [march_avg, feb_avg, ...]
         accum = 0.0, weight = 0.0

After W1: accum += 500 × 0.100 = 50.0,   weight = 0.100
After W2: accum += 480 × 0.233 = 161.8,  weight = 0.333
After W3: accum += 520 × 0.233 = 282.9,  weight = 0.567
After W4: accum += 510 × 0.233 = 401.7,  weight = 0.800
After W5: accum += 490 × 0.200 = 499.7,  weight = 1.000
  → finalize: april_avg = 499.7 / 1.000 = 499.7 m³/s
  → shift_lag_state: lag[0] = 499.7, lag[1] = march_avg
  → reset accum, seed: accum = 490 × 0.032 = 15.7, weight = 0.032

After M2: accum += z_may × 0.968 = ...,  weight = 1.000
  → finalize: may_avg = ...
  → terminal FCF evaluates with lag[0] = may_avg, lag[1] = 499.7
```

Lags are frozen at `[march_avg, feb_avg, ...]` for all of W1–W5. The
PAR formula's AR contribution is the same for all weekly stages. The
accumulator correctly produces the weighted monthly average including
the boundary-straddling days.

**Backward pass with opening tree clamping:**

- W1–W5: `effective_openings = min(configured, 1) = 1` (single forecast)
- M2: `effective_openings = min(configured, N)` (full GEVAZP tree)

Weekly backward passes solve 1 LP per trial point (no redundancy).
The monthly backward pass uses the full scenario tree.

### Uniform scenario count constraint and interim workaround

The current `ExternalScenarioLibrary` enforces a uniform scenario count
across all stages (`external.rs` validation V3.4). Production DECOMP has
1 scenario for weekly stages and N for the monthly stage, which violates
this constraint.

**Interim workaround:** Pad the weekly stages to N scenarios by
replicating the single forecast N times. The forward pass already handles
`n_scenarios < forward_passes` via modular wrapping — with N identical
scenarios, `hash(iteration, scenario) % N` always yields the same inflow
regardless of the index. The opening tree clamping rule (above) ensures
the backward pass uses `effective_openings = 1` despite having N
nominally distinct (but identical) scenarios.

**Cost of padding:** Memory for N × n_weekly_stages × n_hydros extra
eta values. For typical sizes (N=2000, 5 weekly stages, 200 hydros) this
is ~16 MB — negligible. No computational cost because the clamping rule
avoids redundant LP solves in the backward pass.

**Long-term direction:** The regularity assumption (one node per stage,
uniform scenario count) simplifies workload distribution and maximizes
parallel performance. However, supporting non-uniform scenario trees
(Markovian graphs, conditional branching) will eventually require a
node-based policy graph where each stage can have multiple nodes with
transition probabilities. This is a deep architectural change that is
deferred — the padding workaround is sufficient for production DECOMP
and does not block the future redesign.

### Mid-season study starts → see Debt 10

When a DECOMP study starts mid-season (e.g., January 5 instead of
January 1), the partial-season observations before the study start must
seed the lag accumulator. See Debt 10 for the `recent_observations`
input design and the rolling revision lifecycle.

### Testing strategy for Debt 6

- **D-accum-01:** PMO_APR_2026_rv0 accumulator trace (the arithmetic from
  the concrete example above). 5 weekly + 1 monthly stage, External
  scenarios, verify lag values at each stage boundary match the trace.
- **D-accum-02:** Uniform monthly study. Verify accumulator degenerates
  to identity (lag[0] = z_inflow, finalize every stage).
- **D-accum-03:** Week straddling month boundary (Jan 28 – Feb 3). Verify
  spillover weight correctly seeds the next period's accumulator.
- **D-accum-04:** PAR(2) with 4 weekly stages. Verify lag[1] preserves
  the previous month's value (not overwritten by weekly inflows).

---

## Debt 7: Mixed-resolution PAR model behavior

> **Status: RESOLVED** in v0.4.3 (`ae4403a`). The mdBook page
> `book/src/guide/stochastic-modeling.md` now documents PAR temporal resolution
> behavior, validation rules, and the recommended chronological-blocks pattern
> for sub-monthly decision granularity.

**Severity:** Design constraint — not a code bug, but important to document.

**Problem:** When multiple stages share `season_id`, with the noise-sharing
fix from Debt 5, the PAR model behaves correctly but with known limitations:

1. **Same inflow rate across sub-period stages:** All weeks in January get
   the same m³/s rate. This is physically reasonable for the monthly average
   but doesn't capture intra-month variability.

2. **All sub-period stages see the same AR contribution:** The lag from
   December feeds into all January weeks equally. No intra-month lag
   propagation. This is consistent with the data resolution.

3. **Historical windows produce identical eta:** For Historical sampling,
   all sub-period stages map to the same observation → identical noise.
   This is correct given the data resolution.

**These are not bugs — they are honest representations of monthly data
applied to sub-monthly stages.** Users who need true weekly variability
should use External scenarios from a dedicated short-term model.

**Chronological blocks as the preferred approach:** Rather than creating
4 weekly SDDP stages, the recommended pattern for mixed-resolution studies
is 1 monthly SDDP stage with 4 weekly chronological blocks:

- Weekly decision granularity in the LP (per-block variables/constraints)
- No Benders cuts between weeks (no cut approximation overhead)
- One noise realization for the month (consistent with data)
- Natural monthly state transition at the stage boundary
- Boundary cuts evaluate against the monthly lag directly

---

## Debt 8: Historical residuals as opening tree noise source

> **Status: RESOLVED** in v0.4.3 (`d4a604d`). New `NoiseMethod::HistoricalResiduals`
> variant copies pre-computed eta vectors from `HistoricalScenarioLibrary` into the
> backward-pass opening tree. Uses hash-based deterministic window selection via
> `derive_opening_seed`. Cholesky correlation skipped (empirical cross-entity
> correlation preserved from actual observations). Openings clamped to `n_windows`
> when fewer than `branching_factor`. The library is pre-built in
> `prepare_stochastic` via standalone `PrecomputedPar::build` to resolve a
> circular dependency. `OpeningTreeInputs` struct bundles `user_tree` and
> `historical_library`.

**Severity:** Missing feature (user-requested).

**Problem:** The backward pass opening tree currently draws noise ε from
synthetic distributions (SAA, LHS, QMC via `NoiseMethod`). A user has
requested the ability to use **historical residuals** instead — the
standardized noise values derived from actual observations.

**What this means concretely:**

Given the PAR model `x_t = μ_m + Σψ_ℓ·(x_{t-ℓ} - μ_{m-ℓ}) + σ_m·ε_t`,
the historical residual for hydro `h`, season `m`, year `y` is:

```
ε_{h,m,y} = (x_{h,m,y} - μ_{h,m} - Σ_ℓ ψ_{h,m,ℓ} · (x_{h,m-ℓ,y} - μ_{h,m-ℓ})) / σ_{h,m}
```

Instead of drawing ε from N(0,1), the opening tree would subsample from the
finite pool `{ε_{·,m,y} : y ∈ selected_years}` for each season `m`.

**Key properties:**

1. **Cross-entity consistency (joint realization):** When the opening tree
   selects year `y` for season `m`, it uses the full vector
   `[ε_{1,m,y}, ε_{2,m,y}, ..., ε_{H,m,y}]` for all hydros. This preserves
   the actual spatial correlation from that year — no parametric correlation
   model needed. This is the natural behavior: real events are joint.

2. **Estimation vs sampling distinction:** The PAR model (μ, σ, ψ) is always
   estimated from the **full** historical record. The year selection only
   controls which residuals are available for backward-pass noise. This
   is analogous to how `HistoricalYears` works for the Historical forward
   sampling scheme.

3. **Finite sample pool and branching factor interaction:** The number of
   available noise vectors per season equals the number of valid historical
   windows. Use the **same strategy as the Historical forward sampler**:
   - Discovery: find all valid window years (same `discover_historical_windows`
     logic, same `HistoricalYears` config for year selection).
   - If `n_windows < branching_factor`: warn (not error) that residuals will
     repeat across openings. Selection wraps via modular arithmetic, just as
     the forward pass wraps `hash(iteration, scenario) % n_windows` when
     `n_windows < forward_passes` (see `window.rs:226-234`).
   - If `n_windows >= branching_factor`: subsample without repetition.
   - Selection is deterministic: `hash(stage, opening_idx) % n_windows`,
     same domain-separated hashing pattern as `select_historical_window`.

4. **Configuration:** Should receive the same information as Historical
   scenario sources: which years to consider for sampling. A new
   `NoiseMethod` variant (e.g., `HistoricalResiduals`) or an extension
   to the existing `NoiseMethod` enum.

**Where existing infrastructure applies:**

- The Historical standardization pipeline (`standardize_historical_windows`)
  already computes eta values that are conceptually these residuals. It uses
  the full PAR noise formula (`solve_par_noise`), producing standardized ε
  values per (window, stage, hydro).
- The `HistoricalYears` config already exists for year selection.
- The `HistoricalScenarioLibrary` stores per-(window, stage, hydro) eta
  values — these are the residuals, organized by window year.
- The `select_historical_window` function in `class_sampler.rs:158` already
  implements the deterministic hash-based window selection pattern.
- The warning for `n_windows < forward_passes` at `window.rs:227-234`
  is the exact model for the `n_windows < branching_factor` case.

**What's new:**

- A `NoiseMethod` variant to select historical residuals for the opening
  tree. The opening tree generator would look up residual vectors by
  (window_year, stage, entity) instead of generating continuous noise.
- The `generate_opening_tree` function needs a code path that, for each
  (stage, opening), selects a window year and copies the corresponding
  residual vector from the `HistoricalScenarioLibrary`.
- No new correlation logic needed — the residuals carry the empirical
  cross-entity correlation from the historical year. The parametric
  Cholesky correlation step is skipped for this noise method.

**Interaction with other debts:**

- **Debt 1 (month0 bug):** The residual computation uses the Historical
  standardization path, which has the `month0()` bug. Must be fixed first.
- **Debt 5 (noise sharing):** Historical residuals naturally share across
  same-`(season_id, year)` stages — picking year 1950 for January gives
  the same ε for all January stages. Consistent with noise-sharing policy.
- **Debt 7 (PAR limitations for sub-monthly):** Historical residuals are at
  the resolution of the historical data. For monthly data, sub-monthly
  stages would share the monthly residual — same limitation, same honesty.

---

## Debt 9: `standardize_external_inflow` advances lags per-stage

> **Status: RESOLVED** in the production-decomp plan (`8f8cf73`).
> `standardize_external_inflow` in `cobre-stochastic/src/sampling/external.rs`
> now accepts `&[StageLagTransition]` and uses frozen-lag + accumulation logic
> matching the runtime forward pass. Round-trip consistency test added:
> standardize then evaluate with a PMO_APR_2026-style mixed weekly+monthly
> layout verifies realized inflows match external targets within solver
> tolerance. All 32 deterministic regression tests pass unchanged.

**Severity:** Setup-time bug — produces incorrect eta values for
mixed-resolution studies with External scenarios.

**Location:** `cobre-stochastic/src/sampling/external.rs:335-347`

**Problem:** The current `standardize_external_inflow` function builds lag
values for the reverse PAR formula by looking at the previous **stage's**
raw external value. For stage 1 (W2), `lag[0] = raw_values[W1]` — a weekly
value. But the PAR's ψ and μ_lag are monthly. The AR term
`ψ · (weekly_value - μ_monthly)` mixes resolutions.

**Correct behavior with frozen lags:** The lag values used for eta
computation should match the PAR model's resolution (monthly for monthly
PAR, quarterly for quarterly PAR):

- Within the same `(season_id, year)` group, the lag buffer stays frozen
  at the previous lag period's values.
- When crossing a lag period boundary, the lag buffer shifts with the
  weighted average of the preceding stages' external inflows.

This is the same `StageLagTransition` precomputed config from Debt 6
applied during setup. The standardization loop becomes:

```
for scenario in 0..n_scenarios:
    reset lag_buf to past_inflows
    reset accum to 0
    for t in 0..n_stages:
        target = external_value[t, scenario, h]
        eta = solve_par_noise(det_base, psi, order, lag_buf, sigma, target)
        library.eta[t, scenario, h] = eta

        // Accumulate for lag period average
        accum[h] += target * stage_lag[t].accumulate_weight
        if stage_lag[t].spillover_weight > 0:
            next_accum[h] += target * stage_lag[t].spillover_weight

        if stage_lag[t].finalize_period:
            period_avg = accum[h] / weight_sum
            shift lag_buf with period_avg
            accum = next_accum; next_accum = 0
```

For uniform monthly studies this degenerates to the current behavior
(weight=1, finalize every stage, lag advances every stage).

**Consistency requirement:** The standardization must use the same
lag-freezing + accumulation logic as the runtime forward pass. If the
runtime freezes lags during January weeks, the standardization must also
freeze lags during January weeks. Otherwise the eta values encode noise
relative to a different AR contribution than what the runtime computes,
and the realized inflows won't match the external targets.

**Testing:** Add a test with 4 weekly + 1 monthly External stages.
Standardize, then run a forward pass with the resulting eta values.
Assert that the LP's realized inflows match the original external targets
within solver tolerance.

---

## Debt 10: Mid-season study start — `recent_observations`

> **Status: RESOLVED** in the production-decomp plan (`08c1a32`).
> `RecentObservation` type added to `cobre-core::initial_conditions` with date
> range and value fields. IO parser in `cobre-io::initial_conditions` handles
> date parsing, value validation (finite, non-negative), and overlap detection.
> Accumulator seeding implemented in `cobre-sddp::lag_transition`: weighted seed
> from recent_observations replaces zero-fill at trajectory start. Backward
> compatible — empty `recent_observations` produces identical behavior to before.
> D28 DECOMP integration test uses recent_observations with mid-season start.

**Severity:** Missing feature — required for rolling DECOMP revisions
(Pattern B).

**Problem:** When a DECOMP study starts mid-season (e.g., January 5
instead of January 1), partial-season observations before the study start
have nowhere to go:

```
Real world:  ... Dec ... | Jan 1-4 (observed: 480 m³/s) | Jan 5 → study
                         |←── 4 days happened ──→|
                         |   not a study stage   |
```

- `past_inflows[0]` = December monthly average (previous complete season) ✓
- January 1–4 is real data but not a study stage → no LP produces a
  `z_inflow` for it
- The accumulator starts at zero → January's lag will average only stages
  from Jan 5 onward, missing 4 days of real data

**Proposed solution: `recent_observations` input**

Extend `initial_conditions.json` (or a companion file) with observed
inflow data from the partial current season preceding the study:

```json
{
  "past_inflows": [
    { "hydro_id": 0, "values_m3s": [dec_avg, nov_avg] }
  ],
  "recent_observations": [
    {
      "hydro_id": 0,
      "start_date": "2025-01-01",
      "end_date": "2025-01-05",
      "value_m3s": 480.0
    }
  ]
}
```

- `past_inflows` — unchanged. Complete seasonal lags for the PAR model.
- `recent_observations` — observed average inflow rate for the partial
  period between the last season boundary and the study start. Carries
  dates so the system can compute the duration and the accumulation weight.

**At setup time:** the recent observations seed the lag accumulator:

```
accum_jan[h] = 480.0 × (4 days × 24h)
weight_jan = 4 × 24h / total_jan_hours
```

When study stages accumulate their January contributions on top of this
seed, the final January average correctly includes all 31 days:

```
jan_avg = (480 × 96h + Σ stage_inflows × stage_hours) / total_jan_hours
```

**For the PAR model:** the lags are still frozen at `past_inflows` values
(December). The recent observations don't affect the PAR formula — they
only seed the accumulator for the current lag period. This is correct:
the PAR conditions on the _previous complete season_, not on partial
current-season data.

**Backward compatibility:**

- Uniform monthly studies starting on the 1st: `recent_observations` is
  empty. The accumulator starts at zero. No impact.
- Non-DECOMP studies: `recent_observations` is optional, defaults to empty.

**Consistency across noise sources:** The `recent_observations` seeding
works identically for External scenarios (Pattern B) and PAR-generated
noise (Pattern C). The accumulator is agnostic to the noise source — it
accumulates the LP's realized inflow weighted by precomputed stage weights.

### Production DECOMP revision lifecycle

In production DECOMP, the PMO for a given month runs multiple revisions
as weeks pass. Each revision drops one weekly stage (that week is now
past) and adds its observed inflow to `recent_observations`. Using April
2026 as the running example:

**PMO_APR_2026_rv0** (study starts 2026-03-28, Saturday):

```
recent_observations: []  (no April days observed yet)
Stages: W1(Mar28-Apr03) W2(Apr04-10) W3(Apr11-17) W4(Apr18-24) W5(Apr25-May01) M2(May)
```

**PMO_APR_2026_rv1** (study starts 2026-04-04, Saturday):

```
recent_observations: [
  { hydro_id: 0, start: "2026-04-01", end: "2026-04-04", value: 500.0 }
]
Stages: W2(Apr04-10) W3(Apr11-17) W4(Apr18-24) W5(Apr25-May01) M2(May)
```

W1 is gone — its April days (Apr 1–3) are now observed data. The
accumulator seeds with 3 days × 500.0 m³/s, then W2–W5 accumulate
on top. March days (Mar 28–31) from the former W1 are irrelevant —
March is fully captured by `past_inflows`.

**PMO_APR_2026_rv2** (study starts 2026-04-11):

```
recent_observations: [
  { hydro_id: 0, start: "2026-04-01", end: "2026-04-04", value: 500.0 },
  { hydro_id: 0, start: "2026-04-04", end: "2026-04-11", value: 480.0 }
]
Stages: W3(Apr11-17) W4(Apr18-24) W5(Apr25-May01) M2(May)
```

**PMO_APR_2026_last** (study starts late April, e.g., 2026-04-25):

```
recent_observations: [
  { hydro_id: 0, start: "2026-04-01", end: "2026-04-04", value: 500.0 },
  { hydro_id: 0, start: "2026-04-04", end: "2026-04-11", value: 480.0 },
  { hydro_id: 0, start: "2026-04-11", end: "2026-04-18", value: 520.0 },
  { hydro_id: 0, start: "2026-04-18", end: "2026-04-25", value: 510.0 }
]
Stages: W5(Apr25-May01) M2(May)
```

Each revision:

1. Drops the earliest weekly stage (that week is now history)
2. Adds that week's observed inflow to `recent_observations`
3. Keeps the monthly stage with GEVAZP scenarios (possibly re-generated
   to condition on the updated observations)

The accumulator seeding grows across revisions, and the study stages
shrink correspondingly, so the total April coverage remains complete:
`seed_hours + Σ stage_hours = total_april_hours`.

**Testing:**

- **D-recent-01:** rv0 (no recent_observations) and rv2 (with seeding).
  Verify that the January lag at finalization includes all 31 days in
  rv2 and only study-stage days in rv0.
- **D-recent-02:** Verify backward compatibility — uniform monthly study
  with empty recent_observations produces identical results to current
  behavior.

---

## SPARTACUS Investigation

**Source:** `~/git/SPTcpp`

### Three-level temporal hierarchy

SPARTACUS decouples three temporal concepts that Cobre currently conflates:

| Concept                              | SPARTACUS                                                                                              | Cobre equivalent                                       |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **Optimization stage** (`IdEstagio`) | LP decision unit. Can span any duration.                                                               | `Stage` — 1:1 with LP solve                            |
| **Study period** (`Periodo`)         | Sub-stage temporal unit. The LP iterates over periods within a stage, creating constraints per period. | None — Cobre's LP has one set of constraints per stage |
| **Stochastic season** (`IdMes`)      | PAR parameter bucket (monthly). Keyed by calendar month.                                               | `season_id` on each stage                              |

The critical innovation is the **study period** layer. A single optimization
stage can contain multiple periods. The LP for that stage has variables and
constraints for each period. The stochastic process generates noise per period,
not per stage.

### How the mapping works

```
Stochastic process timeline (monthly):
  Jan-2024  Feb-2024  Mar-2024  ...
     │         │         │
     │         │         │     ← PAR model keyed by IdMes
     │         │         │
     ▼         ▼         ▼
  Noise₁    Noise₂    Noise₃   ← one realization per stochastic period
     │         │         │
     │         │         │     ← getIteradores(period_stage) filters
     ▼         ▼         ▼        stochastic periods into stages
  ┌──────────┐ ┌──┐ ┌──────────┐
  │ Stage 0  │ │S1│ │ Stage 2  │  ← optimization stages
  │ (weekly  │ │  │ │ (monthly │
  │  ×4)     │ │  │ │  ×1)     │
  └──────────┘ └──┘ └──────────┘
```

Setup code (`C_ModeloOtimizacao_multiestagio_estocastico.cpp:108-113`):

```cpp
// Get full stochastic timeline for scenario 1
horizonte_proc_estoc_completo = getElementosMatriz(
    IdProcessoEstocastico_1,
    AttMatriz_mapeamento_espaco_amostral,
    IdCenario_1, Periodo(), IdRealizacao());

// Filter to just the stochastic periods within this stage's span
horizonte_proc_estoc = horizonte_proc_estoc_completo
    .getIteradores(period_stage);
```

Each stage then iterates over its stochastic periods to build LP constraints.
The LP itself has period-level granularity within stages
(`criarRestricoesHidraulicas` at line 4762 takes `a_period` parameters).

### What happens for weekly stages with monthly stochastic process

All four weekly stages within January call `.getIteradores(period_stage)` and
get back the same monthly period (Jan-2024). They all share the same stochastic
realization. The LP for each week uses the same monthly inflow rate (m³/s).

This is an honest representation of the data resolution: if you only have
monthly historical data, sub-monthly stages cannot have independent noise.
They share the monthly signal. This is different from generating 4 independent
noise draws at monthly scale (which overstates independence).

### PAR model organization

All PAR parameters are keyed by `IdMes` (month), not by stage:

- `media_serie_temporal[IdMes]` — seasonal mean
- `desvio_serie_temporal[IdMes]` — seasonal std
- `coeficiente_auto_correlacao[IdMes × lag]` — AR coefficients per month
- `correlacao[IdVariavelAleatoria × IdMes]` — cross-entity correlation per month

The noise generation pipeline:

1. White noise `ruido_branco_espaco_amostral[Periodo × IdRealizacao]`
2. Correlated noise `ruido_correlacionado_espaco_amostral[Periodo × IdRealizacao]`
3. Scenario realization `cenarios_realizacao_transformada_espaco_amostral[Periodo × IdCenario]`

All indexed by `Periodo`, then mapped to stages via the filtering mechanism.

### Key differences from Cobre

| Aspect                     | SPARTACUS                                   | Cobre                                  |
| -------------------------- | ------------------------------------------- | -------------------------------------- |
| LP per stage               | Iterates over sub-stage periods             | One constraint set per stage           |
| Noise per stage            | One per stochastic period within stage      | One per stage                          |
| PAR parameters             | Per month (`IdMes`), independent of stages  | Per stage (`InflowModel.stage_id`)     |
| Sub-monthly stages         | Share monthly noise (honest)                | Independent draws at monthly σ (wrong) |
| Stage ↔ stochastic mapping | Explicit `getIteradores()` filtering        | Implicit via `season_id`               |
| Period arithmetic          | Rich `Periodo` class with duration encoding | Calendar dates on stages               |

### Implications for Cobre

After discussion, the "stochastic period" concept proposed initially
**collapses into the existing `season_id` mechanism**. The information
needed to determine noise sharing is already present:

- `season_id` groups stages by recurring season
- Stage dates identify which year each occurrence belongs to
- Same `(season_id, year)` = same stochastic period = same noise

No new temporal entity is required. The behavioral change is a policy
in the `ForwardSampler`: stages sharing `(season_id, year)` reuse ξ
instead of drawing independently.

**What Cobre takes from SPARTACUS:**

1. The noise-sharing principle for sub-period stages (Debt 5)
2. Confirmation that sub-stage temporal resolution is handled in the LP,
   not by multiplying SDDP stages (chronological blocks = SPARTACUS periods)
3. The clean separation: stages define cut boundaries, blocks define
   decision granularity, seasons define stochastic parameters

**What Cobre does NOT need from SPARTACUS:**

1. A separate `Periodo` type — `season_id` + stage dates suffice
2. Sub-stage period iteration in the solver — chronological blocks serve
   this role with better LP template reuse
3. A `getIteradores()`-style dynamic mapping — the mapping is static from
   the stage definitions

### The DECOMP coupling constraint

DECOMP-like studies impose an additional requirement beyond noise sharing.
The boundary cuts from a monthly model have monthly inflow lags as state
variables. At the coupling point, the lag must be at the monthly resolution
expected by the cuts, not at the stage resolution.

**When noise is shared** (monthly stochastic data, weekly stages): trivial.
All weekly stages have the same inflow rate, which is the monthly lag.

**When noise differs per stage** (weekly External scenarios): the state
transition at the coupling boundary must aggregate:

```
monthly_lag = Σ(inflow_w × hours_w) / Σ(hours_w)
```

**The chronological-blocks alternative avoids this entirely:**
Use 1 monthly stage with 4 weekly chronological blocks instead of 4 weekly
stages. The LP makes weekly decisions internally, and the state transition
at the stage boundary naturally produces a monthly-resolution lag. No
aggregation needed. No cut approximation between weeks. This is the
recommended pattern for DECOMP-like studies in Cobre.

---

## Opportunity 11: External initial state sampling

**Severity:** Design opportunity — large computational savings for
long-term planning studies.

### The problem: pre-study burn-in

Long-term monthly planning studies (NEWAVE-like) commonly need to start
from a "steady state regime" rather than from specific initial conditions,
to avoid bias from recent operative state. The current approach in the
industry is to add pre-study years: e.g., 10 years of SDDP stages before
the 5 years of actual interest. This is extremely costly — the pre-study
stages require the full SDDP machinery (cuts, forward passes, backward
passes) for results that are entirely discarded. The only purpose is to
let reservoir levels and inflow lags converge to a statistically
representative regime.

### Proposed solution: pool of initial states

Generalize the existing External scenario pattern to initial conditions.
Instead of a single deterministic `(storage, past_inflows)`, provide a
**pool of N possible initial states**. Each forward pass samples one from
the pool. The SDDP algorithm builds cuts valid across the state space
explored by these diverse starting points.

This is not a new concept — it's the same "pool of realizations" pattern
that External scenarios already use for inflows, loads, and NCS.

### How it works

**Input:** An optional `external_initial_states/` directory (or parquet
files) alongside `initial_conditions.json`:

- `external_initial_storage.parquet`:
  `(scenario_id: INT32, hydro_id: INT32, value_hm3: DOUBLE)`
- `external_initial_lags.parquet`:
  `(scenario_id: INT32, hydro_id: INT32, lag: INT32, value_m3s: DOUBLE)`
- `external_recent_observations.parquet` (optional, for mid-season starts):
  `(scenario_id: INT32, hydro_id: INT32, start_date: DATE, end_date: DATE, value_m3s: DOUBLE)`

Each `scenario_id` defines one complete initial state. Picking scenario 42
gives storage_42 + lags_42 + recent_obs_42 for all hydros jointly.

When `external_initial_states/` is present, `initial_conditions.json` is
ignored (or used as fallback when the directory is absent).

**Sampling:** Same hash-based deterministic selection as Historical windows:

```
initial_state_idx = hash(iteration, scenario) % n_initial_states
```

Same wrapping behavior: if `n_initial_states < forward_passes`, warn and
wrap. Same deterministic reproducibility guarantees.

**Forward pass:** At the start of each trajectory, load the sampled
initial state into `ws.current_state`. Everything else proceeds normally.

**Backward pass:** The trial point at stage 0 comes from the forward
pass's sampled initial state (as it does today — the backward pass uses
forward pass states as trial points). Cuts at stage 0 are built at
different operating points across iterations, improving cut quality.

### Lower bound computation

With a single initial state, the lower bound is one solve of stage 0.

With N initial states, two options:

1. **Average over pool:** Solve stage 0 for each initial state in the pool,
   take the mean. Cost: N × (one LP solve). For N=100 and a 5-year study
   with 60 stages, this is 100 extra solves vs 60×iterations solves in the
   main loop — negligible.

2. **Sample-based:** Use the forward-pass sample of initial states and
   report the average lower bound across sampled trajectories. Converges
   to the true expected cost as iterations increase.

Option 1 is more precise for convergence checks. Option 2 is simpler and
sufficient for practical purposes.

### What it buys

| Approach                 | Stages | Cost (relative)         |
| ------------------------ | ------ | ----------------------- |
| 10yr burn-in + 5yr study | 180    | 3×                      |
| 5yr study + external IS  | 60     | 1× + N LP solves for LB |

For a typical study with N=50–200 initial states, the savings are ~2.5×
on total computation.

### Generating the initial state pool

The framework is agnostic to how the pool is generated. Possible sources:

- **Prior SDDP policy:** Run simulation with an existing policy, sample
  terminal states at the end of each simulated year. These represent the
  steady-state distribution under optimal operation.
- **Rolling operational data:** Take observed reservoir levels and inflow
  histories from the last N years of actual system operation.
- **ML models:** Train on historical operative data to generate plausible
  initial states that capture correlations between reservoir levels across
  the system.
- **Previous study's terminal states:** Chain studies — the terminal
  states of last month's planning study become the initial states of
  this month's study.

### Interaction with other debts

- **Debt 3 (past_inflows metadata):** External initial states make the
  lag resolution explicit — the parquet schema carries lag indices with
  values at the appropriate resolution.
- **Debt 10 (recent_observations):** The external initial states can
  include per-scenario recent observations, solving the mid-season
  start problem for each initial state independently.
- **Debt 5 (noise sharing):** Independent of noise sharing. Initial
  state sampling adds uncertainty in the starting point; noise sharing
  controls uncertainty within the trajectory.

### Why not a new concept

The sampling pattern (pool + hash-based selection + wrapping) is identical
to External scenarios. The only difference is what's being sampled:

| Existing                                               | Proposed                                             |
| ------------------------------------------------------ | ---------------------------------------------------- |
| External inflow scenarios: pool of inflow trajectories | External initial states: pool of starting conditions |
| Sampled per (iteration, scenario)                      | Sampled per (iteration, scenario)                    |
| `hash % n_scenarios`                                   | `hash % n_initial_states`                            |
| Parquet input                                          | Parquet input                                        |

---

## Debt 12: Terminal-stage external FCF (boundary cuts)

> **Status: RESOLVED** in the production-decomp plan (`667c240`).
> `BoundaryPolicy` config added to `cobre-io::config` with `path` and
> `source_stage` fields under `policy.boundary`. `load_boundary_cuts` and
> `inject_boundary_cuts` implemented in `cobre-sddp::policy_load` — loads cuts
> from a source Cobre policy checkpoint and injects them as fixed boundary
> conditions at the terminal stage. Wired into both CLI (`run.rs`) and Python
> bindings (`cobre-python/src/run.rs`). Integration test verifies LB does not
> degrade when boundary cuts are added. D28 DECOMP test exercises boundary
> cut composition end-to-end.

**Severity:** Missing feature — required for production DECOMP (Pattern B)
and any study that couples with an outer model.

### The problem

Production DECOMP has a short horizon: ~4 weekly stages + 1 monthly stage.
The future cost beyond the monthly stage comes from a **monthly model's
future cost function** (FCF) — in the traditional pipeline, NEWAVE's
boundary cuts. Without these terminal-stage cuts, the monthly stage's LP
has no information about the long-term future, producing myopic decisions.

Currently, Cobre's `policy_load.rs` supports loading cuts from a previous
**Cobre** run (warm start / resume from checkpoint), but only as initial
cuts for the same study. There is no mechanism for loading cuts from a
different study as terminal-stage boundary conditions.

### Proposed solution: Cobre→Cobre coupling via policy checkpoint

Rather than designing a format-specific loader for external models, the
feature loads an **existing Cobre policy checkpoint** as the terminal-stage
FCF. The production pipeline becomes:

1. Run a **monthly Cobre study** (equivalent to NEWAVE's role) → produces
   policy checkpoint with cuts for each monthly stage
2. Run a **weekly+monthly Cobre study** (equivalent to DECOMP's role) →
   loads the monthly policy's stage-N cuts as terminal-stage FCF

Both studies use the same entity IDs, same state variable layout
(storage + inflow lags per hydro), same cut format. The mapping is
trivial when both are Cobre studies.

### Interface

- **Input:** A Cobre policy checkpoint path + a stage index indicating
  which stage's cuts to import as the terminal FCF. For example, "load
  stage 2's cuts from the monthly policy" to get February's future cost
  approximation as the boundary for the weekly+monthly study.
- **Behavior:** Imported cuts are **fixed** — they are not updated by the
  SDDP training algorithm. They are added to the terminal stage's cut
  pool at setup time and remain unchanged throughout training.
- **Validation:** State dimension compatibility between the two studies.
  The terminal-stage state variables (storage + lags) must match the
  imported cut coefficients in dimension and ordering.
- **Lower bound:** The terminal-stage LP evaluates imported + own cuts.
  The lower bound computation includes the terminal FCF contribution.

### What this enables

| Pipeline step        | Old (NEWAVE + DECOMP)           | New (Cobre + Cobre)                                 |
| -------------------- | ------------------------------- | --------------------------------------------------- |
| Monthly policy       | NEWAVE (external)               | Cobre monthly study                                 |
| Monthly cuts         | NEWAVE FCF (proprietary format) | Cobre policy checkpoint                             |
| Weekly+monthly study | DECOMP (reads NEWAVE cuts)      | Cobre weekly+monthly study (reads Cobre checkpoint) |

The user can also chain studies: a long-term study produces cuts, a
medium-term study loads those as terminal FCF and produces its own cuts,
a short-term study loads the medium-term cuts. Each link is a standard
Cobre policy checkpoint.

### Interaction with lag accumulation

The terminal-stage FCF cuts have coefficients π_lag at monthly resolution.
The lag accumulation mechanism (Debt 6) ensures that the state vector's
lag values at the terminal stage are monthly averages (from the
accumulator), so the imported cut coefficients evaluate correctly.

---

## Debt 13: Multi-resolution PAR transition (monthly→quarterly)

**Severity:** Architectural gap — required for Pattern D (long-term
multi-resolution studies). Deferred until Debt 6 accumulator is proven
in production.

**Problem:** Long-term planning studies benefit from monthly stages in the
near term (years 1–5) transitioning to quarterly stages in the far term
(years 6–10+). Each resolution requires its own PAR model: 12-season
monthly PAR fitted on monthly observations, 4-season quarterly PAR fitted
on quarterly-aggregated observations. The transition boundary requires
converting the lag state from monthly to quarterly resolution.

### What works without changes

- **Disjoint season_id spaces:** Monthly seasons 0–11 and quarterly
  seasons 12–15 in a single `Custom` `SeasonMap` (see Debt 4).
- **Independent PAR models:** The `PrecomputedPar` is indexed by
  `(season_id, hydro)`. Seasons 0–11 get monthly parameters; seasons
  12–15 get quarterly parameters. The fitting pipeline (Debt 2) handles
  aggregation.
- **The accumulator mechanism:** `StageLagTransition` computes lag period
  boundaries from the `SeasonMap`. During quarterly stages, the lag period
  is a quarter: three monthly stages accumulate, the third finalizes.
  This is the standard accumulator pattern from Debt 6 with different
  period boundaries.

### What requires new design: lag state resolution transition

At the monthly→quarterly boundary (last monthly stage → first quarterly
stage), the lag state must convert from monthly to quarterly resolution.

**The specific problem (lag depth at the transition):**

Suppose the quarterly PAR model has order 2. When entering the first
quarterly stage (Q1 of year 6), the lag state needs:

```
lag[0] = Q4 average  (Oct+Nov+Dec of year 5)  ← accumulator provides this ✓
lag[1] = Q3 average  (Jul+Aug+Sep of year 5)  ← where does this come from?
```

The monthly PAR model with order 2 only retains the last 2 monthly lags
(Dec, Nov). Q3 = avg(Jul, Aug, Sep) requires lags from 4–6 months back —
they fell off the shift register long ago. This is the same P2 (lag depth
overflow) problem from Debt 6, but at a different resolution boundary.

### Proposed solution: multi-period accumulator ring buffer

The accumulator runs throughout the monthly phase, building quarterly
aggregates in a side buffer even while the monthly PAR model is active:

```rust
/// Extended accumulator for resolution transitions.
/// Pre-allocated in ScratchBuffers, consumed on the hot path.
struct LagAccumulator {
    /// Current lag period accumulation (length: n_hydros).
    current: Vec<f64>,
    current_weight: f64,

    /// Ring buffer of completed lag values at downstream resolution.
    /// Length: n_hydros × n_completed_slots.
    /// Only non-empty when a resolution transition is upcoming.
    completed_lags: Vec<f64>,
    n_completed: usize,
    n_completed_slots: usize,  // = downstream PAR order
}
```

**During uniform monthly phases (no transition upcoming):**
`completed_lags` is empty, `n_completed_slots = 0`. The accumulator
operates exactly as the simple single-period buffer from Debt 6. Zero
overhead — one multiply-add per hydro per stage.

**During the pre-transition window (last L_q quarters before transition):**
The accumulator begins building quarterly values. Each monthly stage
accumulates into the current quarterly period. At quarter boundaries, the
completed quarterly average is pushed into `completed_lags`. After L_q
quarters, the ring buffer holds all the quarterly lag values needed at the
transition point.

**At the transition point:** The lag state is rebuilt from
`completed_lags`. `lag[0]` = most recent completed quarterly average,
`lag[1]` = next most recent, etc. The monthly lag state is discarded.

**Setup-time precomputation:** The pre-transition window start is fully
determinable at setup time: `transition_stage - (L_q × 3)` monthly stages
back. The `StageLagTransition` for stages in this window gets an
additional `accumulate_downstream: bool` flag indicating that the stage
should also accumulate into the downstream ring buffer.

### Concrete example (monthly→quarterly, PAR order 2)

```
Study: 60 monthly stages (years 1-5) + 20 quarterly stages (years 6-10)
Quarterly PAR order: 2 → need Q3 and Q4 of year 5 at transition

Pre-transition window: stages 54-59 (Jul-Dec year 5)
  Stage 54 (Jul): accum Q3, weight += jul_fraction
  Stage 55 (Aug): accum Q3, weight += aug_fraction
  Stage 56 (Sep): accum Q3, finalize → completed_lags[0] = Q3_avg
  Stage 57 (Oct): accum Q4, weight += oct_fraction
  Stage 58 (Nov): accum Q4, weight += nov_fraction
  Stage 59 (Dec): accum Q4, finalize → completed_lags[1] = Q4_avg

At stage 60 (Q1 year 6):
  lag[0] = completed_lags[1] = Q4_avg  ✓
  lag[1] = completed_lags[0] = Q3_avg  ✓
  Quarterly PAR formula: z = μ_Q1 + ψ₁·(Q4 - μ_Q4) + ψ₂·(Q3 - μ_Q3) + σ·ε
```

**During stages 54–59:** the monthly PAR model uses its own monthly lags
normally (shift register operates at monthly resolution as usual). The
downstream accumulator runs independently in the scratch buffer. No
interference between the two.

### Cost analysis

- **Pre-transition window:** `n_hydros` extra multiply-adds per stage for
  the downstream accumulator. Same order as the primary accumulator.
- **At quarterly finalization (every 3 stages):** `n_hydros` divisions +
  ring buffer push. Negligible.
- **No transition in study:** `completed_lags` is empty. Zero overhead.
  The `accumulate_downstream` flag is `false` for all stages; the branch
  is never taken on the hot path.

### Dependencies

- **Debt 2:** Quarterly PAR model requires aggregated observations.
- **Debt 4:** Season_id validation must understand disjoint ranges.
- **Debt 6:** The base accumulator mechanism must exist before extending
  it with the ring buffer.
- **No impact on Pattern B or C:** The multi-resolution transition is
  orthogonal to DECOMP and weekly-PAR patterns. The `LagAccumulator`
  degenerates to the simple Debt 6 buffer when no transition is present.

### Testing

- **D-multires-01:** 12 monthly + 4 quarterly stages with PAR(1).
  Verify that the quarterly lag at the transition equals the weighted
  average of the last 3 monthly realized inflows.
- **D-multires-02:** Same with PAR(2). Verify both quarterly lags (Q4
  and Q3) are correct at the transition.
- **D-multires-03:** Uniform monthly study (no transition). Verify zero
  overhead — `completed_lags` is never allocated, `accumulate_downstream`
  is false for all stages.

---

## Note on scenario generation

In production DECOMP, a program called GEVAZP generates the external
scenario tree for the monthly stage. GEVAZP receives recent observations,
the monthly inflow history, and the weekly forecast, then produces
scenarios with temporal and spatial correlation conditioned on this
information.

**Cobre does not replicate GEVAZP.** The external scenario tree is
provided as input — Cobre is agnostic to how it was generated. However,
the design must support two distinct scenario generation patterns:

1. **All-External (Pattern B):** Weekly stages get a single external
   forecast. The monthly stage gets an external scenario tree from GEVAZP
   (or any other tool). The PAR model provides the baseline for
   standardization (computing eta values) but does not generate noise.

2. **PAR-generated (Pattern C):** The PAR model generates inflow noise
   for all stages. `past_inflows` provides the initial lag values.
   `recent_observations` seeds the accumulator. No external scenario
   generator needed — the stochastic process is internal to Cobre.

Both patterns use the same lag accumulation mechanism (Debt 6), the same
`recent_observations` seeding (Debt 10), and the same frozen-lag behavior.
The difference is only in the noise source: External eta values vs
PAR-generated ξ. The `ForwardSampler`'s `ClassSampler` dispatch already
handles this distinction per entity class.

The user may also mix patterns: use External scenarios for the first
month (weekly stages with forecast) and PAR-generated noise for months
beyond. This works naturally because the `ClassSampler` operates per
(stage, entity class) — the same forward pass can use different noise
sources at different stages.

---

## Relationship between debts

```
Debt 1 (month0 bug)
  └── Fix independently. Required for any non-monthly uniform study.
      Straightforward: thread SeasonMap into Historical pipeline.

Debt 2 (observation aggregation + validation)
  ├── Validation: reject coarser-than-season observations (can't disaggregate)
  ├── Aggregation: finer-than-season observations → weighted average per season
  ├── Required for Pattern D (monthly obs → quarterly PAR model)
  └── Prerequisite for Debt 13 (quarterly PAR needs aggregated observations)

Debt 3 (past_inflows metadata)
  └── Low priority. Current positional convention works. Revisit when
      dual-resolution lag support is needed.

Debt 4 (season_id consistency across heterogeneous season spaces)
  ├── V4.1: every stage season_id must exist in season_definitions
  ├── V4.2: observation coverage per season (after Debt 2 aggregation)
  ├── V4.3: resolution consistency within season groups
  ├── V4.4: contiguity within resolution bands (warn on gaps)
  └── Must support disjoint ranges (0-11 monthly + 12-15 quarterly)

Debt 5 (noise sharing for same-season stages)
  ├── ForwardSampler policy: same (season_id, year) → same ξ
  ├── Required for Pattern C (weekly stages + PAR). NOT required for
  │   Pattern B (production DECOMP with all-External scenarios).
  ├── Must be consistent in both forward and backward pass (opening
  │   tree shares ξ for same-(season_id, year) stages within each opening)
  └── No new data structures needed

Debt 6 (lag resolution at coupling boundaries)
  ├── StageLagTransition precomputed at setup from SeasonMap boundaries
  ├── Accumulator + frozen lags for both External and PAR noise sources
  ├── Backward pass does NOT need accumulator (one-step lookahead only)
  ├── Opening tree clamping: effective_openings = min(configured, available)
  ├── Uniform n_scenarios workaround: pad weekly stages with identical copies
  │   (interim; long-term: node-based policy graph)
  └── Critical for Patterns B, C, and D

Debt 7 (PAR behavior documentation)
  └── Document the known limitations, not a code fix.
      Users needing sub-monthly variability use External scenarios.

Debt 8 (historical residuals as opening tree noise)
  ├── New NoiseMethod variant using observed residuals
  ├── Depends on Debt 1 (month0 fix) for correct residual computation
  ├── Naturally consistent with Debt 5 (noise sharing by year)
  ├── Opening tree clamping applies (finite residual pool)
  └── Infrastructure partially exists (Historical standardization pipeline)

Debt 9 (standardize_external_inflow lag advancement)
  ├── Setup-time bug: advances lags per-stage, not per-lag-period
  ├── Same fix as Debt 6: use StageLagTransition + frozen lags
  └── Must be consistent with runtime forward pass behavior

Debt 10 (mid-season study start — recent_observations)
  ├── past_inflows only carries complete seasonal lags
  ├── Partial-season observations before study start have nowhere to go
  ├── New input: recent_observations seeds the lag accumulator
  ├── Works identically for External and PAR noise sources
  └── Required for rolling DECOMP revisions (rv0 → rv1 → rv2 ...)

Opportunity 11 (external initial state sampling)
  ├── Pool of (storage, lags, recent_obs) sampled per forward pass
  ├── Replaces 10yr pre-study burn-in with direct state sampling
  ├── Same pattern as External scenarios (pool + hash + wrapping)
  ├── Interacts with Debt 10 (per-scenario recent_observations)
  └── Enables ML/statistical approaches to steady-state estimation

Debt 12 (terminal-stage external FCF)
  ├── Load Cobre policy checkpoint as terminal-stage boundary cuts
  ├── Fixed cuts, not updated during training
  ├── Enables Cobre→Cobre coupling (monthly study → weekly+monthly study)
  ├── Lag accumulation (Debt 6) ensures correct lag values at terminal
  │   stage for boundary cut evaluation
  └── Critical for Pattern B (production DECOMP)

Debt 13 (multi-resolution PAR transition)
  ├── Monthly→quarterly lag state conversion at resolution boundary
  ├── LagAccumulator ring buffer builds downstream-resolution lags
  │   during pre-transition window (L_q quarters before boundary)
  ├── Depends on Debt 2 (aggregated observations for quarterly PAR)
  ├── Depends on Debt 4 (disjoint season_id ranges)
  ├── Depends on Debt 6 (base accumulator mechanism)
  ├── Zero overhead when no resolution transition in study
  └── Critical for Pattern D; orthogonal to Patterns B and C
```

## Recommended study patterns

### Pattern A: Chronological blocks (simplest, when weekly cuts not needed)

```
┌─────────────────────────────────────┐  ┌───────────┐
│            Stage 0 (Month 1)        │  │  Stage 1   │
│  Block 0: Week 1 (168h)            │  │  (Month 2) │
│  Block 1: Week 2 (168h)  Chrono    │  │  1 block   │──► terminal FCF
│  Block 2: Week 3 (168h)  blocks    │  │            │    (monthly lags)
│  Block 3: Week 4 (192h)            │  │            │
│                                     │  │            │
│  season_id = 0 (January)            │  │ season_id  │
│  1 noise draw (monthly PAR)         │  │  = 1 (Feb) │
│  Weekly decisions via blocks        │  │            │
│  No intra-month Benders cuts        │  │            │
└─────────────────────────────────────┘  └───────────┘
         │                                      │
         └── state transition ──────────────────┘
             (monthly storage + monthly inflow lag)
```

Advantages: one LP per month, optimal intra-month allocation, no noise
sharing needed, no accumulator needed (one stage = one lag transition).

### Pattern B: Weekly stages (when weekly cuts ARE needed)

```
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌───────────┐
│Stage 0 │→│Stage 1 │→│Stage 2 │→│Stage 3 │→│Stage 4 │→│  Stage 5  │
│  W1    │ │  W2    │ │  W3    │ │  W4    │ │  W5    │ │    M2     │
│168h    │ │168h    │ │168h    │ │168h    │ │168h    │ │  ~720h    │
│        │ │        │ │        │ │        │ │        │ │           │
│sid=3   │ │sid=3   │ │sid=3   │ │sid=3   │ │sid=3   │ │  sid=4    │
│ext:1   │ │ext:1   │ │ext:1   │ │ext:1   │ │ext:1   │ │  ext:N    │
│        │ │        │ │        │ │        │ │        │ │           │
│ cuts ← │ │ cuts ← │ │ cuts ← │ │ cuts ← │ │ cuts ← │ │ term FCF  │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └───────────┘
  frozen     frozen     frozen     frozen    finalize     finalize
  lags       lags       lags       lags      april lag    may lag

Noise: External (1 scenario/week, N scenarios/month)
Lags:  Frozen at past_inflows during April, accumulator finalizes at W5
Cuts:  Weekly Benders cuts at each stage boundary
       Terminal FCF at Stage 5 from monthly Cobre policy
```

Each weekly stage may contain parallel blocks for sub-weekly decision
granularity (peak/off-peak dispatch). With PAR noise (Pattern C), noise
sharing applies across same-`(season_id, year)` stages. With External
scenarios (Pattern B), noise sharing is irrelevant — each stage uses
its external forecast.

Opening tree clamping ensures the backward pass is efficient:
`effective_openings = min(configured, n_available_scenarios)`.
Weekly stages with 1 scenario get 1 opening (no redundancy).
