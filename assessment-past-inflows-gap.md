# Assessment: Past Inflows Gap for PAR(p) Lag Initialization

**Date:** 2026-03-18
**Status:** Implementation gap (specs exist, code does not)

---

## Summary

When using PAR(p) models with order >= 1, the SDDP algorithm needs past inflow values to initialize the autoregressive lag buffer at stage 0. The cobre-docs specs define a mechanism for this (via `inflow_history.parquet` and pre-study stages), but the implementation initializes all lags to zero. This doesn't break the algorithm — it just means the policy is computed without recent hydrological tendency.

## What the Specs Say

The specs define two complementary mechanisms:

1. **`inflow_history.parquet`** — raw historical inflows per hydro per date. The scenario generation spec (`scenario-generation.md` §1.2, Step 4) defines:

   ```
   lag_state[h][ℓ] = historical_inflow[h][t₀ − ℓ]  for ℓ = 1..max_order
   ```

2. **Pre-study stages** — negative-ID stages in `stages.json` (`input-scenarios.md` §1.3) representing historical periods before the study horizon, used exclusively for PAR lag initialization.

Past inflows are intentionally **not** in `initial_conditions.json`. Lag initialization is sourced from `inflow_history.parquet` cross-referenced with pre-study stage dates.

### Minor Spec Gap

The specs say `inflow_history.parquet` is "optional" but don't specify what happens when `inflow_lags: true` and no history is provided. Recommendation: error if PAR order > 0 and no history exists.

## Current Behavior

- `build_initial_state` in `cobre-sddp/src/setup.rs:896` allocates lag slots but fills them with `0.0`
- Zero lags = equivalent to assuming zero inflows in pre-study stages
- Algorithm correctness is preserved; only the initial tendency is lost
- Subsequent stages propagate actual computed inflows as lags, so the effect diminishes over time

## What Needs to Change

### Gap 1: cobre-bridge — Convert `vazpast.dat` (Python)

**Repo:** `cobre-rs/cobre-bridge`
**Effort:** ~1 point

NEWAVE stores recent past inflows in `vazpast.dat` (last year's monthly values per gauging station). This file is distinct from `vazoes.dat` (long historical record for PAR fitting).

**Tasks:**

- Add `convert_past_inflows()` to `converters/stochastic.py`
- Read `Vazpast` from inewave (or `Vazoes` filtered to the last `max_order` months before study start)
- Map gauging stations to hydro IDs using `Confhd.usinas.posto`
- Output rows to `scenarios/inflow_history.parquet` with schema `(hydro_id: int32, date: date32, value_m3s: float64)`
- Generate pre-study stages with negative IDs in `stages.json`

**Can be folded into:** ticket-010 (stochastic conversion) as an amendment, or a new mini-ticket in epic-03.

### Gap 2: cobre (Rust) — Consume `inflow_history.parquet` for Lag Init

**Repo:** `cobre-rs/cobre`
**Effort:** ~2 points

**Tasks:**

1. **cobre-io**: Add loader for `inflow_history.parquet` (the Parquet reader already exists at `crates/cobre-io/src/scenarios/inflow_history.rs` — verify it returns data indexed by hydro and date)

2. **cobre-io**: Add Layer 4/5 validation:
   - If `state_variables.inflow_lags: true` and PAR order > 0, require `inflow_history.parquet` to exist
   - Verify history covers at least `max_par_order` months before `stages[0].start_date`
   - Verify all hydro IDs with PAR coefficients have history rows

3. **cobre-sddp**: Update `build_initial_state` in `setup.rs` to:
   - Read `inflow_history.parquet`
   - For each hydro `h` and lag `ℓ = 1..max_order`:
     - Find the date = `stage_0_start_date - ℓ months`
     - Look up `historical_inflow[h][date]`
     - Set `state[indexer.inflow_lags.start + h * max_order + (ℓ-1)] = value`

4. **Tests**: Update deterministic test cases if any use PAR order > 0 (most use AR(0), so likely no change needed)

## NEWAVE File Mapping

| NEWAVE File   | Purpose                                   | Cobre Target                                        |
| ------------- | ----------------------------------------- | --------------------------------------------------- |
| `vazoes.dat`  | Long historical record (decades)          | PAR model fitting (`inflow_seasonal_stats.parquet`) |
| `vazpast.dat` | Recent past inflows (last year, by month) | Lag initialization (`inflow_history.parquet`)       |
| `dger.dat`    | `num_anos_pre_estudo`                     | Pre-study stage count                               |

## Impact If Not Fixed

- **Training:** Policy computed without recent hydrological memory at stage 0. For long horizons (60+ stages), the effect is minor since lags are populated from stage 1 onward.
- **Simulation:** First-stage simulated inflows don't reflect actual recent conditions. This matters for short-term operational planning where the initial state significantly affects the first few stages.
- **Validation against NEWAVE:** Cobre results for the first few stages may diverge more than necessary from NEWAVE, making the ticket-012 comparison harder to interpret.

## Recommended Approach

1. **Amend ticket-010** in cobre-bridge to include `vazpast.dat` conversion (small scope addition)
2. **Defer the Rust-side fix** to a post-v0.2.0 ticket (the algorithm works without it, and the v0.2.0 validation comparison uses a 10% tolerance that should absorb the initial-stage divergence)
3. **Or** create a new ticket in the current plan if the validation in ticket-012 shows the divergence is too large
