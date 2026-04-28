# PAR(p)-A Inflow History Classification (TIPO Detection)

**Status:** Draft — not yet implemented
**Scope:** `cobre-stochastic` (PAR/PAR-A fitting), `cobre-sddp` (estimation orchestration)

## Motivation

When validating cobre's PAR(p)-A fit against NEWAVE's `parpvaz.dat` report
on the `pmo_abr_26` case, we observed large discrepancies in selected
orders, PACFs, and seasonal stats for plants with regulated or anomalous
inflow histories. The most striking example is **BELO MONTE / April**:

| Quantity    | Cobre    | NEWAVE   |
| ----------- | -------- | -------- |
| Mean (m³/s) | 12161.32 | 13900.00 |
| Std (m³/s)  | 2659.68  | 0.00     |
| Order       | non-zero | 0        |
| PACF lag 1  | non-zero | 0        |

Inspection of `parpvaz.dat` revealed that NEWAVE pre-classifies every
`(plant, calendar_month)` pair into one of five behavioral types and
applies type-specific overrides to the seasonal stats and order-selection
pipeline. cobre currently treats every series identically, which produces
divergent fits on plants with operational caps, ecological-flow floors,
or sparse history.

This document specifies the classification rules, the per-type overrides,
and how they slot into cobre's existing fitting pipeline.

## NEWAVE TIPO Catalog

The legend appears verbatim in `parpvaz.dat` (lines 167–172 in the
`pmo_abr_26` report):

```
TIPO 0 - SEM COMPORTAMENTO ESPECIFICO A SER DESTACADO
TIPO 1 - HISTORICO COM VAZOES CONSTANTES/NULAS
TIPO 2 - HISTORICO COM MAIS DE 10% DE VAZOES NEGATIVAS
TIPO 4 - VAZOES CONSTANTES EM MAIS DE 50% DO HISTORICO
TIPO 5 - HISTORICO BI-MODAL
```

NEWAVE's parpvaz header attaches one TIPO code per `(plant, month)` plus
a "constant value" used by TIPO 1 / TIPO 4 overrides. Example for
PIMENTAL (line 164):

```
314 PIMENTAL  302    1 4 0 0 4 1 1 1 1 4 1 1   1100 1600 0 0 2900 1600 1100 900 750 700 800 900
```

Decoded as `(month → TIPO, constant)`:

| Month | TIPO | Constant |
| ----- | ---- | -------- |
| Jan   | 1    | 1100     |
| Feb   | 4    | 1600     |
| Mar   | 0    | —        |
| Apr   | 0    | —        |
| May   | 4    | 2900     |
| Jun   | 1    | 1600     |
| Jul   | 1    | 1100     |
| Aug   | 1    | 900      |
| Sep   | 1    | 750      |
| Oct   | 4    | 700      |
| Nov   | 1    | 800      |
| Dec   | 1    | 900      |

## Classification Rules

The classifier runs once, before `estimate_seasonal_stats`. Inputs:

- `observations[s]: &[f64]` — historical values for calendar month `s`,
  one entry per year (after the bridge has windowed history to
  `[hist_start, study_start)`).

Output: `Tipo` enum + optional `constant_value: f64`.

### TIPO 0 — Default

```
Tipo::Default
```

No specific behavior. Apply unchanged sample mean/std and run the standard
PACF order-selection pipeline.

### TIPO 1 — Constant or Null History

**Detect when:** every `(year, month)` observation in the series equals
the same value, OR every observation is zero/missing.

```
all observations identical (within ε)
  → Tipo::Constant { value: observations[0] }
```

Use a small absolute tolerance (`ε = 1e-6`) for the equality check —
NEWAVE uses exact equality on stored single-precision floats, but we
have IEEE-754 doubles read from parquet, so a tolerance is safer.

**Override:**

- `mean = constant_value`
- `std = 0`
- AR order = 0
- Annual coefficient = 0
- No noise injection at this stage during simulation

### TIPO 2 — Pathological Negative History

**Detect when:** more than 10% of observations are strictly negative.

```
fraction_negative(observations) > 0.10
  → Tipo::ManyNegative
```

Negative incremental inflows are physically meaningful (a plant losing
water to upstream evaporation/diversion can have small negative
incrementals), but a series dominated by negatives indicates the
incremental construction is broken upstream (e.g., the bridge subtracted
a wrong upstream posto). NEWAVE flags these to prevent fitting on
nonsense data.

**Override:** Same as TIPO 1 (degenerate fit), but additionally:

- Emit a warning in `cobre.log` listing every TIPO 2 plant/month for
  operator attention
- Use the **historical sample mean** as the constant (not zero), so the
  scenario library still produces nonzero flows

This warning is operationally important — TIPO 2 usually means upstream
data is wrong, not that the plant should be silenced.

### TIPO 4 — Cap-Saturated History

**Detect when:** the modal value of the series accounts for more than
50% of all observations AND that modal value equals or exceeds the
99th percentile.

```
mode_count(observations) / n_observations > 0.50
  AND mode_value >= percentile(observations, 99)
  → Tipo::Saturated { value: mode_value }
```

The dual condition rules out cases where many observations cluster at a
non-cap value (e.g., a plant with frequent zero-flow periods, which is
better captured as TIPO 1 with `value = 0`).

**Override:**

- `mean = mode_value` (the cap)
- `std = 0`
- AR order = 0
- Annual coefficient = 0

The std=0 propagates structural zeros into adjacent months' PACF rows
(any lag whose conditioning month has std=0 contributes 0 to the
partial correlation), which is exactly NEWAVE's observed behavior on
BELO MONTE.

### TIPO 5 — Bimodal History (Future Work)

**Detect when:** the empirical distribution has two distinct modes with
combined mass > 80%.

This is rare and out of scope for the initial implementation. We will
emit a warning but proceed with the default fit. Add a follow-up ticket
for proper bimodal handling.

## Cobre Implementation Plan

### Type Definition

In `crates/cobre-stochastic/src/par/fitting.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HistoryClass {
    Default,
    Constant { value: f64 },
    ManyNegative { sample_mean: f64 },
    Saturated { cap: f64 },
}

pub struct ClassifiedSeasonalStats {
    pub mean_m3s: f64,
    pub std_m3s: f64,
    pub class: HistoryClass,
}
```

### Classification Function

```rust
pub fn classify_history(observations: &[f64]) -> HistoryClass {
    if observations.is_empty() {
        return HistoryClass::Constant { value: 0.0 };
    }

    // TIPO 1 first — exact constancy supersedes other classifiers
    let first = observations[0];
    if observations.iter().all(|&v| (v - first).abs() < 1e-6) {
        return HistoryClass::Constant { value: first };
    }

    // TIPO 2 — pathological negative content
    let n = observations.len() as f64;
    let n_neg = observations.iter().filter(|&&v| v < 0.0).count() as f64;
    if n_neg / n > 0.10 {
        let mean = observations.iter().sum::<f64>() / n;
        return HistoryClass::ManyNegative { sample_mean: mean };
    }

    // TIPO 4 — cap saturation
    if let Some((mode_value, mode_count)) = mode_with_count(observations) {
        if (mode_count as f64) / n > 0.50 {
            let p99 = percentile(observations, 0.99);
            if mode_value >= p99 {
                return HistoryClass::Saturated { cap: mode_value };
            }
        }
    }

    HistoryClass::Default
}
```

The `mode_with_count` helper sorts observations and runs a single pass
to find the longest run of equal values. Tolerance handling: round to
1 m³/s (raw VAZOES.DAT precision) before mode counting.

### Wire Into Estimation

`estimate_seasonal_stats` in `crates/cobre-stochastic/src/par/fitting.rs`
becomes the single point that returns classification alongside stats:

```rust
pub fn estimate_seasonal_stats_classified(
    observations_by_season: &[Vec<f64>],
) -> Vec<ClassifiedSeasonalStats> {
    observations_by_season
        .iter()
        .map(|obs| {
            let class = classify_history(obs);
            let (mean, std) = match class {
                HistoryClass::Constant { value } => (value, 0.0),
                HistoryClass::Saturated { cap } => (cap, 0.0),
                HistoryClass::ManyNegative { sample_mean } => (sample_mean, 0.0),
                HistoryClass::Default => sample_mean_std(obs),
            };
            ClassifiedSeasonalStats { mean_m3s: mean, std_m3s: std, class }
        })
        .collect()
}
```

### Order-Selection Short-Circuit

In `crates/cobre-sddp/src/estimation.rs`, the order-selection loop
(`reduce_entity_orders_annual` and the upstream PACF computation) must
short-circuit when `class != Default`:

```rust
if !matches!(stats[season_id].class, HistoryClass::Default) {
    // Force order = 0, no AR coefficients, no annual coefficient
    set_zero_order(season_id);
    continue;
}
```

This is the single hot-path change required. Everything else
(`conditional_facp_partitioned`, `select_order_pacf_annual`,
`estimate_periodic_ar_annual_coefficients`) already handles std=0
correctly because of the `validate_order_contributions` plumbing
landed in commit 4854630.

### LP Construction

When `class != Default`, the LP for that stage gets:

- `inflow[hydro, stage] = mean_m3s` as a fixed parameter (not a state
  variable depending on lags)
- No noise term
- No annual lag dependency

This is consistent with NEWAVE's reported behavior (zero residual
variance, deterministic forecast).

### Reporting

Persist classification to a new output file
`output/stochastic/inflow_history_classification.parquet` with schema:

```
hydro_id      INT32
stage_id      INT32
tipo          INT32   -- 0, 1, 2, 4 matching NEWAVE codes
class_value   FLOAT64 -- the constant/cap value when applicable, NaN otherwise
n_observations INT32
fraction_negative FLOAT64
modal_count   INT32
```

This makes the classification auditable and debuggable. Both CLI and
Python parity required (per `CLAUDE.md` rule).

## Test Plan

### Unit Tests (`crates/cobre-stochastic/src/par/fitting.rs`)

1. `classify_history_default` — random Gaussian → `Default`
2. `classify_history_constant_zero` — all zeros → `Constant { 0.0 }`
3. `classify_history_constant_nonzero` — all 1100 → `Constant { 1100.0 }`
4. `classify_history_many_negative` — 11% strictly negative → `ManyNegative`
5. `classify_history_cap_saturated` — 60% at 13900, rest below
   → `Saturated { 13900.0 }`
6. `classify_history_cap_below_threshold` — 49% at cap → `Default`
7. `classify_history_low_mode_high_floor` — 60% at 0 (well below p99)
   → `Constant { 0.0 }` (TIPO 1 catches before TIPO 4)

### Integration Tests (`crates/cobre-sddp/tests/`)

1. `parpa_tipo4_belo_monte_april` — fixture replicates a TIPO 4 scenario,
   asserts `mean=cap`, `std=0`, `order=0`, no AR coefficients, no annual
   coefficient.
2. `parpa_tipo1_pimental_dry_season` — fixture replicates TIPO 1 across
   multiple months, asserts cascade through `validate_order_contributions`
   reduction does not crash on zero-std conditioners.
3. `parpa_tipo2_warning_emitted` — asserts the warning channel reports
   the offending plant/month.

### Parity Test

Compare cobre's classification output for `convertido_parpa` against the
parpvaz.dat header table parsed via inewave. Acceptance: every
`(plant, month)` pair matches NEWAVE's TIPO assignment, with documented
exceptions for any TIPO-5 plants (skipped in initial implementation).

## Out of Scope

- **TIPO 5 (bimodal)** handling — emit warning, proceed with default fit.
  Track separately.
- **Order-selection convergence** beyond TIPO. Initial probing on
  CAMARGOS (TIPO 0 across all months) shows cobre and NEWAVE still
  disagree on order in 8/12 months (e.g., December: cobre 5, NEWAVE 1;
  June: cobre 6, NEWAVE 2). Closing that gap requires investigating the
  PACF significance threshold, the order-selection criterion, and the
  contribution-reduction termination condition — separate from TIPO
  detection. File as follow-up tickets.
- **Per-month minimum-flow overrides** (NEWAVE's "VAZMIN" mechanism) —
  these are constraint-side, not stochastic-side, and belong in the
  hydro constraint converter rather than PARP-A fitting.

## Acceptance Criteria

1. Classification matches NEWAVE's parpvaz header for every plant/month
   on `convertido_parpa` (TIPO 0/1/2/4; TIPO 5 noted as warning).
2. BELO MONTE April reports `mean=13900, std=0, order=0` after
   classification.
3. PIMENTAL months Jan, Feb, May–Dec report `std=0, order=0` matching
   NEWAVE's TIPO 1/4 assignments.
4. CAMARGOS — TIPO 0 across all months — shows no behavioral change vs.
   pre-classification (regression baseline preserved).
5. New parquet output `inflow_history_classification.parquet` written
   by both CLI and Python paths.
6. CHANGELOG entry under unreleased.
