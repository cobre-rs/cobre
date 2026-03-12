# Stochastic Modeling

Hydrothermal dispatch is inherently uncertain. Reservoir inflows depend on
rainfall and snowmelt that cannot be known in advance, and electrical load
varies in ways that are predictable in aggregate but noisy at any given moment.
A dispatch policy that ignores uncertainty will systematically under-prepare for
dry periods and over-commit thermal capacity in wet years.

Cobre addresses this by treating inflows and loads as stochastic processes.
During training, the solver samples many scenario trajectories and builds a
policy that performs well across the distribution of possible futures — not
just for a single forecast. The stochastic layer is responsible for generating
those scenario trajectories in a statistically sound, reproducible way.

The stochastic models are driven by historical statistics provided by the user
in the `scenarios/` directory of the case. If no `scenarios/` directory is
present, Cobre falls back to white-noise generation using only the stage
definitions in `stages.json`. For any study with real hydro plants, providing
historical inflow statistics is strongly recommended.

---

## The `scenarios/` Directory

The `scenarios/` directory sits alongside the other input files in the case
directory:

```
my_study/
  config.json
  stages.json
  ...
  scenarios/
    inflow_seasonal_stats.parquet
    load_seasonal_stats.parquet
    inflow_ar_coefficients.parquet    (only when ar_order > 0)
    inflow_history.parquet            (alternative to pre-computed stats)
```

The directory is optional. When it is absent, Cobre generates independent
standard-normal noise at each stage for each hydro plant and scales it by
a default standard deviation — effectively treating all uncertainty as
white noise. This is sufficient for verifying a case loads correctly, but
is not representative of real inflow dynamics.

When `scenarios/` is present, Cobre reads the Parquet files and fits a
Periodic Autoregressive (PAR(p)) model for each hydro plant and each bus.
The fitted model generates correlated, seasonally-varying inflow and load
trajectories that reflect the historical statistics you supply.

---

## Inflow Statistics

`inflow_seasonal_stats.parquet` provides the seasonal distribution of
historical inflows for every (hydro plant, stage) pair.

### Schema

| Column     | Type   | Nullable | Description                                             |
| ---------- | ------ | -------- | ------------------------------------------------------- |
| `hydro_id` | INT32  | No       | Hydro plant identifier (matches `id` in `hydros.json`)  |
| `stage_id` | INT32  | No       | Stage identifier (matches `id` in `stages.json`)        |
| `mean_m3s` | DOUBLE | No       | Seasonal mean inflow in m³/s                            |
| `std_m3s`  | DOUBLE | No       | Seasonal standard deviation in m³/s (must be >= 0)      |
| `ar_order` | INT32  | No       | Number of AR lags in the PAR(p) model (0 = white noise) |

The file must contain exactly one row per `(hydro_id, stage_id)` pair.
Every hydro plant defined in `hydros.json` must have a row for every stage
defined in `stages.json`. The validator will reject the case if any
combination is missing.

For the `1dtoy` example, the file has 4 rows — one for each of the four
monthly stages — for the single hydro plant `UHE1` (hydro_id = 0).

### Inspecting the file

```python
# Polars
import polars as pl
df = pl.read_parquet("scenarios/inflow_seasonal_stats.parquet")
print(df)
```

```python
# Pandas
import pandas as pd
df = pd.read_parquet("scenarios/inflow_seasonal_stats.parquet")
print(df)
```

```sql
-- DuckDB
SELECT * FROM read_parquet('scenarios/inflow_seasonal_stats.parquet');
```

```r
# R with arrow
library(arrow)
df <- read_parquet("scenarios/inflow_seasonal_stats.parquet")
print(df)
```

---

## Load Statistics

`load_seasonal_stats.parquet` provides the seasonal distribution of
electrical demand at each bus. It drives the stochastic load model used
during training and simulation.

### Schema

| Column     | Type   | Nullable | Description                                             |
| ---------- | ------ | -------- | ------------------------------------------------------- |
| `bus_id`   | INT32  | No       | Bus identifier (matches `id` in `buses.json`)           |
| `stage_id` | INT32  | No       | Stage identifier (matches `id` in `stages.json`)        |
| `mean_mw`  | DOUBLE | No       | Seasonal mean load in MW                                |
| `std_mw`   | DOUBLE | No       | Seasonal standard deviation in MW (must be >= 0)        |
| `ar_order` | INT32  | No       | Number of AR lags in the PAR(p) model (0 = white noise) |

One row per `(bus_id, stage_id)` pair is required. Every bus in `buses.json`
must have a row for every stage. The load mean and standard deviation determine
both the expected demand level and how much it varies across scenarios in each
stage.

---

## The PAR(p) Model

PAR(p) stands for Periodic Autoregressive model of order p. It is the
standard model for hydro inflow time series in long-term hydrothermal planning
because inflows have two key properties the model captures well: seasonal
patterns (wet seasons and dry seasons recur predictably each year) and
autocorrelation (a wet month tends to be followed by another wet month, and
vice versa).

### What `ar_order` controls

The `ar_order` column in the seasonal statistics files sets the number of
autoregressive lags for each (entity, stage) pair.

**`ar_order = 0` — white noise.** The inflow at each stage is drawn
independently from a normal distribution with the specified mean and standard
deviation. There is no memory between stages: knowing last month's inflow
tells you nothing about this month's. This is the simplest setting and
appropriate when you lack historical data to fit AR coefficients, or when
the inflow series shows very little autocorrelation.

**`ar_order > 0` — periodic autoregressive.** The inflow at each stage
depends on the inflows at the preceding p stages, weighted by coefficients
that reflect the seasonal autocorrelation structure. A wet period is
followed by another wet period with the probability implied by the
coefficients. Higher AR orders capture longer-range dependencies: `ar_order
= 1` captures month-to-month persistence, `ar_order = 2` adds two-month
memory, and so on. Most hydro inflow series are well-described by `ar_order
= 1` or `ar_order = 2`.

### AR coefficients file

When any stage in `inflow_seasonal_stats.parquet` has `ar_order > 0`, Cobre
also requires an `inflow_ar_coefficients.parquet` file in the `scenarios/`
directory. This file contains the fitted AR coefficients in standardized form
(as produced by the Yule-Walker equations). The schema and the fitting
procedure are documented in the
[Case Format Reference](../reference/case-format.md).

The `1dtoy` example uses `ar_order = 0` for all stages, so no coefficients
file is needed.

### When to use higher AR orders

In general:

- Use `ar_order = 0` when historical data is short or when you want to
  establish a baseline with the simplest possible model.
- Use `ar_order = 1` for most real hydro systems. Monthly inflows have
  strong one-month autocorrelation, and a first-order model captures the
  bulk of it.
- Use `ar_order = 2` or higher when the inflow series shows multi-month
  persistence (common in systems with large upstream catchments or snowmelt
  storage). Validate with autocorrelation plots of your historical data.
- Setting `ar_order > 0` with `std_m3s = 0` is a validation error — the
  model requires non-zero variance to be identifiable.

For the theoretical derivation of the PAR(p) model, see
[Stochastic Modeling](https://cobre-rs.github.io/cobre-docs/theory/stochastic-modeling.html)
and [PAR(p) Autoregressive Models](https://cobre-rs.github.io/cobre-docs/theory/par-model.html)
in the methodology reference.

---

## Estimation from History

Instead of supplying pre-computed seasonal statistics in
`inflow_seasonal_stats.parquet`, you can provide raw historical inflow
observations and let Cobre estimate the PAR(p) parameters for you.

### Input: `inflow_history.parquet`

Place `inflow_history.parquet` in the `scenarios/` directory. The schema
and required column types are documented in the
[Case Format Reference](../reference/case-format.md). Each row represents
one historical observation of inflow at a given hydro plant and stage.

### What Cobre estimates

When `inflow_history.parquet` is present, Cobre performs the following
estimation steps automatically before building the scenario model:

1. **Seasonal statistics** — mean and standard deviation are computed from
   the historical observations for each (hydro plant, stage) pair. These
   replace the values you would otherwise provide in
   `inflow_seasonal_stats.parquet`.

2. **AR order selection** — Cobre evaluates candidate orders and selects
   the best fit per (hydro plant, stage) using AIC (Akaike Information
   Criterion). This avoids overfitting in series with little autocorrelation
   and captures meaningful persistence where it exists.

3. **AR coefficients** — Coefficients for the selected order are estimated
   via the Levinson-Durbin algorithm applied to the partial autocorrelation
   structure of the standardised historical series.

4. **Spatial correlation** — The contemporaneous correlation between
   hydro plants is estimated from the historical residuals after AR fitting.
   The resulting correlation matrix is used by the Cholesky-based noise
   generator in exactly the same way as a manually specified
   `correlation.json`.

### History vs. pre-computed stats: choose one

`inflow_history.parquet` and `inflow_seasonal_stats.parquet` are mutually
exclusive inputs for the inflow model. Providing both in the same
`scenarios/` directory is a validation error. Use history-based estimation
when raw observations are available and you want Cobre to handle the
statistical fitting; use pre-computed stats when you have already fitted
the model externally or when you need precise control over the parameters.

---

## Correlation

Hydro plants that share a watershed tend to have correlated inflows: when
the upstream basin receives heavy rainfall, all plants along the river
benefit simultaneously. Ignoring this correlation can cause the optimizer to
underestimate the risk of a system-wide dry spell.

### Default behavior: independent noise

When no correlation configuration is provided, Cobre treats each hydro
plant's inflow as independent of all others. Each plant draws its own
noise realization at each stage without any coupling. This is the correct
setting for the `1dtoy` example, which has only one hydro plant.

### Configuring spatial correlation

For multi-plant systems, Cobre supports Cholesky-based spatial correlation.
A correlation model is specified in `correlation.json` in the case directory
and defines named correlation groups, each with a symmetric positive-definite
correlation matrix.

```json
{
  "method": "cholesky",
  "profiles": {
    "default": {
      "groups": [
        {
          "name": "basin_south",
          "entities": [
            { "type": "inflow", "id": 0 },
            { "type": "inflow", "id": 1 }
          ],
          "matrix": [
            [1.0, 0.7],
            [0.7, 1.0]
          ]
        }
      ]
    }
  }
}
```

Entities not listed in any group retain independent noise. Multiple profiles
can be defined and scheduled to activate for specific stages (for example,
using a wet-season correlation structure in January through March and a
dry-season structure for the remaining months). Detailed correlation
configuration documentation will be added with future multi-plant example
cases.

---

## Stochastic Load

Electrical load at each bus can be modeled as a stochastic process in
addition to, or independently of, inflow uncertainty. When
`load_seasonal_stats.parquet` is present in the `scenarios/` directory,
Cobre applies a noise model to bus demand during training and simulation.

### How load noise works

Load noise uses the same PAR(p) framework as inflows. For each bus and each
stage, Cobre draws a noise realisation from the PAR(p) model specified by
the bus's `ar_order`, `mean_mw`, and `std_mw` columns in
`load_seasonal_stats.parquet`. This realisation is then applied as a
multiplicative factor on the base demand for that bus and stage: the sampled
load replaces the deterministic demand value during scenario generation.

Because the PAR(p) model is the same for both inflows and loads, load
variability captures the same seasonal structure and stage-to-stage
persistence that inflow models do. A bus with `ar_order = 0` gets
independently drawn demand noise at each stage; a bus with `ar_order = 1`
gets demand noise that has month-to-month persistence consistent with the
historical standard deviation.

### Optional: deterministic loads without the file

`load_seasonal_stats.parquet` is entirely optional. When the file is absent,
Cobre treats all bus demands as deterministic: the demand at each bus and
stage is the fixed value from the case data, with no noise applied. This is
the correct setting for studies where load uncertainty is negligible or where
you want to isolate inflow uncertainty in isolation.

---

## Scenario Count and Seeds

### `num_scenarios` in `stages.json`

Each stage in `stages.json` has a `num_scenarios` field that controls how
many scenario branches are pre-generated for the opening scenario tree used
during the backward pass. A larger value gives the backward pass more
diverse inflow realizations to evaluate cuts against, at the cost of a
proportionally larger opening tree in memory. For the `1dtoy` example this
is set to 10. Production studies typically use 50 to 200.

### `forward_passes` in `config.json`

The `forward_passes` field in `config.json` controls how many scenario
trajectories are sampled during each training iteration's forward pass.
This is distinct from `num_scenarios`: the forward pass draws new
trajectories on each iteration using a deterministic per-iteration seed,
while `num_scenarios` controls the pre-generated backward-pass tree.

### The `seed` field

The `seed` field in the `training` section of `config.json` is the base
seed for all stochastic generation in the run:

```json
{
  "training": {
    "forward_passes": 50,
    "seed": 42,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }]
  }
}
```

The default value is `42` when `seed` is omitted. When a seed is provided,
every run with the same case directory and the same seed produces
bitwise-identical scenarios, training trajectories, and simulation results.
This reproducibility is guaranteed regardless of the number of MPI ranks,
because each rank derives its scenario seeds independently from the base
seed using a deterministic hash — no inter-rank coordination is required.

To get a non-reproducible run (different scenarios each time), set
`"seed": null` in `config.json`. Cobre will then derive the base seed from
OS entropy at startup.

---

## Inflow Non-Negativity

Normal distributions used in PAR(p) models have unbounded support: even with
a positive mean, there is a non-zero probability of drawing a negative noise
realisation that, after applying the AR dynamics, produces a negative inflow
value. Negative inflow has no physical meaning and, if uncorrected, would
violate water balance constraints in the LP.

Cobre provides two available methods for handling negative inflow realisations,
controlled by the `modeling.inflow_non_negativity.method` field in `config.json`.

### Penalty method (default)

The **penalty method** adds a high-cost slack variable to each water balance
row. When the solver encounters a scenario where the inflow would be negative,
it draws on this virtual inflow at the penalty cost rather than violating the
balance constraint. The penalty cost is configurable via the
`inflow_non_negativity` field in the case configuration; the default keeps it
high enough that the slack is used only when necessary.

In practice, the penalty is rarely activated in well-specified studies. It acts
as a backstop for low-probability tail realisations. This method is
**Recommended** for production use.

### Truncation method

Available since v0.1.1, the **truncation method** evaluates the full inflow
value before constructing the LP and clamps any negative result to zero. The
water balance row receives the clamped inflow directly; no slack variable is
added and no penalty cost is incurred. To enable truncation, set the method
field in `config.json`:

```json
{
  "modeling": {
    "inflow_non_negativity": {
      "method": "truncation"
    }
  }
}
```

Truncation eliminates the penalty cost for tail realisations at the expense
of introducing a small bias: scenarios where the true inflow would be
slightly negative are treated as zero-inflow scenarios, which is
conservative but physically interpretable. For most well-specified studies,
both methods produce similar results because negative realisations are rare.

### Truncation with penalty (deferred)

A combined **truncation with penalty** method — which applies bounded slack
variables on top of the clamped LP bounds — is not yet available in the
current release. It remains on the roadmap for a future version.

For the mathematical theory behind all three methods, see the
[Inflow Non-Negativity](https://cobre-rs.github.io/cobre-docs/theory/inflow-non-negativity.html)
page in the methodology reference, or Oliveira et al. (2022), _Energies_
15(3):1115.

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — introductory walkthrough of the `scenarios/` directory and Parquet schemas
- [Configuration](./configuration.md) — full documentation of `config.json` fields including `seed` and `forward_passes`
- [cobre-stochastic](../crates/stochastic.md) — internal architecture of the stochastic crate: PAR preprocessing, Cholesky correlation, opening tree, and seed derivation
