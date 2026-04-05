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
    inflow_ar_coefficients.parquet    (when PAR model order > 0)
    inflow_history.parquet            (alternative to pre-computed stats)
    non_controllable_stats.parquet    (stochastic NCS availability)
    external_inflow_scenarios.parquet (per-class external inflow)
    external_load_scenarios.parquet   (per-class external load)
    external_ncs_scenarios.parquet    (per-class external NCS)
    correlation.json
    noise_openings.parquet            (user-supplied opening tree, optional)
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

| Column     | Type   | Nullable | Description                                            |
| ---------- | ------ | -------- | ------------------------------------------------------ |
| `hydro_id` | INT32  | No       | Hydro plant identifier (matches `id` in `hydros.json`) |
| `stage_id` | INT32  | No       | Stage identifier (matches `id` in `stages.json`)       |
| `mean_m3s` | DOUBLE | No       | Seasonal mean inflow in m³/s (must be finite)          |
| `std_m3s`  | DOUBLE | No       | Seasonal standard deviation in m³/s (must be >= 0)     |

The file must contain exactly one row per `(hydro_id, stage_id)` pair.
Every hydro plant defined in `hydros.json` must have a row for every stage
defined in `stages.json`. The validator will reject the case if any
combination is missing. The AR model order (number of lags) is determined
from the `inflow_ar_coefficients.parquet` file when present, not from this file.

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

| Column     | Type   | Nullable | Description                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------------- |
| `bus_id`   | INT32  | No       | Bus identifier (matches `id` in `buses.json`)                       |
| `stage_id` | INT32  | No       | Stage identifier (matches `id` in `stages.json`)                    |
| `mean_mw`  | DOUBLE | No       | Seasonal mean load in MW (must be finite)                           |
| `std_mw`   | DOUBLE | No       | Seasonal standard deviation in MW (must be >= 0, 0 = deterministic) |

One row per `(bus_id, stage_id)` pair is required. Every bus in `buses.json`
must have a row for every stage. The load mean and standard deviation determine
both the expected demand level and how much it varies across scenarios in each
stage. A `std_mw` of `0.0` indicates deterministic load for that bus-stage pair.

---

## The PAR(p) Model

PAR(p) stands for Periodic Autoregressive model of order p. It is the
standard model for hydro inflow time series in long-term hydrothermal planning
because inflows have two key properties the model captures well: seasonal
patterns (wet seasons and dry seasons recur predictably each year) and
autocorrelation (a wet month tends to be followed by another wet month, and
vice versa).

### What the AR order controls

The AR order (number of autoregressive lags) is determined by the
`inflow_ar_coefficients.parquet` file. If the file is absent or contains
no coefficients for a given `(hydro_id, stage_id)`, the model defaults to
white noise (order 0). When estimated from history, the order is selected
automatically via PACF (see [Estimation from History](#estimation-from-history)).

**Order 0 — white noise.** The inflow at each stage is drawn
independently from a normal distribution with the specified mean and standard
deviation. There is no memory between stages: knowing last month's inflow
tells you nothing about this month's. This is the simplest setting and
appropriate when you lack historical data to fit AR coefficients, or when
the inflow series shows very little autocorrelation.

**Order > 0 — periodic autoregressive.** The inflow at each stage
depends on the inflows at the preceding p stages, weighted by coefficients
that reflect the seasonal autocorrelation structure. A wet period is
followed by another wet period with the probability implied by the
coefficients. Higher AR orders capture longer-range dependencies: order 1
captures month-to-month persistence, order 2 adds two-month memory, and so
on. Most hydro inflow series are well-described by order 1 or 2.

### AR coefficients file

When a non-trivial AR model is desired, Cobre requires an
`inflow_ar_coefficients.parquet` file in the `scenarios/` directory. This
file contains the fitted AR coefficients in standardized form (as produced
by the periodic Yule-Walker equations). The schema and the fitting procedure
are documented in the [Case Format Reference](../reference/case-format.md).

The `1dtoy` example has no AR coefficients file, so all inflows use white
noise (order 0).

### When to use higher AR orders

In general:

- Use order 0 when historical data is short or when you want to establish
  a baseline with the simplest possible model.
- Use order 1 for most real hydro systems. Monthly inflows have strong
  one-month autocorrelation, and a first-order model captures the bulk of
  it.
- Use order 2 or higher when the inflow series shows multi-month
  persistence (common in systems with large upstream catchments or snowmelt
  storage). Validate with autocorrelation plots of your historical data.
- AR coefficients require `std_m3s > 0` in the corresponding seasonal
  statistics — zero variance makes the model non-identifiable.

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
   the best fit per (hydro plant, stage) using the periodic partial
   autocorrelation function (PACF) with a 95% significance threshold.
   This avoids overfitting in series with little autocorrelation and
   captures meaningful persistence where it exists.

3. **AR coefficients** — Coefficients for the selected order are estimated
   by solving the periodic Yule-Walker matrix system, which correctly
   accounts for the non-Toeplitz covariance structure of periodic
   autoregressive processes.

4. **Spatial correlation** — The contemporaneous correlation between
   hydro plants is estimated from the historical residuals after AR fitting.
   The resulting correlation matrix is used by the Cholesky-based noise
   generator in exactly the same way as a manually specified
   `correlation.json`.

### History vs. pre-computed stats: choose one

`inflow_history.parquet` and `inflow_seasonal_stats.parquet` serve different
roles in the inflow model. When only `inflow_history.parquet` is present
(and `inflow_seasonal_stats.parquet` is absent), Cobre activates the
estimation path and derives seasonal statistics and AR coefficients from the
historical data. When `inflow_seasonal_stats.parquet` is present, it is used
directly regardless of whether `inflow_history.parquet` is also present.
Use history-based estimation when raw observations are available and you want
Cobre to handle the statistical fitting; use pre-computed stats when you have
already fitted the model externally or when you need precise control over the
parameters.

---

## Correlation

Hydro plants that share a watershed tend to have correlated inflows: when
the upstream basin receives heavy rainfall, all plants along the river
benefit simultaneously. Ignoring this correlation can cause the optimizer to
underestimate the risk of a system-wide dry spell. Correlation can also be
configured between load buses and between NCS entities.

### Default behavior: independent noise

When no correlation configuration is provided, Cobre treats each entity's
noise as independent of all others. Each entity draws its own noise
realization at each stage without any coupling. This is the correct setting
for the `1dtoy` example, which has only one hydro plant.

### Configuring spatial correlation

For multi-entity systems, Cobre supports Cholesky-based spatial correlation.
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

### Valid entity types

The `"type"` field in each entity reference must be one of:

- `"inflow"` — hydro inflow series (entity `id` matches `id` in `hydros.json`)
- `"load"` — stochastic load demand (entity `id` matches `id` in `buses.json`)
- `"ncs"` — non-controllable source availability (entity `id` matches `id` in
  `non_controllable_sources.json`)

### Same-type enforcement

All entities within a single correlation group must share the same entity
type. Mixing entity types — for example, placing an `"inflow"` entity and a
`"load"` entity in the same group — is not supported and produces a
`StochasticError::InvalidCorrelation` error at case load time. If you want to
correlate inflow with load, define separate groups with the same correlation
structure for each class.

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
stage, Cobre draws a noise realization scaled by the bus's `mean_mw` and
`std_mw` values from `load_seasonal_stats.parquet`. This realization is then
applied as a multiplicative factor on the base demand for that bus and stage:
the sampled load replaces the deterministic demand value during scenario
generation.

A bus with `std_mw = 0` gets deterministic demand at each stage; a bus with
`std_mw > 0` gets demand noise proportional to the standard deviation.

### Optional: deterministic loads without the file

`load_seasonal_stats.parquet` is entirely optional. When the file is absent,
Cobre treats all bus demands as deterministic: the demand at each bus and
stage is the fixed value from the case data, with no noise applied. This is
the correct setting for studies where load uncertainty is negligible or where
you want to isolate inflow uncertainty in isolation.

---

## Stochastic NCS Availability

Non-controllable sources (wind, solar, run-of-river) can have stochastic
available generation. When `scenarios/non_controllable_stats.parquet` is
present, Cobre samples a per-scenario availability factor for each NCS entity
and applies it to the entity's `max_generation_mw`.

### Schema

The file provides one row per `(ncs_id, stage_id)` pair:

| Column     | Type   | Nullable | Description                                                     |
| ---------- | ------ | -------- | --------------------------------------------------------------- |
| `ncs_id`   | INT32  | No       | NCS entity ID (matches `id` in `non_controllable_sources.json`) |
| `stage_id` | INT32  | No       | Stage identifier (matches `id` in `stages.json`)                |
| `mean`     | DOUBLE | No       | Mean availability factor (dimensionless, must be in [0, 1])     |
| `std`      | DOUBLE | No       | Standard deviation of availability factor (must be >= 0)        |

### How it works

For each forward and backward pass scenario, Cobre draws a standard normal
noise value `η` from the opening tree and computes:

```text
A_r = max_generation_mw × clamp(mean + std × η, 0, 1)
```

The result `A_r` is then multiplied by the per-block factor from
`scenarios/non_controllable_factors.json` (default 1.0) to produce the
final NCS column upper bound:

```text
col_upper = A_r × block_factor
```

With `std = 0`, the availability is deterministic at `mean × max_generation_mw`,
making the stochastic pipeline a strict generalization of the deterministic
`ncs_bounds.parquet` approach.

### Optional: deterministic NCS without the file

When `non_controllable_stats.parquet` is absent, NCS availability is
deterministic: the LP column upper bound comes from `constraints/ncs_bounds.parquet`
(or defaults to `max_generation_mw`). No per-scenario variation occurs.

---

## Seeds and Reproducibility

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

### Dual-Seed Architecture

Cobre uses two independent seeds, each controlling a different part of the
stochastic pipeline:

**`training.tree_seed` in `config.json`** — the base seed for the opening
scenario tree. This seed governs all backward-pass openings and, when the
sampling scheme is `in_sample` (the default), also governs the forward-pass
scenario selection. When the same case is run with the same `tree_seed`, the
opening tree is bitwise identical across runs, regardless of the number of MPI
ranks.

**`scenario_source.seed` in `stages.json`** — the forward seed used when the
sampling scheme is `out_of_sample`, `historical`, or `external`. This seed
controls the noise generated on-the-fly during each forward pass. It is
completely independent of `tree_seed`: changing it does not affect the
backward-pass tree, and changing `tree_seed` does not affect the forward pass.

Both seeds are optional. When omitted, Cobre derives the seed from OS entropy
at startup, producing a non-reproducible run. To make a run fully reproducible,
specify both seeds explicitly:

```json
// config.json
{
  "training": {
    "tree_seed": 42,
    "forward_passes": 50,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }]
  }
}
```

```json
// stages.json
{
  "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.12 },
  "scenario_source": {
    "seed": 99,
    "inflow": { "scheme": "out_of_sample" },
    "load": { "scheme": "in_sample" },
    "ncs": { "scheme": "in_sample" }
  },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 100,
      "sampling_method": "lhs"
    }
  ]
}
```

When `tree_seed` is set to `null` in `config.json`, Cobre derives the base
seed from OS entropy at startup, producing a different opening tree each run.
The same applies to `scenario_source.seed`: a `null` value means the
forward-pass noise is non-reproducible even if `tree_seed` is fixed.

---

## Noise Methods

The `sampling_method` field in each stage entry of `stages.json` controls
how noise vectors are generated within that stage when building the opening
scenario tree. This is orthogonal to the sampling scheme (see
[Sampling Schemes](#sampling-schemes) below), which controls where the
forward-pass noise comes from. The noise method controls the algorithm;
the sampling scheme controls the source.

The default method is `"saa"` when `sampling_method` is omitted.

### SAA — Sample Average Approximation

SAA (Sample Average Approximation) is pure Monte Carlo sampling. Each opening
draws an independent sequence of standard-normal values from a `Pcg64`
generator seeded deterministically from the stage and opening index. There
is no coordination between openings; each is drawn without knowledge of the
others.

SAA is the simplest and most general method. It works for any dimension count
and any branching factor, and it has no restrictions on `num_scenarios`. Use
SAA as your baseline when you are uncertain which method to choose, or when
your branching factor is small (fewer than 50 scenarios per stage).

Configure SAA by setting `"sampling_method": "saa"` (or by omitting the
field, since SAA is the default).

### LHS — Latin Hypercube Sampling

LHS (Latin Hypercube Sampling) is stratified sampling. For a stage with
`N = num_scenarios` openings, each dimension is divided into `N`
equal-probability strata `[k/N, (k+1)/N)` for `k = 0, …, N-1`. Exactly one
sample is placed within each stratum, and a Fisher-Yates shuffle independently
assigns strata to openings for every dimension. The result is marginal
uniformity: when you project all `N` noise vectors onto any single dimension,
the resulting samples cover the entire range of the standard-normal
distribution uniformly, with no stratum left empty.

LHS is the recommended choice for moderate branching factors (50 to 200
scenarios per stage). It reduces the variance of sample-average estimates
compared to SAA for the same `N`, which typically means a better-converged
backward-pass cut approximation for the same computational budget. LHS works
for any dimension count.

Configure LHS by setting `"sampling_method": "lhs"` in the stage entry.

### QMC-Sobol

QMC-Sobol uses Sobol quasi-random sequences, which are low-discrepancy
sequences that fill the unit hypercube more evenly than independent random
draws. Cobre implements the Joe-Kuo 2010 direction number dataset with
Matousek linear scrambling. The scrambling applies an affine transformation
`x' = a·x + b (mod 2^32)` with seed-derived parameters to each dimension,
breaking correlations between dimensions while preserving the low-discrepancy
property. The batch generator uses a Gray-code recurrence for O(1) updates
per point.

QMC-Sobol provides a faster convergence rate than both SAA and LHS for smooth
integrands, meaning that a smaller branching factor can achieve equivalent
policy quality. The convergence benefit is strongest when `num_scenarios` is a
power of 2 (32, 64, 128, 256, …), because Sobol sequences have optimal
2-equidistribution properties at powers of 2. You can use other values of
`num_scenarios`, but the theoretical convergence advantage is reduced.

QMC-Sobol supports up to 21,201 dimensions. If your system dimension (the
total number of hydro plants, load buses, and NCS entities) exceeds 21,201,
Cobre will return an error and refuse to run. In practice, this limit is
never reached in hydrothermal planning models.

Configure QMC-Sobol by setting `"sampling_method": "qmc_sobol"`.

### QMC-Halton

QMC-Halton uses Halton sequences, another family of low-discrepancy
sequences. Each dimension uses a distinct prime base: dimension 1 uses
base 2, dimension 2 uses base 3, dimension 3 uses base 5, and so on. The
prime bases are computed at initialization time using the sieve of
Eratosthenes (`sieve_primes`). Cobre applies Owen-style random digit
scrambling to each dimension: a random permutation table is applied to each
digit position in each dimension, breaking the correlation artifacts that
affect plain Halton sequences at high dimensions (sometimes called the
"Halton curse"). Permutation tables are derived deterministically from the
stage seed.

QMC-Halton has no dimension limit — it can handle arbitrarily many
dimensions by sieving as many primes as needed. This makes it a good
alternative to QMC-Sobol for very high-dimensional cases, though in practice
the dimension limit of QMC-Sobol (21,201) is rarely reached. The convergence
properties of QMC-Halton are similar to QMC-Sobol but the scrambling
approach differs; some integrands favor one over the other.

Configure QMC-Halton by setting `"sampling_method": "qmc_halton"`.

### Selective (Reserved)

The `"selective"` method is reserved for future use. It is intended to
support representative scenario selection (clustering-based methods), but
the required infrastructure is not yet implemented. If you configure a stage
with `"sampling_method": "selective"`, Cobre will return an error for the
opening tree generator. In the out-of-sample forward pass, it falls back to
SAA and emits a diagnostic warning.

### Comparison

| Method     | Convergence rate  | Dimension limit | Scenario count        | Best for                                  |
| ---------- | ----------------- | --------------- | --------------------- | ----------------------------------------- |
| SAA        | O(N^{-1/2})       | None            | Any                   | General use, small branching factors      |
| LHS        | Better than SAA   | None            | Any (50–200 typical)  | Moderate scenario counts, any dimension   |
| QMC-Sobol  | O(N^{-1} log^d N) | 21,201          | Powers of 2 preferred | Best convergence, low-to-medium dimension |
| QMC-Halton | O(N^{-1} log^d N) | None            | Any                   | High-dimension alternative to Sobol       |
| Selective  | N/A               | N/A             | N/A                   | Not implemented; reserved for future use  |

### Per-Stage Method Configuration

The `sampling_method` field is set per stage in `stages.json`. Different
stages in the same study can use different methods. This is useful when you
want a high-quality low-discrepancy method for the near-term stages (where
policy quality matters most) while using the simpler SAA for distant stages
where the investment decisions are less sensitive to sampling quality.

The following example configures a two-stage study where stage 0 uses LHS
and stage 1 uses QMC-Sobol:

```json
{
  "policy_graph": { "type": "finite_horizon", "annual_discount_rate": 0.12 },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 100,
      "sampling_method": "lhs"
    },
    {
      "id": 1,
      "start_date": "2024-02-01",
      "end_date": "2024-03-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 696 }],
      "num_scenarios": 128,
      "sampling_method": "qmc_sobol"
    }
  ]
}
```

Mixed configurations are fully supported. Cobre applies each stage's method
independently when building the opening tree.

---

## Sampling Schemes

The sampling scheme controls where the forward-pass noise comes from. This
is a different concept from the noise method: the noise method controls the
algorithm used to generate noise vectors for the opening tree, while the
sampling scheme controls whether the forward pass reuses the pre-generated
tree, generates fresh noise on-the-fly, replays historical observations, or
reads from an externally supplied file.

Each entity class — inflow, load, and NCS — independently specifies its
forward-pass noise source. The sampling scheme is configured in `stages.json`
under the top-level `scenario_source` object using a per-class format:

```json
// stages.json
{
  "scenario_source": {
    "seed": 42,
    "inflow": { "scheme": "in_sample" },
    "load": { "scheme": "in_sample" },
    "ncs": { "scheme": "in_sample" }
  }
}
```

All three class keys (`"inflow"`, `"load"`, `"ncs"`) default to
`"in_sample"` when absent. The `"seed"` field is shared across all classes
and is required when any class uses `"out_of_sample"`, `"historical"`, or
`"external"`.

### InSample (default)

With `"scheme": "in_sample"`, the forward pass reuses the pre-generated
opening tree. At each `(iteration, scenario, stage)` triple, the solver
selects one opening from the tree using a deterministic per-iteration hash
derived from `tree_seed`. The backward pass and the forward pass see the
same set of noise realizations: the same scenarios that were used to build
cuts are the scenarios against which the forward trajectories are evaluated.

InSample is the default when `scenario_source` is absent from `stages.json`.
It is simple to configure, requires no additional seed, and is appropriate for
most studies. The main limitation is that the forward pass cannot evaluate the
policy on noise realizations outside the opening tree, which can lead to an
optimistic bias when the branching factor is small.

### OutOfSample

With `"scheme": "out_of_sample"`, the forward pass generates fresh noise
on-the-fly at each `(iteration, scenario, stage)` triple. The fresh noise is
drawn from the same distribution as the opening tree but is independent of it
— the forward pass never looks at the tree. Each call derives a unique noise
vector from `scenario_source.seed`, the iteration index, the scenario index,
and the stage ID. The per-stage `sampling_method` controls which algorithm
(SAA, LHS, QMC-Sobol, or QMC-Halton) is used to generate the fresh noise.

OutOfSample requires `scenario_source.seed` to be set. Configure it as follows:

```json
// stages.json
{
  "scenario_source": {
    "seed": 99,
    "inflow": { "scheme": "out_of_sample" },
    "load": { "scheme": "in_sample" },
    "ncs": { "scheme": "in_sample" }
  }
}
```

OutOfSample is preferred when you want to evaluate policy quality on scenarios
that are independent of the scenarios used to build the policy. This avoids
the in-sample optimism that arises with small branching factors, where the
policy has effectively "seen" all the noise realizations during training.
OutOfSample is especially useful during simulation, where you want an unbiased
estimate of the policy's expected cost on new scenarios.

### Historical

With `"scheme": "historical"`, the forward pass replays standardized noise
derived from historical inflow observations stored in `inflow_history.parquet`.
This allows you to evaluate the policy against actual historical sequences —
what would the policy have done during the drought of 1953 or the wet year of
1974?

Historical sampling applies only to the inflow class. The load and NCS classes
configure their own schemes independently and are unaffected by the inflow
class using Historical.

#### Window discovery

A "window" is a starting year `y` for which every hydro plant in the study has
a complete sequence of historical observations covering the entire study period
(plus the PAR model lag order of pre-study seasons needed to seed the AR
state). Cobre discovers valid windows by scanning `inflow_history.parquet` and
checking completeness for every candidate starting year.

When `historical_years` is absent from `scenario_source`, Cobre auto-discovers
all valid windows from the history file. If the history file covers years 1940
through 2010 and the study spans 12 monthly stages, then every year for which
the history is complete (accounting for the required pre-window lag seasons)
becomes a valid window.

#### Configuring `historical_years`

To restrict the pool of candidate windows, set `historical_years` in
`scenario_source`. Two forms are supported:

**Explicit list** — specify the exact starting years to use:

```json
// stages.json
{
  "scenario_source": {
    "seed": 7,
    "inflow": { "scheme": "historical" },
    "load": { "scheme": "in_sample" },
    "ncs": { "scheme": "in_sample" },
    "historical_years": [1940, 1953]
  }
}
```

**Inclusive range** — specify a contiguous span of starting years:

```json
// stages.json
{
  "scenario_source": {
    "seed": 7,
    "inflow": { "scheme": "historical" },
    "load": { "scheme": "in_sample" },
    "ncs": { "scheme": "in_sample" },
    "historical_years": { "from": 1940, "to": 2010 }
  }
}
```

In both forms, Cobre validates each candidate year against the history file
and silently discards years for which the data is incomplete. If no valid
windows remain after filtering, Cobre returns a `StochasticError::InsufficientData`
error. When the number of valid windows is smaller than `forward_passes`, a
diagnostic warning is emitted and windows are repeated across forward passes.

#### Lag seeding (`apply_initial_state`)

For PAR models with order > 0, the first stage of each forward pass requires
historical inflow values from the stages immediately before the window's start
year — the "pre-study" lags. Historical sampling uses the raw historical
observations at those pre-window stages directly as the PAR state vector. This
means the AR dynamics of the first forward stage are initialized from the real
historical record rather than from a generated value, preserving the continuity
invariant between pre-window history and the replayed scenario.

#### How the HistoricalScenarioLibrary is used

At case load time, Cobre constructs a `HistoricalScenarioLibrary` by inverting
the PAR(p) model for every valid (window, stage) pair: it computes the
standardized noise value `η = (obs − deterministic_base − Σ ψ[ℓ]·lag[ℓ]) / σ`
using the raw historical inflow as lags. The resulting eta values are stored in
a flat buffer indexed by `(window, stage, hydro)`. During the forward pass, the
`ClassSampler::Historical` variant selects a window deterministically from the
seed and iteration/scenario indices, then retrieves the pre-computed eta slice
for each stage without any per-step recomputation.

### External

With `"scheme": "external"`, the forward pass reads pre-generated scenario
realizations from per-class Parquet files in the `scenarios/` directory. This
enables integration with external scenario generation tools — for example, a
climate model, a market forecast engine, or a bespoke sampling framework — and
injects their output directly into the Cobre forward pass.

Each entity class that uses External sampling requires its own file. The three
files and their schemas are:

#### `external_inflow_scenarios.parquet`

| Column        | Type   | Nullable | Description                                                  |
| ------------- | ------ | -------- | ------------------------------------------------------------ |
| `stage_id`    | INT32  | No       | Stage identifier (matches `id` in `stages.json`)             |
| `scenario_id` | INT32  | No       | Zero-based scenario index (0 to n_scenarios − 1)             |
| `hydro_id`    | INT32  | No       | Hydro plant ID (matches `id` in `hydros.json`)               |
| `value_m3s`   | DOUBLE | No       | Inflow realization in m³/s for this (stage, scenario, hydro) |

#### `external_load_scenarios.parquet`

| Column        | Type   | Nullable | Description                                            |
| ------------- | ------ | -------- | ------------------------------------------------------ |
| `stage_id`    | INT32  | No       | Stage identifier (matches `id` in `stages.json`)       |
| `scenario_id` | INT32  | No       | Zero-based scenario index (0 to n_scenarios − 1)       |
| `bus_id`      | INT32  | No       | Bus ID (matches `id` in `buses.json`)                  |
| `value_mw`    | DOUBLE | No       | Load realization in MW for this (stage, scenario, bus) |

#### `external_ncs_scenarios.parquet`

| Column        | Type   | Nullable | Description                                                     |
| ------------- | ------ | -------- | --------------------------------------------------------------- |
| `stage_id`    | INT32  | No       | Stage identifier (matches `id` in `stages.json`)                |
| `scenario_id` | INT32  | No       | Zero-based scenario index (0 to n_scenarios − 1)                |
| `ncs_id`      | INT32  | No       | NCS entity ID (matches `id` in `non_controllable_sources.json`) |
| `value`       | DOUBLE | No       | Availability realization for this (stage, scenario, NCS)        |

#### External standardization

Cobre does not use the raw values from external files directly. Before the
forward pass can use them, each value is converted to the same standardized
noise space (eta) that the PAR model and the opening tree use internally:

- **Inflow** — full PAR(p) inversion via `solve_par_noise`: the observed
  value is converted to `η = (obs − deterministic_base − Σ ψ[ℓ]·lag[ℓ]) / σ`
  using the fitted PAR model coefficients and seasonal statistics.
- **Load** — simple z-score normalization: `η = (value − mean) / std` using
  the `mean_mw` and `std_mw` from `load_seasonal_stats.parquet`.
- **NCS** — simple z-score normalization: `η = (value − mean) / std` using
  the `mean` and `std` from `non_controllable_stats.parquet`.

The resulting eta values are stored in an `ExternalScenarioLibrary` — one
per class — and the `ClassSampler::External` variant retrieves them by
`(stage, scenario)` index during the forward pass.

#### Configuring External sampling

```json
// stages.json
{
  "scenario_source": {
    "seed": 1,
    "inflow": { "scheme": "external" },
    "load": { "scheme": "external" },
    "ncs": { "scheme": "in_sample" }
  }
}
```

Each class is configured independently. In the example above, inflow and load
use external files while NCS uses the in-sample opening tree.

---

## User-Supplied Opening Trees

By default, Cobre generates the backward-pass opening tree internally using
SipHash-derived seeds and the spatial correlation Cholesky factor. If you need
to supply your own noise realizations — for cross-tool comparison, sensitivity
analysis, or round-trip replay — you can place `scenarios/noise_openings.parquet`
in the case directory before running.

When the file is present, Cobre loads the opening tree from it instead of
calling the internal generator. When the file is absent, the default generator
runs as usual.

### Schema

The file has exactly four columns:

| Column          | Type   | Required | Description                                                             |
| --------------- | ------ | -------- | ----------------------------------------------------------------------- |
| `stage_id`      | INT32  | Yes      | Zero-based stage index (0 to n_stages − 1)                              |
| `opening_index` | UINT32 | Yes      | Zero-based opening index within the stage (0 to openings_per_stage − 1) |
| `entity_index`  | UINT32 | Yes      | Zero-based entity index in system dimension order                       |
| `value`         | DOUBLE | Yes      | Noise realization for this (stage, opening, entity) triple              |

### Entity ordering

The `entity_index` column follows the system dimension convention:

1. Hydro entities, sorted by canonical ID (ascending)
2. Load buses, sorted by canonical ID (ascending)
3. NCS entities, sorted by canonical ID (ascending)

This matches the ordering used by Cobre's internal opening tree generator. The
file stores only indices, not entity identifiers, so an incorrect ordering
causes silent value misassignment rather than a schema error. Double-check the
entity ordering when constructing the file externally.

### Use cases

- **Cross-tool comparison.** Generate a set of noise realizations in an external
  tool and inject them into Cobre to compare policy quality on identical scenarios.
- **Sensitivity analysis.** Construct an extreme scenario (for example, all
  hydros at minimum inflow for the entire study) and evaluate how the policy
  responds.
- **Round-trip replay.** Export the opening tree that Cobre used in a training
  run with `exports.stochastic: true` in `config.json`, copy `output/stochastic/noise_openings.parquet`
  to `scenarios/`, and re-run to reproduce the exact same backward-pass context.
  See [Exporting Stochastic Artifacts](./running-studies.md#exporting-stochastic-artifacts)
  for the complete workflow.

### Interaction with `tree_seed`

The `training.tree_seed` field in `config.json` remains required even when a
user-supplied opening tree is present. The opening tree and forward-pass noise
are independent: `tree_seed` governs the forward-pass scenario sampling
performed by `sample_forward()`, which uses SipHash seeds derived independently
of the opening tree. Supplying a custom opening tree has no effect on forward-pass
noise.

### Limitations

- Partial-stage override is not supported. You must supply openings for all
  study stages. If you want to replace a subset of stages while keeping the rest
  internally generated, you must supply a complete tree and duplicate the
  internally generated values for the unmodified stages.
- User-supplied noise is used as-is. The Cholesky spatial correlation factor
  is not applied again. You are responsible for any spatial correlation structure
  encoded in the values you supply.

The file schema and validation rules are documented in the `noise_openings.rs` module.

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

### Truncation with penalty

A combined **truncation with penalty** method is available, configured by
setting `method` to `"truncation_with_penalty"` in `config.json`:

```json
{
  "modeling": {
    "inflow_non_negativity": {
      "method": "truncation_with_penalty",
      "penalty_cost": 1000.0
    }
  }
}
```

This method applies both truncation and a bounded slack variable: the inflow
is clamped to zero and a slack penalised at `penalty_cost` is added, providing
a smooth backstop for extreme tail realisations.

For the mathematical theory behind all three methods, see the
[Inflow Non-Negativity](https://cobre-rs.github.io/cobre-docs/theory/inflow-non-negativity.html)
page in the methodology reference, or Oliveira et al. (2022), _Energies_
15(3):1115.

---

## Related Pages

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) — introductory walkthrough of the `scenarios/` directory and Parquet schemas
- [Configuration](./configuration.md) — full documentation of `config.json` fields including `tree_seed` and `forward_passes`
- [cobre-stochastic](../crates/stochastic.md) — internal architecture of the stochastic crate: PAR preprocessing, Cholesky correlation, opening tree, and seed derivation
- [ADR: Noise Method Dispatch and Forward Sampler](../../docs/design/adr-noise-method-forward-sampler.md) — design decisions behind the noise method enum, per-stage dispatch, and OutOfSample forward sampler
