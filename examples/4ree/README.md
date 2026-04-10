# 4ree — Four-Region Brazilian Interconnected System

This example models the four-region Brazilian interconnected power system
(SUDESTE, SUL, NORDESTE, NORTE) with hydro and thermal generation over a
12-month planning horizon (January–December 2015).

The source data is the `4ree` example from the
[sddp-lab](https://github.com/rjmalves/sddp-lab) reference implementation,
located at `example/4ree/` in that repository.

## System Summary

| Entity   | Count | Notes                                         |
| -------- | ----- | --------------------------------------------- |
| Buses    | 4     | SUDESTE (0), SUL (1), NORDESTE (2), NORTE (3) |
| Hydros   | 4     | One per region, independent cascades          |
| Thermals | 126   | All original thermals, remapped to 4 buses    |
| Lines    | 2     | SUDESTE-SUL and SUDESTE-NORDESTE only         |
| Stages   | 12    | Monthly, Jan 2015 – Dec 2015                  |

## Usage

Validate the case (checks all 5 validation layers):

```sh
cobre validate examples/4ree
```

Run the optimization:

```sh
cobre run examples/4ree
```

## Conversion Decisions

### Bus ID remapping

sddp-lab uses 1-indexed bus IDs; Cobre uses 0-indexed IDs. The mapping is:

| sddp-lab ID | sddp-lab name | Cobre ID | Cobre name |
| ----------- | ------------- | -------- | ---------- |
| 1           | SUDESTE       | 0        | SUDESTE    |
| 2           | SUL           | 1        | SUL        |
| 3           | NORDESTE      | 2        | NORDESTE   |
| 4           | NORTE         | 3        | NORTE      |
| 5           | NOFICT1       | excluded | —          |

All `bus_id` references in hydros, thermals, and lines are remapped accordingly.
Thermal IDs are also remapped from 1-indexed (sddp-lab) to 0-indexed (Cobre).

### Bus 5 (NOFICT1) exclusion

sddp-lab includes a fictitious aggregation node NOFICT1 (id=5) with zero load
that acts as an intermediate hub connecting northern generation to southern load
centers. Cobre does not model fictitious nodes.

Decision: exclude NOFICT1 entirely.

- **Thermals**: all 126 thermals in sddp-lab are connected to real buses 1–4;
  none were connected to bus 5, so no thermal reassignment was needed.
- **Lines**: six of the ten sddp-lab lines involve NOFICT1 as source or target:
  - `SUDESTE_NOFICT1` (1→5, 4000 MW)
  - `NORDESTE_NOFICT1` (3→5, 3500 MW)
  - `NORTE_NOFICT1` (4→5, 10000 MW)
  - `NOFICT1_SUDESTE` (5→1, 2940 MW)
  - `NOFICT1_NORDESTE` (5→3, 3300 MW)
  - `NOFICT1_NORTE` (5→4, 4407 MW)

  These six lines are excluded because converting them to direct inter-region
  connections without knowledge of the actual physical routing would introduce
  spurious transmission paths. The four northern exports from NORTE that transit
  through NOFICT1 to SUDESTE are lost in this approximation, which means NORTE
  generation is effectively isolated in this model.

- **Remaining lines**: two real-to-real lines survive:
  - `SUDESTE_SUL` / `SUL_SUDESTE` merged into one bidirectional line
    (direct: 7500 MW, reverse: 5470 MW)
  - `SUDESTE_NORDESTE` / `NORDESTE_SUDESTE` merged into one bidirectional line
    (direct: 1000 MW, reverse: 600 MW)

  The sddp-lab model used paired unidirectional lines for asymmetric capacity.
  Cobre's `capacity.direct_mw` / `capacity.reverse_mw` fields encode both
  directions in a single line entry.

### Inflow model (NOT converted)

sddp-lab uses "Naive" inflow scenarios with per-season LogNormal marginal
distributions and identity Gaussian copulas (independent hydros). Cobre uses
PAR(p) with additive normal noise.

Converting LogNormal(mu, sigma) parameters to PAR(0) normal parameters requires
moment-matching (`mean = exp(mu + sigma^2/2)`), but the resulting distributions
have fundamentally different tail shapes, making convergence bound comparisons
unreliable.

Decision: provide seasonal statistics via the `scenarios/` directory and run
with stochastic inflows using PAR(p). The `scenarios/inflow_seasonal_stats.parquet`
file supplies per-season means and standard deviations derived from the sddp-lab
LogNormal parameters via moment-matching. The resulting inflow distributions differ
from the original LogNormal tails, so convergence bounds remain incomparable with
sddp-lab, but the model produces physically plausible hydro dispatch rather than
zero-inflow drawdown.

### Risk measure

sddp-lab's 4ree uses CVaR (alpha=0.5, lambda=0.5). This example uses the default
Expectation (risk-neutral) risk measure. CVaR is also available via `stages.json`.
The two objective functions are not directly comparable even with matching risk
measures due to the differences in inflow distributions noted above.

### Discount rate

sddp-lab's graph edges all have `discount_rate: 0.0`. Cobre's `stages.json` sets
`annual_discount_rate: 0.0` to match.

### Spillage penalty

The sddp-lab `hydros.csv` lists `spillage_penalty = 1` ($/hm³) for all hydros.
The global spillage penalty in `penalties.json` is set to 1.0 $/hm³.

### Initial storage

Initial reservoir storage values are taken directly from `hydros.csv`:

| Hydro (Cobre ID) | Initial storage (hm³) |
| ---------------- | --------------------- |
| 0 (SUDESTE)      | 38343.9               |
| 1 (SUL)          | 10068.8               |
| 2 (NORDESTE)     | 9030.2                |
| 3 (NORTE)        | 5161.9                |

## Known Limitations

- **Results are NOT comparable to sddp-lab**: different stochastic model
  (PAR(p) normal vs. lognormal), different risk measure (Expectation vs. CVaR),
  and excluded NOFICT1 transit lines all mean the objective values and dispatch
  patterns will differ.
- **NORTE is isolated**: without the NOFICT1 transit lines, NORTE generation
  cannot reach SUDESTE or NORDESTE. NORTE hydro and thermal generation can only
  serve NORTE's own load.
