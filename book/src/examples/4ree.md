# The 4ree Example

The `4ree` case ships in `examples/4ree/` in the Cobre repository. It models the
four-region Brazilian interconnected power system — SUDESTE, SUL, NORDESTE, and
NORTE — with hydro and thermal generation over a 12-month planning horizon
(January–December 2015). The source data is the `4ree` example from the
[sddp-lab](https://github.com/rjmalves/sddp-lab) reference implementation.

This case is larger and more structurally complex than the `1dtoy` example. It
exercises the multi-bus power balance, bidirectional transmission line constraints,
and independent hydro cascades. It is intended for structural validation of the LP
formulation against a real-world system topology, not for producing physically
meaningful dispatch results (see [Known Limitations](#known-limitations)).

---

## System Description

| Element      | Count | Details                                                           |
| ------------ | ----- | ----------------------------------------------------------------- |
| Buses        | 4     | SUDESTE (0), SUL (1), NORDESTE (2), NORTE (3)                     |
| Hydro plants | 4     | One per region, independent cascades, constant productivity       |
| Thermals     | 126   | All original sddp-lab thermals, remapped to 4 buses               |
| Lines        | 2     | SUDESTE-SUL (7500/5470 MW) and SUDESTE-NORDESTE (1000/600 MW)     |
| Stages       | 12    | Monthly, January 2015 – December 2015, 1 block per stage          |
| Simulation   | 100   | Post-training evaluation over 100 independently sampled scenarios |

The system has four independent hydro cascades, each with a single reservoir
serving its own region. Transmission is limited to two bidirectional inter-region
lines, both anchored at SUDESTE. There is no direct connection from NORTE to any
other region (see [NORTE Isolation](#bus-5-nofict1-exclusion)).

Initial reservoir storage values come directly from the sddp-lab source data:

| Hydro plant | Region   | Initial storage (hm³) |
| ----------- | -------- | --------------------- |
| 0           | SUDESTE  | 38343.9               |
| 1           | SUL      | 10068.8               |
| 2           | NORDESTE | 9030.2                |
| 3           | NORTE    | 5161.9                |

---

## Network Topology

The four Brazilian regions connect through SUDESTE as the central hub. Only the
two real-to-real transmission lines survive from the original sddp-lab model after
excluding the NOFICT1 fictitious node (see [Conversion Decisions](#conversion-decisions)):

```
 SUL ──────────── SUDESTE ──────────── NORDESTE
                     │
                  (isolated)
                   NORTE
```

The SUDESTE-SUL line has asymmetric capacity: 7500 MW in the SUDESTE→SUL direction
and 5470 MW in the reverse direction. The SUDESTE-NORDESTE line carries 1000 MW
direct and 600 MW reverse. NORTE has no transmission connection to the rest of
the system in this model.

---

## Input Files

### `config.json`

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": {
    "forward_passes": 1,
    "stopping_rules": [
      {
        "type": "iteration_limit",
        "limit": 256
      }
    ]
  },
  "simulation": {
    "enabled": true,
    "num_scenarios": 100
  },
  "modeling": {
    "inflow_non_negativity": {
      "method": "none"
    }
  }
}
```

`forward_passes: 1` draws one scenario trajectory per training iteration, standard
for single-cut SDDP. The iteration limit is 256 — higher than the 1dtoy case to
allow more cuts to accumulate across the 12-stage horizon. No convergence-based
stopping rule is configured because the case runs with deterministic zero inflows,
so the objective converges to a deterministic LP optimal rather than a stochastic
bound.

`modeling.inflow_non_negativity.method: "none"` allows the PAR(p) noise model to
produce negative samples without truncation. This setting is inherited from the
1dtoy configuration but has no practical effect here because no `scenarios/`
directory is present and inflow is effectively zero throughout.

---

### `stages.json` (excerpt — Stages 0 and 1)

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/stages.schema.json",
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.0
  },
  "stages": [
    {
      "id": 0,
      "start_date": "2015-01-01",
      "end_date": "2015-02-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 10
    },
    {
      "id": 1,
      "start_date": "2015-02-01",
      "end_date": "2015-03-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 672 }],
      "num_scenarios": 10
    }
  ]
}
```

The remaining ten stages follow the same pattern covering March 2015 through
December 2015. Each stage has one load block (`SINGLE`) whose `hours` value
matches the calendar month length.

`annual_discount_rate: 0.0` matches the sddp-lab source data, which used zero
discount on all policy graph edges. The 1dtoy case uses 12% annual discount;
this case uses 0%, so costs are summed directly across stages without discounting.

---

## Usage

Validate the case (checks all five validation layers):

```sh
cobre validate examples/4ree
```

Run training and simulation:

```sh
cobre run examples/4ree
```

To write output to an explicit directory:

```sh
cobre run examples/4ree --output examples/4ree/output
```

The run produces the same output directory structure as the 1dtoy case:
`output/training/`, `output/simulation/`, and `output/policy/`. See
[Output Structure](./1dtoy.md#output-structure) in the 1dtoy page for the
full file listing.

With 12 stages and 126 thermals the LP is substantially larger than 1dtoy. On a
modern laptop the run typically completes within a few minutes for 256 iterations.

---

## Conversion Decisions

The 4ree case was converted from the sddp-lab reference implementation. Several
structural decisions were made during the conversion; understanding them is
necessary for correctly interpreting the results.

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

sddp-lab includes a fictitious aggregation node NOFICT1 (id=5) with zero load that
acts as an intermediate hub connecting northern generation to southern load centers.
Cobre does not model fictitious nodes.

All 126 thermals in sddp-lab connect to real buses 1–4; none were attached to
bus 5, so no thermal reassignment was needed.

Six of the ten sddp-lab lines involve NOFICT1 as a source or target:

| sddp-lab line    | Direction | Capacity |
| ---------------- | --------- | -------- |
| SUDESTE_NOFICT1  | 1 → 5     | 4000 MW  |
| NORDESTE_NOFICT1 | 3 → 5     | 3500 MW  |
| NORTE_NOFICT1    | 4 → 5     | 10000 MW |
| NOFICT1_SUDESTE  | 5 → 1     | 2940 MW  |
| NOFICT1_NORDESTE | 5 → 3     | 3300 MW  |
| NOFICT1_NORTE    | 5 → 4     | 4407 MW  |

These six lines are excluded. Converting them to direct inter-region connections
without knowledge of the actual physical routing would introduce spurious
transmission paths. The practical consequence is that NORTE generation cannot
reach SUDESTE or NORDESTE — the NORTE region becomes isolated in this model.

### Line merging

The original sddp-lab model used paired unidirectional lines to represent
asymmetric capacity. Cobre's `capacity.direct_mw` and `capacity.reverse_mw` fields
encode both directions in a single line entry. The two surviving real-to-real
lines are:

| Cobre line name  | direct_mw | reverse_mw |
| ---------------- | --------- | ---------- |
| SUDESTE_SUL      | 7500      | 5470       |
| SUDESTE_NORDESTE | 1000      | 600        |

The `direct` direction is defined as from the lower bus ID to the higher bus ID
(SUDESTE→SUL, SUDESTE→NORDESTE).

### Inflow model

sddp-lab uses per-season LogNormal marginal distributions with independent hydros
for its 4ree inflow scenarios. Cobre v0.1.x uses PAR(p) with additive normal noise.
Converting LogNormal(mu, sigma) parameters to PAR(0) normal parameters requires
moment-matching, but the resulting distributions have fundamentally different tail
shapes, making convergence bound comparisons unreliable.

Decision: run with **deterministic inflows** — no `scenarios/` directory is
provided. The PAR model produces zero inflow at each stage, so hydro generation
is driven entirely by initial storage drawdown. This is physically unrealistic but
sufficient to verify the LP structure and that validation passes. The full
stochastic conversion will be revisited when Cobre supports lognormal inflow
distributions.

### Risk measure

The sddp-lab 4ree case uses CVaR (alpha=0.5, lambda=0.5). Cobre v0.1.x implements
only the Expectation (risk-neutral) risk measure. The two objective functions are
not comparable, so numerical results from this case cannot be cross-validated
against sddp-lab output.

### Discount rate

sddp-lab's policy graph edges all carry `discount_rate: 0.0`. The `stages.json`
`annual_discount_rate: 0.0` field matches this, so costs are accumulated without
discounting across the 12-month horizon.

### Spillage penalty

The sddp-lab `hydros.csv` lists `spillage_penalty = 1` ($/hm³) for all hydros.
The global spillage penalty in `penalties.json` is set to 1.0 $/hm³ to match.

---

## Known Limitations

**Results are not comparable to sddp-lab.** Three structural differences make
objective values and dispatch patterns incomparable: deterministic versus lognormal
inflow model, Expectation versus CVaR risk measure, and the excluded NOFICT1
transit lines. Use this case for LP structural validation only.

**NORTE is isolated.** Without the NOFICT1 transit lines, NORTE generation cannot
reach SUDESTE or NORDESTE. NORTE hydro and thermal output can only serve NORTE's
own load. This understates inter-regional trade relative to the actual Brazilian
system.

**Zero inflow throughout.** Running without a `scenarios/` directory means the PAR
model produces zero inflow at every stage. The LP dispatches entirely from initial
reservoir storage, which drains rapidly across the 12-month horizon. This produces
an LP feasibility stress test rather than a physically realistic energy study.
