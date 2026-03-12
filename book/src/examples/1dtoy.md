# The 1dtoy Example

The `1dtoy` case ships in `examples/1dtoy/` in the Cobre repository. It is the
smallest complete hydrothermal dispatch problem that exercises every stage of the
workflow: input loading, five-layer validation, stochastic training, and post-training
simulation. The case solves in under a second and produces inspectable output files.

This page is a self-contained annotated reference. For the pedagogical walkthrough
that explains each file field by field, see [Anatomy of a Case](../tutorial/anatomy-of-a-case.md).
For the complete schema reference, see [Case Format Reference](../reference/case-format.md).

---

## System Description

| Element      | Count | Details                                                                            |
| ------------ | ----- | ---------------------------------------------------------------------------------- |
| Buses        | 1     | `SIN` — single copper-plate node, no transmission constraints                      |
| Hydro plants | 1     | `UHE1` — 1000 hm³ reservoir, 50 MW capacity, constant productivity (1 MW per m³/s) |
| Thermals     | 2     | `UTE1` at 5 $/MWh (15 MW), `UTE2` at 10 $/MWh (15 MW)                              |
| Lines        | 0     | Single-bus model, no transmission lines                                            |
| Stages       | 4     | Monthly, January–April 2024, 10 scenarios per stage during training                |
| Simulation   | 100   | Post-training evaluation over 100 independently sampled scenarios                  |

The system has 80 MW of total dispatchable capacity (50 MW hydro + 15 MW UTE1 +
15 MW UTE2). The initial reservoir level is 83.222 hm³ — about 8.3% of maximum
capacity — creating a low-storage starting condition where the solver must weigh
immediate turbine dispatch against the risk of running short in later stages.

The merit order is: hydro (zero fuel cost) first, then UTE1 (5 $/MWh), then
UTE2 (10 $/MWh), then deficit (1000 $/MWh as last resort). The solver learns
this ordering implicitly through the Benders cuts it generates.

---

## Input Files

### `config.json`

```json
{
  "$schema": "https://cobre-rs.github.io/cobre/schemas/config.schema.json",
  "training": {
    "forward_passes": 1,
    "stopping_rules": [
      {
        "type": "iteration_limit",
        "limit": 128
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

`forward_passes: 1` draws one scenario trajectory per training iteration, which
is standard for single-cut SDDP. The only stopping rule is an `iteration_limit`
of 128, so the pre-generated output ran all 128 iterations. In a production study
you would add a convergence-based rule such as `"type": "relative_gap", "tol": 0.01`
to stop early when the optimality gap closes below 1%.

`modeling.inflow_non_negativity.method: "none"` allows the PAR(p) noise model to
produce negative inflow samples without truncation. This is appropriate when inflow
values are already log-transformed or when the scenario generation method handles
non-negativity separately.

For the full configuration schema, see [Configuration](../guide/configuration.md).

---

### `stages.json` (excerpt — Stage 0)

```json
{
  "$schema": "https://cobre-rs.github.io/cobre/schemas/stages.schema.json",
  "policy_graph": {
    "type": "finite_horizon",
    "annual_discount_rate": 0.12
  },
  "stages": [
    {
      "id": 0,
      "start_date": "2024-01-01",
      "end_date": "2024-02-01",
      "blocks": [{ "id": 0, "name": "SINGLE", "hours": 744 }],
      "num_scenarios": 10
    }
  ]
}
```

The remaining three stages follow the same pattern, covering February, March, and
April 2024 with `hours` values matching each calendar month (696 for February 2024,
744 for March, 720 for April).

`policy_graph.type: "finite_horizon"` produces a linear stage chain — Stage 0 feeds
Stage 1, Stage 1 feeds Stage 2, and Stage 3 has zero terminal value. The
`annual_discount_rate: 0.12` applies a 12% annual discount when aggregating costs
across stages, converting monthly LP costs to a comparable present-value basis.

Each stage has one load block named `SINGLE`. The `hours` field converts power
(MW) to energy (MWh) in the LP objective: 744 hours × MW = MWh of energy produced
or consumed. A multi-block stage (e.g., peak/off-peak) would list multiple entries
in the `blocks` array.

---

### `system/hydros.json`

```json
{
  "$schema": "https://cobre-rs.github.io/cobre/schemas/hydros.schema.json",
  "hydros": [
    {
      "id": 0,
      "name": "UHE1",
      "bus_id": 0,
      "downstream_id": null,
      "reservoir": {
        "min_storage_hm3": 0.0,
        "max_storage_hm3": 1000.0
      },
      "outflow": {
        "min_outflow_m3s": 0.0,
        "max_outflow_m3s": 50.0
      },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 1.0,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 50.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 50.0
      }
    }
  ]
}
```

`UHE1` is a standalone tailwater plant (`downstream_id: null`). The reservoir
can hold 0–1000 hm³. Total outflow (turbined plus spilled) is capped at 50 m³/s,
representing the physical river channel capacity below the dam.

The `constant_productivity` turbine model converts flow to power linearly:
power (MW) = flow (m³/s) × `productivity_mw_per_m3s`. With a factor of 1.0,
turbining 30 m³/s yields exactly 30 MW. More accurate production functions use
the FPHA model with a reservoir geometry table, but constant productivity is
sufficient for this tutorial system.

For the hydro field reference, see [Case Format Reference](../reference/case-format.md).

---

### `system/thermals.json` (abbreviated)

```json
{
  "thermals": [
    {
      "id": 0,
      "name": "UTE1",
      "bus_id": 0,
      "cost_segments": [{ "capacity_mw": 15.0, "cost_per_mwh": 5.0 }],
      "generation": { "min_mw": 0.0, "max_mw": 15.0 }
    },
    {
      "id": 1,
      "name": "UTE2",
      "bus_id": 0,
      "cost_segments": [{ "capacity_mw": 15.0, "cost_per_mwh": 10.0 }],
      "generation": { "min_mw": 0.0, "max_mw": 15.0 }
    }
  ]
}
```

Two single-segment thermals at different costs create a two-step merit order above
zero-cost hydro. In each LP solve the solver dispatches UTE1 before UTE2 because
it is cheaper, and it will only reach UTE2 when hydro and UTE1 combined cannot
meet demand.

---

### `initial_conditions.json`

```json
{
  "storage": [{ "hydro_id": 0, "value_hm3": 83.222 }],
  "filling_storage": []
}
```

The initial reservoir level is 83.222 hm³, about 8.3% of the 1000 hm³ maximum.
This low starting level is deliberate: it forces the solver to learn a policy
that conserves water in early stages when the reservoir is nearly empty while
still meeting demand. The `filling_storage` array is empty because there are no
filling reservoirs (non-generating upstream storage) in this case.

---

## Convergence Behavior

The pre-generated output in `examples/1dtoy/output/training/` ran 128 iterations
and stopped at the iteration limit (no convergence-based stopping rule is configured
in `config.json`).

```
Training summary (from output/training/_manifest.json):
  Iterations completed:    128
  Termination reason:      iteration_limit
  Convergence achieved:    false
  Cuts generated:          387
  Cuts active:             384
```

To test for convergence, add a `relative_gap` rule alongside the iteration limit:

```json
{
  "training": {
    "forward_passes": 1,
    "stopping_rules": [
      { "type": "iteration_limit", "limit": 200 },
      { "type": "relative_gap", "tol": 0.01 }
    ]
  }
}
```

With this configuration on the 1dtoy case, the solver typically converges within
roughly 100 iterations (gap closes below 1%). Exact iteration counts vary with the
random seed. Numerical values like gap percentages are stochastic — your run will
differ from any pre-recorded reference values.

The `convergence.parquet` file in the training output records lower bound, upper
bound, and gap at every iteration, so you can plot convergence progress after the
run.

---

## Output Structure

After `cobre run examples/1dtoy --output examples/1dtoy/output`, the output
directory contains three subdirectories:

```
output/
  training/
    _manifest.json          # Run metadata: status, iterations, convergence, cuts
    metadata.json           # Problem dimensions, solver version, timing
    convergence.parquet     # Per-iteration lower bound, upper bound, gap
    timing/                 # Per-stage, per-iteration solver timing
    dictionaries/           # Variable and entity dictionaries for output parsing
    _SUCCESS                # Zero-byte sentinel written on clean completion
  simulation/
    _manifest.json          # Simulation metadata: total/completed/failed scenarios
    buses/                  # Bus dispatch results (Hive-partitioned by scenario)
      scenario_id=0000/
        data.parquet
      ...
      scenario_id=0099/
        data.parquet
    hydros/                 # Hydro dispatch results (storage, turbined, spilled)
    thermals/               # Thermal dispatch results (generation by segment)
    costs/                  # Per-stage costs
    inflow_lags/            # Inflow lag state variables used in each scenario
    _SUCCESS
  policy/
    basis/                  # LP basis snapshots for warm-starting
    cuts/                   # FlatBuffers policy checkpoint (Benders cuts)
    metadata.json           # Policy version and dimensions
```

### Key files

| File                                           | What it contains                                                           |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| `training/_manifest.json`                      | Run status, iteration count, convergence result, cut pool statistics       |
| `training/metadata.json`                       | Problem dimensions (stages, hydros, thermals, buses), Cobre version        |
| `training/convergence.parquet`                 | Lower bound, upper bound, gap per iteration — use this to plot convergence |
| `simulation/buses/scenario_id=N/data.parquet`  | Bus-level demand, generation, deficit per stage for scenario N             |
| `simulation/hydros/scenario_id=N/data.parquet` | Storage level, turbined flow, spillage per stage for scenario N            |
| `simulation/costs/scenario_id=N/data.parquet`  | Total cost per stage for scenario N                                        |
| `policy/cuts/`                                 | Saved Benders cuts — load this with `--policy` to warm-start a future run  |

### Querying results

All Parquet files are readable with any columnar query tool:

```python
import polars as pl

# Convergence plot data
df = pl.read_parquet("examples/1dtoy/output/training/convergence.parquet")
print(df.head())

# Hydro dispatch for scenario 0
df = pl.read_parquet(
    "examples/1dtoy/output/simulation/hydros/scenario_id=0000/data.parquet"
)
print(df)
```

```sql
-- DuckDB: average reservoir storage across all 100 simulation scenarios
SELECT stage_id, AVG(storage_hm3) AS mean_storage
FROM read_parquet('examples/1dtoy/output/simulation/hydros/*/data.parquet')
GROUP BY stage_id
ORDER BY stage_id;
```

For the complete output schema reference, see [Output Format](../reference/output-format.md).

---

## Running the Example

The example directory already contains pre-generated output. To reproduce it from
scratch:

```bash
# Validate the input files
cobre validate examples/1dtoy

# Run training and simulation (overwrites the existing output)
cobre run examples/1dtoy --output examples/1dtoy/output
```

To scaffold a fresh copy of the 1dtoy case into a new directory:

```bash
cobre init --template 1dtoy my_study
cobre validate my_study
cobre run my_study --output my_study/output
```
