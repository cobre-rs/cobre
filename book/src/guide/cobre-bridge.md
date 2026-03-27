# cobre-bridge: Case Conversion

**cobre-bridge** is a standalone Python package that converts power system case
data from legacy formats to the Cobre input format. It currently supports
conversion from the data format used by Brazilian hydrothermal dispatch tools.

The package is maintained in a separate repository:
[github.com/cobre-rs/cobre-bridge](https://github.com/cobre-rs/cobre-bridge).

---

## Installation

```bash
pip install cobre-bridge
```

To enable post-conversion validation with the Cobre solver:

```bash
pip install cobre-bridge cobre-python
```

---

## Converting a Case

The `convert` subcommand reads a source case directory and writes a complete
Cobre case directory:

```bash
cobre-bridge convert newave /path/to/source/case /path/to/output/case
```

### Options

| Flag         | Description                                               |
| ------------ | --------------------------------------------------------- |
| `--validate` | Run `cobre validate` on the output after conversion.      |
| `--force`    | Overwrite the destination directory if it already exists. |
| `--verbose`  | Enable detailed logging output.                           |

### What Gets Converted

The conversion pipeline transforms approximately 20 input files into a
complete Cobre case directory. The mapping covers:

| Source Concept              | Cobre Entity             | Output File                             |
| --------------------------- | ------------------------ | --------------------------------------- |
| Hydro plant configuration   | `HydroPlant`             | `system/hydros.json`                    |
| Thermal plant configuration | `ThermalUnit`            | `system/thermals.json`                  |
| Subsystem definitions       | `Bus`                    | `system/buses.json`                     |
| Inter-area exchange limits  | `Line`                   | `system/lines.json`                     |
| Non-controllable sources    | `NonControllableSource`  | `system/non_controllable_sources.json`  |
| Historical inflow records   | PAR(p) inflow model      | `scenarios/inflow_history.parquet`      |
| Demand time series          | Load seasonal statistics | `scenarios/load_seasonal_stats.parquet` |
| Study horizon configuration | Stage definitions        | `stages.json`                           |
| Solver parameters           | Config                   | `config.json`                           |
| Reservoir bounds/overrides  | Per-stage hydro bounds   | `constraints/hydro_bounds.parquet`      |
| Thermal maintenance windows | Per-stage thermal bounds | `constraints/thermal_bounds.parquet`    |
| Transmission capacity       | Per-stage line bounds    | `constraints/line_bounds.parquet`       |
| VminOP / electric / AGRINT  | Generic LP constraints   | `constraints/generic_constraints.json`  |

### Output Directory Structure

```
output/
  config.json
  stages.json
  penalties.json
  initial_conditions.json
  system/
    hydros.json
    thermals.json
    buses.json
    lines.json
    non_controllable_sources.json
    hydro_production_models.json       (when applicable)
    hydro_geometry.parquet             (forebay/tailrace curves)
  scenarios/
    inflow_seasonal_stats.parquet
    inflow_history.parquet
    load_seasonal_stats.parquet
    load_factors.json
    non_controllable_stats.parquet
    non_controllable_factors.json
  constraints/
    generic_constraints.json
    generic_constraint_bounds.parquet
    hydro_bounds.parquet
    thermal_bounds.parquet
    line_bounds.parquet
    exchange_factors.json
```

Not all files are always produced. Optional files (e.g., `hydro_production_models.json`,
generic constraints) are written only when the source data contains the relevant
configuration.

---

## Comparing Results

After running both the source tool and Cobre on the same case, the `compare`
subcommand checks LP bounds for consistency:

```bash
cobre-bridge compare newave /path/to/source/sintese /path/to/cobre/output \
  --tolerance 1e-3
```

| Flag            | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `--tolerance`   | Absolute tolerance for bound comparison (default: `1e-3`).       |
| `--output PATH` | Write a detailed diff report as a Parquet file.                  |
| `--summary`     | Print only summary counts, not individual mismatches.            |
| `--variables`   | Filter to specific variables (e.g., `storage_min,turbined_max`). |

The comparison reads the source tool's synthesis output and Cobre's
`training/dictionaries/bounds.parquet`, aligns entities by name, and reports
any mismatches beyond the tolerance.

---

## Python API

For programmatic use, import the conversion pipeline directly:

```python
from pathlib import Path
from cobre_bridge.pipeline import convert_newave_case

report = convert_newave_case(
    src=Path("/path/to/source/case"),
    dst=Path("/path/to/output/case"),
)
print(report)  # ConversionReport with entity counts and warnings
```

---

## Conversion Details

### Entity ID Remapping

Source systems typically use 1-based integer IDs. cobre-bridge remaps all
entity IDs to 0-based integers in a deterministic order derived from the
source configuration files. This ensures consistent output regardless of
file ordering.

### Fictitious Plant Filtering

Plants marked as fictitious in the source data (used internally by some
tools for accounting purposes) are automatically excluded from the
conversion output.

### Risk Measure Support

When the source case configures risk-averse optimization (CVaR), cobre-bridge
converts the alpha and lambda parameters to per-stage `risk_measure` entries
in `stages.json`. Three modes are supported:

- **Disabled** -- all stages use `"expectation"`.
- **Constant** -- all stages use the same CVaR parameters.
- **Temporal** -- per-stage alpha/lambda values, with fallback to constants
  when a stage override is zero.

### Generic Constraints

Three types of user-defined constraints are converted and merged into a
single `generic_constraints.json` file with sequential IDs:

- **VminOP** -- minimum stored energy constraints (weighted sum of storage
  across a group of reservoirs).
- **Electric** -- operational constraints on hydro generation and line flows.
- **AGRINT** -- group dispatch constraints for thermal and hydro plants.

---

## Dependencies

| Package        | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `inewave`      | Reads legacy fixed-width and binary input files |
| `pyarrow`      | Writes Parquet output tables                    |
| `pandas`       | DataFrame manipulation during conversion        |
| `cobre-python` | Optional: post-conversion validation            |

---

## See Also

- [Anatomy of a Case](../tutorial/anatomy-of-a-case.md) -- what each output file controls
- [Configuration](./configuration.md) -- all `config.json` fields
- [Case Directory Format](../reference/case-format.md) -- complete input schema reference
