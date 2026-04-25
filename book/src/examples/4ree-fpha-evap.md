# The 4ree-fpha-evap Example

The `4ree-fpha-evap` case ships in `examples/4ree-fpha-evap/` in the Cobre repository.
It extends the four-region [`4ree`](./4ree.md) topology by enabling the FPHA hydro
production model on two of the four reservoirs and adding surface-area evaporation
losses on three of them, demonstrating how non-constant head productivity and
volumetric evaporation interact in the LP formulation.

This case is intended for structural validation of the FPHA and evaporation pipelines
on a non-trivial, multi-bus topology — not for producing physically meaningful dispatch
results. The underlying power system, inflow model, and training configuration are the
same as the base `4ree` case; only the hydro modelling options differ.

---

## System Description

The four-region topology (buses, transmission lines, thermals, stages, scenario
sampling) is identical to [`4ree`](./4ree.md). The differences in this case are
confined to the hydro plant modelling options:

| Element            | Base 4ree             | This case                                          |
| ------------------ | --------------------- | -------------------------------------------------- |
| Hydro 0 (SUDESTE)  | Constant productivity | FPHA, precomputed hyperplanes; evaporation enabled |
| Hydro 1 (SUL)      | Constant productivity | FPHA, precomputed hyperplanes; evaporation enabled |
| Hydro 2 (NORDESTE) | Constant productivity | Constant productivity; evaporation enabled         |
| Hydro 3 (NORTE)    | Constant productivity | Constant productivity; no evaporation              |

**FPHA (precomputed mode)**: hydros 0 and 1 use FPHA hyperplanes stored in
`system/fpha_hyperplanes.parquet`. The LP generation constraint is replaced by a set
of supporting hyperplanes in the `(turbined_flow, storage, generation)` space, making
hydro productivity a function of average reservoir head rather than a fixed coefficient.

**Evaporation**: hydros 0, 1, and 2 carry per-season evaporation coefficients
(mm/season) and reference volumes (hm³) in `system/hydros.json`. The solver adds a
linearised surface-area evaporation term to each reservoir's water-balance equation,
reducing stored volume by a volume proportional to the evaporation coefficient and
an interpolated surface area.

**Production model assignment**: the `system/hydro_production_models.json` file
assigns FPHA to hydros 0 and 1 for all stages. Hydros 2 and 3 retain the default
constant-productivity model. Hydro 1 uses seasonal assignment; hydro 0 uses a
stage-range covering the entire horizon.

### `config.json`

The training configuration is identical to the base `4ree` case: 4 forward passes per
iteration, an iteration limit of 256, `in_sample` scenario sources with `seed: 42`,
and 100 simulation scenarios. No FPHA-specific configuration appears in `config.json`;
FPHA is activated entirely through the system files.

---

## Usage

Validate the case:

```sh
cobre validate examples/4ree-fpha-evap
```

Run training and simulation:

```sh
cobre run examples/4ree-fpha-evap
```

The output directory structure is the same as the base `4ree` case:
`output/training/`, `output/simulation/`, and `output/policy/`.

---

## Known Limitations

**Results are not comparable to the base 4ree case.** Switching hydros 0 and 1 from
constant productivity to FPHA changes the LP structure and the effective productivity
at each storage level, so objective values and dispatch patterns will differ from the
base case even under identical inflow realisations.

**NORTE remains isolated.** The NOFICT1 exclusion from the base `4ree` case applies
here unchanged; see [Known Limitations](./4ree.md#known-limitations) in the 4ree page.
