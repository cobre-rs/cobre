# Deterministic Regression Suite

The `examples/deterministic/` directory ships 27 hand-built regression cases — `d01`
through `d30`, with indices `d12`, `d17`, and `d18` currently unoccupied — that anchor
the solver against analytically derived expected costs. Each case has minimal stochastic
structure (typically a single scenario per stage) so the optimal cost is computable by
hand and used as a fixed-point reference in the test suite.

These cases are not intended for production-style policy training. They are regression
anchors: any change to the solver, LP builder, or stochastic pipeline that perturbs a
deterministic case cost is flagged as a behavioural change. The test suite runs all 27
cases under `cargo nextest run --workspace` and compares each result against its stored
expected cost.

The suite covers a progression from the simplest possible thermal-only system up through
multi-hydro cascades, transmission constraints, FPHA production models, evaporation,
PAR(p) inflow fitting, block factors, non-controllable sources, generic constraints,
operational violation penalties, stage-decomposition patterns, and discount-rate
accounting. New modelling features are expected to add one or more cases at the end of
the sequence.

## Case Index

| Directory                         | Focus                                             | Notes                                                                                                                  |
| --------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `d01-thermal-dispatch`            | Thermal-only dispatch                             | No hydro plants; establishes the cheapest baseline cost.                                                               |
| `d02-single-hydro`                | Single hydro plant                                | Minimal hydro case with constant productivity.                                                                         |
| `d03-two-hydro-cascade`           | Two-plant hydro cascade                           | Verifies cascade water-balance: outflow from upstream plant becomes inflow to downstream.                              |
| `d04-transmission`                | Transmission constraints                          | Adds a transmission line with binding capacity to verify flow limits and marginal costs.                               |
| `d05-fpha-constant-head`          | FPHA with precomputed hyperplanes (constant head) | Hydro generation modelled via precomputed FPHA hyperplanes; head is fixed so hyperplanes degenerate to a single plane. |
| `d06-fpha-variable-head`          | FPHA with precomputed hyperplanes (variable head) | Head varies with reservoir level; verifies multi-plane FPHA selection and average-storage constraint.                  |
| `d07-fpha-computed`               | FPHA in computed mode                             | FPHA hyperplanes generated from hydro geometry at solve time rather than precomputed.                                  |
| `d08-evaporation`                 | Reservoir evaporation                             | Linearised surface-area evaporation loss; verifies water-balance accounting of evaporated volume.                      |
| `d09-multi-deficit`               | Multiple deficit buses                            | More than one bus with potential supply shortfall; verifies independent deficit variables per bus.                     |
| `d10-inflow-nonnegativity`        | Inflow non-negativity                             | Tests the inflow non-negativity enforcement methods when PAR(p) noise can produce negative samples.                    |
| `d11-water-withdrawal`            | Water withdrawal                                  | Verifies volumetric water withdrawal from a reservoir modelled as a non-generation outflow demand.                     |
| `d13-generic-constraint`          | Generic linear constraint                         | Regression case for user-defined generic linear constraints across system entities.                                    |
| `d14-block-factors`               | Block load and generation factors                 | Verifies per-block scaling factors applied to load and generation limits across intraday blocks.                       |
| `d15-non-controllable-source`     | Non-controllable source (NCS)                     | Regression case for stochastic non-controllable generation with availability factors.                                  |
| `d16-par1-lag-shift`              | PAR(1) lag-shift                                  | Verifies correct lag indexing when fitting PAR(1) models with a non-zero season offset.                                |
| `d19-multi-hydro-par`             | Multi-hydro PAR(p) inflow                         | Regression case for PAR(p) fitting applied to multiple hydro plants simultaneously.                                    |
| `d20-operational-violations`      | Operational violation penalties                   | Verifies penalty cost accounting when operational limits (e.g., min outflow) are relaxed with a penalty.               |
| `d21-min-outflow-regression`      | Minimum outflow constraint                        | Regression case confirming minimum turbine outflow constraints are respected in dispatch.                              |
| `d22-per-block-min-outflow`       | Per-block minimum outflow                         | Minimum outflow constraints applied individually to each intraday load block.                                          |
| `d23-bidirectional-withdrawal`    | Bidirectional water withdrawal                    | Water withdrawal that can both remove from and return flow to a reservoir within the balance equation.                 |
| `d24-productivity-override`       | Productivity model override                       | Per-plant override of the default hydro productivity model via `hydro_production_models.json`.                         |
| `d25-discount-rate`               | Non-zero discount rate                            | Verifies that a positive annual discount rate is applied correctly to inter-stage cost accumulation.                   |
| `d26-estimated-par2`              | Estimated PAR(2) model                            | Regression case for PAR(2) inflow fitting from historical scenario data.                                               |
| `d27-per-stage-thermal-cost`      | Per-stage thermal cost                            | Thermal units with costs that vary by stage; verifies stage-indexed cost lookup in the LP.                             |
| `d28-decomp-weekly-monthly`       | Weekly-to-monthly decomposition                   | Stage pattern with weekly substages grouped into monthly master stages.                                                |
| `d29-pattern-c-weekly-par`        | Weekly stages with PAR(p)                         | PAR(p) inflow model applied to a weekly-resolution stage sequence (pattern C).                                         |
| `d30-pattern-d-monthly-quarterly` | Monthly-to-quarterly decomposition                | Stage pattern grouping monthly stages into quarterly planning periods (pattern D).                                     |

## Running the Suite

The deterministic cases are included in the standard workspace test run:

```sh
cargo nextest run --workspace
```

Each case is driven by a test that loads the directory, runs training and simulation,
and compares the result against the expected cost stored in the test source. Cases with
longer runtimes are gated behind the `slow-tests` feature flag and are skipped in the
default run.
