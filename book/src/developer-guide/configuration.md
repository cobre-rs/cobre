# Configuration

This section contains the configuration reference that maps every tunable parameter exposed by Cobre to its effect on LP subproblem construction, solver behavior, and algorithm variant selection. Configuration is split across two files -- `config.json` for global solver parameters and `stages.json` for temporal structure and per-stage options -- and the reference documents all valid values, defaults, and LP-level consequences for each setting.

## Spec Index

| Spec                                                                  | Description                                                                                                                                  |
| --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [Configuration Reference](./configuration/configuration-reference.md) | Complete mapping of configuration options to LP effects, with JSON examples, formulation-to-config traceability, and variable correspondence |

## Cross-Section References

The configuration options documented here reference definitions from three other specification sections. Modeling options and training parameters map to formulations in [Mathematical Formulations](./math.md) (cut selection strategies, stopping rules, risk measures, discount rates). Data file paths and JSON schemas are defined in [Data Model](./data-model.md) (input directory structure, stages schema, penalty system). Algorithm variant selection and validation rules are specified in [Architecture](./architecture.md) (extension points, scenario generation, cut management implementation).
