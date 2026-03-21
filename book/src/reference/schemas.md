# JSON Schemas

The following JSON Schema files describe the structure of each JSON input file
in a Cobre case directory. Download them and point your editor's JSON Schema
validation setting at the appropriate file to get autocompletion, hover
documentation, and inline error highlighting while authoring case inputs.

> For a complete description of each file's fields and validation rules, see the
> [Case Directory Format](./case-format.md) reference page.

## Available schemas

| Schema file                                                                             | Input file                                | Description                                                                                                            |
| --------------------------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [config.schema.json](../schemas/config.schema.json)                                     | `config.json`                             | Study configuration: training parameters, stopping rules, cut selection, simulation settings, and export flags         |
| [penalties.schema.json](../schemas/penalties.schema.json)                               | `penalties.json`                          | Global penalty cost defaults for bus deficit, line exchange, hydro violations, and non-controllable source curtailment |
| [stages.schema.json](../schemas/stages.schema.json)                                     | `stages.json`                             | Temporal structure of the study: stage sequence, load blocks, policy graph horizon, and scenario source configuration  |
| [buses.schema.json](../schemas/buses.schema.json)                                       | `system/buses.json`                       | Electrical bus registry: bus identifiers, names, and optional entity-level deficit cost tiers                          |
| [lines.schema.json](../schemas/lines.schema.json)                                       | `system/lines.json`                       | Transmission line registry: line identifiers, source/target buses, and directional MW capacity bounds                  |
| [hydros.schema.json](../schemas/hydros.schema.json)                                     | `system/hydros.json`                      | Hydro plant registry: reservoir bounds, outflow limits, generation model parameters, and cascade linkage               |
| [thermals.schema.json](../schemas/thermals.schema.json)                                 | `system/thermals.json`                    | Thermal plant registry: generation bounds and linear cost coefficients                                                 |
| [energy_contracts.schema.json](../schemas/energy_contracts.schema.json)                 | `system/energy_contracts.json`            | Bilateral energy contract registry (optional entities)                                                                 |
| [non_controllable_sources.schema.json](../schemas/non_controllable_sources.schema.json) | `system/non_controllable_sources.json`    | Intermittent (non-dispatchable) generation source registry (optional entities)                                         |
| [pumping_stations.schema.json](../schemas/pumping_stations.schema.json)                 | `system/pumping_stations.json`            | Pumping station registry (optional entities)                                                                           |
| [production_models.schema.json](../schemas/production_models.schema.json)               | `system/production_models.json`           | Production model selection, FPHA hyperplane config, and per-stage productivity overrides (optional)                    |
| [initial_conditions.schema.json](../schemas/initial_conditions.schema.json)             | `initial_conditions.json`                 | Initial reservoir storage, past inflows for PAR lag initialization                                                     |
| [correlation.schema.json](../schemas/correlation.schema.json)                           | `scenarios/correlation.json`              | Inter-site inflow correlation matrix for scenario generation                                                           |
| [generic_constraints.schema.json](../schemas/generic_constraints.schema.json)           | `constraints/generic_constraints.json`    | User-defined linear constraints over LP variables with optional slack penalties                                        |
| [exchange_factors.schema.json](../schemas/exchange_factors.schema.json)                 | `constraints/exchange_factors.json`       | Block-level line capacity multipliers for directional exchange limits                                                  |
| [load_factors.schema.json](../schemas/load_factors.schema.json)                         | `scenarios/load_factors.json`             | Block-level load scaling factors for bus-stage demand profiles                                                         |
| [non_controllable_factors.schema.json](../schemas/non_controllable_factors.schema.json) | `scenarios/non_controllable_factors.json` | Block-level NCS availability scaling factors per source per stage per block                                            |

## Using schemas in your editor

### VS Code

Add a `json.schemas` entry to your workspace `.vscode/settings.json`:

```json
{
  "json.schemas": [
    {
      "fileMatch": ["config.json"],
      "url": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json"
    },
    {
      "fileMatch": ["system/hydros.json"],
      "url": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/hydros.schema.json"
    }
  ]
}
```

Alternatively, add a `$schema` key directly inside each JSON file:

```json
{
  "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/book/src/schemas/config.schema.json",
  "training": {
    "forward_passes": 192,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 200 }]
  }
}
```

### Neovim (via `jsonls`)

Configure `json.schemas` in your `nvim-lspconfig` setup for `jsonls` following
the same URL pattern shown above.

### JetBrains IDEs

Go to **Preferences > Languages & Frameworks > Schemas and DTDs > JSON Schema
Mappings**, add a new mapping, paste the schema URL, and select the file pattern.

## Regenerating schemas

The schema files in `book/src/schemas/` are generated from the Rust type
definitions in `cobre-io`. To regenerate them after modifying the input types,
run:

```
cargo run -p cobre-cli -- schema export --output-dir book/src/schemas/
```
