# cobre-io

<span class="status-alpha">alpha</span>

`cobre-io` is the case directory loader for the Cobre ecosystem. It provides the
[`load_case`](#load_case) function, which reads a case directory from disk and
produces a fully-validated [`cobre_core::System`] ready for use by downstream
solver and analysis crates.

The crate owns the entire input path: JSON and Parquet parsing, five layers of
validation, three-tier penalty and bound resolution, scenario model assembly, and
optional parameter estimation from historical data. No other crate reads input
files. Every crate downstream of `cobre-io` receives a structurally sound `System`
with all foreign keys resolved and all domain rules verified.

## Module overview

| Module               | Purpose                                                                           |
| -------------------- | --------------------------------------------------------------------------------- |
| `config`             | `Config` struct and `parse_config` — reads `config.json`                          |
| `system`             | Entity parsers for buses, lines, hydros, thermals, and stub types                 |
| `extensions`         | Hydro production model extensions (FPHA hyperplanes, geometry tables)             |
| `scenarios`          | Inflow and load statistical model loading, assembly, and history-based estimation |
| `constraints`        | Stage-varying bound and penalty override loading from Parquet                     |
| `penalties`          | Global penalty defaults parser (`penalties.json`)                                 |
| `stages`             | Stage sequence and policy graph loading (`stages.json`)                           |
| `initial_conditions` | Reservoir initial storage loading                                                 |
| `validation`         | Five-layer validation pipeline and `ValidationContext`                            |
| `resolution`         | Three-tier penalty and bound resolution into O(1) lookup tables                   |
| `pipeline`           | Orchestrator that wires all layers into a single `load_case` call                 |
| `report`             | Structured validation report generation                                           |
| `broadcast`          | System serialization and deserialization for MPI broadcast                        |
| `output`             | Output result types for simulation and training data                              |

## `load_case`

```rust
pub fn load_case(path: &Path) -> Result<System, LoadError>
```

Loads a power system case directory and returns a fully-validated `System`.

`path` must point to the case root directory. That directory must contain
`config.json`, `penalties.json`, `stages.json`, `initial_conditions.json`, the
`system/` subdirectory, the `scenarios/` subdirectory, and the `constraints/`
subdirectory. See [Case directory structure](#case-directory-structure) for the
full layout.

`load_case` executes the following sequence:

1. **Layer 1 — Structural validation.** Checks that all required files exist on
   disk and records which optional files are present. Missing required files
   produce [`LoadError::ConstraintError`] entries. Missing optional files are
   silently noted in the file manifest without error.
2. **Layer 2 — Schema validation.** Parses every present file, verifies required
   fields, types, and value ranges. Returns [`LoadError::IoError`] for read
   failures and [`LoadError::ParseError`] for malformed JSON or invalid Parquet.
   Schema violations produce [`LoadError::ConstraintError`] entries.
3. **Layer 3 — Referential integrity.** Verifies that every cross-entity ID
   reference resolves to a known entity. Dangling foreign keys produce
   [`LoadError::ConstraintError`] entries.
4. **Layer 4 — Dimensional consistency.** Checks that optional per-entity files
   provide coverage for every entity that needs them (for example, that inflow
   statistical parameters exist for every hydro plant, and that load seasonal
   statistics cover every bus for every stage). Coverage gaps produce
   [`LoadError::ConstraintError`] entries.
5. **Layer 5 — Semantic validation.** Enforces domain business rules: acyclic
   hydro cascade topology, penalty ordering (lower tiers may not exceed upper),
   PAR model stationarity, stage count consistency, estimation prerequisites, and
   other invariants. Violations produce [`LoadError::ConstraintError`] entries.
6. **Resolution.** After all five layers pass, three-tier penalty and bound
   resolution is performed. The result is pre-resolved lookup tables embedded in
   the `System` for O(1) solver access.
7. **Scenario assembly.** Inflow and load statistical models are assembled from
   the parsed seasonal statistics and autoregressive coefficients. When
   `inflow_history.parquet` is present and `inflow_seasonal_stats.parquet` is
   absent, the estimation pipeline derives seasonal statistics and AR coefficients
   from the historical data before assembly.
8. **System construction.** `SystemBuilder::build()` is called with the fully
   resolved data. Any remaining structural violations (duplicate IDs, broken
   cascade) surface as a final [`LoadError::ConstraintError`].

All validation diagnostics across Layers 1 through 5 are collected by
`ValidationContext` before failing. When `load_case` returns an error, the error
message contains every problem found, not just the first one.

### Minimal example

```rust,no_run
use cobre_io::load_case;
use std::path::Path;

let system = load_case(Path::new("path/to/my_case"))?;
println!("Loaded {} buses, {} hydros", system.n_buses(), system.n_hydros());
```

### Return type

On success, `load_case` returns a `cobre_core::System` — an immutable,
`Send + Sync` container holding all entity registries, topology graphs,
pre-resolved penalty and bound tables, scenario models, and the stage sequence.
All entity collections are in canonical ID-sorted order.

On failure, `load_case` returns a `LoadError`. See [Error handling](#error-handling)
for the full set of variants and when each occurs.

## Case directory structure

A valid case directory has the following layout:

```
my_case/
├── config.json                          # Solver configuration (required)
├── penalties.json                       # Global penalty defaults (required)
├── stages.json                          # Stage sequence and policy graph (required)
├── initial_conditions.json              # Reservoir storage at study start (required)
├── system/
│   ├── buses.json                       # Electrical buses (required)
│   ├── lines.json                       # Transmission lines (required)
│   ├── hydros.json                      # Hydro plants (required)
│   ├── thermals.json                    # Thermal plants (required)
│   ├── non_controllable_sources.json    # Intermittent sources (optional)
│   ├── pumping_stations.json            # Pumping stations (optional)
│   └── energy_contracts.json           # Bilateral contracts (optional)
├── extensions/
│   ├── hydro_geometry.parquet           # Reservoir geometry tables (optional)
│   ├── production_models.json           # FPHA production function configs (optional)
│   └── fpha_hyperplanes.parquet         # FPHA hyperplane coefficients (optional)
├── scenarios/
│   ├── inflow_seasonal_stats.parquet    # PAR model seasonal statistics (required)
│   ├── inflow_ar_coefficients.parquet   # PAR autoregressive coefficients (required)
│   ├── inflow_history.parquet           # Historical inflow series (optional)
│   ├── load_seasonal_stats.parquet      # Load model seasonal statistics (optional)
│   ├── load_factors.parquet             # Load scaling factors (optional)
│   ├── correlation.json                 # Cross-series correlation model (optional)
│   └── external_scenarios.parquet       # Pre-generated external scenarios (optional)
└── constraints/
    ├── hydro_bounds.parquet             # Stage-varying hydro bounds (optional)
    ├── thermal_bounds.parquet           # Stage-varying thermal bounds (optional)
    ├── line_bounds.parquet              # Stage-varying line bounds (optional)
    ├── pumping_bounds.parquet           # Stage-varying pumping bounds (optional)
    ├── contract_bounds.parquet          # Stage-varying contract bounds (optional)
    ├── generic_constraints.json         # User-defined LP constraints (optional)
    ├── generic_constraint_bounds.parquet # Bounds for generic constraints (optional)
    ├── exchange_factors.parquet         # Block exchange factors (optional)
    ├── penalty_overrides_hydro.parquet  # Stage-varying hydro penalty overrides (optional)
    ├── penalty_overrides_bus.parquet    # Stage-varying bus penalty overrides (optional)
    ├── penalty_overrides_line.parquet   # Stage-varying line penalty overrides (optional)
    └── penalty_overrides_ncs.parquet    # Stage-varying NCS penalty overrides (optional)
```

For the full JSON and Parquet schemas for each file, see the
[Case Format Reference](../reference/case-format.md).

## Validation pipeline

The five layers run in sequence. Earlier layers gate later ones: if Layer 1 finds
a missing required file, the file is not parsed in Layer 2. All diagnostics across
all layers are collected before returning.

```
Case directory
      │
      ▼
┌─────────────────────────────────────────────────┐
│  Layer 1 — Structural                           │
│  Does each required file exist on disk?         │
│  Records optional-file presence in FileManifest.│
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Layer 2 — Schema                               │
│  Parse JSON and Parquet. Check required fields, │
│  types, and value ranges. Collect schema errors.│
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Layer 3 — Referential integrity                │
│  All cross-entity ID references must resolve.   │
│  (e.g., hydro.bus_id must exist in buses list)  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Layer 4 — Dimensional consistency              │
│  Optional per-entity files must cover every     │
│  entity that needs them. Load cross-validation  │
│  checks bus coverage when load stats present.   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Layer 5 — Semantic                             │
│  Domain business rules: acyclic cascade,        │
│  penalty ordering, PAR stationarity, stage      │
│  count consistency, estimation prerequisites,   │
│  and other invariants.                          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼ (all layers pass)
              Resolution + Assembly
              System construction
                     │
                     ▼
              Ok(System)
```

### What each layer checks

**Layer 1 (Structural):** Verifies that the four root-level required files
(`config.json`, `penalties.json`, `stages.json`, `initial_conditions.json`) and
the four required entity files (`system/buses.json`, `system/lines.json`,
`system/hydros.json`, `system/thermals.json`) exist. Optional files are noted in
the `FileManifest` but their absence is not an error. The `FileManifest` is
passed to Layer 2 so that optional-file parsers are only called when the files
are present.

**Layer 2 (Schema):** Parses every file found by Layer 1. For JSON files,
deserialization uses serde with strict field requirements — missing required
fields and unknown field values surface immediately. For Parquet files, column
presence and data types are verified. Post-deserialization checks catch domain
range violations (for example, negative capacity values) that serde cannot
express. All parse and schema errors are collected by `ValidationContext`.

**Layer 3 (Referential integrity):** Checks all cross-entity foreign-key
references. Examples: every `hydro.bus_id` must name a bus in the bus registry;
every `line.source_bus_id` and `line.target_bus_id` must resolve; every
`pumping_station.source_hydro_id` and `destination_hydro_id` must resolve;
every bound override row's entity ID must match a known entity. All broken
references are collected before returning.

**Layer 4 (Dimensional consistency):** Verifies cross-file entity coverage. When
`scenarios/inflow_seasonal_stats.parquet` is present, every hydro plant must
have at least one row of statistics. When `scenarios/inflow_ar_coefficients.parquet`
is present, the AR order must be consistent with the number of coefficient rows.

**Load file cross-validation:** When `scenarios/load_seasonal_stats.parquet` is
present, every bus in the system must have a row for every study stage. A bus
that is present in `buses.json` but missing from `load_seasonal_stats.parquet`
for any stage produces a `DimensionMismatch` error. This ensures that the load
model covers the full spatial and temporal extent of the case before any
downstream model is built.

Other coverage checks ensure that optional per-entity Parquet files do not
silently omit entities.

**Layer 5 (Semantic):** Enforces domain invariants that span multiple files or
require reasoning about the system as a whole:

- **Acyclic cascade.** The hydro `downstream_id` graph must be a directed forest
  (no cycles). A topological sort detects cycles.
- **Penalty ordering.** Violation penalty tiers must be ordered: lower-tier
  penalties may not exceed upper-tier penalties for the same entity.
- **PAR model stationarity.** Seasonal inflow statistics must satisfy the
  stationarity requirements of the PAR(p) model.
- **Stage count consistency.** The number of stages must match across
  `stages.json`, scenario data, and any stage-varying Parquet files.
- **Estimation prerequisites.** When the estimation path is active (see
  [Estimation pipeline](#estimation-pipeline)), three additional rules are
  enforced:
  - `season_definitions` must be present in `stages.json` so that historical
    observations can be grouped by season for fitting.
  - Every hydro plant in `hydros.json` must have at least one observation in
    `inflow_history.parquet`; hydros with no history cannot be estimated
    (`BusinessRuleViolation`).
  - Each `(hydro, season)` group is checked for a minimum number of observations
    (configurable via `estimation.min_observations_per_season`); groups below the
    threshold produce a `ModelQuality` warning.

## Estimation pipeline

When `scenarios/inflow_history.parquet` is present in the case directory and
`scenarios/inflow_seasonal_stats.parquet` is **absent**, `load_case` activates
the estimation path. In this mode, the seasonal statistics and AR coefficients
required by the scenario model are derived automatically from the historical
inflow series rather than being read from pre-computed Parquet files.

The trigger condition is checked after Layers 1 through 5 complete:

```
inflow_history.parquet present
    AND inflow_seasonal_stats.parquet absent
        → estimation path active
```

When the estimation path is inactive (explicit stats files are provided),
`inflow_history.parquet` is loaded and stored on `ScenarioData.inflow_history`
but does not influence model assembly. This allows downstream consumers to access
the raw historical series without re-triggering estimation.

### Estimation configuration types

The `config.json` file accepts an optional `"estimation"` section that controls
the fitting procedure. All fields have defaults and the section may be omitted
entirely.

| Field                         | Type                 | Default | Description                                                             |
| ----------------------------- | -------------------- | ------- | ----------------------------------------------------------------------- |
| `max_order`                   | `u32`                | `6`     | Maximum autoregressive lag order considered during model selection      |
| `order_selection`             | `"aic"` or `"fixed"` | `"aic"` | Criterion for selecting the AR order: AIC minimization or fixed maximum |
| `min_observations_per_season` | `u32`                | `30`    | Minimum observations required per `(entity, season)` group              |

The `estimation` configuration is accessible at `config.estimation` after
`parse_config`. The `min_observations_per_season` threshold is used both during
Layer 5 validation (to emit a `ModelQuality` warning for sparse groups) and
during the fitting procedure itself (to skip groups below the threshold).

### Season map requirement

The estimation path groups historical observations by season in order to fit
season-specific AR models. This requires the `season_definitions` field to be
present in `stages.json`. If `season_definitions` is absent when estimation is
active, Layer 5 emits a `BusinessRuleViolation` before fitting begins.

## Penalty and bound resolution

After all five validation layers pass, `load_case` resolves the three-tier
penalty and bound cascades into flat lookup tables embedded in the `System`.

### Three-tier cascade

Penalty and bound values follow a three-tier precedence cascade:

```
Tier 1 — Global defaults (penalties.json)
    ↓ overridden by
Tier 2 — Entity-level overrides (system/*.json fields)
    ↓ overridden by
Tier 3 — Stage-varying overrides (constraints/penalty_overrides_*.parquet)
```

Tier-1 and tier-2 resolution happen during entity parsing (Layer 2). By the time
the resolution step runs, each entity struct already holds its tier-2 resolved
value in the relevant penalty or bound field.

The resolution step applies tier-3 stage-varying overrides from the optional
Parquet files. For each `(entity, stage)` pair, the resolved value is:

- The tier-3 override from the Parquet row, if a row exists for that pair.
- Otherwise, the tier-2 value already stored in the entity struct.

### Sparse expansion

Tier-3 overrides are stored sparsely: a Parquet row only needs to exist for
stages where the override differs from the entity-level value. The resolution
step expands this sparse representation into a dense
`[n_entities × n_stages]` array for O(1) solver lookup at construction time.

### Result

Resolution produces two pre-resolved tables stored on `System`:

- `ResolvedPenalties` — per-(entity, stage) penalty values for buses, hydros,
  lines, and non-controllable sources.
- `ResolvedBounds` — per-(entity, stage) upper and lower bound values for
  hydros, thermals, lines, pumping stations, and energy contracts.

Both tables use dense flat arrays with positional entity indexing (entity
position in the canonical ID-sorted slice becomes its array index).

## `Config` struct

`Config` is the in-memory representation of `config.json`. Use `parse_config` to
load it independently of `load_case`:

```rust,no_run
use cobre_io::config::parse_config;
use std::path::Path;

let cfg = parse_config(Path::new("my_case/config.json"))?;
println!("forward_passes = {:?}", cfg.training.forward_passes);
```

`Config` has seven sections:

| Section                  | Type                         | Default    | Purpose                                                |
| ------------------------ | ---------------------------- | ---------- | ------------------------------------------------------ |
| `modeling`               | `ModelingConfig`             | `{}`       | Inflow non-negativity treatment method and cost        |
| `training`               | `TrainingConfig`             | (required) | Iteration count, stopping rules, cut selection         |
| `upper_bound_evaluation` | `UpperBoundEvaluationConfig` | `{}`       | Inner approximation upper-bound evaluation settings    |
| `policy`                 | `PolicyConfig`               | fresh mode | Policy directory path, warm-start / resume mode        |
| `simulation`             | `SimulationConfig`           | disabled   | Post-training simulation scenario count and output     |
| `exports`                | `ExportsConfig`              | all on     | Flags controlling which output files are written       |
| `estimation`             | `EstimationConfig`           | `{}`       | AR model fitting settings for history-based estimation |

### Mandatory fields

Two fields in `training` have no defaults and must be present in `config.json`.
`parse_config` returns `LoadError::SchemaError` if either is absent:

- `training.forward_passes` — number of scenario trajectories per iteration (integer, `>= 1`)
- `training.stopping_rules` — list of stopping rule entries (must include at least one
  `iteration_limit` rule)

### Stopping rules

The `training.stopping_rules` array accepts four rule types, identified by the
`"type"` field:

| Type              | Required fields                                                       | Stops when                                    |
| ----------------- | --------------------------------------------------------------------- | --------------------------------------------- |
| `iteration_limit` | `limit: u32`                                                          | Iteration count reaches `limit`               |
| `time_limit`      | `seconds: f64`                                                        | Wall-clock time exceeds `seconds`             |
| `bound_stalling`  | `iterations: u32`, `tolerance: f64`                                   | Lower bound improvement falls below tolerance |
| `simulation`      | `replications`, `period`, `bound_window`, `distance_tol`, `bound_tol` | Policy and bound have both stabilized         |

Multiple rules combine according to `training.stopping_mode`: `"any"` (default,
OR semantics — stop when any rule triggers) or `"all"` (AND semantics — stop only
when all rules trigger simultaneously).

### Policy modes

The `policy.mode` field controls warm-start behavior:

| Mode           | Behavior                                                                     |
| -------------- | ---------------------------------------------------------------------------- |
| `"fresh"`      | (default) Start from scratch; no policy files are read                       |
| `"warm_start"` | Load existing cuts and states from `policy.path` as a starting approximation |
| `"resume"`     | Resume an interrupted run from the last checkpoint                           |

When `mode` is `"warm_start"` or `"resume"`, `load_case` also validates policy
compatibility: the stored policy's entity counts, stage count, and cut dimensions
must match the current case. Mismatches return `LoadError::PolicyIncompatible`.

## Error handling

All errors returned by `load_case` and its internal parsers are variants of
`LoadError`:

### `IoError`

```
I/O error reading {path}: {source}
```

Occurs when a required file exists in the file manifest but cannot be read from
disk (file not found, permission denied, or other OS-level I/O failure). Fields:
`path: PathBuf` (the file that failed) and `source: std::io::Error` (the
underlying error).

**When it occurs:** Layer 1 or Layer 2, when `std::fs::read_to_string` or a
Parquet reader returns an error for a required file.

### `ParseError`

```
parse error in {path}: {message}
```

Occurs when a file is readable but its content is malformed — invalid JSON
syntax, unexpected end of input, or an unreadable Parquet column header. Fields:
`path: PathBuf` and `message: String` (description of the parse failure).

**When it occurs:** Layer 2, during initial deserialization of JSON or Parquet
files before any field-level validation runs.

### `SchemaError`

```
schema error in {path}, field {field}: {message}
```

Occurs when a file parses successfully but a field violates a schema constraint:
a required field is missing, a value is outside its valid range, or an enum
discriminator names an unknown variant. Fields: `path: PathBuf`,
`field: String` (dot-separated path to the offending field, e.g.,
`"hydros[3].bus_id"`), and `message: String`.

**When it occurs:** Layer 2, during post-deserialization validation. Also
returned by `parse_config` when `training.forward_passes` or
`training.stopping_rules` is absent.

### `CrossReferenceError`

```
cross-reference error: {source_entity} in {source_file} references
non-existent {target_entity} in {target_collection}
```

Occurs when an entity ID field references an entity that does not exist in the
expected registry. Fields: `source_file: PathBuf`, `source_entity: String` (e.g.,
`"Hydro 'H1'"`), `target_collection: String` (e.g., `"bus registry"`), and
`target_entity: String` (e.g., `"BUS_99"`).

**When it occurs:** Layer 3 (referential integrity). All broken references across
all entity types are collected before returning.

### `ConstraintError`

```
constraint violation: {description}
```

A catch-all for collected validation errors from any of the five layers, and for
`SystemBuilder::build()` rejections. The `description` field contains all error
messages joined by newlines, each prefixed with its `[ErrorKind]`, source file,
optional entity identifier, and message text.

**When it occurs:** After any validation layer collects one or more error-severity
diagnostics, or when `SystemBuilder::build()` finds duplicate IDs or a cascade
cycle in the final construction step.

### `PolicyIncompatible`

```
policy incompatible: {check} mismatch — policy has {policy_value},
system has {system_value}
```

Occurs when a warm-start or resume policy file is structurally incompatible with
the current case. The four compatibility checks are: hydro count, stage count,
cut dimension, and entity identity hash. Fields: `check: String` (name of the
failing check), `policy_value: String`, and `system_value: String`.

**When it occurs:** After all five validation layers pass, when
`policy.mode` is `"warm_start"` or `"resume"` and the stored policy fails a
compatibility check.

## Design notes

**Collect-all validation.** Unlike parsers that short-circuit on the first error,
all five validation layers collect diagnostics into a shared `ValidationContext`
before failing. When `load_case` returns a `ConstraintError`, the `description`
field contains every problem found in a single report. This avoids the
frustrating fix-one-error-re-run-repeat cycle on large cases.

**File-format split.** Entity identity data (IDs, names, topology, static
parameters) lives in JSON. Time-varying and per-stage data (bounds, penalty
overrides, statistical parameters, scenarios) lives in Parquet. JSON is easy to
read and edit by hand; Parquet handles large numeric tables efficiently. The two
formats complement each other without overlap.

**Resolution separates concerns.** The three-tier cascade is resolved once at
load time into dense arrays, not at every solver call. Downstream solver crates
call `system.penalties().hydro(entity_idx, stage_idx)` and get an `f64` with no
branching, no hash lookups, and no tier logic. The complexity of the cascade is
entirely contained in `cobre-io`.

**Declaration-order invariance.** All entity collections are sorted by ID before
`SystemBuilder::build()` is called. Any `System` built from the same entities,
regardless of the order they appear in the input files, produces a structurally
identical result with identical pre-resolved tables.

**Estimation as a loading mode.** The estimation path is triggered by the
presence of `inflow_history.parquet` combined with the absence of
`inflow_seasonal_stats.parquet`. This design allows callers to switch between
the explicit-stats path (provide pre-computed files) and the estimation path
(provide raw history) without any code changes — only the files present in the
case directory determine which path runs.
