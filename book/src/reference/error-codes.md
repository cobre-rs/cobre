# Error Codes Reference

`cobre-io` reports two kinds of errors: `LoadError` variants (the top-level
`Result<System, LoadError>` returned by `load_case`) and `ErrorKind` values
(diagnostic categories collected by `ValidationContext` during the five-layer
validation pipeline).

For an explanation of how the validation pipeline works and when each error
phase runs, see [cobre-io](../crates/io.md).

---

## `LoadError` variants

`LoadError` is the top-level error type returned by `load_case` and by every
individual file parser. There are 6 variants, ordered by the pipeline phase
in which they typically occur.

### `IoError`

**When it occurs:** A required file exists in the file manifest but cannot be
read from disk — file not found, permission denied, or other OS-level I/O
failure. Occurs in Layer 1 (structural) or Layer 2 (schema) when
`std::fs::read_to_string` or a Parquet reader returns an error.

**Display format:**

```
I/O error reading {path}: {source}
```

**Fields:**

| Field    | Type             | Description                             |
| -------- | ---------------- | --------------------------------------- |
| `path`   | `PathBuf`        | Path to the file that could not be read |
| `source` | `std::io::Error` | Underlying OS I/O error                 |

**Example:**

```
I/O error reading system/hydros.json: No such file or directory (os error 2)
```

**Resolution:** Verify the file exists in the case directory. Check that the
process has read permissions for the directory and file. For `load_case`, the
case root must contain all 8 required files (see [Case Format](case-format.md)).

---

### `ParseError`

**When it occurs:** A file is readable but its content is malformed — invalid
JSON syntax, unexpected end of input, or an unreadable Parquet column header.
Occurs in Layer 2 (schema) during initial deserialization before any
field-level validation runs.

**Display format:**

```
parse error in {path}: {message}
```

**Fields:**

| Field     | Type      | Description                                     |
| --------- | --------- | ----------------------------------------------- |
| `path`    | `PathBuf` | Path to the file that failed to parse           |
| `message` | `String`  | Human-readable description of the parse failure |

**Example:**

```
parse error in stages.json: expected `:` at line 5 column 12
```

**Resolution:** Open the file in a JSON validator or Parquet viewer. The message
contains the location of the syntax error. For JSON files, a trailing comma,
missing closing brace, or unquoted key are common causes.

---

### `SchemaError`

**When it occurs:** A file parses successfully but a field violates a schema
constraint: a required field is missing, a value is outside its valid range, or
an enum discriminator names an unknown variant. Occurs in Layer 2
(schema) during post-deserialization validation. Also returned by `parse_config`
when `training.forward_passes` or `training.stopping_rules` is absent.

**Display format:**

```
schema error in {path}, field {field}: {message}
```

**Fields:**

| Field     | Type      | Description                                                            |
| --------- | --------- | ---------------------------------------------------------------------- |
| `path`    | `PathBuf` | Path to the file containing the invalid entry                          |
| `field`   | `String`  | Dot-separated path to the offending field (e.g., `"hydros[3].bus_id"`) |
| `message` | `String`  | Human-readable description of the violation                            |

**Example:**

```
schema error in config.json, field training.forward_passes: required field is missing
```

```
schema error in system/buses.json, field buses[1].id: duplicate id 5 in buses array
```

**Resolution:** The `field` value identifies the exact location of the problem.
Check that required fields are present and that values fall within documented
ranges. For `config.json`, `training.forward_passes` and
`training.stopping_rules` are mandatory and have no defaults.

---

### `CrossReferenceError`

**When it occurs:** An entity ID field references an entity that does not exist
in the expected registry. Occurs in Layer 3 (referential integrity). All broken
references across all entity types are collected before returning.

**Display format:**

```
cross-reference error: {source_entity} in {source_file} references
non-existent {target_entity} in {target_collection}
```

**Fields:**

| Field               | Type      | Description                                                                            |
| ------------------- | --------- | -------------------------------------------------------------------------------------- |
| `source_file`       | `PathBuf` | Path to the file that contains the dangling reference                                  |
| `source_entity`     | `String`  | String identifier of the entity that holds the broken reference (e.g., `"Hydro 'H1'"`) |
| `target_collection` | `String`  | Name of the registry that was expected to contain the target (e.g., `"bus registry"`)  |
| `target_entity`     | `String`  | String identifier of the entity that could not be found (e.g., `"BUS_99"`)             |

**Example:**

```
cross-reference error: Hydro 'FURNAS' in system/hydros.json references
non-existent BUS_99 in bus registry
```

**Resolution:** The `target_entity` does not exist in the `target_collection`.
Either add the missing entity to its registry file, or correct the ID reference
in `source_file`. Common causes: a bus was deleted from `system/buses.json`
but a hydro, thermal, or line still references its old ID.

---

### `ConstraintError`

**When it occurs:** A catch-all for all validation diagnostics collected by
`ValidationContext` across any of the five layers, and for `SystemBuilder::build()`
rejections. The `description` field contains every collected error message joined
by newlines, each prefixed with its `[ErrorKind]`, source file, optional entity
identifier, and message text.

**Display format:**

```
constraint violation: {description}
```

**Fields:**

| Field         | Type     | Description                           |
| ------------- | -------- | ------------------------------------- |
| `description` | `String` | All error messages joined by newlines |

**Example:**

```
constraint violation: [FileNotFound] system/hydros.json: required file 'system/hydros.json' not found in case directory
[SchemaViolation] system/buses.json (bus_42): missing field bus_id
```

**Resolution:** Read every line in `description` — each line is a separate
problem. Address them all and re-run. The `[ErrorKind]` prefix identifies the
category of each problem; see the `ErrorKind` catalog below for resolution
guidance per category.

---

### `PolicyIncompatible`

**When it occurs:** After all five validation layers pass, when `policy.mode` is
`"warm_start"` or `"resume"` and the stored policy file is structurally
incompatible with the current case. The four compatibility checks are: hydro
count, stage count, cut dimension, and entity identity hash.

**Display format:**

```
policy incompatible: {check} mismatch — policy has {policy_value}, system has {system_value}
```

**Fields:**

| Field          | Type     | Description                                                     |
| -------------- | -------- | --------------------------------------------------------------- |
| `check`        | `String` | Name of the failing compatibility check (e.g., `"hydro count"`) |
| `policy_value` | `String` | Value recorded in the policy file                               |
| `system_value` | `String` | Value present in the current system                             |

**Example:**

```
policy incompatible: hydro count mismatch — policy has 42, system has 43
```

**Resolution:** The stored policy was produced by a run with a different system
configuration. Options:

- Set `policy.mode` to `"fresh"` to start from scratch without loading the policy.
- Revert the system change that caused the mismatch.
- Delete the policy directory and start fresh.

---

## `ErrorKind` values

`ErrorKind` categorises the validation problem within the `ValidationContext`
diagnostic system. Every `ValidationEntry` carries one `ErrorKind`. When
`ValidationContext::into_result()` produces a `ConstraintError`, each line in
`description` is prefixed with the `ErrorKind` in debug format (e.g., `[FileNotFound]`).

There are 14 `ErrorKind` values. Two (`UnusedEntity` and `ModelQuality`) default
to `Severity::Warning` — they are reported but do not block execution. All others
default to `Severity::Error` and must be resolved before `load_case` succeeds.

### `FileNotFound`

**Default severity:** Error

**What triggers it:** A file that is required by the case structure is missing
from the case directory. Emitted by Layer 1 (structural validation) for each of
the 8 required files that is not found on disk.

**Example message:** `required file 'system/hydros.json' not found in case directory`

**Resolution:** Create the missing file in the correct subdirectory. The 8
required files are: `config.json`, `penalties.json`, `stages.json`,
`initial_conditions.json`, `system/buses.json`, `system/lines.json`,
`system/hydros.json`, and `system/thermals.json`.

---

### `ParseError`

**Default severity:** Error

**What triggers it:** A file exists and was read but could not be parsed —
invalid JSON syntax, an unreadable Parquet header, or an unknown enum variant
in a tagged JSON union. Emitted by Layer 2 (schema validation) when the
initial deserialization of a file fails.

**Example message:** `parse error in stages.json: expected `:` at line 5 column 12`

**Resolution:** Fix the syntax error in the indicated file. Use a JSON linter
or Parquet viewer to find the exact location. For JSON files, common causes are
trailing commas, missing quotation marks, or mismatched braces.

---

### `SchemaViolation`

**Default severity:** Error

**What triggers it:** A file parses successfully but a field fails a schema
constraint: a required field is missing, a value is outside its valid range
(e.g., negative capacity, non-positive penalty cost), or a field contains an
unexpected type. Emitted by Layer 2 (schema validation) during post-deserialization
validation.

**Example message:** `schema error in system/buses.json, field buses[2].deficit_segments[0].cost: penalty value must be > 0.0, got -100.0`

**Resolution:** Correct the value in the indicated field. Field paths use
dot-notation and zero-based array indices. Consult the
[Case Format](case-format.md) page for valid ranges and required fields.

---

### `InvalidReference`

**Default severity:** Error

**What triggers it:** A cross-entity foreign-key reference points to an entity
that does not exist in the expected registry. For example, a hydro plant's
`bus_id` references a bus that is not in `system/buses.json`. Emitted by Layer
3 (referential integrity).

**Example message:** `Hydro 'FURNAS' references non-existent bus BUS_99 in bus registry`

**Resolution:** Either add the referenced entity to its registry file, or
correct the ID in the referencing file. Check all ID references: `hydros.bus_id`,
`thermals.bus_id`, `lines.source_bus_id`, `lines.target_bus_id`,
`hydros.downstream_id`.

---

### `DuplicateId`

**Default severity:** Error

**What triggers it:** Two entities within the same registry share the same ID.
IDs must be unique within each entity type. Emitted by Layer 2 (schema validation)
when duplicate IDs are detected within a single file.

**Example message:** `duplicate id 5 in buses array`

**Resolution:** Assign a unique ID to each entity. IDs are integers; use any
non-negative value as long as each is unique within its registry file.

---

### `InvalidValue`

**Default severity:** Error

**What triggers it:** A field value falls outside its valid range or violates a
value constraint that is specific to the field's domain. Examples: a reservoir's
`min_storage_hm3` exceeds `max_storage_hm3`, or a stage has `num_scenarios: 0`.
Emitted by Layer 2 (schema validation).

**Example message:** `min_storage_hm3 (8000.0) must be <= max_storage_hm3 (5000.0)`

**Resolution:** Correct the field value to be within the valid range. Consult
the [Case Format](case-format.md) page for documented constraints. For storage
bounds, ensure min <= max. For scenario counts, ensure num_scenarios >= 1.

---

### `CycleDetected`

**Default severity:** Error

**What triggers it:** A directed graph contains a cycle. The primary case is the
hydro cascade: the `downstream_id` links among hydro plants must form a directed
forest (no cycles). A cycle would mean plant A drains into plant B which drains
back into plant A. Detected by topological sort in Layer 5 (semantic validation).

**Example message:** `hydro cascade contains a cycle involving plants: [H1, H2, H3]`

**Resolution:** Review the `downstream_id` chain for the listed plants and remove
the cycle. Every hydro cascade must be a directed tree rooted at plants with no
downstream (tailwater discharge).

---

### `DimensionMismatch`

**Default severity:** Error

**What triggers it:** A cross-file coverage check fails. For example, when
`scenarios/inflow_seasonal_stats.parquet` is present, every hydro plant must
have at least one row of statistics. A mismatch means an optional per-entity
file provides data for some entities but not all that require it. Emitted by
Layer 4 (dimensional consistency).

**Example message:** `hydro 'ITAIPU' has no inflow seasonal statistics`

**Resolution:** Add the missing rows to the Parquet file. Every hydro plant that
is active during the study must appear in `inflow_seasonal_stats.parquet` when
that file is present.

---

### `BusinessRuleViolation`

**Default severity:** Error

**What triggers it:** A domain-specific business rule is violated that cannot
be expressed as a simple range constraint. Examples: penalty tiers must be
monotonically ordered (lower-tier penalties may not exceed upper-tier penalties
for the same entity), PAR model stationarity requirements are violated, or
stage count is inconsistent across files. Emitted by Layer 5 (semantic validation).

**Example message:** `penalty tier ordering violated for hydro 'FURNAS': spillage_cost (500.0) exceeds storage_violation_below_cost (100.0)`

**Resolution:** Read the message carefully — it describes the specific rule that
was violated and which entities are involved. For penalty ordering, ensure that
costs increase from lower-priority to higher-priority tiers. For stationarity,
verify that the PAR model parameters satisfy the required statistical properties.

---

### `WarmStartIncompatible`

**Default severity:** Error

**What triggers it:** A warm-start policy is structurally incompatible with the
current system. The four compatibility checks are: hydro count, stage count, cut
dimension, and entity identity hash. The policy was produced by a run with a
different system configuration. This `ErrorKind` is the `ValidationContext`
counterpart to the `LoadError::PolicyIncompatible` variant.

**Example message:** `warm-start policy has 42 hydros but current system has 43`

**Resolution:** See `PolicyIncompatible` under `LoadError` above.

---

### `ResumeIncompatible`

**Default severity:** Error

**What triggers it:** A resume state (checkpoint) is incompatible with the current
run configuration. The checkpoint may have been produced by a run with a different
`config.json` or a different system, making it impossible to resume from that
state consistently.

**Example message:** `resume checkpoint iteration 150 is beyond current iteration_limit 100`

**Resolution:** Either adjust `config.json` to be consistent with the checkpoint
(e.g., increase the iteration limit), or set `policy.mode` to `"fresh"` to
discard the checkpoint and start a new run.

---

### `NotImplemented`

**Default severity:** Error

**What triggers it:** A feature referenced in the input files is recognized by
the schema but not yet implemented in the current version of Cobre. This is used
during development to surface unimplemented feature requests from valid input.

**Example message:** `hydro production model 'fpha' is not yet implemented`

**Resolution:** Avoid using the unimplemented feature until it is available.
Check the [project roadmap](../overview/roadmap.md) for the implementation timeline.
Alternatively, use the currently supported alternatives (e.g., `"constant_productivity"`
instead of `"fpha"` for hydro generation models).

---

### `UnusedEntity`

**Default severity:** Warning (does not block execution)

**What triggers it:** An entity is defined in a registry file but appears to be
inactive — for example, a thermal plant with `max_generation_mw: 0.0` for all
stages. The entity is valid but contributes nothing to the model. Reported as a
warning to alert the user to possible input errors or unintentional inclusions.

**Example message:** `thermal 'OLD_PLANT' has max_generation_mw = 0.0 and will contribute no generation`

**Resolution:** Either remove the entity from the registry file or set a
non-zero generation capacity if the omission was accidental. If the entity is
intentionally inactive, this warning can be ignored.

---

### `ModelQuality`

**Default severity:** Warning (does not block execution)

**What triggers it:** A statistical quality concern is detected in the input
model. Examples: residual bias in the PAR model seasonal statistics, high
autocorrelation residuals, or an AR order that is suspiciously large for the
data. These do not prevent execution but may indicate that the model needs
recalibration.

**Example message:** `residual bias detected in inflow_seasonal_stats for hydro 'FURNAS' at stage 0: mean residual 45.2 m3/s`

**Resolution:** Review the flagged model parameters. Consider recalibrating the
PAR model for the affected hydro plants. Warnings of this type do not prevent
the solver from running, but they may indicate that the stochastic model does
not accurately represent historical inflows.

---

## Severity reference

| Severity | Effect                                | `ErrorKind` values                                 |
| -------- | ------------------------------------- | -------------------------------------------------- |
| Error    | Prevents `load_case` from succeeding  | All kinds except `UnusedEntity` and `ModelQuality` |
| Warning  | Reported but does not block execution | `UnusedEntity`, `ModelQuality`                     |

To inspect warnings after a successful `load_case`, call
`ValidationContext::warnings()` before calling `into_result()`. Warnings are
not surfaced in the `Result` returned by `load_case`; they must be read from
the context directly.
