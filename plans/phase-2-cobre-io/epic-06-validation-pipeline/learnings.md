# Epic 06 Learnings: 5-Layer Validation Pipeline

**Epic**: epic-06-validation-pipeline
**Date extracted**: 2026-03-04
**Tickets covered**: 029 (schema), 030 (referential), 031 (dimensional), 032 (semantic-hydro/thermal), 033 (semantic-stages/penalties/scenarios)
**Files added**: `crates/cobre-io/src/validation/schema.rs` (+1055 lines), `crates/cobre-io/src/validation/referential.rs` (+1415 lines), `crates/cobre-io/src/validation/dimensional.rs` (+1285 lines), `crates/cobre-io/src/validation/semantic.rs` (+2909 lines), `crates/cobre-io/src/validation/mod.rs` (modified)
**Tests at end of epic 06**: 504 unit tests + 92 doc-tests in cobre-io, all green, zero clippy warnings

---

## Patterns Established

### Validation Layer Function Signature
- All layer entry points are `pub(crate) fn validate_<layer>(data: &ParsedData, ctx: &mut ValidationContext)` — infallible (no `Result`), takes `ParsedData` by shared reference and mutates the context by pushing errors/warnings.
- `validate_schema` is the exception: it also takes `case_root: &Path` and `manifest: &FileManifest`, and returns `Option<ParsedData>`.
- See `crates/cobre-io/src/validation/referential.rs:37`, `crates/cobre-io/src/validation/dimensional.rs:31`, `crates/cobre-io/src/validation/semantic.rs:72`.

### Error-Count Guard for `validate_schema`
- Rather than threading a `bool has_error` through all 33 parse calls, `validate_schema` snapshots `ctx.errors().len()` at entry and compares at exit: `if ctx.errors().len() > error_count_before { return None; }`. This is O(1) and eliminates any risk of missing an error.
- See `crates/cobre-io/src/validation/schema.rs:236, 557`.

### `parse_or_error` and `optional_or_error` Helpers (Layer 2)
- Two private helpers eliminate repeated `match result { Ok(v) => Some(v), Err(e) => { map_load_error(e, path, ctx); None } }` chains across 33 parse calls.
- `parse_or_error<T>(Result<T, LoadError>, path, ctx) -> Option<T>` handles required files.
- `optional_or_error<T, F, D>(present: bool, parse_fn: F, default_fn: D, path, ctx) -> T` handles optional files, calling `default_fn()` when `present == false`.
- See `crates/cobre-io/src/validation/schema.rs:625-667`.

### Sentinel `GlobalPenaltyDefaults` for Continued Parsing
- When `penalties.json` fails to parse (required by `parse_buses`, `parse_hydros`, `parse_lines`, `parse_non_controllable_sources`), a `sentinel_penalties()` function returns a minimal valid `GlobalPenaltyDefaults` filled with `1.0` everywhere. This allows those parsers to be attempted and collect their own independent errors in the same pass.
- See `crates/cobre-io/src/validation/schema.rs:669-700`.

### `HashSet<i32>` Lookup Tables Built at Entry (Layers 3-5)
- All referential and dimensional validation functions build `HashSet<i32>` lookup tables from entity ID slices at the very top: `let bus_ids: HashSet<i32> = data.buses.iter().map(|b| b.id.0).collect();`. All rule checks use `.contains(id)` for O(1) membership.
- For 2-D coverage (entity+stage pairs), use `HashSet<(i32, i32)>`.
- See `crates/cobre-io/src/validation/referential.rs:40-53`, `crates/cobre-io/src/validation/dimensional.rs:48-57`.

### Inline Rule Grouping with Section Comments
- Each validation function uses `// ── Rules N-M: <group name> ────` comments to separate logical rule groups. This allows a single large function to contain 13-16 rules while remaining auditable against the spec table.
- Preferred over splitting into one function per rule, because most rule groups share a single `HashSet` or aggregated cost variable.
- See `crates/cobre-io/src/validation/semantic.rs:82-504`.

### Delegated Sub-functions for Multi-Step Rule Groups (Layer 5a)
- `validate_semantic_hydro_thermal` delegates each rule group to a private helper: `check_cascade_acyclic`, `check_hydro_bounds`, `check_lifecycle_consistency`, `check_filling_config`, `check_geometry_monotonicity`, `check_fpha_constraints`, `check_thermal_generation_bounds`.
- This structure produces readable top-level orchestration and testable helper functions.
- See `crates/cobre-io/src/validation/semantic.rs:72-80`.

### Kahn's Algorithm for Cycle Detection (Layer 5a, Rule 1)
- Cascade acyclicity is checked with Kahn's topological sort, not by calling `CascadeTopology::build` (which silently ignores cycles). Local `HashMap<i32, Vec<i32>>` adjacency and `HashMap<i32, usize>` in-degree are built from `hydro.downstream_id`. If `visited_count < all_ids.len()` after the sort, a cycle exists. Cycle participants are all nodes with `in_degree > 0` after the sort.
- See `crates/cobre-io/src/validation/semantic.rs:89-177`.

### `slice.windows(2)` for Consecutive-Pair Checks on Pre-Sorted Data
- Geometry monotonicity (rules 8-10) and FPHA plane-count checks (rule 11) both rely on the fact that rows are pre-sorted by key. Groups are identified with a `current_key + while` scan (O(n) no allocation), and consecutive pairs within a group are checked with `group.windows(2)`.
- See `crates/cobre-io/src/validation/semantic.rs:337-483`.

### Aggregated Per-Check Penalty Ordering Warnings (Layer 5b, Rules 6-10)
- Penalty ordering checks aggregate violations across all hydros into a single warning per violated check: collect `violations: Vec<(id, higher, lower)>`, find worst case by `max_by(|a,b| (b.lower - b.higher).partial_cmp(...)`, emit one `ctx.add_warning` with count and worst-case entity.
- This follows the spec: one warning per violated ordering pair, not one per hydro.
- `const PROB_TOLERANCE: f64 = 1e-6` and `const CORR_TOLERANCE: f64 = 1e-9` are module-level constants for floating-point comparisons.
- See `crates/cobre-io/src/validation/semantic.rs:680-844`.

### `correlation.json` as `Option<CorrelationModel>` (not `Vec`)
- Correlation is the only optional JSON file (all other optional files are Parquet row vecs). It is handled specially in `validate_schema` with an inline `if manifest.scenarios_correlation_json { match ... }` block rather than the `optional_or_error` helper.
- See `crates/cobre-io/src/validation/schema.rs:418-430`.

### `#[allow(dead_code)]` on All Validation Functions
- Until `load_case` (ticket-036) wires up the full pipeline, all Layer 2-5 entry functions carry `#[allow(dead_code)]` so the linter does not reject CI. This is the correct pattern when a module is implemented ahead of its caller.
- See `crates/cobre-io/src/validation/schema.rs:227`, `referential.rs:36`, `semantic.rs:71, 528`.

---

## Architectural Decisions

### Decision: Infallible Layer Functions, No `Result`
- **Chose**: All layer functions are infallible, push to `ctx`, and return nothing (or `Option<ParsedData>` for schema).
- **Rejected**: Returning `Result<(), LoadError>` from each layer (would require callers to chain `?` and break the collect-all-errors contract).
- **Rationale**: The entire point of the validation pipeline is to collect all errors before failing. A fallible layer function would undermine this by forcing early exit at the first error from any rule.

### Decision: Single `ParsedData` Struct Carrying All 33 Parsed Outputs
- **Chose**: `pub(crate) struct ParsedData` with one field per input file (33 total), passed by `&ParsedData` to all subsequent layers.
- **Rejected**: Returning individual vecs from `validate_schema` or threading data through a mutable state struct.
- **Rationale**: A single data bundle makes the pipeline stages composable and type-safe. All subsequent validation layers receive exactly what they need without extra coupling.
- **Key caveat**: `ParsedData` is `pub(crate)` only, never exposed to downstream crates.
- See `crates/cobre-io/src/validation/schema.rs:68-144`.

### Decision: `validate_semantic_*` Split into 5a (hydro/thermal) and 5b (stages/penalties/scenarios)
- **Chose**: Two separate entry functions in one `semantic.rs` file.
- **Rejected**: One monolithic function (too long) or four separate files (too much boilerplate).
- **Rationale**: The two groups validate entirely disjoint domains and can be implemented in parallel (tickets 032 and 033 are independent). A single file avoids `use` import duplication.

### Decision: No Helper Macro for Referential Rule Repetition
- **Chose**: Explicit inline rule blocks with section comments.
- **Rejected**: `macro_rules! check_ref! { ... }` macro to reduce boilerplate across 30 rules.
- **Rationale**: Rule messages and field names differ enough that a macro would need many parameters, making it harder to read than the explicit inline form. 30 rules with identical structure are still readable with section separators.
- See `crates/cobre-io/src/validation/referential.rs:54-555`.

---

## Files and Structures Created

- `crates/cobre-io/src/validation/schema.rs` — `ParsedData` (33-field data bundle), `validate_schema`, `map_load_error`, `parse_or_error`, `optional_or_error`, `sentinel_penalties`
- `crates/cobre-io/src/validation/referential.rs` — `validate_referential_integrity` (30 cross-reference rules)
- `crates/cobre-io/src/validation/dimensional.rs` — `validate_dimensional_consistency` (8 coverage rules)
- `crates/cobre-io/src/validation/semantic.rs` — `validate_semantic_hydro_thermal` (13 rules), `validate_semantic_stages_penalties_scenarios` (16 rules), plus 9 private helper functions
- `crates/cobre-io/src/validation/mod.rs` — adds `pub mod` declarations for 4 new submodules; re-exports `validate_schema` and `ParsedData` at `pub(crate)` level

---

## Conventions Adopted

- **Error entity string format**: `"<Type> <id>"` (e.g., `"Hydro 42"`, `"Line 7"`, `"Thermal 3"`). No suffix, no padding. Consistent across all 5 layers.
- **Error message format for referential violations**: `"<source_type> <source_id> references non-existent <target_type> <target_id> via field '<field_name>'"`.
- **Error message format for bound violations**: `"<entity_str>: <field_a> (<val_a>) > <field_b> (<val_b>); <explanation>"` — always includes both values and a human-readable reason.
- **Warning format for penalty ordering**: `"Penalty ordering violation: <higher_name> (<val>) should be > <lower_name> (<val>) -- N hydro(s) affected, worst case: Hydro <id>"`.
- **Tolerance constants**: `PROB_TOLERANCE = 1e-6` (probability sums), `CORR_TOLERANCE = 1e-9` (correlation matrix symmetry/diagonal).
- **Guard before optional data rules**: every rule block on optional data opens with `if data.field.is_empty() { return; }` or `if let Some(ref x) = data.field { ... }`.
- **Study stage filter**: `data.stages.stages.iter().filter(|s| s.id >= 0)` — pre-study stages (negative IDs) are never included in coverage or validation checks.

---

## Surprises and Deviations

### Rule Count Expansion: 26 -> 30 Cross-Reference Rules (ticket-030)
- The ticket specified 26 cross-reference rules in its main body. The actual spec contained 4 additional rules for penalty override files (rules 27-30: `BusPenaltyOverrideRow`, `LinePenaltyOverrideRow`, `HydroPenaltyOverrideRow`, `NcsPenaltyOverrideRow`). These are listed in the ticket's rule table but the overview said 26.
- Implementation correctly implemented all 30 rules.
- See `crates/cobre-io/src/validation/referential.rs:491-555`.

### Penalty Override File `pub(crate)` Visibility Required Upstream Changes
- `BusPenaltyOverrideRow`, `LinePenaltyOverrideRow`, `HydroPenaltyOverrideRow`, `NcsPenaltyOverrideRow` were originally private inside `crates/cobre-io/src/constraints/`. They had to be elevated to `pub(crate)` in their source files so `schema.rs` and `referential.rs` could import them.
- The unstaged changes to `bounds.rs`, `generic_bounds.rs`, and `penalty_overrides.rs` are minor visibility bumps (single-line each).

### `GlobalPenaltyDefaults` is Not `Clone` by Default
- `parse_buses`, `parse_hydros`, `parse_lines`, and `parse_non_controllable_sources` all require `&GlobalPenaltyDefaults`. When `penalties.json` fails to parse, a sentinel is needed. This required `GlobalPenaltyDefaults` to implement `Clone` (or be constructed fresh via `sentinel_penalties()`). The sentinel approach was chosen to avoid adding `Clone` to a `cobre-core` type.
- See `crates/cobre-io/src/validation/schema.rs:669-700`.

### `validate_semantic_hydro_thermal` Needs `#[allow(clippy::too_many_lines)]` at Two Levels
- Both the public entry function and `check_stage_structure` required `#[allow(clippy::too_many_lines)]` independently. This is expected for functions checking 13-16 rules inline. It is not a code smell — it is the correct pattern for a validation layer.

---

## Recommendations for Future Epics

### Epic 07 (Integration and Output): Wire Up the Full Pipeline
- `load_case` in ticket-036 will be the first caller of all five validation layers. The function signature should be `fn load_case(case_root: &Path) -> Result<CaseData, LoadError>` and the body should be: `validate_structure` → `validate_schema` → `validate_referential_integrity` → `validate_dimensional_consistency` → `validate_semantic_hydro_thermal` + `validate_semantic_stages_penalties_scenarios` → `ctx.into_result()`.
- All `#[allow(dead_code)]` annotations in `crates/cobre-io/src/validation/*.rs` should be removed when `load_case` is implemented.
- `ParsedData` will be consumed by the output and resolution layer. Consider whether `load_case` returns `ParsedData` directly or a higher-level `CaseData` struct that wraps it.

### Epic 07: Integration Test Infrastructure
- All five ticket test suites use `tempfile::TempDir` + `write_file` helpers to build minimal valid case directories. Epic 07's integration tests should reuse the `make_valid_case` helper from `crates/cobre-io/src/validation/schema.rs:811-824` — or extract it to a shared `tests/fixtures.rs` module to avoid duplication.

### Epic 08 (Documentation): Book Page for the Validation Pipeline
- The 5-layer pipeline is a user-visible feature. The book page should describe: what each layer checks, how to read validation error messages, and the distinction between `ErrorKind::ModelQuality` warnings (non-blocking) and error kinds (blocking).
- The module-level doc table in `crates/cobre-io/src/validation/semantic.rs:7-44` is a good starting point.
