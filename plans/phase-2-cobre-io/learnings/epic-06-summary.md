# Accumulated Learnings: Epics 02-06 (cobre-io Foundation through 5-Layer Validation Pipeline)

**Last updated**: 2026-03-04
**Covers**: epic-02 (tickets 009-013), epic-03 (tickets 014-018), epic-04 (tickets 019-023, 048), epic-05 (tickets 024-028), epic-06 (tickets 029-033)
**Tests at end of epic 06**: 504 unit tests + 92 doc-tests in cobre-io, all green, zero clippy warnings

---

## Parsing Patterns (All Epics)

- **4-step JSON pipeline** (canonical): `fs::read_to_string` -> `serde_json::from_str` -> `validate_raw(&raw, path)?` -> `Ok(convert(raw))`. No exceptions. See `crates/cobre-io/src/system/buses.rs:117-129`.
- **`Raw*` intermediate types**: every JSON file gets private `RawFile` + `Raw*` structs that derive `Deserialize` only. Never put serde annotations on `cobre-core` types. See `crates/cobre-io/src/penalties.rs:37-111`.
- **Validate on raw type, convert afterward**: validation references JSON field paths (e.g., `"constraints[1].expression"`) only available on the raw type. See `crates/cobre-io/src/constraints/generic.rs:152-165`.
- **Sort in `convert()`**: every function producing a collection sorts by declared key before returning. See `crates/cobre-io/src/constraints/generic.rs:272-274`.
- **Parquet reader pattern**: `File::open` -> `ParquetRecordBatchReaderBuilder::try_new` -> `build()` -> `for batch in reader`. See `crates/cobre-io/src/scenarios/inflow_stats.rs:107-178`.
- **`parse_*` + `load_*` wrapper pair**: `parse_*` takes `&Path` and always fails on missing file; `load_*` takes `Option<&Path>` and returns empty for `None`. See `crates/cobre-io/src/constraints/mod.rs:49-358`.

## Serde Design Decisions (All Epics)

- **Tagged enums**: `#[serde(tag = "<spec_field>", rename_all = "snake_case")]`. Tag field name from JSON spec. See `crates/cobre-io/src/system/hydros.rs:147`.
- **`flatten` for sibling tag + payload fields**: when discriminator and data fields are at the same JSON level. See `crates/cobre-io/src/extensions/production_models.rs:174-203`.
- **Untagged enums for mixed string/object fields**: `#[serde(untagged)]`, string variant first. See `crates/cobre-io/src/stages.rs:252-266`.
- **Mandatory fields via `Option<T>` + post-validation**: produces better error messages than serde missing-field errors. See `crates/cobre-io/src/config.rs:734-751`.
- **Sense/discriminant validation manually**: fixed string sets (e.g., `">="`) are validated in `validate_raw` and mapped in `convert()` to produce `SchemaError` with accurate field paths. See `crates/cobre-io/src/constraints/generic.rs:183-193`.

## Error Type Rules (All Epics)

- **Never implement `From<io::Error>`**: use `LoadError::io(path, e)` at every I/O callsite. See `crates/cobre-io/src/error.rs:103-124`.
- **`ParseError` vs `SchemaError`**: `ParseError` = malformed syntax/missing serde field; `SchemaError` = domain constraint violation detected post-deserialization.
- **Detect unknown discriminants**: match `"unknown variant"` in serde error message string to reclassify as `SchemaError`. See `crates/cobre-io/src/config.rs:694-695`.
- **Field path format**: dot-separated with bracketed indices: `"constraints[2].expression"`. Parquet: `"<filename>[{row}].<column>"`.
- **Duplicate ID detection**: `HashSet<i32>` loop; error field is `"<array>[{i}].id"`.

## Parquet Infrastructure (Epics 04-05)

- **Dependencies**: `arrow = { version = "55", features = ["prettyprint"] }` and `parquet = { version = "55", features = ["arrow"] }`. See `crates/cobre-io/Cargo.toml:17-18`.
- **Centralized extraction helpers**: `pub(crate) mod parquet_helpers` in `crates/cobre-io/src/parquet_helpers.rs` — `extract_required_int32`, `extract_required_float64`, `extract_required_date32`, `extract_optional_int32`, `extract_optional_float64`.
- **Float validation for bounds**: `value.is_finite()` only — bounds can be zero or negative. **For penalties**: `value > 0.0 && value.is_finite()` — penalties must be strictly positive.
- **Multi-key sort**: `.sort_by` with `.then_with(|| ...)` chaining. See `crates/cobre-io/src/constraints/generic_bounds.rs`.
- **Parquet test helpers**: `ArrowWriter` — never commit binary `.parquet` fixtures.

## Constraints Module Organization (Epic 05)

- **Flat submodule layout**: `bounds.rs`, `penalty_overrides.rs`, `generic.rs`, `generic_bounds.rs`, `exchange_factors.rs`. All `load_*` wrappers in `constraints/mod.rs`. See `crates/cobre-io/src/constraints/`.
- **Module-level Parquet schema docs**: every Parquet parser opens with a Markdown table (column, type, required, description). See `crates/cobre-io/src/constraints/bounds.rs:1-65`.
- **Deferred validation documented inline**: every parser module has a "Deferred validations" section naming which layer owns unchecked rules. See `crates/cobre-io/src/constraints/penalty_overrides.rs:67-70`.

## Expression Parser Pattern (Epic 05)

- **Hand-written recursive descent**: no `nom`, `pest`. Grammar: tokenize -> `Vec<Token>` -> `parse_terms`. See `crates/cobre-io/src/constraints/generic.rs:292-750`.
- **Expression validated twice**: once in `validate_raw` (accurate `SchemaError` field paths), again in `convert()` (infallible production). See `crates/cobre-io/src/constraints/generic.rs:155-165`.
- **`pub(crate) fn parse_expression`**: exposed at crate visibility so Layer 5 can reuse it. See `crates/cobre-io/src/constraints/generic.rs:292`.

## Resolution Module Pattern (Epic 05)

- **Infallible resolver functions**: `resolve_penalties` and `resolve_bounds` return the container directly (no `Result`). Unknown IDs and out-of-range stages are silently skipped — that is Layer 3's concern. See `crates/cobre-io/src/resolution/penalties.rs:134-146`.
- **`ResolvedPenalties::new` / `ResolvedBounds::new` requires single default**: guard `n_stages == 0` with early return before fill loop. See `crates/cobre-io/src/resolution/penalties.rs:213-232`.
- **Override application pattern**: `for row in overrides { let Some(&idx) = index.get(&row.entity_id) else { continue; }; ... }`. See `crates/cobre-io/src/resolution/penalties.rs:276-320`.

## Validation Pipeline Patterns (Epic 06 - NEW)

- **Infallible layer functions**: all layer entry points are `pub(crate) fn validate_<layer>(data: &ParsedData, ctx: &mut ValidationContext)` — no `Result`, push to `ctx`. See `crates/cobre-io/src/validation/referential.rs:37`.
- **Error-count guard** in `validate_schema`: snapshot `ctx.errors().len()` at entry, compare at exit. Cleaner than threading `has_error: bool`. See `crates/cobre-io/src/validation/schema.rs:236, 557`.
- **`parse_or_error` + `optional_or_error` helpers**: eliminate repeated `match result { Ok -> Some, Err -> map_load_error + None }` across 33 parse calls. See `crates/cobre-io/src/validation/schema.rs:625-667`.
- **Sentinel `GlobalPenaltyDefaults`**: when `penalties.json` fails, `sentinel_penalties()` provides a minimal valid struct (all `1.0`) so penalty-dependent parsers still run and collect their own errors. See `crates/cobre-io/src/validation/schema.rs:669-700`.
- **`HashSet<i32>` lookup tables built at entry**: `let bus_ids: HashSet<i32> = data.buses.iter().map(|b| b.id.0).collect();` pattern repeated for every entity type. `HashSet<(i32, i32)>` for 2-D entity+stage pairs. See `crates/cobre-io/src/validation/referential.rs:40-53`.
- **Kahn's algorithm for cascade acyclicity**: local `HashMap<i32, Vec<i32>>` adjacency + `HashMap<i32, usize>` in-degree. Do NOT use `CascadeTopology::build` (silently ignores cycles). See `crates/cobre-io/src/validation/semantic.rs:89-177`.
- **`slice.windows(2)` + `current_key` scan for grouped pre-sorted data**: used for geometry monotonicity (rules 8-10) and FPHA plane-count (rule 11). O(n), no extra allocation. See `crates/cobre-io/src/validation/semantic.rs:337-483`.
- **Aggregated penalty ordering warnings**: collect all violations into `Vec<(id, higher, lower)>`, emit one `ctx.add_warning` per check with count and worst-case entity. See `crates/cobre-io/src/validation/semantic.rs:680-844`.
- **`#[allow(dead_code)]` on all pipeline functions**: required until `load_case` (ticket-036) wires up the caller. Remove these suppressions in epic-07. See `crates/cobre-io/src/validation/schema.rs:227`.
- **Study stage filter**: `data.stages.stages.iter().filter(|s| s.id >= 0)` — pre-study stages (negative IDs) never included in coverage or semantic checks.

## Validation Error Conventions (Epic 06 - NEW)

- **Entity string**: `"<Type> <id>"` (e.g., `"Hydro 42"`, `"Line 7"`). No suffix, no padding. Consistent across all layers.
- **Referential error message**: `"<source_type> <id> references non-existent <target_type> <target_id> via field '<field>'"`.
- **Bound violation message**: `"<entity_str>: <field_a> (<val_a>) > <field_b> (<val_b>); <explanation>"`.
- **Penalty ordering warning**: `"Penalty ordering violation: <higher> (<val>) should be > <lower> (<val>) -- N hydro(s) affected, worst case: Hydro <id>"`.
- **Float tolerances**: `const PROB_TOLERANCE: f64 = 1e-6` (probability sums), `const CORR_TOLERANCE: f64 = 1e-9` (correlation matrices). See `crates/cobre-io/src/validation/semantic.rs:542-546`.

## Scope Boundaries (What Parsers Do NOT Validate)

- **No cross-reference validation in parsers**: `bus_id`, `hydro_id`, etc. existence checks are Layer 3. Construct `EntityId(raw.field_id)` directly in `convert()`.
- **No semantic validation in parsers**: min < max checks, penalty ordering, AR lag contiguity — all Layer 5.
- **No duplicate-key validation in Parquet parsers**: duplicate `(entity_id, stage_id)` pairs in bounds/override files are Layer 3.
- **No tier-1/tier-2 resolution in the resolver**: `resolve_penalties` reads already-resolved entity fields. It only overlays tier-3 stage overrides.

## Test Conventions (All Epics)

- **JSON tests**: `write_json(content: &str) -> NamedTempFile` helper; `const VALID_JSON` for error-path tests.
- **Parquet tests**: `make_batch` + `write_parquet` + `schema` helpers; test valid, missing column, invalid value, empty file, declaration-order invariance.
- **Validation tests**: `tempfile::TempDir` + `write_file(root, relative, content)` helpers; `make_valid_case(dir)` populates all 8 required files. See `crates/cobre-io/src/validation/schema.rs:800-824`.
- **Error match pattern**: `match &err { LoadError::SchemaError { field, message, .. } => { assert!(field.contains(...)); assert!(message.contains(...)); } other => panic!(...) }`.
- **Clippy allow on test modules**: `#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines, clippy::doc_markdown)]` — always add this header.
- **f64 equality**: never `assert_eq!` with f64. Use `(actual - expected).abs() < f64::EPSILON`.

## Clippy Suppressions Required (All Epics)

- `#[allow(clippy::struct_excessive_bools)]` — structs with 4+ bool fields.
- `#[allow(clippy::similar_names)]` — physics coefficient fields.
- `#[allow(clippy::struct_field_names)]` — spec-mirroring fields with common prefix.
- `#[allow(dead_code)]` on `String` inner value of `#[serde(untagged)]` string enum variant.
- `#[allow(clippy::too_many_arguments, clippy::too_many_lines)]` — resolver functions (9-11 params, 300+ lines) and validation functions checking 13-16 rules inline.
- `#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]` — validated f64 -> i32/usize casts in expression parser.

## Epic 07 Integration Readiness

- `load_case` = `validate_structure` -> `validate_schema` -> `validate_referential_integrity` -> `validate_dimensional_consistency` -> `validate_semantic_hydro_thermal` + `validate_semantic_stages_penalties_scenarios` -> `ctx.into_result()`.
- Remove all `#[allow(dead_code)]` in `crates/cobre-io/src/validation/*.rs` when `load_case` is wired.
- Reuse `make_valid_case` from `schema.rs` tests (or extract to `tests/fixtures.rs`) for integration test setup.
- `ParsedData` will be consumed by output/resolution layer; decide whether `load_case` returns `ParsedData` directly or wraps it in a `CaseData` struct.
