# Accumulated Learnings: Epics 02-07 (cobre-io Foundation through Integration and Output)

**Last updated**: 2026-03-04
**Covers**: epic-02 (tickets 009-013), epic-03 (tickets 014-018), epic-04 (tickets 019-023, 048), epic-05 (tickets 024-028), epic-06 (tickets 029-033), epic-07 (tickets 034-038)
**Tests at end of epic-07**: integration tests in `crates/cobre-io/tests/` (6 integration + 3 invariance), plus all prior unit/doc-tests; zero clippy warnings

---

## Parsing Patterns (Epics 02-06, unchanged)

- **4-step JSON pipeline** (canonical): `fs::read_to_string` -> `serde_json::from_str` -> `validate_raw(&raw, path)?` -> `Ok(convert(raw))`. No exceptions. See `crates/cobre-io/src/system/buses.rs:117-129`.
- **`Raw*` intermediate types**: every JSON file gets private `RawFile` + `Raw*` structs that derive `Deserialize` only. Never put serde annotations on `cobre-core` types. See `crates/cobre-io/src/penalties.rs:37-111`.
- **Validate on raw type, convert afterward**: validation references JSON field paths only available on the raw type. See `crates/cobre-io/src/constraints/generic.rs:152-165`.
- **Sort in `convert()`**: every function producing a collection sorts by declared key before returning. See `crates/cobre-io/src/constraints/generic.rs:272-274`.
- **Parquet reader pattern**: `File::open` -> `ParquetRecordBatchReaderBuilder::try_new` -> `build()` -> `for batch in reader`. See `crates/cobre-io/src/scenarios/inflow_stats.rs:107-178`.
- **`parse_*` + `load_*` wrapper pair**: `parse_*` takes `&Path` and always fails on missing file; `load_*` takes `Option<&Path>` and returns empty for `None`. See `crates/cobre-io/src/constraints/mod.rs:49-358`.

## Serde Design Decisions (Epics 02-06, unchanged)

- **Tagged enums**: `#[serde(tag = "<spec_field>", rename_all = "snake_case")]`. Tag field name from JSON spec. See `crates/cobre-io/src/system/hydros.rs:147`.
- **`flatten` for sibling tag + payload fields**: when discriminator and data fields are at the same JSON level. See `crates/cobre-io/src/extensions/production_models.rs:174-203`.
- **Untagged enums for mixed string/object fields**: `#[serde(untagged)]`, string variant first. See `crates/cobre-io/src/stages.rs:252-266`.
- **Mandatory fields via `Option<T>` + post-validation**: produces better error messages than serde missing-field errors. See `crates/cobre-io/src/config.rs:734-751`.
- **Sense/discriminant validation manually**: fixed string sets validated in `validate_raw`, mapped in `convert()`. See `crates/cobre-io/src/constraints/generic.rs:183-193`.

## Error Type Rules (Epics 02-06, unchanged)

- **Never implement `From<io::Error>`**: use `LoadError::io(path, e)` at every I/O callsite. See `crates/cobre-io/src/error.rs:103-124`.
- **`ParseError` vs `SchemaError`**: `ParseError` = malformed syntax/missing serde field; `SchemaError` = domain constraint violation detected post-deserialization.
- **Detect unknown discriminants**: match `"unknown variant"` in serde error message string to reclassify as `SchemaError`. See `crates/cobre-io/src/config.rs:694-695`.
- **Field path format**: dot-separated with bracketed indices: `"constraints[2].expression"`. Parquet: `"<filename>[{row}].<column>"`.
- **Broadcast / report error path conventions** (NEW in epic-07): use angle-bracket synthetic paths for non-file errors: `"<broadcast>"` for postcard errors, `"<report>"` for JSON serialization errors. See `crates/cobre-io/src/broadcast.rs:60` and `crates/cobre-io/src/report.rs:93`.

## Parquet Infrastructure (Epics 04-05, unchanged)

- **Dependencies**: `arrow = { version = "55", features = ["prettyprint"] }` and `parquet = { version = "55", features = ["arrow"] }`. See `crates/cobre-io/Cargo.toml:17-18`.
- **Centralized extraction helpers**: `pub(crate) mod parquet_helpers` in `crates/cobre-io/src/parquet_helpers.rs`.
- **Float validation for bounds**: `value.is_finite()` only. **For penalties**: `value > 0.0 && value.is_finite()`.
- **Parquet test helpers**: `ArrowWriter` — never commit binary `.parquet` fixtures.

## Validation Pipeline Patterns (Epic 06, unchanged)

- **Infallible layer functions**: all entry points are `pub(crate) fn validate_<layer>(data: &ParsedData, ctx: &mut ValidationContext)` — no `Result`, push to `ctx`. See `crates/cobre-io/src/validation/referential.rs:37`.
- **Error-count guard** in `validate_schema`: snapshot `ctx.errors().len()` at entry, compare at exit. See `crates/cobre-io/src/validation/schema.rs:236`.
- **`parse_or_error` + `optional_or_error` helpers**: eliminate repeated match patterns across 33 parse calls. See `crates/cobre-io/src/validation/schema.rs:625-667`.
- **Sentinel `GlobalPenaltyDefaults`**: when `penalties.json` fails, provide minimal valid struct so penalty-dependent parsers still collect their own errors. See `crates/cobre-io/src/validation/schema.rs:669-700`.
- **Kahn's algorithm for cascade acyclicity**: local `HashMap<i32, Vec<i32>>` adjacency + `HashMap<i32, usize>` in-degree. See `crates/cobre-io/src/validation/semantic.rs:89-177`.

## Pipeline Orchestrator Pattern (Epic 07 - NEW)

- **Separate `pipeline.rs` module**: `load_case` in `lib.rs` is a one-liner delegating to `pub(crate) fn run_pipeline(path: &Path)` in `crates/cobre-io/src/pipeline.rs`. This keeps the public API surface thin and orchestration logic separate. See `crates/cobre-io/src/lib.rs:121` and `crates/cobre-io/src/pipeline.rs:40`.
- **Short-circuit pattern for `None` from `validate_schema`**: use `let Some(data) = data else { return ctx.into_result().map(|()| unreachable!(...)); };`. The `map(|()| unreachable!())` makes the type checker happy for `Result<System, LoadError>` without any dummy value. See `crates/cobre-io/src/pipeline.rs:49-53`.
- **Stage count filter**: `data.stages.stages.iter().filter(|s| s.id >= 0).count()` — pre-study stages (negative IDs) are excluded from resolution. Never pass the raw `len()`. See `crates/cobre-io/src/pipeline.rs:70`.
- **`CorrelationModel::default()` over `.unwrap_or_default()`**: use `.unwrap_or_else(CorrelationModel::default)` (clippy pedantic requires explicit path over closure) or `.unwrap_or_default()` depending on clippy version in effect. See `crates/cobre-io/src/pipeline.rs:120`.
- **Dead-code annotation removal**: all `#[allow(dead_code)]` on validation layer functions are removed when `pipeline.rs` wires them. This is the signal that integration is complete. Confirmed in validation module files.
- **`ctx.into_result()` is consumed**: after calling it for the early-exit case, the `ctx` is gone. Warnings are silently discarded in the public API — callers wanting diagnostics must use `generate_report` before calling `into_result`. Document this design decision inline.

## Postcard Serialization Pattern (Epic 07 - NEW)

- **`postcard = { version = "1", features = ["alloc"] }`**: use `alloc` feature, NOT `std` feature. `to_allocvec` requires only `alloc`. See `crates/cobre-io/Cargo.toml`.
- **`rebuild_indices()` is mandatory after deserialization**: `System`'s `HashMap` lookup fields are `#[serde(skip)]`. Without calling `rebuild_indices()`, all `system.bus(id)` lookups return `None` silently. This is the only post-deserialization step required. See `crates/cobre-io/src/broadcast.rs:98`.
- **`PartialEq` on `System` skips index fields**: two Systems can pass `assert_eq!` even if one has empty indices. Always test an actual lookup (e.g., `system.bus(EntityId(1)).is_some()`) after round-trip, not just equality. See `crates/cobre-io/src/broadcast.rs` tests.
- **Doc-example pattern for public API**: `broadcast.rs` uses `/// # Examples` with ```` ```rust ```` (not ```` ```rust,ignore ````) because the functions work without side effects in doctest context. Use `/// ``` rust,ignore` only for functions requiring filesystem/MPI. See `crates/cobre-io/src/broadcast.rs:44-57`.

## Integration Test Architecture (Epic 07 - NEW)

- **External test files in `tests/`**: integration tests live in `crates/cobre-io/tests/integration.rs` and `invariance.rs`, not in `src/`. They are invoked with `cargo test --test integration -p cobre-io` or `cargo test --test invariance`. See `crates/cobre-io/tests/`.
- **Shared helpers module via `mod helpers;`**: both `integration.rs` and `invariance.rs` include `mod helpers;` which resolves to `crates/cobre-io/tests/helpers/mod.rs`. This is the standard Rust pattern for sharing test utilities across multiple integration test files. See `crates/cobre-io/tests/helpers/mod.rs`.
- **Private test constants cannot cross crate boundaries**: `VALID_*_JSON` constants in `src/validation/schema.rs` tests are private. Integration tests redefine them in `tests/helpers/mod.rs` as `pub const`. Do not attempt to import them. See `crates/cobre-io/tests/helpers/mod.rs:30-101`.
- **`make_referential_violation_case` built by composition**: build the full multi-entity case, then overwrite a single file with the invalid content. Avoids duplicating all JSON. See `crates/cobre-io/tests/helpers/mod.rs:277-306`.
- **Error display assertions**: for `LoadError::ConstraintError`, use `err.to_string().contains(...)` rather than matching on fields — `ConstraintError` joins all error messages into a single `description` string. See `crates/cobre-io/tests/integration.rs:128-134`.
- **`#![allow(...)]` at crate level in test files**: use `#![allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines, clippy::doc_markdown)]` at the top of each integration test file (not `#[allow(...)]` on the module). See `crates/cobre-io/tests/integration.rs:9-14`.

## Declaration-Order Invariance Testing (Epic 07 - NEW)

- **Fixed reverse permutations, not randomized**: shuffle by reversing array order in JSON strings. Deterministic, reproducible, sufficient for the invariant. See `crates/cobre-io/tests/invariance.rs:37-154`.
- **Shuffle only multi-element arrays**: single-entity arrays (one hydro, one thermal, one line) produce no meaningful shuffle. Only buses (2 entities) and stages (2 entities) are reordered. Document this in the helper doc comment. See `crates/cobre-io/tests/invariance.rs:31-36`.
- **Verify with ID-based lookups, not positional**: use `system.bus(EntityId(1))` and `system.bus(EntityId(2))`, not `system.buses()[0]`. Positional checks would pass even without sorting. See `crates/cobre-io/tests/invariance.rs:185-207`.
- **`stages.json` nesting**: only the `stages` array is reversed; the `transitions` array inside `policy_graph` must keep `source_id`/`target_id` semantics intact. Never reorder transitions. See `crates/cobre-io/tests/invariance.rs:44-73`.

## Validation Report Pattern (Epic 07 - NEW)

- **`generate_report` takes `&ValidationContext`, not owned**: callers can still call `ctx.into_result()` after generating the report. This is important — the report is an optional diagnostic view, not a replacement for error propagation. See `crates/cobre-io/src/report.rs:121`.
- **`ErrorKind` variant name via `format!("{:?}", entry.kind)`**: the `Debug` repr gives exact variant names like `"FileNotFound"`, `"ParseError"`. Callers (CLI, TUI, MCP) can switch on these strings. See `crates/cobre-io/src/report.rs:126`.
- **`Serialize` only, never `Deserialize`**: `ValidationReport` and `ReportEntry` are output-only types. Adding `Deserialize` would be misleading and unused. See `crates/cobre-io/src/report.rs:37-67`.
- **Output module stub**: `crates/cobre-io/src/output/mod.rs` is a deliberate stub with only a doc comment. Phase 7 output writers (Hive-partitioned Parquet, FlatBuffers FCF) will be added here. Do not implement anything until Phase 7 spec is read.

## Scope Boundaries (What Parsers Do NOT Validate, unchanged from epics 02-06)

- **No cross-reference validation in parsers**: `bus_id`, `hydro_id`, etc. existence checks are Layer 3.
- **No semantic validation in parsers**: min < max checks, penalty ordering — all Layer 5.
- **No tier-1/tier-2 resolution in the resolver**: only overlays tier-3 stage overrides.

## Test Conventions (All Epics, extended in epic-07)

- **JSON tests**: `write_json(content: &str) -> NamedTempFile` helper; `const VALID_JSON` for error-path tests.
- **Parquet tests**: `make_batch` + `write_parquet` + `schema` helpers; test valid, missing column, invalid value, empty file, declaration-order invariance.
- **Integration tests**: `TempDir` + `write_file(root, relative, content)` helpers; `make_minimal_case` + `make_multi_entity_case` + `make_referential_violation_case` in `tests/helpers/mod.rs`. See `crates/cobre-io/tests/helpers/mod.rs`.
- **Error match pattern**: use `err.to_string().contains(...)` for `ConstraintError` (multi-message join); use `match &err { LoadError::ParseError { .. } => ... }` for specific variant matching.
- **f64 equality**: never `assert_eq!` with f64. Use `(actual - expected).abs() < f64::EPSILON`.

## Clippy Suppressions Required (All Epics, unchanged)

- `#[allow(clippy::struct_excessive_bools)]`, `#[allow(clippy::similar_names)]`, `#[allow(clippy::struct_field_names)]`
- `#[allow(dead_code)]` on `String` inner value of `#[serde(untagged)]` string enum variant.
- `#[allow(clippy::too_many_arguments, clippy::too_many_lines)]` — resolver and validation functions.
- `#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]` — expression parser.
- `#![allow(dead_code)]` in `tests/helpers/mod.rs` — helper functions are not all used by every test file.

## Guidance for Epics 08-09

- **Epic 08 (docs)**: `lib.rs` crate-level doc already describes the 5-layer pipeline accurately. Module docs in `broadcast.rs`, `report.rs`, `output/mod.rs` cite DEC-003 and SS6.2 — use them as templates. The book page for `cobre-io` should cover `load_case`, `serialize_system`/`deserialize_system`, `generate_report`, and the planned `output` module scope.
- **Epic 09 (language review)**: `cobre-io` public API is solver-neutral (`load_case`, `run_pipeline`, `ValidationReport`). The "MPI broadcast" mention in `broadcast.rs` is implementation context, not a solver coupling — keep it. Audit `crates/cobre-io/src/scenarios/` module names for any SDDP-specific terminology.
