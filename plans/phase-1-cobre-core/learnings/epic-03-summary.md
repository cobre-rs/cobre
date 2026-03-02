# Epic 03: Validation and Testing -- Learnings

## What Worked Well

1. **Validation decomposed into per-entity helper functions**: The
   `validate_cross_references` umbrella function delegates to six private helpers
   (`validate_line_refs`, `validate_hydro_refs`, `validate_thermal_refs`, etc.),
   each taking only the slice and indices it needs. This kept individual functions
   short and made them easy to test in isolation. See
   `crates/cobre-core/src/system.rs`.

2. **`clippy::too_many_arguments` suppressed with rationale**: The
   `validate_cross_references` coordinator takes 9 arguments (one per entity
   collection plus two indices). Rather than introducing an intermediate wrapper
   struct just to satisfy clippy, the team used `#[allow(clippy::too_many_arguments)]`
   with an explanatory comment. This is the correct call when the function has a
   well-defined, non-extensible parameter set.

3. **Cycle detection via topological-order length comparison**: Rather than
   modifying `CascadeTopology::build()` to return `Result`, cycle detection is done
   in `SystemBuilder::build()` by comparing `cascade.topological_order().len()` to
   `self.hydros.len()`. The set difference (hydros not in the topo order) gives the
   cycle participants directly, with no DFS needed. See
   `crates/cobre-core/src/system.rs` around line 87.

4. **Validation phase ordering prevents false positives**: The four-phase ordering
   (duplicate check -> cross-reference validation -> topology + cycle check ->
   filling validation) prevents the cycle detector from seeing invalid references as
   phantom cycle participants. If cross-reference validation fails, build returns
   early before CascadeTopology is built.

5. **Integration test file as first external consumer**: Creating
   `crates/cobre-core/tests/integration.rs` immediately surfaced an issue: all
   existing unit-test helpers (`make_hydro`, `make_thermal`, etc.) used placeholder
   `bus_id = EntityId(0)` without supplying a corresponding Bus. Once cross-reference
   validation was live, every pre-existing unit test that built a System with hydros
   or thermals needed a `make_bus(0)` entry. The fix was systematic and improved
   test data quality.

6. **Dual-signature helper pattern for test helpers**: When existing
   `make_entity(id)` helpers needed to stay but also needed per-field control, the
   team introduced `make_entity_with_params(id, field1, field2)` and had the
   simple form delegate to it: `fn make_hydro(id) { make_hydro_on_bus(id, 0) }`.
   See `crates/cobre-core/src/system.rs` test module. This preserved backward
   compatibility with all existing tests while enabling new targeted tests.

## Issues Encountered

1. **Scope adherence score penalised for formatter-only changes**: ticket-009
   scored `scope_adherence = 0.0` because the diff touched 3 files beyond
   `system.rs`: `lib.rs` (import reorder by `rustfmt`), `error.rs` (line reflow
   by `rustfmt`), and `topology/cascade.rs` (import reorder). All changes were
   cosmetic, not semantic. The guardian scoring rule assigns 0.0 for two or more
   extra files, regardless of whether the changes are formatter-driven. Future
   tickets should explicitly list "formatter may reflow adjacent imports" as a
   known side-effect if the ticket targets a file near a heavily-imported module.

2. **Pre-existing unit tests broke when cross-reference validation went live**:
   Four tests in the `system.rs` unit test block (`test_lookup_by_id`,
   `test_lookup_missing_id`, `test_cascade_accessible`, `test_all_entity_lookups`)
   used `make_hydro(id)` which hardcodes `bus_id = EntityId(0)` but did not supply
   a Bus with id=0. After ticket-009, these tests started failing. The specialist
   correctly fixed them by adding `make_bus(0)` to each affected builder call. This
   was an expected but unplanned ripple effect of adding validation to a previously
   permissive builder.

3. **`const fn` constraint surfaced by adding `PartialEq`**: The `Send + Sync`
   compile-time check used `fn` (not `const fn`) closures. When `PartialEq` was
   added to `System` in ticket-011, an implicit promotion promoted the block to
   a `const` context that required `const fn`. The fix was to change
   `fn assert_send_sync` and `fn check` to `const fn` in the static check block.
   This is a minor but non-obvious interaction between const-context promotion and
   generic function definitions.

## Patterns Established

- **Validation order in `SystemBuilder::build()`**: duplicate check -> cross-reference
  validation (early return) -> topology construction + cycle detection ->
  filling config validation -> final error gate. Any future validation pass added
  to Phase 2 should be inserted before topology construction if it does not depend
  on topology, or between topology and the final gate if it does.
  File: `crates/cobre-core/src/system.rs`.

- **Per-entity-type validator functions**: Each entity type's cross-references are
  validated by a dedicated private function
  (`validate_line_refs`, `validate_hydro_refs`, etc.) that accepts only the slice
  and indices it uses. The umbrella `validate_cross_references` is the sole caller.
  This pattern makes it trivial to add validation for a new entity type.

- **`HashSet<EntityId>` for set-difference cycle detection**: Building a
  `HashSet` from the topological order and filtering the full hydro list is the
  established pattern for identifying cycle participants.
  File: `crates/cobre-core/src/system.rs`.

- **Integration tests in `tests/`**: External integration tests live in
  `crates/cobre-core/tests/integration.rs` and access only the public API.
  All test helpers are redefined locally (not shared with unit test helpers).
  This is the established pattern for all future integration tests in the crate.

- **`make_entity_on_field(id, field)` + `make_entity(id)` delegation**: When
  unit-test helpers need both a simple form and a parameterized form, define the
  parameterized version as the implementation and have the simple form call it.
  This avoids duplication and enables targeted tests without breaking existing tests.

## Files and Structures Created

- `crates/cobre-core/tests/integration.rs` -- New external integration test file.
  Contains 7 integration tests covering declaration-order invariance, realistic
  multi-entity system construction, validation rejection (cross-reference, cycle,
  filling config, diversion reference), and a large-order invariance test.

- `crates/cobre-core/src/system.rs` -- Extended with:
  - `PartialEq` on `System`
  - `validate_cross_references(...)` + 6 per-type helpers
  - `validate_filling_configs(...)`
  - Inline cycle detection block after `CascadeTopology::build()`
  - 15 new unit tests in the `#[cfg(test)] mod tests` block

## Conventions Adopted

- **`#[allow(clippy::too_many_arguments)]` on coordinator functions**: When a
  function is a thin coordinator that dispatches to sub-validators, the argument
  count matches the number of entity types and is not a design smell. Suppress
  the lint at the function level with a comment.

- **Error reason strings use the field name verbatim**: `ValidationError::InvalidFillingConfig`
  reasons use the exact field name in the message:
  `"filling_inflow_m3s must be positive"` and `"filling requires entry_stage_id to be set"`.
  This makes programmatic matching in tests reliable (`reason.contains("entry_stage_id")`).

- **Validation tests verify message content with `contains()`**: Tests for reason
  strings use `reason.contains("substring")` rather than exact string equality.
  This prevents test brittleness if the message is rephrased while preserving
  the key field name in the message.

## Surprises and Deviations

- **ticket-009 touched 3 files beyond `system.rs` due to `rustfmt`**: The declared
  scope was `system.rs` only. The formatter reflowed adjacent imports in `lib.rs`,
  `error.rs`, and `cascade.rs`. All changes were cosmetic. The guardian correctly
  scored `scope_adherence = 0.0` under the rubric, giving ticket-009 an overall
  quality score of 0.75. No code changes were incorrect; this was a scoring
  artefact of formatter side-effects in a strictly-interpreted scope rule.

- **ticket-011 added 2 tests beyond the 4 specified**: The spec required 4
  integration tests; the implementation delivered 7. The extra tests covered
  `test_large_order_invariance`, `test_invalid_filling_config_rejected`, and
  `test_diversion_invalid_reference_rejected`. These were directly motivated by
  the acceptance criteria of tickets-009 and 010 and validated code paths that
  the four specified tests did not exercise. The guardian scored ticket-011 at 1.0.

- **`const fn` promotion for Send + Sync check**: Adding `PartialEq` to `System`
  triggered a Rust 2024 edition constraint: the `const _: ()` block containing
  non-const closures required them to be `const fn`. This was not anticipated in
  the ticket. The specialist fixed it inline. See `crates/cobre-core/src/system.rs`
  around line 83.

## Recommendations for Future Epics

1. **Phase 2 validation additions**: When `cobre-io` adds stage-varying data, a
   new validation pass checking `entry_stage_id` / `exit_stage_id` ranges against
   actual stage counts should be inserted between cross-reference validation and
   topology construction. Follow the `validate_cross_references` decomposition
   pattern in `crates/cobre-core/src/system.rs`.

2. **Scope adherence and `rustfmt`**: When a ticket modifies a file that uses a
   large `use` block (like `lib.rs` re-exports), formatter side-effects will touch
   adjacent lines. Explicitly list those secondary files as "may be touched by
   formatter" in the Key Files section to avoid a 0.0 scope adherence score.

3. **Integration test helpers**: `crates/cobre-core/tests/integration.rs` now
   contains a complete set of public-API test helpers for all 7 entity types.
   Future integration tests in the same file can reuse these helpers directly
   without duplication. Do not make unit-test helpers public to share them.

4. **Declaration-order invariance tests are cheap to extend**: The pattern in
   `test_declaration_order_invariance` and `test_large_order_invariance` (build
   twice with different orderings, assert `==`) is low-cost and should be added
   to any future ticket that introduces new sorted entity collections or changes
   the canonical ordering logic.

## Metrics

| Ticket     | Readiness | Quality | Notes                                             |
| ---------- | --------- | ------- | ------------------------------------------------- |
| ticket-009 | 1.00      | 0.75    | scope_adherence=0.0 due to formatter side-effects |
| ticket-010 | 0.98      | 0.955   | test_delta weighted at 0.5 (add test coverage)    |
| ticket-011 | 0.98      | 1.00    | 7 integration tests, all pass                     |

Mean readiness: 0.987 | Mean quality: 0.902 | New tests added: 22 unit + 7 integration = 29

Phase 1 cumulative: 88 unit + 7 integration = 95 tests total, all passing.
