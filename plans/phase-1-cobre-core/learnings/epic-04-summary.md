# Phase 1 cobre-core -- Accumulated Learnings (through Epic 04)

## Architecture and Design Decisions

- `cobre-core` is the zero-dependency shared data model; all other crates consume it read-only
- Enum dispatch over `Box<dyn Trait>` for all closed variant sets (DEC-001)
- `EntityId(i32)` newtype: cheap hash/copy, no `Ord` (explicit `sort_by_key(|e| e.id.0)` at sort sites)
- `Send + Sync` enforced at compile time via `const fn assert_send_sync::<System>()` in `const _: ()` block
- Clarity-first representation in `cobre-core`; performance-adapted views in `cobre-sddp` (not yet built)
- `HydroPenalties` stores 11 pre-resolved `f64` fields; `HydroPenaltyOverrides` mirrors with `Option<f64>` for loading

## Patterns Confirmed Across Epics

- **Module layout**: one file per entity in `entities/`, re-exported via `entities/mod.rs` and `lib.rs`
- **Test pattern**: `#[cfg(test)] mod tests` at bottom of each file with `use super::*;`
- **Field docs**: units in backtick-wrapped brackets -- ``/// Capacity [`MW`]`` (prevents clippy `doc_markdown` warnings)
- **HasId private trait**: used for generic `build_index` and `check_duplicates` on entity collections
- **Topology construction**: `Topology::build(&[Entity])` pattern -- takes borrowed slices, returns owned topology
- **Validation order in `SystemBuilder::build()`**: duplicate check > cross-reference > cascade topology + cycle > filling config; insert new passes before topology if topology-independent
- **Per-entity validator functions**: `validate_hydro_refs`, `validate_line_refs`, etc., each taking only the slices they need, called by `validate_cross_references` coordinator
- **Cycle detection**: compare `cascade.topological_order().len()` to `hydros.len()`; set-difference gives participants
- **Integration tests**: `crates/cobre-core/tests/integration.rs` uses only public API; helpers are local, not shared with unit tests
- **Audit rubric (five-level)**: CONFORM / NAMING / EXTRA / GAP / OUT-OF-SCOPE for spec compliance audits

## Key Files

- `crates/cobre-core/src/entity_id.rs` -- `EntityId` newtype
- `crates/cobre-core/src/entities/` -- 7 entity files (bus, line, hydro, thermal, pumping_station, energy_contract, non_controllable)
- `crates/cobre-core/src/system.rs` -- `System`, `SystemBuilder`, all validation passes
- `crates/cobre-core/src/topology/cascade.rs` -- `CascadeTopology` (Kahn's algorithm, sorted ready queue)
- `crates/cobre-core/src/topology/network.rs` -- `NetworkTopology`, `BusLineConnection`, `BusGenerators`, `BusLoads`
- `crates/cobre-core/src/penalty.rs` -- `GlobalPenaltyDefaults`, `HydroPenaltyOverrides`, 5 resolution functions
- `crates/cobre-core/src/error.rs` -- `ValidationError` (6 variants; `InvalidPenalty` defined but not triggered -- GAP-001)
- `crates/cobre-core/tests/integration.rs` -- 7 integration tests (order invariance, validation rejection)
- `book/src/crates/core.md` -- 476-line crate documentation page (design principles, entities, API, code examples)
- `plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md` -- 808-line Phase 1 conformance audit

## Conventions Adopted

- `String::from("x")` anti-pattern -- use `"x".to_string()` in tests
- `#[allow(clippy::too_many_arguments)]` on coordinator validators with a comment explaining why
- `#[allow(clippy::struct_field_names)]` on `NetworkTopology` (fields all share `bus_` prefix -- intentional)
- Error reason strings use the exact field name verbatim; tests use `reason.contains("field_name")`
- Edition 2024 closures: `|(_, &deg)| deg == 0` is disallowed; use `|&(_, deg)| *deg == 0`
- `const fn` required for functions in `const _: ()` blocks (Rust 2024 edition constraint)
- Phase tracker test count comes from `cargo test --workspace` output, not from accumulated learnings records
- Book status banner stays `experimental` for implemented crates until the API is stabilized

## Recurring Issues and Fixes

- **Formatter side-effects inflate scope_adherence score to 0.0**: `rustfmt` touching adjacent imports in `lib.rs`/`error.rs` when the declared scope is `system.rs` only. Fix: explicitly list secondary files in ticket's Key Files if they are likely to be reformatted.
- **Unit tests break when validation goes live**: tests using `make_hydro(id)` with `bus_id=0` but no Bus in the builder will fail once cross-reference validation is active. Fix: add `make_bus(0)` to each affected builder call.
- **`const fn` promotion on `PartialEq` addition**: adding `PartialEq` to `System` triggers promotion of closures in `const _: ()` to require `const fn`. Fix: change `fn assert_send_sync` and `fn check` to `const fn`.
- **Numeric counts in acceptance criteria**: always double-check field counts against the enumerated list in the code block (ticket-003 said "21 fields" but listed 22).

## Known Gaps (carry forward to Phase 2)

- **GAP-001**: `ValidationError::InvalidPenalty` is defined in `crates/cobre-core/src/error.rs` but `SystemBuilder::build()` does not check for negative penalty values. Phase 2 (`cobre-io`) should trigger this in the entity deserialization/validation step.
- **`DisconnectedBus` check deferred**: `ValidationError::DisconnectedBus` variant exists; the check requires network topology context and belongs in the `cobre-io` load pipeline (Phase 2).
- **Stage range validation deferred**: `entry_stage_id`/`exit_stage_id` range checks require actual stage count from `cobre-io` (Phase 2).

## Quality Scores by Epic

| Epic | Tickets | Mean Readiness | Mean Quality | Tests Added                 |
| ---- | ------- | -------------- | ------------ | --------------------------- |
| 01   | 4       | 0.965          | 0.988        | 39                          |
| 02   | 4       | 0.990          | 1.000        | 49                          |
| 03   | 3       | 0.987          | 0.902        | 16 (unit) + 7 (integration) |
| 04   | 3       | 1.000          | 1.000        | 0 (documentation only)      |

Phase 1 total: 108 tests, all passing. Spec conformance: 89.3% exact CONFORM, 9.6% intentional NAMING differences, 0.6% minor GAP.
