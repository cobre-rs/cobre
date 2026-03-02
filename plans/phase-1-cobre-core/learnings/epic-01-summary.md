# Epic 01: Foundation Types -- Learnings

## What Worked Well

1. **Spec-first struct definitions**: Providing exact Rust code blocks in tickets minimized ambiguity. Specialists implemented field-by-field without deviation.

2. **Progressive ticket ordering**: EntityId first, then Bus/Line (simple), then Hydro (complex), then remaining entities. Each ticket built cleanly on the prior.

3. **Explicit derive lists in tickets**: Specifying `#[derive(Debug, Clone, Copy, PartialEq)]` vs `#[derive(Debug, Clone, PartialEq)]` prevented guesswork.

4. **Clippy pedantic compliance from the start**: All 4 tickets achieved zero clippy warnings on first pass (after doc-markdown fixes for unit brackets like `[MW]` and `[$/MWh]`).

## Issues Encountered

1. **Field count typo**: Ticket-003 acceptance criterion said "21 fields" but listed 22. The struct code block also had 22. Clarification was needed -- future tickets should double-check numeric counts against enumerated lists.

2. **Doc-markdown lint for unit brackets**: `[MW]` and `[$/MWh]` in doc comments trigger clippy `doc_markdown` warnings because `MWh` looks like an unresolved link. The fix is backtick-quoting: ``[`MW`]`` or using escaped brackets. This pattern recurs across all entity types.

3. **String::from vs .to_string() in tests**: Multiple specialists used `String::from("name")` which is verbose. Code simplifier normalized to `.to_string()`. Future tickets can mention this preference.

## Patterns Established

- **Module layout**: One file per entity type in `entities/`, re-exports through `entities/mod.rs` and `lib.rs`
- **Test pattern**: `#[cfg(test)] mod tests` at bottom of each entity file with `use super::*;`
- **Field documentation**: Units in backtick-wrapped brackets: ``/// Capacity [`MW`]``
- **No serde**: Confirmed that pure data structs with no serialization work cleanly
- **EntityId usage**: `EntityId::from(i32)` for construction in tests, `EntityId(raw)` for direct construction

## Metrics

| Ticket     | Readiness | Quality | Tests Added |
| ---------- | --------- | ------- | ----------- |
| ticket-001 | 0.94      | 0.95    | 12          |
| ticket-002 | 0.98      | 1.00    | 7           |
| ticket-003 | 0.98      | 1.00    | 10          |
| ticket-004 | 0.96      | 1.00    | 10          |

Mean readiness: 0.965 | Mean quality: 0.988 | Total tests: 39

## Recommendations for Epic 02

1. **Penalty resolution (ticket-005)**: Will need to reference `HydroPenalties` and global defaults. Ensure the ticket specifies the exact `GlobalPenaltyDefaults` struct fields.
2. **Cascade topology (ticket-006)**: Uses `Hydro.downstream_id` field. The `Option<EntityId>` pattern is established.
3. **Network topology (ticket-007)**: Uses Bus, Line, and all entity `bus_id` fields. All cross-reference types are now defined.
4. **SystemBuilder (ticket-008)**: All entity types are available. The builder can accept `Vec<Bus>`, `Vec<Line>`, etc.
