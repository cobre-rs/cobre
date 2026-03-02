# ticket-013 Write cobre-core Software Book Page

## Context

### Background

The Cobre project maintains a software book in `book/` (mdBook) containing user-facing documentation. Per CLAUDE.md: "Theory pages in cobre-docs are written ahead of implementation (they define the contract); software pages in `book/` are written during or after implementation (they describe what exists)." The current `book/src/crates/core.md` is a placeholder that says "This crate is not yet implemented." With Phase 1 complete across epics 01-03, cobre-core now has a full implementation: 7 entity types, `EntityId`, `System` container with `SystemBuilder`, `CascadeTopology`, `NetworkTopology`, penalty resolution, cross-reference validation, cascade cycle detection, filling config validation, and declaration-order invariance. The placeholder must be replaced with comprehensive documentation.

### Relation to Epic

This is the second ticket of epic-04 (Spec Audit and Documentation). It depends on ticket-012 (the spec audit) because the audit report confirms exactly what was implemented and identifies any deviations. The book page should describe the actual implementation, not the spec's target -- the audit bridges that gap.

### Current State

`book/src/crates/core.md` contains:

```markdown
# cobre-core

<span class="status-experimental">experimental</span>

This crate is not yet implemented. See the [specification](https://cobre-rs.github.io/cobre-docs/) for the target design.
```

The `book/src/SUMMARY.md` already has an entry for `cobre-core` pointing to `./crates/core.md`, so no SUMMARY changes are needed.

The `book/src/crates/overview.md` has a brief description: "Entity model (buses, hydros, thermals, lines)" but lacks detail.

## Specification

### Requirements

Rewrite `book/src/crates/core.md` to be a comprehensive crate documentation page. The page must cover:

1. **Status banner**: Change from "experimental / not yet implemented" to "experimental" only (the crate is implemented but the API may change).

2. **Overview section**: What cobre-core is, its role in the ecosystem (shared data model consumed read-only by all other crates), and the dual-nature design principle (clarity-first in cobre-core, performance-adapted views in cobre-sddp).

3. **Entity types section**: Description of each of the 7 entity types with their fields, organized by category:
   - **Fully modeled**: Bus, Line, Thermal, Hydro
   - **Stub entities**: PumpingStation, EnergyContract, NonControllableSource (type exists but contributes no LP variables in the minimal viable solver)

4. **Supporting types section**: Enums (`HydroGenerationModel`, `TailraceModel`, `HydraulicLossesModel`, `EfficiencyModel`, `ContractType`) and structs (`DeficitSegment`, `ThermalCostSegment`, `GnlConfig`, `DiversionChannel`, `FillingConfig`, `HydroPenalties`, `TailracePoint`).

5. **EntityId section**: The newtype pattern, why `i32` not `String`, why no `Ord`.

6. **System and SystemBuilder section**: How to construct a `System`, what validation the builder performs (duplicate detection, cross-reference validation, cascade cycle detection, filling config validation), what the builder produces (immutable `System` with canonical ordering and O(1) lookup).

7. **Topology section**: `CascadeTopology` (downstream/upstream adjacency, topological ordering) and `NetworkTopology` (bus-line incidence, bus generators/loads maps).

8. **Penalty resolution section**: Three-tier cascade (global defaults -> entity overrides -> stage overrides), what Phase 1 implements (first two tiers), the `GlobalPenaltyDefaults` struct, resolution functions.

9. **Validation section**: The validation phases in `SystemBuilder::build()` (duplicate check -> cross-reference -> topology + cycle -> filling config -> final gate), the error types.

10. **Design principles section**: Declaration-order invariance, canonical ID ordering, `Send + Sync` guarantee, immutability after construction.

11. **Public API summary section**: Listing the categories of methods on `System` (collection accessors, count queries, entity lookup by ID, topology accessors).

### Inputs/Props

- The audit report from ticket-012 (`plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md`)
- The source code in `crates/cobre-core/src/`
- The 3 epic learnings files

### Outputs/Behavior

A rewritten `book/src/crates/core.md` that replaces the placeholder content. The file must be valid mdBook markdown and render correctly when `mdbook build book/` is run.

### Error Handling

Not applicable -- this is a documentation task.

## Acceptance Criteria

- [ ] Given the file `book/src/crates/core.md` exists with placeholder content, when this ticket is completed, then `book/src/crates/core.md` contains at least 150 lines of documentation content covering all 11 sections listed in Requirements.
- [ ] Given the rewritten `book/src/crates/core.md`, when `mdbook build book/` is run from the repo root, then the build succeeds with exit code 0 and no "file not found" or broken link warnings for internal book links.
- [ ] Given the rewritten `book/src/crates/core.md`, when the EntityId section is inspected, then it describes the newtype pattern, the inner `i32` type, and why `Ord` is not derived.
- [ ] Given the rewritten `book/src/crates/core.md`, when the System section is inspected, then it describes `SystemBuilder` with its 4 validation phases (duplicate, cross-reference, cascade cycle, filling config).
- [ ] Given the rewritten `book/src/crates/core.md`, when the Entity types section is inspected, then all 7 entity types are listed, with Hydro having the most detail (it is the most complex entity type with 22 fields).

## Implementation Guide

### Suggested Approach

1. Read the audit report from ticket-012 to understand exactly what was implemented and any deviations from spec.
2. Read `crates/cobre-core/src/lib.rs` to see the public API surface (re-exports).
3. Read `crates/cobre-core/src/system.rs` to understand the `System` public API and `SystemBuilder` validation pipeline.
4. Write the page section by section, following the order in the Requirements list.
5. Use Rust code blocks (` ```rust `) for type signatures and example usage.
6. Reference the methodology specification at `https://cobre-rs.github.io/cobre-docs/` for deeper details, but ensure the book page is self-contained for understanding the crate's purpose and API.
7. Run `mdbook build book/` to verify the page renders correctly.

### Key Files to Modify

- `book/src/crates/core.md` -- rewrite from placeholder to comprehensive documentation

### Key Files to Read

- `plans/phase-1-cobre-core/epic-04-spec-audit-and-documentation/audit-report.md` (from ticket-012)
- `crates/cobre-core/src/lib.rs`
- `crates/cobre-core/src/entity_id.rs`
- `crates/cobre-core/src/entities/hydro.rs`
- `crates/cobre-core/src/entities/bus.rs`
- `crates/cobre-core/src/entities/thermal.rs`
- `crates/cobre-core/src/entities/line.rs`
- `crates/cobre-core/src/entities/energy_contract.rs`
- `crates/cobre-core/src/entities/pumping_station.rs`
- `crates/cobre-core/src/entities/non_controllable.rs`
- `crates/cobre-core/src/system.rs`
- `crates/cobre-core/src/topology/cascade.rs`
- `crates/cobre-core/src/topology/network.rs`
- `crates/cobre-core/src/penalty.rs`
- `crates/cobre-core/src/error.rs`

### Patterns to Follow

- Follow the style of other mdBook documentation sites: clear headings, brief introductions before details, code examples where helpful.
- Use tables for summarizing entity fields rather than listing every field in prose.
- Use the `<span class="status-experimental">experimental</span>` banner pattern already established in the book.
- Reference rustdoc with `cargo doc --workspace --no-deps --open` for API-level detail rather than duplicating full method signatures.
- Keep the tone factual and concise -- the book describes what exists, not what is planned.

### Pitfalls to Avoid

- Do NOT describe Phase 2+ features as if they are implemented (e.g., serde, stage-varying penalties, `ResolvedPenalties`, `Stage`, `PolicyGraph`)
- Do NOT modify `book/src/SUMMARY.md` -- the entry for `crates/core.md` already exists
- Do NOT modify any Rust source code -- this ticket is documentation-only
- Do NOT write guide-level tutorials (those belong in `book/src/guide/` and require cross-crate features)
- Do NOT use emojis in the documentation

## Testing Requirements

### Unit Tests

Not applicable -- documentation-only ticket.

### Integration Tests

Not applicable.

### E2E Tests (if applicable)

Verify that `mdbook build book/` completes successfully after the page is written.

## Dependencies

- **Blocked By**: ticket-012 (the audit report informs what to document)
- **Blocks**: ticket-014 (introduction update references cobre-core as implemented)

## Effort Estimate

**Points**: 2
**Confidence**: High
