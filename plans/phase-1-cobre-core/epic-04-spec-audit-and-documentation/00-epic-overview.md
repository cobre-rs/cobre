# Epic 04: Spec Audit and Documentation

## Goal

Verify that the `cobre-core` implementation fully and correctly implements everything required by the Phase 1 spec reading list, write software book pages documenting what was built, and update the project phase tracker to mark Phase 1 as complete. This epic closes the feedback loop between specification and implementation before the project advances to Phases 2-5.

## Scope

- **Spec audit**: Systematically compare the implementation in `crates/cobre-core/src/` against all 9 spec files in the Phase 1 reading list. Identify any gaps, deviations, missing fields, or missing validation rules. Produce a structured audit report documenting conformance and any findings.
- **Software book -- cobre-core page**: Rewrite `book/src/crates/core.md` from its current placeholder ("not yet implemented") to a comprehensive crate documentation page covering the entity model, SystemBuilder, validation pipeline, topology structures, penalty resolution, and public API.
- **Software book -- introduction update**: Update `book/src/introduction.md` to reflect that Phase 1 is complete and cobre-core is implemented.
- **Phase tracker update**: Update the CLAUDE.md phase tracker table to mark Phase 1 as complete.

## Non-Goals

- Fixing implementation gaps found during the audit (those would be separate tickets if needed)
- Writing documentation for crates other than cobre-core
- Updating cobre-docs spec files (those are the source of truth and do not change based on implementation)
- Writing user-guide pages (guide/\* files) -- those cover cross-crate workflows not yet available

## Tickets

1. **ticket-012**: Audit cobre-core implementation against Phase 1 spec reading list
2. **ticket-013**: Write cobre-core software book page
3. **ticket-014**: Update introduction and phase tracker

## Dependencies

- **Blocked By**: All of epic-03 (tickets 009-011, all completed)
- **Blocks**: Nothing -- this is the final epic of Phase 1

## Success Criteria

- Audit report documents conformance status for every spec section in the Phase 1 reading list
- `book/src/crates/core.md` contains a comprehensive crate documentation page (not a placeholder)
- `book/src/introduction.md` reflects Phase 1 completion
- `CLAUDE.md` phase tracker shows Phase 1 as complete
- `mdbook build book/` succeeds without errors
