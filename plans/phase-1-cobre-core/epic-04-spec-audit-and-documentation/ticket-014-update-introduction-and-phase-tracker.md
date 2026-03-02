# ticket-014 Update Introduction and Phase Tracker

## Context

### Background

With Phase 1 complete (all 11 implementation tickets and the spec audit done), two project-level files need updating to reflect the current state: the software book's introduction page and the CLAUDE.md phase tracker table. These updates are the final step before Phase 1 is officially closed.

The CLAUDE.md file contains a phase tracker table (under "### Phase tracker") that currently shows all 8 phases as "not started". Phase 1 should be marked as complete. The "### Current phase" section should be updated to indicate that Phase 1 is done and Phases 2, 3, and 5 are the next candidates (they can proceed in parallel).

The `book/src/introduction.md` file says "Cobre is in early development. The architecture is specified, the crate boundaries are defined, and implementation is progressing through the 8-phase build sequence." This should be updated to reflect that Phase 1 (cobre-core) is complete.

### Relation to Epic

This is the third and final ticket of epic-04 (Spec Audit and Documentation). It depends on ticket-013 (the book page) because that ticket establishes the cobre-core documentation that the introduction can reference.

### Current State

`CLAUDE.md` phase tracker (lines 129-142):

```markdown
### Phase tracker

<!-- UPDATE THIS TABLE as phases are completed -->

| Phase | Status      | Notes                                        |
| ----- | ----------- | -------------------------------------------- |
| 1     | not started | All crates are empty stubs                   |
| 2     | not started | Blocked by Phase 1                           |
| 3     | not started | Can start after Phase 1 (parallel with 2, 5) |
| 4     | not started | Blocked by Phase 3                           |
| 5     | not started | Can start after Phase 1 (parallel with 2, 3) |
| 6     | not started | Blocked by Phases 1-5                        |
| 7     | not started | Blocked by Phase 6                           |
| 8     | not started | Blocked by Phase 7                           |
```

`CLAUDE.md` current phase section (lines 144-146):

```markdown
### Current phase

**Phase 1: cobre-core** -- Implementation begins with the foundation data model.
```

`book/src/introduction.md` current status section (lines 14-16):

```markdown
## Current status

Cobre is in early development. The architecture is specified, the crate boundaries are defined, and implementation is progressing through the [8-phase build sequence](https://cobre-rs.github.io/cobre-docs/specs/overview/implementation-ordering.html).
```

## Specification

### Requirements

1. **Update CLAUDE.md phase tracker table**: Change Phase 1 from "not started" to "complete" and update the Notes column from "All crates are empty stubs" to "Entity model, System, topology, validation, penalty resolution -- 95 tests". Update Phase 2 notes to "Ready to start" (no longer blocked). Update Phase 3 and 5 notes to "Ready to start" (no longer blocked).

2. **Update CLAUDE.md current phase section**: Change from "Phase 1: cobre-core -- Implementation begins with the foundation data model." to indicate that Phase 1 is complete and Phases 2, 3, and 5 are the next candidates.

3. **Update book/src/introduction.md**: Update the "Current status" section to reflect that Phase 1 (cobre-core) is complete, with a brief description of what it delivers. Keep the link to the 8-phase build sequence.

### Inputs/Props

- Current content of `CLAUDE.md` (the phase tracker and current phase sections)
- Current content of `book/src/introduction.md`

### Outputs/Behavior

- Modified `CLAUDE.md` with updated phase tracker and current phase sections
- Modified `book/src/introduction.md` with updated status section

### Error Handling

Not applicable -- these are simple text edits.

## Acceptance Criteria

- [ ] Given `CLAUDE.md` has Phase 1 status "not started", when this ticket is completed, then the phase tracker table shows Phase 1 as "complete" with notes mentioning the 95 tests.
- [ ] Given `CLAUDE.md` has Phases 2, 3, and 5 as "not started" with blocking notes, when this ticket is completed, then Phases 2, 3, and 5 show "Ready to start" (or equivalent) in their Notes column, with Phase 2 still listed as "not started" status but unblocked, and similarly for 3 and 5.
- [ ] Given `CLAUDE.md` current phase says "Phase 1: cobre-core -- Implementation begins", when this ticket is completed, then the current phase section indicates Phase 1 is complete and lists the next candidates (Phases 2, 3, 5).
- [ ] Given `book/src/introduction.md` says "Cobre is in early development", when this ticket is completed, then the status section mentions that cobre-core (Phase 1) is complete and describes what it delivers (entity model, system container, validation, topology).

## Implementation Guide

### Suggested Approach

1. Read `CLAUDE.md` and locate the phase tracker table and current phase section.
2. Edit the phase tracker table:
   - Phase 1: Status = "complete", Notes = "Entity model, System, topology, validation, penalty resolution -- 95 tests"
   - Phase 2: Notes = "Ready to start (depends on Phase 1, complete)"
   - Phase 3: Notes = "Ready to start (depends on Phase 1, complete; parallel with 2, 5)"
   - Phase 5: Notes = "Ready to start (depends on Phase 1, complete; parallel with 2, 3)"
3. Edit the current phase section to reflect Phase 1 completion.
4. Read `book/src/introduction.md` and edit the "Current status" section.
5. Run `mdbook build book/` to verify the introduction page renders correctly.

### Key Files to Modify

- `CLAUDE.md` -- phase tracker table (around lines 133-142) and current phase section (around lines 144-146)
- `book/src/introduction.md` -- current status section (around lines 14-16)

### Patterns to Follow

- Keep the phase tracker table format consistent with the existing markdown table
- Keep the introduction page concise -- one paragraph for the status update
- Use the established convention of referencing the build sequence link

### Pitfalls to Avoid

- Do NOT modify any sections of CLAUDE.md other than the phase tracker table and current phase section
- Do NOT modify any Rust source code
- Do NOT change the book SUMMARY.md
- Do NOT claim phases 2, 3, 5 are "in progress" -- they are "ready to start" but not started
- Do NOT change the phase tracker Status column for phases 2-8 -- they are still "not started" (only their Notes should indicate they are now unblocked)

## Testing Requirements

### Unit Tests

Not applicable -- documentation-only ticket.

### Integration Tests

Not applicable.

### E2E Tests (if applicable)

Verify that `mdbook build book/` completes successfully after the introduction update.

## Dependencies

- **Blocked By**: ticket-013 (the book page for cobre-core should exist before the introduction references it)
- **Blocks**: Nothing -- this is the final ticket of Phase 1

## Effort Estimate

**Points**: 1
**Confidence**: High
