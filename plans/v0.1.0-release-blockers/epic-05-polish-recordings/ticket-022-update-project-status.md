# ticket-022: Update PROJECT-STATUS.md

## Context

### Background

`docs/PROJECT-STATUS.md` was last comprehensively updated after Phase 8 completion and then partially updated by ticket-026 (which added the "Inflow non-negativity: penalty method only" section and a truncation deferral note). Since then, Epic 01 (rayon thread parallelism) has been completed, adding intra-rank thread parallelism to the forward pass, backward pass, and simulation pipeline. The "Critical Gaps for v0.1.0" section still lists intra-rank thread parallelism as "BLOCKING for release" even though it has been implemented.

Epic 03 (4ree example case) was deferred entirely -- the user will prepare it manually post-v0.1.0. The document should not reference 4ree as a completed milestone.

### Relation to Epic

Epic 05 is the final polish pass. Updating PROJECT-STATUS.md ensures the document accurately reflects the state of the codebase at the v0.1.0 tag point.

### Current State

- `docs/PROJECT-STATUS.md` exists with sections: Overall Status, Per-Crate Implementation Status, Test Coverage Summary, Key Milestones Achieved (phases 1-8), Dependency Graph, Critical Gaps for v0.1.0, Links
- The "Critical Gaps" section has two subsections: "Intra-rank thread parallelism (BLOCKING for release)" and "Inflow non-negativity: penalty method only"
- The test count says "1851 tests" but the workspace now has 1947 tests (after Epic 01 rayon parallelism changes)
- `CLAUDE.md` phase tracker still says "Critical gap: intra-rank thread parallelism" as BLOCKING -- this should also be updated
- `docs/ROADMAP.md` already exists (created by ticket-026) with sections for deferred work

## Specification

### Requirements

1. Update the "Critical Gaps for v0.1.0" section:
   - Mark "Intra-rank thread parallelism" as RESOLVED -- rayon-based thread parallelism is now implemented for forward pass, backward pass, and simulation pipeline. Reference the `--threads` CLI flag and `COBRE_THREADS` env var.
   - Keep the "Inflow non-negativity: penalty method only" subsection (it was updated by ticket-026 and accurately describes the current state)
2. Update the workspace test count from 1851 to the current count (1947 as of this writing -- verify by running `cargo test --workspace --all-features`)
3. Update the `cobre-sddp` test count in the "Test Coverage Summary" table (previously 479, now higher after Epic 01 added determinism tests and parallelism tests)
4. Update the `cobre-cli` test count in the "Test Coverage Summary" table (previously 110, now higher after Epic 01 added `--threads` flag tests)
5. Add a "v0.1.0 Release Blockers" subsection under "Key Milestones Achieved" documenting: rayon thread parallelism (Epic 01), inflow non-negativity penalty method (Epic 02), documentation completion (Epic 04)
6. Update `CLAUDE.md` to remove the "Critical gap: intra-rank thread parallelism" section (or mark it as resolved)

### Inputs/Props

- Current `docs/PROJECT-STATUS.md` content
- Current `CLAUDE.md` content (phase tracker section)
- Output of `cargo test --workspace --all-features` for accurate test counts

### Outputs/Behavior

- `docs/PROJECT-STATUS.md` accurately reflects the v0.1.0 state: no false "blocking" gaps, correct test counts
- `CLAUDE.md` no longer claims intra-rank parallelism is blocking

### Error Handling

Not applicable -- this is a documentation-only change.

## Acceptance Criteria

- [ ] Given `docs/PROJECT-STATUS.md` is read, when searching for "BLOCKING for release", then that phrase does not appear in the document
- [ ] Given `docs/PROJECT-STATUS.md` is read, when searching for "Intra-rank thread parallelism", then the section is marked as resolved (contains "Resolved" or "Implemented") and mentions `--threads` CLI flag
- [ ] Given `docs/PROJECT-STATUS.md` is read, when searching for the workspace test total, then the number matches the output of `cargo test --workspace --all-features 2>&1 | grep "test result:" | awk -F'[; ]' '{sum += $4} END {print sum}'` (currently 1947)
- [ ] Given `CLAUDE.md` is read, when searching for "BLOCKING for v0.1.0 release", then that phrase does not appear in the "Critical gap" section (the section is either removed or updated to say "Resolved")

## Implementation Guide

### Suggested Approach

1. Run `cargo test --workspace --all-features` and record the per-crate test counts from the `test result:` lines
2. Edit `docs/PROJECT-STATUS.md`:
   a. Change the "Intra-rank thread parallelism" subsection heading from "(BLOCKING for release)" to "(RESOLVED)". Replace the body with a brief description: rayon-based parallelism implemented in Epic 01 for forward pass, backward pass, and simulation; controlled via `--threads N` CLI flag or `COBRE_THREADS` env var; default is 1 thread (conservative)
   b. Update the test count in "Overall Status" and in each row of the "Test Coverage Summary" table
   c. Add a subsection under "Key Milestones Achieved" for the v0.1.0 release blocker work
3. Edit `CLAUDE.md`:
   a. In the "Critical gap: intra-rank thread parallelism" section, update text to indicate it is resolved
   b. Update the phase tracker comments if needed

### Key Files to Modify

- `docs/PROJECT-STATUS.md` -- update critical gaps, test counts, add milestone
- `CLAUDE.md` -- update critical gap section

### Patterns to Follow

- Maintain the existing document structure and heading hierarchy in `PROJECT-STATUS.md`
- Use the same table format for test counts as the existing "Test Coverage Summary" table
- Keep factual, concise descriptions in milestone entries (match the style of existing Phase 1-8 entries)

### Pitfalls to Avoid

- Do not claim 4ree example is completed -- Epic 03 was deferred
- Do not modify the "Inflow non-negativity" subsection -- it was already updated by ticket-026
- Do not change sections of `CLAUDE.md` outside the critical gap and phase tracker areas
- Run `cargo test --workspace --all-features` to get actual test counts rather than guessing

## Testing Requirements

### Unit Tests

Not applicable -- documentation-only change.

### Integration Tests

Not applicable.

### E2E Tests (if applicable)

Not applicable.

## Dependencies

- **Blocked By**: Epic 01 (completed), Epic 02 (completed)
- **Blocks**: ticket-023-final-review-pass.md, ticket-025-document-deferred-hpc-optimizations-roadmap.md

## Effort Estimate

**Points**: 1
**Confidence**: High
