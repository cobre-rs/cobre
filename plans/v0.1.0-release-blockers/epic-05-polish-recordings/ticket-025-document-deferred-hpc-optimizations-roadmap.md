# ticket-025: Document Deferred HPC Optimizations in Roadmap

## Context

### Background

The v0.1.0 release implements baseline thread parallelism via rayon (Epic 01), but several HPC performance optimizations are intentionally deferred. These optimizations represent significant future improvement opportunities and should be documented in the roadmap to set expectations and avoid knowledge loss. The `docs/ROADMAP.md` file was created by ticket-026 and already has an "HPC Optimizations" section with three bullet points (NUMA-aware allocation, work-stealing tuning, MPI+threads interaction). This section needs to be expanded with the full set of deferred optimizations.

### Relation to Epic

Epic 05 is the final polish for v0.1.0. This ticket ensures the roadmap captures all known performance improvement opportunities, providing a clear path for post-release HPC work.

### Current State

- `docs/ROADMAP.md` exists with four sections: "Inflow Truncation Methods", "HPC Optimizations", "Post-MVP Crates", "Algorithm Extensions"
- The "HPC Optimizations" section currently has 3 items: NUMA-aware allocation, work-stealing tuning, MPI+threads interaction
- The software book at `book/src/SUMMARY.md` does not have a roadmap entry -- there is no link from the book to `docs/ROADMAP.md`
- rayon-based thread parallelism is the baseline (implemented in Epic 01), controlled via `--threads` flag and `COBRE_THREADS` env var

## Specification

### Requirements

1. Expand the "HPC Optimizations" section in `docs/ROADMAP.md` to include all 8 deferred optimizations: NUMA-aware thread/memory placement, MPI one-sided communication, OpenMP interop evaluation, GPU acceleration, asynchronous cut exchange, profile-guided optimization (PGO), SIMD vectorization for hot paths, memory pool/arena allocation
2. Each optimization entry must include: brief description (2-3 sentences), expected benefit/impact, prerequisites or dependencies on other work
3. Add a priority ordering note: distinguish near-term (likely v0.1.x/v0.2.x) from longer-term (v0.3+) items
4. Add a "Roadmap" link from the software book -- either a new page `book/src/reference/roadmap.md` that links to `docs/ROADMAP.md`, or a cross-reference from an existing reference page

### Inputs/Props

- Current `docs/ROADMAP.md` content
- The 8 optimization items from the original outline ticket
- Epic 01 learnings (rayon baseline, partition strategy, per-thread workspaces)

### Outputs/Behavior

- `docs/ROADMAP.md` "HPC Optimizations" section is comprehensive with all 8 items, each with description, expected impact, and prerequisites
- The software book links to the roadmap from the reference section

### Error Handling

Not applicable -- documentation-only change.

## Acceptance Criteria

- [ ] Given `docs/ROADMAP.md` is read, when counting the bullet-point items under "HPC Optimizations", then there are at least 8 distinct optimization items
- [ ] Given `docs/ROADMAP.md` is read, when inspecting each optimization entry, then each one contains at minimum: a bold title, a description sentence, and an "Expected impact" or "Prerequisites" note
- [ ] Given `docs/ROADMAP.md` is read, when searching for priority grouping, then the section distinguishes near-term (v0.1.x/v0.2.x) items from longer-term (v0.3+) items
- [ ] Given `book/src/SUMMARY.md` is read, when searching for "roadmap" (case-insensitive), then there is a link to a roadmap-related page in the book

## Implementation Guide

### Suggested Approach

1. Edit `docs/ROADMAP.md` and expand the "HPC Optimizations" section. Keep the existing 3 items but expand their descriptions and add the 5 missing items. Organize into two priority tiers:

   **Near-term (v0.1.x / v0.2.x):**
   - NUMA-aware thread/memory placement (highest single-node impact)
   - Work-stealing tuning (profile rayon granularity)
   - Memory pool / arena allocation (reduce per-iteration allocation pressure)
   - Profile-guided optimization (PGO) builds (free performance from compiler)
   - MPI + threads interaction validation (correctness prerequisite)

   **Longer-term (v0.3+):**
   - SIMD vectorization for cut evaluation hot paths
   - Asynchronous cut exchange (overlap communication/computation)
   - GPU acceleration (requires GPU solver backends)
   - MPI one-sided communication (MPI_Put/MPI_Get for cuts)
   - OpenMP interop evaluation

2. For each item, write: title (bold), 2-3 sentence description, "Expected impact" line, "Prerequisites" line.

3. Create `book/src/reference/roadmap.md` with a brief introduction and a link to `docs/ROADMAP.md` (or the GitHub URL `https://github.com/cobre-rs/cobre/blob/main/docs/ROADMAP.md`). Add it to `book/src/SUMMARY.md` under the Reference section.

4. Run `mdbook build book/` to verify the new page renders correctly.

### Key Files to Modify

- `docs/ROADMAP.md` -- expand HPC Optimizations section
- `book/src/reference/roadmap.md` (new file) -- roadmap cross-reference page
- `book/src/SUMMARY.md` -- add roadmap link under Reference section

### Patterns to Follow

- Match the existing `docs/ROADMAP.md` formatting style (Markdown with `---` separators between major sections, bold item titles, indented sub-notes)
- Match the existing `book/src/reference/` page style for the new reference page
- Use spec references where applicable (e.g., `cobre-docs/src/specs/hpc/parallelism-model.md`)

### Pitfalls to Avoid

- Do not modify the cobre-docs repository -- this ticket only touches files in the cobre repo
- Do not remove or restructure the existing 4 sections in `docs/ROADMAP.md` -- only expand the "HPC Optimizations" section and add the book cross-reference
- Do not duplicate the full ROADMAP.md content into the book page -- the book page should be a brief pointer with a link

## Testing Requirements

### Unit Tests

Not applicable -- documentation-only change.

### Integration Tests

Not applicable.

### E2E Tests (if applicable)

Manual verification: `mdbook build book/` succeeds and the reference/roadmap page renders with a working link.

## Dependencies

- **Blocked By**: ticket-022-update-project-status.md (coordinate to avoid conflicting edits to docs/)
- **Blocks**: ticket-023-final-review-pass.md

## Effort Estimate

**Points**: 1
**Confidence**: High
