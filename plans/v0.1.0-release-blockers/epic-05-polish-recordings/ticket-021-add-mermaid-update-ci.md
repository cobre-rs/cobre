# ticket-021: Add mdbook-mermaid and Update CI

## Context

### Background

The software book (`book/`) uses mdBook 0.5.2 but has no Mermaid diagram support. The architecture documentation and crate overview pages would benefit from embedded diagrams (e.g., the crate dependency graph, data flow diagrams). The `mdbook-mermaid` preprocessor (v0.17.0) renders Mermaid fenced code blocks into inline SVGs during book build.

Note: `mdbook-admonish` was skipped in ticket-015 due to incompatibility with mdBook 0.5.x. The `mdbook-mermaid` v0.17.0 binary uses a JS-based approach (injecting Mermaid JS into the page) rather than pre-rendering, so it is compatible with mdBook 0.5.x. This was verified locally: `cargo install mdbook-mermaid` installs v0.17.0 which provides `mdbook-mermaid install <dir>` and `mdbook-mermaid supports html`.

### Relation to Epic

Epic 05 is about polish for the v0.1.0 release. Adding Mermaid support enables richer documentation diagrams in the software book without manual SVG maintenance.

### Current State

- `book/book.toml` has no preprocessor entries -- only `[book]`, `[build]`, `[output.html]`, and `[output.html.fold]` sections
- No files in `book/src/` contain Mermaid fenced code blocks
- `.github/workflows/docs.yml` builds the book using `peaceiris/actions-mdbook@v2` with `mdbook-version: "latest"` and does not install any preprocessors
- The `cobre-docs` repository is out of scope for this ticket (this ticket covers only the software book in the `cobre` repo)

## Specification

### Requirements

1. Add `mdbook-mermaid` preprocessor configuration to `book/book.toml`
2. Update `.github/workflows/docs.yml` to install `mdbook-mermaid` before the book build step
3. Add one Mermaid diagram to an existing book page to verify the preprocessor works end-to-end (the crate overview page `book/src/crates/overview.md` is the natural place for the workspace dependency graph)

### Inputs/Props

- `book/book.toml` -- current book configuration
- `.github/workflows/docs.yml` -- current CI workflow for book deployment
- `book/src/crates/overview.md` -- crate overview page for the example diagram

### Outputs/Behavior

- `mdbook build book/` succeeds locally and renders Mermaid blocks as diagrams
- The docs CI workflow installs `mdbook-mermaid` and the book builds without errors
- The crate overview page contains a rendered dependency graph diagram

### Error Handling

- If `mdbook-mermaid` is not installed, `mdbook build book/` will fail with a clear preprocessor-not-found error -- this is expected and documented in the book README

## Acceptance Criteria

- [ ] Given `book/book.toml` is read, when inspecting its content, then it contains a `[preprocessor.mermaid]` section with `command = "mdbook-mermaid"`
- [ ] Given `.github/workflows/docs.yml` is read, when inspecting the `build` job steps, then there is a step that runs `cargo install mdbook-mermaid` (or equivalent) before the `mdbook build book/` step
- [ ] Given `book/src/crates/overview.md` is read, when inspecting its content, then it contains a fenced code block with language `mermaid` depicting the workspace crate dependency graph
- [ ] Given the command `mdbook build book/` is run locally with `mdbook-mermaid` installed, when the build completes, then the output directory `book/output/` contains the rendered overview page with a Mermaid diagram (the HTML contains `class="mermaid"` or a `<svg>` element from Mermaid rendering)

## Implementation Guide

### Suggested Approach

1. Run `mdbook-mermaid install book/` to auto-configure `book.toml` (this adds the `[preprocessor.mermaid]` section and copies the `mermaid-init.js` and `mermaid.min.js` files into `book/`). Verify the changes to `book.toml` and commit the generated files.
2. If `mdbook-mermaid install` adds files to `book/` that should be tracked, add them to git. If it modifies `book.toml` in unexpected ways, manually add only the `[preprocessor.mermaid]` section.
3. Add a Mermaid diagram to `book/src/crates/overview.md`. Use the dependency graph from `docs/PROJECT-STATUS.md` as the source, converting the ASCII art into a Mermaid `graph TD` block. Example:

```
\`\`\`mermaid
graph TD
    core[cobre-core]
    io[cobre-io]
    solver[cobre-solver]
    comm[cobre-comm]
    stochastic[cobre-stochastic]
    sddp[cobre-sddp]
    cli[cobre-cli]
    ferrompi[ferrompi]

    core --> io
    core --> stochastic
    core --> solver
    core --> comm
    ferrompi --> comm
    io --> sddp
    solver --> sddp
    comm --> sddp
    stochastic --> sddp
    sddp --> cli
\`\`\`
```

4. Update `.github/workflows/docs.yml` to install `mdbook-mermaid` before building. Add a step after the mdBook installation step:

```yaml
- name: Install mdbook-mermaid
  run: cargo install mdbook-mermaid --locked
```

5. Test locally: `mdbook build book/` and open `book/output/index.html` to verify the diagram renders.

### Key Files to Modify

- `book/book.toml` -- add mermaid preprocessor config
- `book/src/crates/overview.md` -- add example Mermaid diagram
- `.github/workflows/docs.yml` -- add mdbook-mermaid install step

### Patterns to Follow

- The `peaceiris/actions-mdbook@v2` action installs mdBook itself; preprocessors must be installed separately via `cargo install`
- Keep the `[preprocessor.mermaid]` section minimal -- the `mdbook-mermaid install` command generates the correct config

### Pitfalls to Avoid

- Do not modify the `cobre-docs` repository -- this ticket is scoped to the `cobre` software book only
- The `mdbook-mermaid install` command may add `additional-js` entries to `[output.html]` -- verify these are correct and do not conflict with existing `additional-css` entries
- Do not use a pinned version in `cargo install mdbook-mermaid` in CI unless there is a known compatibility issue -- `--locked` is sufficient for reproducibility
- Verify that `mdbook-mermaid supports html` returns exit code 0, confirming HTML renderer compatibility

## Testing Requirements

### Unit Tests

Not applicable -- this is configuration and documentation, not Rust code.

### Integration Tests

Not applicable.

### E2E Tests (if applicable)

Manual verification: run `mdbook build book/` locally and confirm the Mermaid diagram renders in the browser. The CI workflow will serve as the automated verification once merged.

## Dependencies

- **Blocked By**: None
- **Blocks**: ticket-023-final-review-pass.md

## Effort Estimate

**Points**: 1
**Confidence**: High
