# Decision: Keep cobre-mcp and cobre-tui as Workspace Stubs

**Date:** 2026-03-28
**Status:** Accepted
**Context:** Assessment v0.2.2 finding #15

## Question

Should `cobre-mcp` and `cobre-tui` remain in the workspace `members` list, or be excluded/feature-gated until implementation begins?

## Current State

- `cobre-mcp/src/main.rs`: 25 lines. `fn main() { todo!("...") }`. MCP server stub.
- `cobre-tui/src/lib.rs`: 31 lines. Doc comments describing architecture. No executable code.
- Both compile successfully and are listed in the workspace `members` array.
- `docs/ROADMAP.md` lists them under "Post-MVP Crates" with no target version.
- CI overhead: negligible (~2s incremental for both stubs combined).

## Options Considered

1. **Keep in workspace** (chosen) -- Zero-cost placeholders. CI overhead is ~2s.
   Preserves architectural intent and dependency graph visibility. No friction
   when development starts.

2. **Move to `exclude`** -- Removes from `cargo build --workspace` and CI.
   Requires re-adding when work starts. Risk of stale `Cargo.toml` metadata
   drifting unnoticed.

3. **Feature-gate** -- Add a `stub-crates` feature to workspace. Over-engineered
   for two crates with no code. Adds configuration complexity for no benefit.

## Decision

Option 1: keep both crates in the workspace as-is. Revisit when v0.3.0
planning begins and either implement or remove them.

## Consequences

- `cargo build --workspace` continues to compile both stubs (~2s).
- `cobre-mcp` binary panics at runtime with `todo!()`. Acceptable since
  no one should run it until implementation begins.
- No `Cargo.toml` changes needed.
- Both crates continue to validate their dependency declarations on every CI run,
  preventing silent metadata drift.
