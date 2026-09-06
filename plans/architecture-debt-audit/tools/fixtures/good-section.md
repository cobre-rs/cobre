# check-anchors fixture — every anchor resolves at the pinned baseline

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — good-section

**CD-801 · Sev B · leaky-boundary · effort S · confidence high**
Path-only anchor: `crates/cobre-solver/src/types.rs`.
Path-plus-line anchor: `crates/cobre-cli/src/commands/run/setup.rs:58`.
Symbol anchor with a parenthesised visibility:
`crates/cobre-cli/src/commands/run/setup.rs::resolve_thread_count`.
- **Alignment:** advances-0a (roadmap Part IV.2)

**CD-802 · Sev C · duplication · effort S · confidence med**
Symbol anchor on a plain `pub struct`: `crates/cobre-solver/src/types.rs::StageTemplate`.
Prose backticks such as `god-fn` and `Sev A` are not anchors and must not be checked.
- **Alignment:** neutral

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — sibling

_(no entries yet)_
