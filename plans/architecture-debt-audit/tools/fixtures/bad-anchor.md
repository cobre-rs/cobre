# check-anchors fixture — two anchors fail at the pinned baseline

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — bad-anchor

**CD-901 · Sev B · leaky-boundary · effort S · confidence high**
Valid anchor, resolves at the baseline:
`crates/cobre-solver/src/types.rs::StageTemplate`.
- **Alignment:** advances-1 (roadmap Part IV.1)

**CD-902 · Sev C · duplication · effort S · confidence low**
Path that does not exist at the baseline:
`crates/cobre-core/src/no_such_module.rs:12`.
Symbol absent from a file that does exist:
`crates/cobre-solver/src/types.rs::NoSuchTemplate`.
Prose backticks such as `god-fn` and `Sev A` are not anchors and must not be checked.
- **Alignment:** neutral
