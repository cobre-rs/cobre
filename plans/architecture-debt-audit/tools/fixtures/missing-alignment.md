# fields-check fixture — one missing Alignment, one invalid, one `conflicts`, one valid

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — missing-alignment

**CD-902 · Sev B · missing-seam · effort M · confidence high**

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** seeded: no Alignment bullet at all.
- **Fix-shape:** none; fixture entry.

**CD-903 · Sev C · duplication · effort S · confidence low**

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** seeded: Alignment spelled outside the vocabulary.
- **Fix-shape:** none; fixture entry.
- **Alignment:** maybe

**CD-904 · Sev B · leaky-boundary · effort L · confidence med**

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** seeded: `conflicts` is valid vocabulary and is held at the owner gate, not here.
- **Fix-shape:** none; fixture entry.
- **Alignment:** conflicts (roadmap §IV.1)

**CD-905 · Sev C · asymmetry · effort S · confidence high**

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** seeded: fully valid entry.
- **Fix-shape:** none; fixture entry.
- **Alignment:** advances-0a (roadmap §IV.2)
