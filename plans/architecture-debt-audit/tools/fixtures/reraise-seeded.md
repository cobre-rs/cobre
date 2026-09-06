# check-reraise fixture — two unjustified re-raises, one justified, one clean

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — reraise-seeded

**CD-900 · Sev C · duplication · effort S · confidence high**

The parallel and chronological water-entry fillers share a skeleton.

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-sddp/src/lp/builder/entries.rs::fill_parallel_water_entries`
- **Evidence:** seeded: the retracted duplication surface, raised again without a Re-raise-of line.
- **Fix-shape:** none; fixture entry.
- **Alignment:** neutral (roadmap §IV.3)

**CD-901 · Sev C · duplication · effort S · confidence high**

Superseded cut-sync public methods are dead surface.

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-sddp/src/cut/cut_sync.rs`
- **Evidence:** seeded: token-overlaps the mirror heading "Superseded cut-sync public methods".
- **Fix-shape:** none; fixture entry.
- **Alignment:** neutral (roadmap §IV.3)

**CD-902 · Sev C · duplication · effort S · confidence high**

The water-entry fillers duplicate their entity-iteration skeleton.

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-sddp/src/lp/builder/entries.rs::fill_parallel_water_entries`
- **Evidence:** seeded: same surface, but justified below.
- **Fix-shape:** none; fixture entry.
- **Alignment:** neutral (roadmap §IV.3)
- **Re-raise-of:** CD-008 — the retraction was scoped to the two LP formulations; this is the shared iteration order only.

**CD-903 · Sev C · asymmetry · effort S · confidence med**

Thread-count resolution lives in the CLI setup module only.

- **Station:** fixture
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-cli/src/commands/run/setup.rs::resolve_thread_count`
- **Evidence:** seeded: clean entry, overlaps no retired item.
- **Fix-shape:** none; fixture entry.
- **Alignment:** advances-0a (roadmap §V.1)
