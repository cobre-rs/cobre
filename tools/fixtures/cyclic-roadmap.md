# check-roadmap-dag fixture — three-row dependency cycle

**Do-not-touch list (fixture):** PD-004 (deferred pending a profile).

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — sddp

**CD-040 · Sev B · missing-seam · effort S · confidence high**

- **Station:** sddp
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** fixture entry.
- **Fix-shape:** none; fixture entry.
- **Alignment:** advances-0a (roadmap §IV.2)
- **Status:** ratified

**CD-041 · Sev B · asymmetry · effort M · confidence high**

- **Station:** sddp
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** fixture entry.
- **Fix-shape:** none; fixture entry.
- **Alignment:** advances-0a (roadmap §IV.2)
- **Status:** kept

**CD-042 · Sev C · leaky-boundary · effort M · confidence med**

- **Station:** sddp
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** fixture entry.
- **Fix-shape:** none; fixture entry.
- **Alignment:** advances-0a (roadmap §IV.2)
- **Status:** sharpened

**CD-045 · Sev B · bad-abstraction · effort L · confidence med**

- **Station:** sddp
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** fixture entry whose fix-shape contradicts the target layering; unscheduled here.
- **Fix-shape:** none; fixture entry.
- **Alignment:** conflicts (roadmap §IV.1)
- **Status:** rejected

**OD-011 · Sev C · duplication · effort S · confidence high**

- **Station:** sddp
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** fixture entry scheduled in the last wave.
- **Fix-shape:** none; fixture entry.
- **Alignment:** neutral
- **Status:** deferred

**OD-012 · Sev C · duplication · effort S · confidence high**

- **Station:** sddp
- **Baseline:** a136840d
- **Anchors:** `crates/cobre-io/src/output/policy/codec.rs`
- **Evidence:** fixture entry that the derived coverage case leaves unscheduled.
- **Fix-shape:** none; fixture entry.
- **Alignment:** neutral
- **Status:** dismissed

## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — unified-roadmap

### Milestones

| Milestone | Wave | Trigger |
| ---------- | ---- | ------- |
| gnl-import | 2 | GNL anticipated-coupling import lands |
| 0a | 4 | engine seam + `study` admission gate |
| 0b | 6 | cobre-model carved out of lp/ |
| 1 | 8 | cobre-core / cobre-solver purification |

### Waves

| Wave | Entry | Findings | Depends on | Phase | Effort | Trigger/deadline |
| ---- | ----- | -------- | ---------- | ----- | ------ | ---------------- |
| 1 | W1-state-family | CD-040 | W3-boundary-frame | serves 0a | S | before the GNL import |
| 2 | W2-output-owner | CD-041 | W1-state-family | serves 0a | M | with the shared writer work |
| 3 | W3-boundary-frame | CD-042 | W2-output-owner | serves 0a | M | with the 0a seam work |
| 5 | W5-late-cleanup | OD-011 | W3-boundary-frame | neutral | S | after the frame lands |
