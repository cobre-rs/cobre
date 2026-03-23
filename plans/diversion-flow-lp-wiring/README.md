# Diversion Flow LP Wiring

Wires diversion flow variables into the SDDP LP formulation. The data model,
input loading, penalty resolution, and simulation output schema are all in place.
This plan adds H*K dense diversion columns between spillage and thermal, couples
them into the water balance (outflow and inflow terms), wires the generic
constraint resolver, and extracts diversion values for simulation output.

**Design reference:** `docs/designs/diversion-flow-lp-wiring.md`

## Tech Stack

- **Rust** (cobre-sddp): LP builder changes in `lp_builder.rs`, `indexer.rs`;
  generic constraint wiring in `generic_constraints.rs`; simulation extraction
  in `simulation/extraction.rs`; deterministic test in `tests/deterministic.rs`

## Epics

| Epic | Name | Tickets | Detail Level | Status |
|------|------|---------|-------------|--------|
| epic-01 | Indexer and Layout | 2 | Detailed | Pending |
| epic-02 | LP Builder Columns and Water Balance | 3 | Detailed | Pending |
| epic-03 | Extraction and Generic Constraints | 2 | Outline | Pending |
| epic-04 | Integration Testing | 2 | Outline | Pending |

## Progress Tracking

| Ticket | Title | Epic | Status | Detail Level | Readiness | Quality | Badge |
|--------|-------|------|--------|-------------|-----------|---------|-------|
| ticket-001 | Add diversion range to StageIndexer | epic-01 | completed | Detailed | 0.94 | -- | -- |
| ticket-002 | Add col_diversion_start to StageLayout | epic-01 | completed | Detailed | 0.90 | -- | -- |
| ticket-001 | Set diversion column bounds and objective | epic-02 | completed | Detailed | 0.92 | -- | -- |
| ticket-002 | Precompute diversion upstream map | epic-02 | completed | Detailed | 0.92 | -- | -- |
| ticket-003 | Add diversion water balance entries | epic-02 | completed | Detailed | 0.90 | -- | -- |
| ticket-001 | Wire HydroDiversion in generic constraint resolver | epic-03 | completed | refined | -- | -- | -- |
| ticket-002 | Extract diversion from simulation LP primal | epic-03 | completed | refined | -- | -- | -- |
| ticket-001 | Add D17 diversion deterministic test | epic-04 | pending | Outline | -- | -- | -- |
| ticket-002 | Verify existing D01-D16 regression suite | epic-04 | pending | Outline | -- | -- | -- |

## Dependency Graph

```
epic-01/ticket-001 (independent -- StageIndexer extension)
     |
     v
epic-01/ticket-002 (depends on ticket-001 for diversion range in indexer)
     |
     v
epic-02/ticket-001 ----+---- epic-02/ticket-002
(column bounds)        |     (upstream map)
     |                 |          |
     v                 v          v
          epic-02/ticket-003
     (water balance entries, depends on both ticket-001 and ticket-002)
               |
               v
     epic-03/ticket-001 (generic constraint resolver)
               |
               v
     epic-03/ticket-002 (simulation extraction)
               |
               v
     epic-04/ticket-001 (D17 deterministic test)
               |
               v
     epic-04/ticket-002 (regression suite verification)
```

## Execution Order

1. `epic-01/ticket-001` -- Add diversion range to StageIndexer
2. `epic-01/ticket-002` -- Add col_diversion_start to StageLayout
3. `epic-02/ticket-001` -- Set diversion column bounds and objective (parallel with ticket-002)
4. `epic-02/ticket-002` -- Precompute diversion upstream map (parallel with ticket-001)
5. `epic-02/ticket-003` -- Add diversion water balance entries
6. `epic-03/ticket-001` -- Wire HydroDiversion in generic constraint resolver
7. `epic-03/ticket-002` -- Extract diversion from simulation LP primal
8. `epic-04/ticket-001` -- Add D17 diversion deterministic test
9. `epic-04/ticket-002` -- Verify existing D01-D16 regression suite

Epic-02 tickets 001 and 002 can execute in parallel after epic-01/ticket-002.
Ticket-003 depends on both ticket-001 and ticket-002.

## Key Constraints

1. **Dense H*K columns** -- all hydros get diversion columns even without diversion (bounds [0,0] presolve-eliminated)
2. **Cache locality** -- diversion placed between spillage and thermal for contiguous access in water balance loop
3. **Mechanical column shift** -- thermal and all subsequent columns shift by +H*K (same pattern as prior layout refactors)
4. **No new rows** -- diversion enters existing water balance rows only
5. **Sign convention** -- +tau for outflow (same as turbine/spillage), -tau for inflow (same as cascade upstream)
6. **Existing D01-D16 tests must not change** -- none have diversion; extra [0,0] columns are presolve-eliminated
