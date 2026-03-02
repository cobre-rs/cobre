# Phase 1: cobre-core Data Model and Registries

## Overview

Phase 1 implements the `cobre-core` crate -- the foundation data model for the Cobre SDDP solver ecosystem. This crate defines all entity types, the System container, topology structures, and entity-level penalty resolution. It has zero in-workspace dependencies and is consumed read-only by all other Cobre crates.

## Tech Stack

- Rust 2024 edition, MSRV 1.85
- No external dependencies (cobre-core is dependency-free)
- Workspace lints: `unsafe_code = "forbid"`, `missing_docs = "warn"`, `clippy::pedantic`, `unwrap_used = "deny"`

## Epics

| Epic    | Name                         | Tickets     | Detail Level | Status    |
| ------- | ---------------------------- | ----------- | ------------ | --------- |
| epic-01 | Foundation Types             | 4 (001-004) | Detailed     | Completed |
| epic-02 | System Struct and Topology   | 4 (005-008) | Detailed     | Completed |
| epic-03 | Validation and Testing       | 3 (009-011) | Refined      | Completed |
| epic-04 | Spec Audit and Documentation | 3 (012-014) | Detailed     | Completed |

## Progress

| Ticket     | Title                                               | Epic    | Status    | Detail Level | Readiness | Quality | Badge      |
| ---------- | --------------------------------------------------- | ------- | --------- | ------------ | --------- | ------- | ---------- |
| ticket-001 | Scaffold crate and define EntityId                  | epic-01 | completed | Detailed     | 0.94      | 0.95    | EXCELLENT  |
| ticket-002 | Define Bus and Line entities                        | epic-01 | completed | Detailed     | 0.98      | 1.00    | EXCELLENT  |
| ticket-003 | Define Hydro entity and supporting types            | epic-01 | completed | Detailed     | 0.98      | 1.00    | EXCELLENT  |
| ticket-004 | Define Thermal, PumpingStation, EnergyContract, NCS | epic-01 | completed | Detailed     | 0.96      | 1.00    | EXCELLENT  |
| ticket-005 | Implement penalty resolution                        | epic-02 | completed | Detailed     | 0.98      | 1.00    | EXCELLENT  |
| ticket-006 | Implement CascadeTopology                           | epic-02 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-007 | Implement NetworkTopology                           | epic-02 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-008 | Implement SystemBuilder and System struct           | epic-02 | completed | Detailed     | 0.98      | 1.00    | EXCELLENT  |
| ticket-009 | Cross-reference validation                          | epic-03 | completed | Refined      | 1.00      | 0.75    | ACCEPTABLE |
| ticket-010 | Cascade and filling validation                      | epic-03 | completed | Refined      | 0.98      | 0.96    | EXCELLENT  |
| ticket-011 | Integration and order-invariance tests              | epic-03 | completed | Refined      | 0.98      | 1.00    | EXCELLENT  |
| ticket-012 | Audit cobre-core against Phase 1 specs              | epic-04 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-013 | Write cobre-core software book page                 | epic-04 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-014 | Update introduction and phase tracker               | epic-04 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |

## Dependency Graph

```
ticket-001 (EntityId, module structure)
    |
    +---> ticket-002 (Bus, Line)
    |         |
    +---> ticket-003 (Hydro)
    |         |
    +---> ticket-004 (Thermal, PumpingStation, EnergyContract, NCS)
              |
    ticket-002 + ticket-003 ---> ticket-005 (Penalty resolution)
    ticket-003 ---> ticket-006 (CascadeTopology)
    ticket-002 + ticket-003 + ticket-004 ---> ticket-007 (NetworkTopology)
    ticket-005 + ticket-006 + ticket-007 ---> ticket-008 (SystemBuilder + System)
    ticket-008 ---> ticket-009 (Cross-reference validation)
    ticket-008 + ticket-009 ---> ticket-010 (Cascade + filling validation)
    ticket-009 + ticket-010 ---> ticket-011 (Integration tests)
    ticket-011 ---> ticket-012 (Spec audit)
    ticket-012 ---> ticket-013 (Software book page)
    ticket-013 ---> ticket-014 (Introduction + phase tracker update)
```

## Spec References

All specs are in `/home/rogerio/git/cobre-docs/src/specs/`:

- `data-model/internal-structures.md` -- Entity struct definitions, System struct, EntityId, topology
- `data-model/input-system-entities.md` -- JSON schemas for all 7 entity types
- `data-model/penalty-system.md` -- Three-tier penalty cascade
- `overview/design-principles.md` -- Declaration-order invariance, numeric representation
- `overview/notation-conventions.md` -- Mathematical notation and index sets
- `math/hydro-production-models.md` -- Constant productivity, FPHA, linearized head
- `data-model/input-hydro-extensions.md` -- Hydro geometry, production models, FPHA hyperplanes
- `math/system-elements.md` -- System element modeling overview
- `math/equipment-formulations.md` -- Per-equipment LP constraint details
