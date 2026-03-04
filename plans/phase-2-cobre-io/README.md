# Phase 2: cobre-io Input Loading and Validation

## Overview

Phase 2 implements the `cobre-io` crate: the input loading and validation pipeline for the Cobre power systems ecosystem. It also generalizes Phase 1 naming to be solver-neutral, extends `cobre-core` with temporal and scenario types, and populates the documentation infrastructure.

## Tech Stack

- **Language**: Rust 2024 edition, MSRV 1.85
- **Crates**: cobre-core (extended), cobre-io (new)
- **Dependencies**: serde, serde_json, chrono, thiserror, arrow, parquet, postcard

## Plan Structure

This is a **progressive plan** with 10 epics. Epics 0-2, 8, and 9 have detailed tickets; epics 3-7 have outline tickets that will be refined with learnings from earlier epics.

| Epic | Name | Tickets | Points | Detail Level |
|------|------|---------|--------|-------------|
| 00 | Phase 1 Generalization | 2 | 3 | Detailed |
| 01 | cobre-core Extension | 6 | 14 | Detailed |
| 02 | cobre-io Foundation | 5 | 12 | Detailed |
| 03 | Entity Registry Loading | 5 | 15 | Refined |
| 04 | Temporal and Scenario Loading | 5 | 12 | Refined |
| 05 | Constraint Loading and Resolution | 5 | 13 | Refined |
| 06 | Validation Pipeline | 5 | 14 | Refined |
| 07 | Integration and Output | 5 | 9 | Refined |
| 08 | Documentation Coherence | 6 | 13 | Detailed |
| 09 | Solver-Neutral Language Review | 3 | 7 | Detailed |
| **Total** | | **47** | **112** | |

## Progress Tracking

| Ticket | Title | Epic | Status | Detail Level | Readiness | Quality | Badge |
|--------|-------|------|--------|-------------|-----------|---------|-------|
| ticket-001 | Generalize cobre-core doc comments | epic-00 | completed | Detailed | 0.96 | 0.96 | EXCELLENT |
| ticket-002 | Generalize cobre-core book page | epic-00 | completed | Detailed | 0.90 | 0.95 | EXCELLENT |
| ticket-003 | Add serde feature flag and chrono dependency | epic-01 | completed | Detailed | 0.96 | 0.95 | EXCELLENT |
| ticket-004 | Define temporal types (Stage, Block, PolicyGraph) | epic-01 | completed | Detailed | 1.00 | 0.95 | EXCELLENT |
| ticket-005 | Define scenario pipeline raw types | epic-01 | completed | Detailed | 1.00 | 1.00 | EXCELLENT |
| ticket-006 | Define InitialConditions and GenericConstraint types | epic-01 | completed | Detailed | 0.98 | 0.88 | ACCEPTABLE |
| ticket-007 | Define ResolvedPenalties and ResolvedBounds types | epic-01 | completed | Detailed | 0.96 | 0.95 | EXCELLENT |
| ticket-008 | Extend System and SystemBuilder with new fields | epic-01 | completed | Detailed | 0.98 | 1.00 | EXCELLENT |
| ticket-009 | Define LoadError enum and crate dependencies | epic-02 | completed | Detailed | 0.98 | 0.88 | ACCEPTABLE |
| ticket-010 | Parse config.json into Config struct | epic-02 | completed | Detailed | 0.97 | 0.95 | EXCELLENT |
| ticket-011 | Parse penalties.json into GlobalPenaltyDefaults | epic-02 | completed | Detailed | 0.97 | 0.95 | EXCELLENT |
| ticket-012 | Parse initial_conditions.json into InitialConditions | epic-02 | completed | Detailed | 0.97 | 1.00 | EXCELLENT |
| ticket-013 | Implement ValidationContext and structural validation | epic-02 | completed | Detailed | 0.98 | 1.00 | EXCELLENT |
| ticket-014 | Parse buses.json and thermals.json | epic-03 | completed | Refined | 0.99 | 1.00 | EXCELLENT |
| ticket-015 | Parse hydros.json entity registry | epic-03 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-016 | Parse lines.json and optional registries | epic-03 | completed | Refined | 0.98 | 1.00 | EXCELLENT |
| ticket-017 | Add Parquet support and parse hydro_geometry | epic-03 | completed | Refined | 0.98 | 0.88 | ACCEPTABLE |
| ticket-018 | Parse production models and FPHA hyperplanes | epic-03 | completed | Refined | 0.98 | 1.00 | EXCELLENT |
| ticket-019 | Parse stages.json into temporal types | epic-04 | completed | Refined | 1.00 | 0.88 | ACCEPTABLE |
| ticket-020 | Parse inflow scenario Parquet files | epic-04 | completed | Refined | 0.96 | 0.98 | EXCELLENT |
| ticket-021 | Parse load scenario files | epic-04 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-022 | Parse correlation.json and external_scenarios | epic-04 | completed | Refined | 0.98 | 1.00 | EXCELLENT |
| ticket-023 | Assemble scenario pipeline data | epic-04 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-024 | Parse entity bounds Parquets with sparse expansion | epic-05 | completed | Refined | 1.00 | 0.95 | EXCELLENT |
| ticket-025 | Parse penalty override Parquet files | epic-05 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-026 | Parse generic constraints and exchange factors | epic-05 | completed | Refined | 0.96 | 1.00 | EXCELLENT |
| ticket-027 | Implement three-tier penalty cascade resolution | epic-05 | completed | Refined | 1.00 | 0.88 | ACCEPTABLE |
| ticket-028 | Implement bound resolution | epic-05 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-029 | Implement schema validation (Layer 2) | epic-06 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-030 | Implement referential integrity validation (Layer 3) | epic-06 | completed | Refined | 1.00 | 0.88 | ACCEPTABLE |
| ticket-031 | Implement dimensional consistency validation (Layer 4) | epic-06 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-032 | Implement semantic validation hydro/thermal (Layer 5a) | epic-06 | completed | Refined | 1.00 | 0.95 | EXCELLENT |
| ticket-033 | Implement semantic validation stages/penalties (Layer 5b) | epic-06 | completed | Refined | 1.00 | 0.88 | ACCEPTABLE |
| ticket-034 | Implement load_case pipeline orchestrator | epic-07 | completed | Refined | 0.96 | 0.94 | EXCELLENT |
| ticket-035 | Implement postcard serialization helpers | epic-07 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-036 | Create integration test fixtures and E2E tests | epic-07 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-037 | Test declaration-order invariance | epic-07 | completed | Refined | 1.00 | 0.93 | EXCELLENT |
| ticket-038 | Implement validation report writer and output stubs | epic-07 | completed | Refined | 1.00 | 1.00 | EXCELLENT |
| ticket-039 | Review cobre-docs specs against implementation | epic-08 | completed | Detailed | 0.90 | 1.00 | EXCELLENT |
| ticket-040 | Update cobre-docs cross-reference index | epic-08 | completed | Detailed | 0.90 | 1.00 | EXCELLENT |
| ticket-041 | Populate cobre-core book page | epic-08 | completed | Detailed | 0.91 | 1.00 | EXCELLENT |
| ticket-042 | Populate cobre-io book page | epic-08 | completed | Detailed | 0.91 | 1.00 | EXCELLENT |
| ticket-043 | Populate reference pages (case-format, error-codes) | epic-08 | completed | Detailed | 0.91 | 1.00 | EXCELLENT |
| ticket-044 | Update project documentation | epic-08 | completed | Detailed | 0.92 | 1.00 | EXCELLENT |
| ticket-045 | Audit and fix SDDP-specific language in cobre-core source | epic-09 | pending | Detailed | 0.95 | -- | -- |
| ticket-046 | Audit and fix SDDP-specific language in cobre-io source | epic-09 | pending | Detailed | 0.94 | -- | -- |
| ticket-047 | Audit and fix SDDP-specific language in documentation | epic-09 | pending | Detailed | 0.94 | -- | -- |
| ticket-048 | Remove MPI config section from Config struct (DEC-018) | emergency | completed | Detailed | 0.95 | 0.95 | EXCELLENT |
