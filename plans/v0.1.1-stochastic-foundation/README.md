# v0.1.1 Stochastic Foundation + Examples

Build the PAR model simulation capability, implement inflow truncation, harden the scenario pipeline, add `cobre summary`, polish the book/documentation, implement stochastic load demand, add inflow estimation from history, and upgrade documentation to reflect all v0.1.1 changes.

## Tech Stack

- **Language**: Rust 2024 edition (MSRV 1.85)
- **Crates affected**: cobre-stochastic, cobre-sddp, cobre-io, cobre-core, cobre-cli, cobre-python
- **Documentation**: mdBook (book/), markdown, cobre-docs methodology reference

## Epics

| Epic    | Name                                    | Tickets | Detail Level |
| ------- | --------------------------------------- | ------- | ------------ |
| epic-01 | Immediate Fixes                         | 3       | Detailed     |
| epic-02 | PAR Model Simulation                    | 2       | Detailed     |
| epic-03 | Inflow Truncation                       | 3       | Refined      |
| epic-04 | Scenario Pipeline Hardening             | 3       | Refined      |
| epic-05 | `cobre summary` Subcommand              | 2       | Refined      |
| epic-06 | Book & Documentation Polish             | 4       | Refined      |
| epic-07 | Simulation Polish & Python Completeness | 4       | Detailed     |
| epic-08 | Load Noise Foundation                   | 2       | Detailed     |
| epic-09 | Load Integration in SDDP                | 5       | Detailed     |
| epic-10 | Load Validation & Testing               | 2       | Refined      |
| epic-11 | PAR Fitting Module                      | 5       | Refined      |
| epic-12 | Estimation Pipeline Integration         | 3       | Refined      |
| epic-13 | Book Documentation Upgrade              | 6       | Detailed     |
| epic-14 | cobre-docs Spec Upgrade                 | 5       | Detailed     |
| epic-15 | Repository Documentation Cleanup        | 4       | Detailed     |

**Total**: 53 tickets (31 detailed, 22 refined)

## Progress

| Ticket     | Title                                              | Epic    | Status    | Detail Level | Readiness | Quality | Badge      |
| ---------- | -------------------------------------------------- | ------- | --------- | ------------ | --------- | ------- | ---------- |
| ticket-001 | Rewrite book introduction                          | epic-01 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-002 | Replace cobre.dev schema URLs                      | epic-01 | completed | Detailed     | 0.96      | 0.91    | EXCELLENT  |
| ticket-003 | Check off cobre-python in README                   | epic-01 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-004 | Implement evaluate_par_inflow                      | epic-02 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-005 | Implement compute_truncation_noise                 | epic-02 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-006 | Add Truncation variant                             | epic-03 | completed | Refined      | 1.00      | 0.88    | ACCEPTABLE |
| ticket-007 | Wire truncation config in cobre-io                 | epic-03 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-008 | Integrate truncation into forward pass             | epic-03 | completed | Refined      | 1.00      | 0.96    | EXCELLENT  |
| ticket-009 | Defaults cascade integration tests                 | epic-04 | completed | Refined      | 1.00      | 0.98    | EXCELLENT  |
| ticket-010 | Audit and document seed handling                   | epic-04 | completed | Refined      | 1.00      | 0.94    | EXCELLENT  |
| ticket-017 | Fix simulation progress bar                        | epic-04 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-011 | Implement manifest/convergence readers             | epic-05 | completed | Refined      | 0.98      | 1.00    | EXCELLENT  |
| ticket-012 | Add cobre summary subcommand                       | epic-05 | completed | Refined      | 1.00      | 0.97    | EXCELLENT  |
| ticket-013 | Deduplicate tutorial/guide overlap                 | epic-06 | completed | Refined      | 1.00      | 0.98    | EXCELLENT  |
| ticket-014 | Add examples section to book                       | epic-06 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-015 | Re-record VHS tapes with brand theme               | epic-06 | completed | Refined      | 0.96      | 0.95    | EXCELLENT  |
| ticket-016 | Set up 4ree case directory                         | epic-06 | completed | Refined      | 0.94      | 1.00    | EXCELLENT  |
| ticket-018 | Fix simulation progress statistics                 | epic-07 | completed | Detailed     | 0.97      | 1.00    | EXCELLENT  |
| ticket-019 | Re-record VHS GIF files                            | epic-07 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-020 | Add FlatBuffers policy reader                      | epic-07 | completed | Detailed     | 0.98      | 1.00    | EXCELLENT  |
| ticket-021 | Add simulation/policy readers to Python            | epic-07 | completed | Detailed     | 0.85      | 0.78    | ACCEPTABLE |
| ticket-022 | Add PrecomputedNormalLp struct                     | epic-08 | completed | Detailed     | 1.00      | 0.91    | EXCELLENT  |
| ticket-023 | Extend stochastic context for load noise           | epic-08 | completed | Detailed     | 1.00      | 0.88    | ACCEPTABLE |
| ticket-024 | Build load factor lookup and extend StageTemplates | epic-09 | completed | Detailed     | 1.00      | 0.95    | EXCELLENT  |
| ticket-025 | Extend PatchBuffer for load balance rows           | epic-09 | completed | Detailed     | 1.00      | 0.88    | ACCEPTABLE |
| ticket-026 | Wire load noise into forward pass                  | epic-09 | completed | Detailed     | 1.00      | 0.78    | ACCEPTABLE |
| ticket-027 | Wire load noise into backward pass                 | epic-09 | completed | Detailed     | 1.00      | 0.90    | EXCELLENT  |
| ticket-028 | Wire load noise into simulation pipeline           | epic-09 | completed | Detailed     | 1.00      | 0.85    | ACCEPTABLE |
| ticket-029 | Add load file cross-validation rules               | epic-10 | completed | Refined      | 1.00      | 0.96    | EXCELLENT  |
| ticket-030 | Add stochastic load integration tests              | epic-10 | completed | Refined      | 1.00      | 0.98    | EXCELLENT  |
| ticket-031 | Implement Levinson-Durbin recursion                | epic-11 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-032 | Implement estimate_seasonal_stats                  | epic-11 | completed | Refined      | 1.00      | 0.88    | ACCEPTABLE |
| ticket-033 | Implement estimate_ar_coefficients                 | epic-11 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-034 | Implement estimate_correlation                     | epic-11 | completed | Refined      | 1.00      | 0.90    | EXCELLENT  |
| ticket-038 | Implement AIC-based AR order selection             | epic-11 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-035 | Add season map and estimation config types         | epic-12 | completed | Refined      | 1.00      | 0.75    | ACCEPTABLE |
| ticket-036 | Wire estimation into assembly pipeline             | epic-12 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-037 | Add history estimation validation rules            | epic-12 | completed | Refined      | 1.00      | 1.00    | EXCELLENT  |
| ticket-039 | Update stochastic-modeling.md for truncation/load  | epic-13 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-040 | Update crates/stochastic.md for new modules        | epic-13 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-041 | Update crates/sddp.md for load noise wiring        | epic-13 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-042 | Add cobre summary to CLI reference and crate page  | epic-13 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-043 | Update crates/io.md for estimation and load        | epic-13 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-044 | Update reference/roadmap.md for v0.1.1 features    | epic-13 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-045 | Update theory/inflow-nonnegativity.md truncation   | epic-14 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-046 | Add stochastic load theory to stochastic-modeling  | epic-14 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-047 | Update theory/par-model.md for estimation          | epic-14 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-048 | Update theory/scenario-generation.md for load      | epic-14 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-049 | Update roadmap/stochastic-enhancements.md          | epic-14 | completed | Detailed     | 1.00      | 0.75    | ACCEPTABLE |
| ticket-050 | Rewrite cobre/README.md roadmap and status          | epic-15 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-051 | Update CLAUDE.md for v0.1.1                         | epic-15 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-052 | Update cobre-python README and pyproject             | epic-15 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |
| ticket-053 | Clean up cobre-docs stale pages                      | epic-15 | completed | Detailed     | 1.00      | 1.00    | EXCELLENT  |

## Dependency Graph

```
Epic 1 (Immediate Fixes) --- no deps, can start immediately
  ticket-001, ticket-002, ticket-003 (all independent)

Epic 2 (PAR Model Simulation) --- no deps, can start immediately
  ticket-004 -> ticket-005

Epic 3 (Inflow Truncation) --- depends on Epic 2
  ticket-006 (independent of Epic 2)
  ticket-007 -> ticket-006
  ticket-008 -> ticket-005, ticket-006, ticket-007

Epic 4 (Scenario Pipeline Hardening) --- independent
  ticket-009, ticket-010 (independent of each other)
  ticket-017 (independent, blocks ticket-015)

Epic 5 (cobre summary) --- independent
  ticket-011 -> ticket-012

Epic 6 (Book & Documentation Polish) --- ticket-016 depends on Epic 3
  ticket-013, ticket-014, ticket-015 (independent)
  ticket-016 -> ticket-008 + user-provided 4ree data

Epic 7 (Simulation Polish & Python Completeness) --- independent
  ticket-018 (independent)
  ticket-019 -> ticket-018
  ticket-020 (independent)
  ticket-021 -> ticket-020

--- Workstream 1: Stochastic Load ---

Epic 8 (Load Noise Foundation) --- no new deps, can start immediately
  ticket-022 (independent)
  ticket-023 -> ticket-022

Epic 9 (Load Integration in SDDP) --- depends on Epic 8
  ticket-024 -> ticket-022, ticket-023
  ticket-025 -> ticket-024
  ticket-026 -> ticket-025
  ticket-027 -> ticket-025
  ticket-028 -> ticket-025

Epic 10 (Load Validation & Testing) --- depends on Epic 9
  ticket-029 -> ticket-028
  ticket-030 -> ticket-029

--- Workstream 2: Inflow Estimation ---

Epic 11 (PAR Fitting Module) --- independent of Workstream 1
  ticket-031 (independent)
  ticket-032 (independent)
  ticket-033 -> ticket-031, ticket-032
  ticket-034 -> ticket-033
  ticket-038 -> ticket-031, ticket-033

Epic 12 (Estimation Pipeline Integration) --- depends on Epic 11
  ticket-035 (independent, can start in parallel with Epic 11)
  ticket-036 -> ticket-034, ticket-035
  ticket-037 -> ticket-036

--- Workstream 3: Documentation Upgrade ---

Epic 13 (Book Documentation Upgrade) --- depends on Epics 1-12 being complete
  ticket-039, ticket-040, ticket-041, ticket-042, ticket-043, ticket-044 (all independent)

Epic 14 (cobre-docs Spec Upgrade) --- depends on Epics 1-12 being complete; independent of Epic 13
  ticket-045, ticket-046, ticket-047, ticket-048, ticket-049 (all independent)

Epic 15 (Repository Documentation Cleanup) --- depends on Epics 1-12 being complete; independent of Epics 13-14
  ticket-050, ticket-051, ticket-052, ticket-053 (all independent)
```

## Related Work (Out of Scope)

- cobre-docs BibTeX citation entry (`~/git/cobre-docs`)
- External/Historical sampling scheme wiring (deferred)
- BIC automatic AR order selection (AIC included in epic-11, BIC deferred)
- Load AR model with temporal correlation (deferred)
- Non-controllable source uncertainty (deferred)
- cobre-python book page (no page exists yet; separate feature)
