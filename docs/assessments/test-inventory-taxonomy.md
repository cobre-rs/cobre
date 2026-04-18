# Test Inventory Taxonomy — Architecture Unification

Purpose: frozen vocabulary for tagging the test-suite inventory; all deletion decisions in the architecture-unification work resolve against the guard labels defined here.

---

## 1. Category

Exactly one category per test. When a test fits two labels, pick the narrower one: `unit` beats `integration`; `regression` beats `unit` when a commit SHA is attached.

| Label             | Definition                                                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `unit`            | Exercises one function or one data structure in isolation.                                                                       |
| `integration`     | Exercises a multi-module contract (e.g., cobre-sddp forward pass + cobre-solver trait) without running the full pipeline.        |
| `e2e`             | Runs the full training/simulation pipeline from config load to output write.                                                     |
| `regression`      | Pinned past-bug reproducer. Must have a linked commit SHA or ticket in a separate notes column.                                  |
| `conformance`     | D-case numerical snapshot (D01–D30) or convertido byte-identical golden output.                                                  |
| `parameter-sweep` | Shares its body with one or more other tests and differs only in input parameters. Candidate for parameterization.               |
| `coverage-matrix` | Combinatorial cross-axis coverage (e.g., MPI × cut-selection × warm-start). Justified only when the axes interact non-trivially. |

---

## 2. Guards

A test may carry one or more guard labels (comma-separated). A test with no deletable guard is tagged `generic`.

| Label                            | Meaning — test depends on this code path                                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `alien-only`                     | Test exercises `WarmStartBasisMode::AlienOnly` or the alien FFI path directly.                                   |
| `non-alien-first`                | Test exercises `WarmStartBasisMode::NonAlienFirst` (the path that stays).                                        |
| `canonical-disabled`             | Test exercises `CanonicalStateStrategy::Disabled`.                                                               |
| `canonical-clearsolver`          | Test exercises `CanonicalStateStrategy::ClearSolver` (the path that stays).                                      |
| `non-baked`                      | Test exercises the legacy (non-baked) template / `add_rows` path.                                                |
| `baked`                          | Test exercises baked templates (the path that stays).                                                            |
| `stored-cut-row-offset`          | Test depends on the non-zero `stored_cut_row_offset` parameter in `reconstruct_basis`.                           |
| `solve-with-basis-trait`         | Test calls `SolverInterface::solve_with_basis` as a distinct trait method (folded into `solve` in a later step). |
| `clear-solver-state-trait`       | Test calls `SolverInterface::clear_solver_state` as a trait method (removed in a later step).                    |
| `add-rows-trait`                 | Test calls `SolverInterface::add_rows` (removed in a later step).                                                |
| `warm-start-config-flag`         | Test depends on parsing `training.solver.warm_start_basis_mode` from config.                                     |
| `canonical-config-flag`          | Test depends on parsing `training.solver.canonical_state` from config.                                           |
| `broadcast-warm-start-field`     | Test depends on `BroadcastConfig::warm_start_basis_mode` wire field.                                             |
| `broadcast-canonical-field`      | Test depends on `BroadcastConfig::canonical_state_strategy` wire field.                                          |
| `training-result-struct-literal` | Test constructs `TrainingResult { ... }` using struct-literal syntax (migrates to builder in a later step).      |
| `fpha-slow`                      | Test is in the `fpha_*` slow-suite group.                                                                        |
| `d-case-determinism`             | Test is one of D01–D30 deterministic regression cases.                                                           |
| `convertido-determinism`         | Test consumes the convertido golden artifact.                                                                    |
| `unified-path`                   | Test targets the new `run_stage_solve` surface (pre-populated empty until that surface exists).                  |
| `generic`                        | Test does not depend on any specific deletable code path. Survives all refactoring steps.                        |

---

## 3. Deletion Mapping

Informational table linking guard labels to the refactoring step at which they are reviewed for deletion. A test carrying none of these guards is retained unless marked for consolidation.

| Step | Guards flagged for deletion in that step                                                                                         |
| ---: | -------------------------------------------------------------------------------------------------------------------------------- |
|    3 | `alien-only`, `warm-start-config-flag`, `broadcast-warm-start-field`                                                             |
|    4 | `canonical-disabled`, `canonical-config-flag`, `broadcast-canonical-field`, `clear-solver-state-trait`, `solve-with-basis-trait` |
|    5 | `non-baked`, `stored-cut-row-offset`, `add-rows-trait`                                                                           |
|    6 | `training-result-struct-literal`                                                                                                 |
|    9 | `parameter-sweep` (consolidation candidates), any `fpha-slow` test missing the slow-tests feature gate                           |
