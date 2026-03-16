---
name: fpha_fitting_phase
description: Key facts about the FPHA hyperplane fitting implementation (feat/fpha-fitting plan, 4 epics)
type: project
---

FPHA fitting landed in 4 commits on feat/fpha-fitting, merged as PR #19 (202b6de..HEAD).
Main files: cobre-sddp/src/fpha_fitting.rs (~4640 lines), hydro_models.rs wiring, cobre-io/src/output/hydro_models.rs (new Parquet writer).

**Why:** Full computed FPHA path from VHA geometry; complement to the precomputed Parquet path.

**How to apply:** When reviewing future FPHA/hydro model changes, note these design choices:

- `validate_computed_prerequisites` requires tailrace + hydraulic_losses + efficiency; intentionally stricter than the math (policy decision, not all tests exercise this).
- `select_planes` early-break (when no valid removal exists) can return > max_planes_per_hydro — the doc says "at most N" but this is violated. However in practice the greedy loop always finds a valid removal before the count reaches max_planes_per_hydro for physically realistic inputs.
- `eprintln!` for kappa < 0.95 warning is a deliberate choice (no structured event system in the preprocessing pipeline); revisit when Phase 9+ adds observability.
- Grid construction formula (q_min = max(1.0, 0.01 _ q_max), s_max = 0.5 _ q_max) is duplicated across sample_tangent_planes, eliminate_redundant, compute_kappa, and compute_grid_errors — all four must be kept in sync.
