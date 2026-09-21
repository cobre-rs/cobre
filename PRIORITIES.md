# Prioritized remediation — 2026-09 quality evaluation (stations core-io + stochastic; §7–§8 extend it to solver-comm, sddp, cli-python; §9 reconciles the build-ci, test-corpus, alignment and perf-sweep results)

Independent validation and prioritization of the 112 findings ratified at the two owner gates
(`stations/core-io/gate.md`, `stations/stochastic/gate.md`). This document ranks; it changes no
register entry. Finding IDs are owned by `BACKLOG.md`; every ID below links to its entry as
`BACKLOG.md:<line>` (line of the `**ID · Sev …**` heading at commit `1baeeadb`).

- **Validated at:** `1baeeadb` (2026-09-11) = register baseline `a136840d` + develop merge `99b3b0d1`
  (`fix(cli): broadcast boundary cuts to all ranks`). The merge touched only
  `cobre-cli/src/commands/run/policy.rs`, `cobre-cli/src/commands/broadcast.rs`,
  `cobre-sddp/src/policy/policy_load.rs`; the one affected anchor (CD-058 resume path) was re-checked.
- **Method:** every entry re-derived against source by the main session (all Tier-1/Tier-2 items
  by hand) plus five read-only validators, one per lens/station. Outcome: **0 refuted, 26 partial
  (count/anchor/severity corrections), 86 verified.** Partials never changed a defect's existence.
- **Status of the evaluation (updated 2026-09-21):** 8 of 11 stations ratified — core-io and
  stochastic (2026-09-08, at `a136840d`; §1–§5 below), solver-comm, sddp and cli-python (2026-09-18,
  at `077dbe2c`; §7), build-ci (STATION 7, 2026-09-19; 19 ids CD-099…CD-115, OD-049, OD-050),
  test-corpus (STATION 8, 2026-09-19; 8 ids TD-074…TD-077, CD-116…CD-118, OD-051; 50 seed folds, 10
  seeds retired to Cleared, §5.1 of `testing-architecture.md` left a Proposal) and
  generalization-alignment (2026-09-20; 243-entry ledger ratified, 12 retags, 9 Part-I entries
  CD-119…CD-127 minted, 0 holds). performance-sweep is executed for every single-process claim
  (2026-09-20: 22 not-material, 6 UNMEASURED/case-infeasible, PD-004 not-material) — only the two
  collective 2x2 rows PD-008 / PD-017 (both already FIXED 2026-09-15) and the verify step remain.
  unified-roadmap (E11) is unrun. `reconciliation` holds the Tier-1…W7 fix waves, the 2026-09-17
  post-plan section and the 2026-09-21 wave + merge section. The tracked mirror
  `docs/design/reserved-seams-and-deferred-debt.md` carries the fixed items of every wave through
  W17 (ID-free); the write-backs the later gates routed to E11 are listed in §8 step 8.
- **Ungated wave status (2026-09-21):** Tiers 6, 7, 8, W17 and W8 are all FIXED on
  `feat/quality-wave-ungated` (58 tickets, `04635b8f..cbc0ca10`, merged to `develop`); the
  `chore/quality-evaluation` ledger (27 commits, `plans/architecture-debt-audit/` only) was merged on
  top at `d2545190`. §9 records what that unblocks and the agreed next wave.
- **Tier 1 status (2026-09-11):** all four FIXED — `fix/quality-tier1` (a729a259 … 0d4c8c22,
  merged to `develop`) plus two follow-ups found during execution on `fix/quality-tier1-followups`:
  19521701 (thermal joins rule 49 by declared-id membership; rule 16 retired) and 3b363161
  (rewrite clears stale checkpoint payloads). Entries carry `- **Status:** fixed` bullets. Both
  follow-ups and CD-072's fix are merged to `develop`.
- **Tier 2 status (2026-09-12):** all eight FIXED plus CD-067; CD-066 partial — `plans/quality-tier2-hotpath`,
  14 tickets, commits 66b9b788 … fe439fc0 (merged to `develop`, which is the next reconciliation
  baseline). Three new findings minted (CD-073, PD-031, TD-034); the wave's own defects and their
  fixes are recorded in the register's reconciliation section.
- **Tier 3 status (2026-09-14):** all eleven FIXED (CD-040, CD-043, CD-045, CD-048, CD-057, OD-019,
  CD-066 remainder, OD-028, OD-029, CD-073, PD-031) — `plans/quality-tier3-footguns`, 10 tickets,
  commits 25603faf … 3e90024f (merged to `develop`, which is the next reconciliation baseline).
  Shipped shapes that differ from §Tier 3: the penalty twin was deleted, not aliased (owner decision);
  the model tables are validated through a new `ValidationError::UnsortedModelTable` and
  `with_scenario_models` became fallible. Two regressions surfaced at the epic boundaries (a pre-build
  reader of `Stage.index` in cobre-io's semantic validation; an unsorted `run_partial_estimation` table)
  and were fixed before the epic commits — see the register's reconciliation section.
- **Not yet a roadmap:** `tools/check-roadmap-dag.py` reads `- **Status:**` bullets on entries
  (only the fixed Tier-1/2/3 entries carry one; every other entry parses as `open`), so the Waves table
  in §5 is an _interim_ schedule in the checker's vocabulary, for the unified-roadmap station to lift.
- **Reconciliation 2026-09-17 (`develop` `2a14fe56..077dbe2c`, merged at `e3535a47`):** two feature plans
  landed, neither a debt wave — boundary policy by calendar date, and the CLI simplification + cobre-python
  review. No ID minted or closed; **CD-029 moves to partial** (Python-parity half fixed), **CD-025 stays
  open with hardened detection** (golden CLI-vs-Python determinism test, import-resolving parity gate).
  Anchor drift on open entries: TD-017 only (pre-existing). Details in the register's post-plan section.
  Next baseline `develop` @ `077dbe2c`.
- **CD-074 (2026-09-17 → 2026-09-19):** FIXED on `develop` at `3356da2d` by the state-canonicalization
  plan (18/18 tickets); the drift tally it first shipped was removed as overengineering (`7fbb3da2`).
  No release has been cut since v0.15.0 (`main` = `a136840d`). Owner decision 2026-09-19: no
  release-scoping cutoff — everything ungated in §7 is planned as one wave and lands as time allows (§8).

---

## 1. Re-ratings that differ from the register

| ID                                                                  | Register                | Validated                        | Why (evidence at HEAD)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------------------------------- | ----------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OD-011 (`BACKLOG.md:2680`)                                          | B                       | **A**                            | `constraints/penalty_overrides_ncs.parquet` is loaded (`structural.rs:311`, `schema.rs:161`), resolved into the per-stage table (`resolution/penalties.rs:380–390`) and never read: the LP objective uses the entity constant at `cobre-sddp/src/lp/builder/columns.rs:1136` while hydro/line/bus read the resolved table at `:347/:687/:722`. The override is silently ignored in every LP. cobre-docs documents the file as the only per-stage tier for NCS curtailment cost.                                                                                                                                                                                                                                                    |
| CD-058 (`BACKLOG.md:2462`)                                          | B                       | **A**                            | Four in-place `std::fs::write` sites at `output/policy/checkpoint.rs:221,227,237,244` and `csv::Writer::from_path` at `output/dictionary.rs:148,256` bypass `output/atomic.rs`. Wider than recorded: `PolicyMode::Resume` reads `output/<policy_path>` (`cli run/policy.rs:198–200`) and `run/outputs.rs:66` rewrites the same directory; the old `manifest.bin` is never removed first. A crash mid-rewrite destroys the only checkpoint. `EventConfig.checkpoint_interval` has no production consumer, so this single end-of-training write is the only checkpoint.                                                                                                                                                              |
| CD-056 (`BACKLOG.md:2441`)                                          | B                       | **A**                            | Hydro/line/pumping/contract/unit-group bound rows whose `stage_id` is not a study stage are dropped with no diagnostic at `resolution/bounds.rs:428,545,583,621` and `resolution/group_bounds.rs:76`; `validation/semantic/block_bounds.rs:98–99` states the gap; `check_bounds_references` has no stage check. Thermal (`semantic/thermal.rs:986`) and NCS (`referential.rs:945`) reject the same defect.                                                                                                                                                                                                                                                                                                                         |
| CD-051 (`BACKLOG.md:2390`)                                          | B                       | B, **user-visible**              | `cobre validate` (`cli validate.rs:381–383`) and Python `validate` (`cobre-python/src/io.rs:271`) resolve only the training scenario source; run setup (`cobre-sddp/src/setup/mod.rs:385–388`) resolves both. An invalid `simulation.scenario_source` passes validate with exit 0 and fails at run, contradicting the parity comment at `validate.rs:385–386`.                                                                                                                                                                                                                                                                                                                                                                     |
| PD-023 / PD-024 (`BACKLOG.md:3285`, `:3297`)                        | B, "setup/fitting-time" | **B, HOT path**                  | `ForwardSampler::sample` is called per (iteration, scenario, stage) from `training/forward_pass_state.rs:969`, `training/forward/enumerated.rs:782`, `simulation/pipeline.rs:928`, `simulation/enumerated.rs:210`. Under `scheme: out_of_sample` the `QmcSobol` arm heap-allocates `build_direction_matrix` + `derive_scramble_params` per draw (`qmc_sobol/mod.rs:277–278`) and `QmcHalton` allocates `sieve_primes` + `build_scramble_tables` (`qmc_halton/mod.rs:281–282`); `class_sampler.rs:311` hard-codes `sobol_ctx = None`. Violates "never allocate on hot paths". Default noise method is SAA (`stages.rs:213`, allocation-free), so no shipped deck is hit today. `out_of_sample.rs:44` "No heap allocation" is false. |
| PD-027 (`BACKLOG.md:3333`)                                          | B, setup                | **B, HOT path (conditional)**    | `apply_group_scan` allocates three vectors when a correlation group exceeds `MAX_STACK_DIM = 64` (`correlation/resolve.rs:426–437`) and is reached per draw from `sampling/mod.rs:224–251`. Not exercised by the example decks (no correlation input in `cobre_rodada`), but realistic for a national inflow group (154 hydros).                                                                                                                                                                                                                                                                                                                                                                                                   |
| PD-025 / PD-026 (`BACKLOG.md:3309`, `:3321`)                        | B, setup                | B, HOT (compute only)            | LHS reshuffles `dim` permutations of `total_scenarios` per point (`tree/lhs.rs:145–150`); the per-entity position scan runs per draw on the same path. No allocation. PD-026's fix-shape misses the `ForwardSampler` entry point.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| CD-043 (`BACKLOG.md:2307`)                                          | B, "latent footgun"     | B, latent **with a live reader** | `par/lag_transition.rs:114` slices by `stage.index` to decide `finalize_period`; a wrong index gives wrong PAR lag weights. Only `cobre-io/src/stages.rs:815` writes the index in production, so still latent.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| CD-045 (`BACKLOG.md:2328`)                                          | B                       | **C**                            | `PrecomputedPar::build` is hash-keyed (`par/precompute.rs:194–223`) and does not depend on model order; order leaks only into policy-export row order. Doc-contract drift.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| CD-067 (`BACKLOG.md:3194`)                                          | B (A-risk)              | **B, not user-visible**          | `referential.rs:447–461` rejects an unknown `entity_type` on the production pipeline (`pipeline.rs:86`), so a mis-cased tag cannot silently disable a group. Stringly-typed maintainability only.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| CD-066 (`BACKLOG.md:3183`)                                          | B (A-risk)              | **C**                            | A literal typo at the `class_name` gate fails loudly (garbled-name rejection or `MissingScenarioSource`), not silently.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| CD-070 (`BACKLOG.md:3224`)                                          | B (A-risk)              | **C**                            | Both resolvers are the same enumerate→(id, i) one-liner; the real drift is which stage set callers feed, which a shared type does not fix.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| PD-007, PD-008, PD-017 (`:2536`, `:2546`, `:2637`)                  | B                       | **C**                            | One-time setup or once-per-run output costs. PD-007 is O(N·depth), not O(N²). PD-017's two `allgatherv` calls (`cli run/simulation.rs:312,328`) run once at simulation end.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| OD-013, OD-016, OD-018, OD-023 (`:2701`, `:2732`, `:2753`, `:2805`) | B                       | **C**                            | No observable effect. OD-016's divergence is unreachable because `stages.json` is required (`structural.rs:154`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| TD-001, TD-018, TD-021 (`:2838`, `:3008`, `:3038`)                  | B                       | **C**                            | Mechanical, small blast radius.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

Held as recorded: OD-019 B (latent silent misassignment), PD-009 B (scales with record depth), CD-068 B,
TD-002/005/007/014/016 B (they compound through one missing mechanism, §Tier 4).

---

## 2. Tiers

### Tier 1 — user-visible correctness and data loss (fix first, byte-neutral for well-formed decks)

| ID     | Register          | Gate                     | Defect                                                       | Fix-shape (validated)                                                                                                                                                                                                                                                                              | Effort | Parity                                                                                  |
| ------ | ----------------- | ------------------------ | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------- |
| OD-011 | `BACKLOG.md:2680` | core-io gate, Sev-B tier | NCS per-stage penalty override never reaches the LP          | Read `ctx.resolved.penalties.ncs_penalties(ncs_sys_idx, stage_idx).curtailment_cost` at `columns.rs:1136`; add an LP-builder test asserting a stage override changes the objective coefficient. Byte-neutral without the override file.                                                            | S      | shared pipeline, automatic                                                              |
| CD-058 | `BACKLOG.md:2462` | core-io gate, Sev-B tier | Checkpoint rewrite is not crash-safe on the resume path      | Route `checkpoint.rs` payload + manifest writes and `dictionary.rs` CSVs through `write_bytes_atomic`; remove the old `manifest.bin` before payload writes (or write to a sibling dir and rename). Keep manifest-last. Add a crash-simulation test (truncated tmp must never replace a good file). | M      | `cobre-python/src/policy.rs:359` exposes `write_policy_checkpoint`; signature unchanged |
| CD-056 | `BACKLOG.md:2441` | core-io gate, Sev-B tier | Out-of-horizon bound rows for five families dropped silently | Add one table-driven stage-axis rule beside `check_bound_block_id_range` (`semantic/block_bounds.rs:100`) keyed off `FamilyMeta`, reusing `StageIdResolver`; absorb the Layer-3 NCS stage/negative check (CD-053 half). CHANGELOG note: previously tolerated rows now fail validation.             | S–M    | shared pipeline                                                                         |
| CD-051 | `BACKLOG.md:2390` | core-io gate, Sev-B tier | validate/run parity gap on the simulation scenario source    | Resolve both sources once in the Layer-2 config gate and report into the validation context; update `cli validate.rs` and `cobre-python/src/io.rs:271`.                                                                                                                                            | M      | Python `validate` must move in lockstep                                                 |

### Tier 2 — hard-rule violations on the forward hot path (non-default config; fix before recommending QMC/LHS)

**2026-09-11:** planned as `plans/quality-tier2-hotpath/` (spec + seam map) on the baseline that includes
CD-072's fix. Owner decisions: per-iteration shared tables on the training session (never per thread),
all three methods hoisted, single precomputed correlation path (twin and scan deleted), riders CD-069
(shared point spec) and CD-066/CD-067 (entity-class enum) included. CD-072 (cross-class seed sharing,
Sev A) was found while mapping the seams and fixed first on `fix/out-of-sample-class-seed`.

**2026-09-12: FIXED.** Every row below is closed (status bullets on the entries; wave summary in the
register's reconciliation section). The shipped shape differs from the fix-shapes in one respect:
the scenario-invariant state lives in per-iteration tables owned by the training and simulation
state structs (`ForwardNoiseTables`, rebuilt once per iteration and shared by `&` through
`SampleRequest`), not in per-thread scratch; the correlation scratch does ride on `ScratchBuffers`
as planned.

| ID                     | Register                  | Defect                                                        | Fix-shape (validated)                                                                                                                                                                                                                                                                                                                                                                            | Effort                                                          |
| ---------------------- | ------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| PD-023, PD-024         | `:3285`, `:3297`          | Sobol/Halton per-draw heap allocation                         | Caller-owned per-thread QMC scratch threaded through `SampleRequest` (precedent: `perm_scratch`), keyed on (forward_seed, iteration, noise_group_id, dim, total_scenarios) and rebuilt on key change; pass `Some(&ctx)` into the existing `sobol_ctx`; add `HaltonPrecomputed`. Add a precomputed-vs-direct bit-equality test (`SobolPrecomputed::new` has zero callers today, including tests). | L (cross-crate plumbing into cobre-sddp's per-thread workspace) |
| PD-025                 | `:3309`                   | LHS O(dim × n²) per stage                                     | Hoist the `dim` permutations into the same per-(iteration, stage) scratch; consume `perm_rng` in the same order so bits are unchanged.                                                                                                                                                                                                                                                           | M (rides on the scratch above)                                  |
| PD-026, PD-027, PD-028 | `:3321`, `:3333`, `:3345` | Position scan and >64 scratch allocation per draw             | Resolve class positions once at `DecomposedCorrelation::build` (or `build_forward_sampler`, which already holds class order slices) so `apply_group_precomputed` is reachable; thread caller-owned gathered/correlated/positions scratch through both entry points (`generate_opening_tree` and `ForwardSampler::sample`).                                                                       | M                                                               |
| CD-068, CD-069         | `:3204`, `:3214`          | Duplicated correlation applier; triplicated point-spec struct | Collapse to one applier and one spec type as part of the same change; retire `apply_correlation` full-vector twin after the per-class path owns the differential test.                                                                                                                                                                                                                           | S–M                                                             |

Determinism guards: `tests/saa_golden_value.rs`, `tests/{halton,sobol,lhs}_integration.rs`,
`tests/reproducibility.rs` pin opening-tree and forward-noise bits; any hoist must reproduce them.

### Tier 3 — latent footguns, one ticket each

**2026-09-14: FIXED.** Every row below is closed (status bullets on the entries; wave summary in the
register's reconciliation section). The penalty-twin row's fix-shape was superseded by the owner's
deletion decision: the sweep touched 34 files / 224 occurrences and compiled after one
`cargo check --all-targets` pass, so the "~130 sites (L) — do not" caution was overstated.

| Ticket                       | IDs                    | Register                  | Fix-shape (validated)                                                                                                                                                                                                                                                                               | Effort |
| ---------------------------- | ---------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Builder owns canonical order | CD-043, CD-045, CD-048 | `:2307`, `:2328`, `:2359` | Reassign `Stage.index` after `SystemBuilder::build` sorts (`system/builder.rs:364`) and drop the `stages.rs:812–816` loop; validate (not sort) the three `*_models` tables as canonical; fix the four doc sites incl. `cobre-core/src/system/mod.rs:3,28`. Validate-not-sort keeps bytes identical. | S–M    |
| Input-file registry          | CD-057, OD-019         | `:2451`, `:2763`          | Replace the positional `FILE_ENTRIES` ↔ `manifest_fields_mut` zip (`validation/structural.rs:376,398`) with a keyed registry (enum key + flag array + named accessors). Order matches today (43/43).                                                                                                | M      |
| Penalty twin                 | CD-040                 | `:2277`                   | `pub type HydroStagePenalties = HydroPenalties;` (S). Deleting the twin outright touches ~130 construction sites (L) — do not.                                                                                                                                                                      | S      |
| Typed entity class           | CD-066 (remainder)     | `:3183`                   | `EntityClass` exists and the correlation side is typed (CD-067 fixed in Tier 2). Remaining: carry `EntityClass` on `ClassSamplerParams` in place of `class_name: &str`, keep the label for diagnostics, derive the class seed from it.                                                              | S      |

### Tier 4 — the one structural lever (unblocks fifteen test-bloat findings)

cobre-core declares `test-support` (`crates/cobre-core/Cargo.toml:29`), every consumer crate already
dev-depends on it, and it gates exactly one method (`Hydro::declare_mirror_unit_group`,
`entities/hydro.rs:276`). A 49-line `Hydro` literal is byte-identical at nine sites; the eight
cobre-stochastic integration binaries carry 626 duplicated helper lines (43 % of their prelude).

| Step                             | IDs resolved                                        | Register                                                    | Content                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------------- | --------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Ratify the standard           | —                                                   | `docs/design/testing-architecture.md` §5 (status: Proposal) | Ratify §5.2 `test-support` convention — DONE 2026-09-15 (D1(a), `feat/quality-tier45-closeout`); the §5.1 homing threshold stays a proposal for the test-corpus wave, so TD-008/TD-031 stay decisions, not tickets, until it is ratified.                                                                                                                                                                                                   |
| 1. Safety net first              | TD-023                                              | `:3059`                                                     | Owner-level contract tests for the six `parquet_helpers.rs` extractors, before any consolidation that relies on them.                                                                                                                                                                                                                                                                                                                       |
| 2. cobre-core builder surface    | TD-002, TD-005, TD-004 (hoist), feeds TD-007/TD-016 | `:2848`, `:2878`, `:2868`                                   | Parameterized `make_bus/make_line/make_hydro/make_thermal/make_ncs/make_contract/make_pumping/make_unit_group/make_stage` + test-only `Default` for `HydroPenalties` and `GlobalPenaltyDefaults`, mirroring `cobre-sddp/tests/common/builders.rs` (`Spec + Default + make_*`), which is binary-local and cannot be shared. `make_hydro` needs `operational_start_date` and mirror-group axes (two variants exist: 2024+mirror ×9, 2020 ×3). |
| 3. cobre-io local fixture module | TD-007, TD-014, TD-016, TD-018, TD-020, TD-009      | `:2898`, `:2968`, `:2988`, `:3008`, `:3028`, `:2918`        | `#[cfg(test)]` crate-level module owning `write_json`/`write_parquet`/JSON corpus/`make_config`/`read_batch`; home `validation/semantic/test_support.rs` one level up.                                                                                                                                                                                                                                                                      |
| 4. cobre-stochastic consumers    | TD-024, TD-025, TD-028, TD-029, TD-030, TD-032      | `:3442`, `:3453`, `:3486`, `:3497`, `:3508`, `:3530`        | Consume the cobre-core surface; `uniform_tree` and `approx_erf/norm_cdf` stay crate-local (`tests/common/` or a cobre-stochastic `test-support` feature). Expose the `CorrelationModel` builder only; `DecomposedCorrelation` callers wrap with `.build()`.                                                                                                                                                                                 |

Guard rails: `tests/reproducibility.rs` (`deterministic_reproducibility`, `declaration_order_invariance`,
`seed_sensitivity`, `infrastructure_genericity_no_sddp_references`), `tests/saa_golden_value.rs`,
`invariance.rs`, the three `declaration_order_invariance` parser tests (TD-009) and
`no_stage_positional_filename_convention_remains` (TD-021) are load-bearing: relocate, never delete.
**TD-011:** fold the `bus2.name` handle into `test_full_case_ordering_invariance` instead of deleting
an order-invariance test.

### Tier 5 — dead-surface sweep (batch; several are licensed public-API breaks)

| Group                                    | IDs                                                                                                                      | Register                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Dead cobre-io surface                    | OD-014, OD-015, OD-016, OD-017, OD-018, OD-020, OD-021, OD-024, OD-025                                                   | `:2712`, `:2722`, `:2732`, `:2743`, `:2753`, `:2774`, `:2785`, `:2815`, `:2826` |
| Dead cobre-core surface                  | OD-010, OD-012, CD-041                                                                                                   | `:2670`, `:2691`, `:2287`                                                       |
| Dead error variants (pub API break gate) | CD-052, OD-031                                                                                                           | `:2401`, `:3430`                                                                |
| Dead output plumbing                     | OD-022 (two `IterationRecord` fields), PD-017 (`partitions_written` + two `allgatherv`)                                  | `:2795`, `:2637`                                                                |
| Wire payload                             | PD-008 + OD-013 as one ticket: `serde(skip)` cascade + network, rebuild in `From<SystemRepr>`                            | `:2546`, `:2701`                                                                |
| Stochastic dead surface                  | OD-026, OD-027, OD-030                                                                                                   | `:3380`, `:3390`, `:3420`                                                       |
| Read/write helper pair                   | CD-047 + CD-063 (reader prologue helper; `ensure_parent_dir` + atomic-write helper — also CD-058's landing zone), CD-049 | `:2349`, `:2514`, `:2369`                                                       |

### Deprioritized (record, do not schedule a push)

- Setup-time Sev-C performance: PD-006, PD-009 (only one that scales with record depth), PD-010,
  PD-011, PD-012, PD-013, PD-014, PD-015, PD-016, PD-018, PD-019, PD-020, PD-021, PD-022, PD-029, PD-030.
  Fold PD-016's `WriterProperties` half with OD-023 and OD-020's `dictionary.rs:76` rebuild (one
  config-threading change).
- Doc / table drift, one sweep — DONE 2026-09-15 (W7): CD-042, CD-044, CD-046, CD-050, CD-053, CD-054, CD-055, CD-059,
  CD-060, CD-062, CD-064, OD-029, OD-028.
- Alignment holds for the generalization station (Epic 9), no code now: CD-061, CD-065, CD-070, CD-071.
- Test-corpus sweep (Epic 8): TD-001, TD-003, TD-006, TD-008, TD-010, TD-011 (fold), TD-012, TD-013,
  TD-015, TD-017, TD-019, TD-022, TD-026, TD-027, TD-031, TD-033.

---

## 3. Register corrections to apply at the reconciliation station

Facts the validators could not reproduce or that the register states wrongly; none changes a verdict.
**Applied 2026-09-11** as `- **Correction (2026-09-11):**` bullets on the entries (reconciliation
section of `BACKLOG.md`); `perf-queue.json` left as a station artifact. Two corrections found while
fixing Tier 1 were added to the table below at the same time.

| Entry                                                                   | Correction                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PD-007 (`:2536`)                                                        | O(N·window depth), not O(N²): the overlap walk breaks past the window end.                                                                                                                                                                                                                                                                                                              |
| PD-023/024/025 measurement lines; `stations/stochastic/perf-queue.json` | Path class is HOT (forward sampler), not "setup/fitting-time".                                                                                                                                                                                                                                                                                                                          |
| CD-045 (`:2328`)                                                        | `PrecomputedPar::build` does not depend on model order (hash-keyed). Three tables, not "seven".                                                                                                                                                                                                                                                                                         |
| CD-047 (`:2349`)                                                        | 77 `try_new` prologue sites in 29 files (incl. tests), not 28 in 19.                                                                                                                                                                                                                                                                                                                    |
| CD-054 (`:2421`)                                                        | `s.id >= 0` occurs 34× crate-wide (2 in `referential.rs`), not "20 copies".                                                                                                                                                                                                                                                                                                             |
| CD-058 (`:2462`)                                                        | Payload writes are NOT covered by manifest-last on the resume path (same directory rewritten, old manifest never removed). Also: the reader enumerates `cuts/`/`basis/`/`states/` and the pool count is only a `debug_assert_eq!` (`fcf.rs::from_deserialized`), so stale payloads from a rewrite with fewer pools or states export off were read silently in release (fixed 3b363161). |
| CD-056 (`:2441`)                                                        | "Thermal's guard is legitimately family-specific (padded resolution region)" is wrong: the padding holds base values only and `resolve_bounds` keys thermal overrides by `stage_index`; thermal's `[0, n)` test was a latent defect for gapped/1-based id sets (fixed 19521701).                                                                                                        |
| CD-060 (`:2482`)                                                        | 10 of 35 schemas declared outside `schemas.rs`; gate list covers 22 of 35 (not 9/34, 21/34).                                                                                                                                                                                                                                                                                            |
| CD-066 (`:3183`)                                                        | Gate is at `sampling/mod.rs:360`; a typo fails loudly.                                                                                                                                                                                                                                                                                                                                  |
| CD-067 (`:3194`)                                                        | Producers at `sampling/mod.rs:231/240/249`; cobre-io rejects unknown tags at `referential.rs:447–461`.                                                                                                                                                                                                                                                                                  |
| OD-017 (`:2743`)                                                        | `BusinessRuleViolation` has 58 `add_error` emissions and 1 `add_warning`; no "sole emission".                                                                                                                                                                                                                                                                                           |
| OD-021 (`:2785`)                                                        | `default_bounds` is public at `cobre_io::output::default_bounds`, not at the crate root. No `JsonSchema` derive in `manifest.rs`.                                                                                                                                                                                                                                                       |
| OD-029 (`:3410`)                                                        | Arities are 11/13/13, not 11/14/12.                                                                                                                                                                                                                                                                                                                                                     |
| TD-004 (`:2868`)                                                        | Comparators are exhaustive today; `clp_determinism.rs` has no `opt_f64_bits_eq` (uses `SolveBits`). Latent.                                                                                                                                                                                                                                                                             |
| TD-008 (`:2908`)                                                        | Mechanical application hits ~15 input-path modules, not seven.                                                                                                                                                                                                                                                                                                                          |
| TD-022 (`:3049`)                                                        | 11 count-only tests (10 with an umbrella row), not seven.                                                                                                                                                                                                                                                                                                                               |
| TD-028 (`:3486`)                                                        | `sampling/mod.rs:670` `make_hydro` is the canonical 9-copy variant, not a divergent outlier.                                                                                                                                                                                                                                                                                            |
| TD-029 (`:3497`)                                                        | `identity_correlation` is at `saa_golden_value.rs:53`, not `:32`.                                                                                                                                                                                                                                                                                                                       |
| TD-031 (`:3519`)                                                        | 16 modules exceed 500 inline test LOC (e.g. `sampling/external.rs` 2312); anchor list is under-scoped.                                                                                                                                                                                                                                                                                  |

New observations not in the register (candidates for the sddp / cli-python stations):
`EventConfig.checkpoint_interval` (`cobre-sddp/src/config.rs:183`) has no production consumer
(re-confirmed 2026-09-17); `SobolPrecomputed::new` (`qmc_sobol/mod.rs:181`) had zero callers including
tests (resolved 2026-09-12); `out_of_sample.rs:44` "No heap allocation" was a false doc claim (removed
2026-09-12). Added 2026-09-17 (cli-python / build-ci): the `training/hydro_models.json` and
`training/model_provenance.json` sidecars lost their only consumer with `cobre summary` and are kept under
the parity rule; `cobre-python` has `doc = false`, so its intra-doc links are never gated; the CLI plan's
parity matrices and fix list under `plans/cli-simplification-python-review/` are ready station inputs.

---

## 4. Test and prod line inventory (context for Tier 4 and the test-corpus station)

| Crate            | `src/` lines | inline `#[cfg(test)]` lines                             | `tests/` lines |
| ---------------- | ------------ | ------------------------------------------------------- | -------------- |
| cobre-core       | 18,671       | 9,430 (51 %)                                            | 668            |
| cobre-io         | 104,066      | 59,778 (57 %)                                           | 8,474          |
| cobre-stochastic | 61,298       | 20,634 (34 %; 21k of prod is the generated Sobol table) | 4,096          |
| cobre-sddp       | 187,059      | 82,263 (44 %)                                           | 79,855         |
| cobre-cli        | 10,873       | 5,454 (50 %)                                            | 4,035          |
| cobre-solver     | 9,608        | 1,471 (15 %)                                            | —              |

cobre-sddp (the largest crate) has not been re-evaluated in this pass; the August audit's Waves 4–7
remain open there (`BACKLOG.md:1898–1922`): setup lifecycle redesign (CD-002/003/004/005), CLI/Python
output hand-mirror (CD-025), inline-test giants (CD-007), god functions (CD-012, CD-014).

The table above is the 2026-09-11 snapshot. Since then (to `077dbe2c`) cobre-cli lost a net 2.3k lines
(two subcommands, three test binaries, the test-only summary oracle) and cobre-python gained a net 0.5k
(golden parity test, checkpoint round-trip tests, `Study` native lifecycle); cobre-sddp grew a net 5.2k
with the date-driven boundary work. Regenerate the table at the next station's inventory step rather
than hand-editing it.

---

## 5. Interim waves (checker vocabulary; not yet the unified-roadmap section)

Milestones as in `BACKLOG.md:33–51`; `gnl-import` is SATISFIED (`44e72b76`, `9db2cbdf`) and must not be re-opened.

| Wave | Entry                       | Findings                                                                                                                                                                                       | Depends on                 | Phase    | Effort | Trigger/deadline                                                                             |
| ---- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | -------- | ------ | -------------------------------------------------------------------------------------------- |
| 1    | W1-tier1-correctness        | OD-011, CD-058, CD-056, CD-051                                                                                                                                                                 | -                          | neutral  | M      | SATISFIED 2026-09-11 (`fix/quality-tier1` merged; follow-ups pending)                        |
| 2    | W2-canonical-order-registry | CD-043, CD-045, CD-048, CD-057, OD-019, CD-040                                                                                                                                                 | W1-tier1-correctness       | neutral  | M      | SATISFIED 2026-09-14 (`plans/quality-tier3-footguns` epic-01, merged to develop at 3e90024f) |
| 3    | W3-forward-sampler-scratch  | PD-023, PD-024, PD-025, PD-026, PD-027, PD-028, CD-068, CD-069                                                                                                                                 | -                          | neutral  | L      | SATISFIED 2026-09-12 (`plans/quality-tier2-hotpath`, merged to develop at fe439fc0)          |
| 4    | W4-typed-class-and-seam     | CD-066 (remainder), OD-028, OD-029 — CD-067 fixed 2026-09-12                                                                                                                                   | W3-forward-sampler-scratch | neutral  | M      | SATISFIED 2026-09-14 (`plans/quality-tier3-footguns` epic-02, incl. CD-073 and PD-031)       |
| 5    | W5-test-support-surface     | TD-023, TD-002, TD-005, TD-004, TD-007, TD-014, TD-016, TD-018, TD-020, TD-009, TD-024, TD-025, TD-028, TD-029, TD-030, TD-032, TD-034 (minted 2026-09-12)                                     | -                          | neutral  | M      | SATISFIED 2026-09-15 (`feat/quality-tier45-closeout`, merged to develop at 54ab986a)         |
| 6    | W6-dead-surface-sweep       | OD-010, OD-012, OD-014, OD-015, OD-016, OD-017, OD-018, OD-020, OD-021, OD-022, OD-024, OD-025, OD-026, OD-027, OD-030, OD-031, CD-041, CD-052, PD-017, PD-008, OD-013, CD-047, CD-063, CD-049 | W1-tier1-correctness       | neutral  | S      | SATISFIED 2026-09-15 (`feat/quality-tier45-closeout`, merged to develop at 54ab986a)         |
| 7    | W7-doc-and-table-drift      | CD-042, CD-044, CD-046, CD-050, CD-053, CD-054, CD-055, CD-059, CD-060, CD-062, CD-064                                                                                                         | W1-tier1-correctness       | neutral  | S      | SATISFIED 2026-09-15 (fixed directly on develop at eb0b82ef + 2a14fe56)                      |
| 8    | W8-setup-perf-opportunistic | PD-006, PD-007, PD-009, PD-010, PD-011, PD-012, PD-013, PD-014, PD-015, PD-016, PD-018, PD-019, PD-020, PD-021, PD-022, PD-029, PD-030, OD-023                                                 | W6-dead-surface-sweep      | neutral  | M      | SATISFIED 2026-09-21 (`feat/quality-wave-ungated`)                                           |
| 9    | W9-test-corpus-sweep        | TD-001, TD-003, TD-006, TD-008, TD-010, TD-011, TD-012, TD-013, TD-015, TD-017, TD-019, TD-021, TD-022, TD-026, TD-027, TD-031, TD-033                                                         | W5-test-support-surface    | neutral  | M      | SUPERSEDED 2026-09-21: E08 ratified 2026-09-19 — TD-017 retired to Cleared; TD-021, TD-031 stay §5.1-gated in W16; the other 14 rows move to W18 (§7 Interim waves)                                                                  |
| 10   | W10-alignment-hold          | CD-061, CD-065, CD-070, CD-071                                                                                                                                                                 | W4-typed-class-and-seam    | serves 1 | -      | ADJUDICATED 2026-09-20: CD-061 `advances-0a` → W15; CD-065, CD-070, CD-071 `advances-1` → W19 (§7 Interim waves)                         |

Every one of the 112 ratified IDs appears in exactly one wave (verified mechanically when this file was written).

---

## 6. Next steps (proposed 2026-09-17, after the reconciliation) — SUPERSEDED by §8 on 2026-09-19

Steps 0 (CD-074), 1 (re-pin), 2 (cli-python) and 3 (sddp) are DONE; step 4 (W8) stays opportunistic;
step 5's order is re-cut in §8 after the three owner gates. Kept for provenance.

Nine of eleven stations are unrun. Ordered by what the tree just made cheap or urgent:

| # | Step | Why now | Inputs already in hand |
| --- | --- | --- | --- |
| 0 | **CD-074 hotfix, then a redesign ticket** (anticipated-commitment delivery reconciliation; user-reported `MPI_Abort` on a `−1.5e-4` MW carried value at a `0` MW floor, `BACKLOG.md:4219`). | Tier-1-shaped: a user-visible abort that loses a multi-hour run over 150 W of noise, and the third constant-recalibration of the same guard — the mechanism (a commitment-relative margin blind to the delivery stage's actual per-block box — positive must-run floors and caps drift the same way zero did; relax-the-LP-bound instead of project-the-pin onto that box; abort instead of tally) is the fix, not the constant. Ahead of every station: it is live in users' hands. | The reporter's log; `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs`; the `.claude/rules/sddp.md` contract to rewrite; `git show -s 7628d3d0 54882130` for the two prior recalibrations. |
| 1 | **Re-pin the evaluation baseline** to `develop` @ `077dbe2c` for the remaining stations (`tools/pin-baseline.sh`; the ratified core-io and stochastic stations keep `a136840d`). | Every remaining station reads a surface the two merged plans rewrote; evaluating at the old pin would re-find fixed defects. | `pin-baseline.sh` exit codes 2–4 guard the pre-conditions. |
| 2 | **cli-python station** (cobre-cli + cobre-python, four lenses). | The surface is freshly settled and pre-audited: 48 argument rows, 30 behaviour rows, 128 docstring claims and a 26-item fix list exist; the station's job is to ratify residuals and mint IDs, not to discover. Re-adjudicate the 16 justified-as-is rows, the orphaned sidecars, CD-025's owner question, and the `doc = false` gate gap. | `plans/cli-simplification-python-review/{parity-arguments,parity-behaviour,docstring-audit,fix-list}.md`; the core-io prompt templates under `stations/core-io/prompts/`. |
| 3 | **sddp station** (cobre-sddp, the largest crate, never re-evaluated in this pass). | Holds the largest open cluster: Wave 4 setup redesign (CD-002/003/004/005), god functions CD-012/CD-014, CD-023 prep inversion, CD-028 orchestration mirror, and the unrecorded `checkpoint_interval` seam; the boundary-policy plan just added 5k lines here. | August Station 1–4 entries (`BACKLOG.md:139–1057`) as the prior register; `measurements/CAL*` bounds for the perf lens. |
| 4 | **W8 setup-perf, opportunistic** (PD-009 first). | Unblocked since W6; no dedicated push. | §Deprioritized list. |
| 5 | **build-ci + test-corpus stations**, then **generalization-alignment** (adjudicates W10 and CD-044's home) and **unified-roadmap** (lifts §5 into the checker's vocabulary). | Order by dependency: test-corpus needs `testing-architecture.md` §5.1 ratified; W10 needs the alignment station. | `tools/check-roadmap-dag.py` for the roadmap section. |

Owner decisions this ordering assumes: CD-074 ships as a hotfix ahead of the `sddp` station and its
redesign is its own ticket (step 0); the remaining stations evaluate at the new pin (step 1); the
cli-python station may cite the plan's matrices as prior evidence instead of re-deriving them; CD-025's
hoist stays Wave 5 (not folded into the cli-python station's fixes).

---

## 7. Stations 4–6 — solver-comm, sddp, cli-python (tiered 2026-09-19)

Scope: the three owner gates of 2026-09-18 (`stations/solver-comm/gate.md`, `stations/sddp/gate.md`,
`stations/cli-python/gate.md`), all at baseline `077dbe2c`. Minted: 20 + 44 + 40 = **104 new ids**
(3 × B (A-risk), 39 × B, 62 × C) plus **26 sharpened or kept prior ids** (sddp 22, cli-python 4);
3 prior ids retired to Cleared (CD-001, CD-003-construction-hop, CD-006); 0 rejected, 1 deferred
(OD-043, trigger CD-004).

**Validation basis.** Unlike §1–§5, no independent re-derivation pass was run: every entry below went
through one read-only defender, a calibration pass (reviewer downgraded 18, upgraded 1) and an owner
gate that answered 94 needs-human items. The gate answers (`Owner decision (2026-09-18, …)` bullets
on the entries) fix the fix-shape; the per-ticket spec scoring of the fix plan is the validation point
for Tier 6, exactly as the Tier-1 execution was for §1. Line refs below are `BACKLOG.md` heading
lines at `3356da2d`; anchors inside entries are symbol-level at `077dbe2c`.

**Tiering rule.** Tiers 6–8 are ordered by what gates them, not by severity: Tier 6 is what users
can hit; Tier 7 is what the owner already said to do now and what nothing else gates; Tier 8 is
what waits for a licensed public-API break. Tiers 9–11 are gated by an unrun station (perf sweep,
alignment, test-corpus). Sev-C entries with no gate and no owner "now" land when the surrounding
code is next touched (§Deprioritized below), the same rule §2 applied.

### Tier 6 — user-visible and latent-correctness (first epics of the wave)

Every row is alignment-neutral in effect: no seam, crate or dispatch topology moves, even where the
entry carries a provisional `advances-0a` tag for Epic 9's bookkeeping.

| ID | Register | Defect | Fix-shape (owner-decided) | Effort | Parity / contract bar |
| --- | --- | --- | --- | --- | --- |
| CD-082 | `:4557` | `cobre validate` exits 0 for three `ResolvedParametersError` classes whenever `policy.boundary` is set: the two validate paths build `StudySetup` against an empty scalar-parameter table, so `MissingSeason`, `PerStageBlockCoverage`, `MissingSpecificProductivity` surface only at `cobre run`. | Make the scalar table a constructor input of `StudySetup::new` / `new_with_boundary_requirements` (unpatched params unrepresentable); the fail-loud leg is a construction-time check at the LP-build / admission site (R6-nh-2) — `ResolvedParameters::get` stays infallible. | M | parity goldens, permute harness, `mpiexec -n 1/2`; both front ends' `validate` gain the three rejections (Python `cobre.io.validate` in lockstep). |
| CD-029 (prior, sharpened) | `:1244`; sharpened at `:5307` | Boundary reconciliation bypasses `PrepPhase` through two hand-mirrored front-end copies; `cobre validate --json` emits NO error object on a boundary reject; the CLI copy reads the checkpoint twice. | Fold the boundary check into cobre-sddp `PrepPhase` as its fourth phase (L3, R4 hold closed) with a fourth `prep_phase_metadata` row (R5-nh-4); both front ends call it; emit the `--json` error object and add a `--json` reject test (R5-nh-20); correct the "four steps" prose. | M | needs-rebaseline: the `--json` object gains a phase — CHANGELOG describes the contract change. |
| CD-091 | `:5647` | The CLI's `LoadError` → kind map is a 3-arm match with a catch-all to exit 4 and NO `--json` object for `ParseError` / `SchemaError`; the bindings keep a private total 5-variant map. | One kind map in cobre-io beside `LoadError` (R5-nh-6); both front ends call it; every CLI early return routes through `emit_validate_json` under `--json`; `CaseValidationError` retired, no alias (R5-nh-5). | M | needs-rebaseline: `--json` `error.phase` vocabulary changes — CHANGELOG. `test_validate.py:222` pins the Python kinds; nothing asserts `CaseValidationError`. |
| CD-095 | `:5719` | `load_simulation` / `load_simulation_arrow` with `entity_type=None` iterate a hand-kept 10-name `ENTITY_TYPES` and silently omit four written families (`hydro_bus_generation`, `in_transit`, `transit_seed`, `anticipated_lanes`) while their docstrings promise all. | cobre-io exposes the family names it writes beside `SIMULATION_FAMILIES`; both readers iterate it. New read-side regression on a deck with travel-time arcs and post-study stages. | S | read-only: all three write-path parity layers hold by construction. |
| CD-096 | `:5733` | `load_convergence` drops `mean_rows_in_lp` while claiming schema parity; the key-presence test cannot see it. | Iterate the parquet file's own schema fields (the arrow sibling already does); regression asserting returned keys == written field names; the arrow doc table states the set by reference. | S | read-only. |
| CD-086 | `:4926` (5c) | Under `Traversal::Enumerated` no `slot_increments` fold and no `sync_stage_metadata`, so `CutPool::record_binding` never fires: the DCS resident-set seed and `enforce_budget`'s eviction key degrade silently on a pairing no gate rejects. | Road (a): typed admission-gate rejection of enumerated + dynamic cut selection beside the existing enumerated preconditions in `setup/mod.rs`; named test + `.claude/rules/sddp.md` entry; eviction reader in scope (R6-nh-19/20). | S–M | byte-neutral on every shipped golden (no deck pairs them). |
| CD-084 | `:4754` (5b) | `fill_anticipated_columns` keys the deposit slot off the raw delivery axis; the three sibling residue owners use `PointResolution::ring_index`. Latent (the diverging deck is rejected at validation) but a wrong column bound compiles. | One ring-residue walker in `lp/builder` (fpha_cursor.rs style) that the row fill, column fill and `build_anticipated_slot_row_pos` all drive (R6-nh-38). | S | byte-neutral against every golden; cites the ring-axis and column-bound-pinning contracts. |
| CD-089 | `:5612` | The simulate-arm gate is written twice in non-equivalent form (CLI: `n_scenarios > 0`; bindings: `config.simulation.enabled && n_scenarios > 0`), agreeing only through the setup normalization. | The engine answers the phase plan as one small owned value (trained-then-simulated / simulate-from-policy / nothing); both L4 entry points consume it; each keeps its own no-op rendering (R5-nh-1). | S | CLI-vs-Python value golden, file-set parity, 18-name floor. |
| CD-090 | `:5629` | The solver-stats log-to-totals fold is duplicated line for line in cobre-cli and cobre-python; the bit-for-bit `total_lp_solves` caveat lives as prose in one copy. | One fold in cobre-sddp `solver_stats.rs` over `&[SolverStatsLogEntry]` with the rank filter as an argument; the caveat becomes a doc + named regression. | S | value golden. |
| OD-040 residue | `:5218` (5d) | Four `cast_sign_loss` allows in `resolve_fitting_bounds` are unmitigated: a negative discretization count wraps past the `< 2` / `< 1` guards into `build_grid`. | Non-negativity validation of the four discretization counts at the cobre-io input boundary (the R6-nh-30 re-file lands as code, recorded on the entry); the 42-site rationale sweep is NOT done (R6-nh-29: mirror prose corrected down to D4 at E11). | S | byte-neutral; CHANGELOG: previously accepted negative counts now fail validation. |

Not in Tier 6 although B or A-risk: CD-004 (single Config projection), CD-005 (StudySetup split behind
a parity re-baseline), CD-025 (shared output orchestration in cobre-io) and CD-079 (StageTemplate
sheds five fields) — they ARE the Phase-0a/0b work and wait for Tier 10; the perf B rows wait for Tier 9.

### Tier 7 — owner-directed now-fixes and ungated Sev-B structure (same plan, later epics)

| Group | IDs | Owner direction at the gate | Effort |
| --- | --- | --- | --- |
| Doc-only corrections that land now | CD-076 (L0 vocabulary reword, R10), CD-028 (`claim_scatter.rs` consumer list, R6-nh-21), CD-083 false rustdoc line, OD-038 + `BlockGrid::advance_fpha_base` rustdoc (R6-nh-13/14), CD-080 README row (keep the mapping unconditional) | now-fix; the CD-083 / OD-038 removals themselves are Tier 8 | S |
| Deletions the owner ordered now | OD-032 `is_homogeneous` + 4 tests (R8); TD-039 `clp_only_smoke.rs` (R11); TD-051 `test_mpi_allgatherv_nonuniform_workers.rs` (R6-nh-24/43); `StudySetup::set_budget` (CD-005 R6-nh-1) | delete now; coverage-neutral | S |
| Ungated Sev-B structure | CD-078 (CLP basis-code constants, one owner in `ffi/clp.rs`); OD-035 (HiGHS retry tolerance pair → one helper, FFI order verbatim, pinned by the escalation tests); CD-085 (typed `commitment_hold_{incoming,outgoing}_col` resolvers, R6-nh-11); CD-087 private half (intra-crate alias retirement, `lp` aliases first, R6-nh-27) | as recorded | S / S / S / M |
| Backward-pass symmetry pair (priors, neutral) | CD-022 (successor-outcome reification inlined twice), CD-015 (by-scenario merge-and-commit tail vs `by_node_finish`), CD-016 delta (the two diverged `slot_increments` fold bounds), CD-014-remnant relief | as recorded at the sddp gate; hot-path code — `.claude/architecture-rules.md` applies | M |
| Test-side items the owner said "now" | TD-057 + TD-065 (`crates/cobre-cli/tests/common/`, R5-nh-22/23); TD-044 (per-binary hoist, R6-nh-8); TD-048 (parameterize in place, R6-nh-17); TD-064 + TD-063 (`*_lines` helpers, delete the `format_*_string` twins, R5-nh-35) | now; the cross-binary lift stays E08's call | S–M |

### Tier 8 — licensed public-API break batch (CD-019 precedent)

Every 0.x minor release licenses a public-API break and the unreleased section already removes the
`report` / `summary` subcommands and their Python mirrors, so the batch is not gated on a release
decision: owner decision 2026-09-19 (no release-scoping cutoff) puts it in the wave after Tiers 6–7.

| ID | Removal | Consumers to update |
| --- | --- | --- |
| OD-033 (B) | CLP hot-start acquire half: `cobre_clp_mark_hot_start`, `cobre_clp_solve_from_hot_start`, the two safe wrappers, their harness tests (R6) | none in production |
| OD-037 (C) | `Col`, `Row` newtypes + 3 tests (R6-nh-12) | `lp/indexer/mod.rs` re-export, lib.rs |
| OD-038 (C) | `FphaRowRange` + smoke test (R6-nh-13) | re-export; docs corrected in Tier 7 |
| CD-083 (C) | `CutManagementConfig::warm_start_cuts` (R6-nh-3) | two production literals, `train_inner` reset, test literals |
| CD-087 public half (C) | `pub use policy::orchestration` (R6-nh-27) | cobre-cli, cobre-python, the corpus, `scripts/ci/check_python_parity.py` literal |
| CD-019 / mirror "Superseded cut-sync public methods" | `sync_cuts`, `pack_local_records`, `sync_packed_records` and their tests (E5 dup-of, no new id) | none in production; `test_mpi_sync_cuts_invariant.rs`'s count-mismatch copy retires with them |

### Tier 9 — performance rows, gated on the performance sweep (E10)

All UNMEASURED by rule; the sweep measures each at its layout on the sanctioned decks
(`measurements/CAL*`) and the fix ticket lands only IMPROVED or NEUTRAL with the three bars green.
Owner shaping already recorded: PD-036 → `NodeGraph` field (R6-nh-6); PD-039 is a warm-start-chain
change that must clear `opening_order_determinism` (R6-nh-15); PD-042 keeps the over-inclusion,
shared-block fix only (R6-nh-23); PD-045 `StageContext` span only (R6-nh-32); PD-050 proceeds as a
supported bulk reader (R5-nh-14); PD-032/PD-034 shape questions (index-scoped CLP bound writer, bulk
basis accessor) are answered by the fix ticket against the profile (R15).

**Measured 2026-09-20 (E10-3, register pin `077dbe2c`, shared 4t recording `measurements/SWEEP-4T/`,
PD-047 on its own 2t enumerated recording).** The profile is LP-solve-bound (HiGHS simplex internals
own every top self-% symbol); the hottest Cobre symbol any claim names is `fill_col_state_patches` at
0.010% of user-space samples. Outcome for this tier:

- **Closed not-material — no fix ticket:** the nine sddp B rows PD-036, PD-038, PD-039, PD-040,
  PD-041, PD-042, PD-044, PD-045 and PD-047, plus the parked PD-004 (`run_enumerated_backward` alloc
  site 0.139% of samples). Each entry carries a dated `→ MEASURED` bullet with the basis; the
  fix-shape stays recorded, unpromoted. The Sev-C record-only rows (PD-035, PD-043, PD-046, PD-048;
  PD-052…PD-055) never had a claim row and stay as recorded (PD-037 dropped at W17).
- **UNMEASURED / case-infeasible — owner call open:** PD-032, PD-033, PD-034 profile the CLP
  backend, which the pinned HiGHS profiling binary does not link (a `--features clp` rebuild is a
  different binary); PD-049, PD-050, PD-051 profile cobre-python, a workspace-excluded cdylib the
  `cobre run` harness never loads (needs a Python-driven profile). Each entry carries a dated
  `→ UNMEASURED / case-infeasible` bullet and a needs-human line. Options: (a) commission the two
  extra profiles (a CLP profiling build on the same decks; a `pytest`-driven profile of the three
  readers), (b) close them by the materiality rule as setup/read-side work outside the training
  wall, or (c) leave UNMEASURED and land the fixes only when the code is next touched. This document
  does not decide; nothing here schedules until the owner does.
- **Collective 2x2 rows** (PD-008, PD-017, core-io) stay `measured: false` in
  `measurements/claim-table.json`; both were FIXED 2026-09-15 (W6), so the pending E10-4 run would
  only confirm the post-fix cost. Low value; the owner may close E10 without it.

The pre-measurement table below is kept for the record.

| Station | B rows (layout `4t` unless noted) | Sev-C rows (record only) |
| --- | --- | --- |
| solver-comm | PD-032 (`cobre_clp_chg_bounds` per-crossing alloc), PD-033 (CLP `add_rows` scratch), PD-034 (`ClpSolver::get_basis` per-element FFI) | — |
| sddp | PD-036, PD-038, PD-039, PD-040, PD-041, PD-042, PD-044, PD-045, PD-047 (enumerated deck, `2t`) | PD-035, PD-037, PD-043, PD-046, PD-048 |
| cli-python | PD-049 (`load_policy` holds the GIL through decode), PD-050 (per-cell type resolution), PD-051 (`cut_matrix` per-element PyFloat) | PD-052, PD-053, PD-054, PD-055 |
| do-not-touch | PD-004 (existence-and-queue only, profile first) | — |

### Tier 10 — the Phase-0a / 0b structural cluster, gated on generalization-alignment (E9)

These are the seams Part V §V.1 (Phase 0a) and §IV.2 (the 0b carve) define; they land as ONE plan
after the alignment station adjudicates the `advances-*` rows in `stations/{sddp,cli-python}/alignment-queue.json`
and `stations/solver-comm/handoffs.json`. Nothing here is blocked on a `conflicts` hold — all four
holds were closed by their alternatives at the gates.

**Adjudicated 2026-09-20 (alignment gate ratified; `alignment/gate.md`, `alignment/alignment-ledger.json`).**
The ledger decided 243 entries: 229 `neutral`, 7 `advances-0a`, 1 `advances-0b`, 6 `advances-1`; 12
retags ratified, 0 holds, 0 overrides. Consequences for this tier:

- **Phase 0a set (W15):** CD-061, CD-088 and the Part-I entries CD-125 (the config type is
  SDDP-shaped; `crates/cobre-io/src/config/mod.rs`) and CD-127 (CLI ↔ orchestration coupling;
  `write_training_outputs`) — plus CD-051, CD-059 (FIXED in W1 / W7) and CD-089, CD-091, CD-095
  (FIXED in W11), which the ledger tags `advances-0a`; the Phase-0a plan inherits them as done seams,
  not work. CD-004,
  CD-005 and CD-025 carry no ledger row (prior ids without an `Alignment` field) and stay in W15 by
  the 2026-09-19 decision above. CD-092 was retagged `neutral` and both carriers it rode (CD-029,
  CD-091) are FIXED, so it is now ungated Sev C — it moves to W18 (§Interim waves).
- **Phase 0b:** CD-099 (build-ci, genericity crate list single-owned) retagged `advances-1 →
  advances-0b`; its fix is a script + prose edit with no seam move, so it lands in W18 and the
  `cobre-model` carve simply inherits a single-owned list.
- **Phase 1 set (W19, after 0b):** CD-065, CD-070, CD-071, CD-079 (StageTemplate shed — the R13
  timing question resolved as Phase 1), TD-040 (its fixture collateral, so it leaves the test-corpus
  slice) and the Part-I entries CD-119, CD-120, CD-122, CD-123, CD-124, CD-126; CD-044 (FIXED in W7)
  is inherited. CD-121 (`HorizonGraph` framing) is `neutral` and
  stays record-only until Phase 1 touches it.
- **lp/ share amendment** (Part IV.2): the fifth-to-a-quarter estimate is superseded by the measured
  42–56% engine-neutral-or-mixed share at `077dbe2c`; extraction stays priced as a rewrite. E11 writes
  it back with the nine dated Part-I amendments.

| ID | Sev | Direction fixed at the gate |
| --- | --- | --- |
| CD-004 (+ OD-043 deferred to it) | B (A-risk) | one Config projection; `BroadcastConfig` and `StudyParams::from_config` stop being hand-kept twins; `broadcast.rs` placement resolves with it (R5-nh-34); CD-024-successor rides it (R4) |
| CD-005 | B | `StudySetup` lifecycle split with the NCS fields moved as one unit, behind a parity re-baseline naming all three bars (R5) |
| CD-025 (+ CD-088 rides it, R6-nh-28) | B | shared output-orchestration entry point in cobre-io (L2): one call-site list of outputs and guards, both front ends wire through it; emit-condition coverage waits for it (R5-nh-18) |
| CD-079 | B (A-risk) | `StageTemplate` sheds `n_state`/`n_transfer`/`n_dual_relevant`/`n_hydro`/`max_par_order`; E9 decides whether it sheds now or once at the 0b carve (R13); TD-040 and TD-037's I.3-8 collateral travel with it |
| CD-092 | C | rendered prep-phase message owned once (rides CD-029 / CD-091) |
| W10 (core-io / stochastic) | — | CD-061, CD-065, CD-070, CD-071 as already scheduled in §5 |

### Tier 11 — test-corpus rows, gated on the test-corpus station (E08)

`testing-architecture.md` §5.1 (homing threshold, binary consolidation) must be ratified first (§Tier 4
step 0 left it a proposal). Rows: solver-comm TD-035, TD-036, TD-037, TD-038, TD-040; sddp TD-041,
TD-042, TD-043, TD-045, TD-046, TD-047, TD-049, TD-050, TD-052, TD-053, TD-054, TD-055, TD-056;
cli-python TD-058, TD-059, TD-060, TD-061, TD-062, TD-066, TD-067, TD-068, TD-069, TD-070, TD-071,
TD-072, TD-073 — plus W9's 17 rows from §5. Owner directions already fixed: TD-053/TD-056 StubComm
pair into cobre-sddp `test_support` (R6-nh-25/36); TD-054 cfg-gated declarative macro (R6-nh-26);
TD-042 uniform spread + one canary literal (R6-nh-7); TD-045 consolidate behind `test-support`
(R6-nh-9, workflow fix at E07); TD-037 `tests/common` module in cobre-solver; TD-047 one extracted
`tests.rs` per module (R6-nh-16); TD-055 checklist to prose (R6-nh-35); TD-059 pinned 1dtoy numbers
retired (R5-nh-36); TD-072 `__file__`-resolved paths (R5-nh-29). TD-039, TD-044, TD-048, TD-051,
TD-057, TD-063, TD-064, TD-065 were pulled forward into Tier 7 by the owner's "now" answers.

**Ratified 2026-09-19 (test-corpus gate; `stations/test-corpus/gate.md`, STATION 8 in the register) —
but `testing-architecture.md` §5.1 stays a Proposal.** The station ran on the 72 queued seeds (50
folds ratified as the entries' current reading, 10 retired to Cleared — TD-004, TD-005, TD-007,
TD-009, TD-014, TD-017, TD-018, TD-023, TD-024, TD-029 — 4 deletion-only fix-shapes refused and
re-stated as folds under the **Rule 2 Boundary carve-out**: a fold that keeps every distinct assertion
in the retained sibling may lower the nextest count; additive shapes may raise it) and minted 8 ids:
TD-074 / TD-075 (mpi-slurm.yml provision-vs-select mismatch and cut/ path filter; Sev B), TD-076,
TD-077, CD-116, CD-117, CD-118, OD-051 (Sev C; the two prose-drift rows route their doc side to E11).
Every homing-threshold question (R12-nh-s51-threshold: TD-021, TD-031 and the CD-007 inline giants)
was answered "not at this gate — §5.1 stays a Proposal"; every other row's fix-shape is settled by
the gate and independent of §5.1. The tier therefore splits:

- **Gated remainder (W16, unchanged trigger):** TD-021 (codec/checkpoint tests homed in `policy/mod.rs`
  — destination form is the §5.1 inline-vs-sibling call) and TD-031 (the homing-threshold entry
  itself), with the deprioritized CD-007 inline-test giants riding the same ratification.
- **Ungated slice → W18 (§Interim waves):** the other 14 W9 rows (TD-017 retired), 30 of the 31 W16
  rows listed above (TD-040 travels with CD-079 to W19), the 8 minted ids and CD-092. Coverage bar for every fold: nextest count parity under
  the Rule 2 Boundary carve-out, stated per ticket with the before-count recorded.

### Deprioritized (Sev C, no gate, no owner "now" — land when the area is next touched)

- solver-comm: CD-075 (lints-table drift checker in `scripts/ci`, "keep forbid + checker" — a build-ci
  station input), CD-077, CD-081, OD-034.
- sddp: OD-036 (`ncs_stochastic_dormant_for_test` visibility), OD-039 (rank-distribution parameter;
  the two-comment correction goes to E07, R6-nh-41), OD-041 (tailrace trio markers, crate-wide pass
  R6-nh-31), OD-042 (`SimulationInputs::new` wrapper), CD-088 rides CD-025.
- cli-python: CD-093 (seven doc fences made honest, no `[lib]`, R5-nh-8), CD-094 (three dead
  re-exports in `summary.rs`), CD-097, CD-098, OD-044, OD-045 (two over-broad cast suppressions; the
  mirror prose goes to E11, R5-nh-19), OD-046, OD-047 (complete the three twins, R5-nh-21), OD-048.
- Prior sddp keeps at C: CD-018, CD-021, CD-023, CD-007 (inline-test giants — E08's homing question),
  CD-012, CD-030, CD-037, CD-038, OD-009; cli-python: CD-002 (downgraded B → C), CD-009.

### Interim waves (checker vocabulary, continues §5's table; for the unified-roadmap station to lift)

| Wave | Entry | Findings | Depends on | Phase | Effort | Trigger/deadline |
| ---- | --- | --- | --- | --- | --- | --- |
| 11 | W11-tier6-user-visible | CD-082, CD-084, CD-086, CD-089, CD-090, CD-091, CD-095, CD-096, OD-040 (+ prior CD-029) | - | neutral | M | SATISFIED 2026-09-21 (`feat/quality-wave-ungated`) |
| 12 | W12-now-fixes-and-ungated-b | CD-076, CD-078, CD-080, CD-085, OD-032, OD-035, TD-039, TD-044, TD-048, TD-051, TD-057, TD-063, TD-064, TD-065 (+ priors CD-028, CD-015, CD-022, CD-016, CD-014-remnant, CD-005's `set_budget`) | W11-tier6-user-visible | neutral | M | SATISFIED 2026-09-21 (`feat/quality-wave-ungated`) |
| 13 | W13-public-api-break-batch | OD-033, OD-037, OD-038, CD-083, CD-087 (+ prior CD-019 cut-sync methods) | W12-now-fixes-and-ungated-b | serves 0b | S | SATISFIED 2026-09-21 (`feat/quality-wave-ungated`) |
| 14 | W14-perf-sweep-gated | PD-032, PD-033, PD-034, PD-036, PD-038, PD-039, PD-040, PD-041, PD-042, PD-044, PD-045, PD-047, PD-049, PD-050, PD-051 | - | neutral | M | MEASURED 2026-09-20 (E10-3): PD-036/038/039/040/041/042/044/045/047 closed not-material, no fix ticket; PD-032/033/034 + PD-049/050/051 UNMEASURED case-infeasible — owner call (§Tier 9). Not satisfied; nothing schedules |
| 15 | W15-phase-0a-0b-structural | CD-061, CD-088, CD-125, CD-127, OD-043 (deferred, trigger CD-004) (+ priors CD-004, CD-005, CD-025; CD-051, CD-059, CD-089, CD-091, CD-095 inherited as FIXED) | W11-tier6-user-visible | serves 0a | L | UNBLOCKED 2026-09-20 (alignment ratified); one Phase-0a plan, after W18 by owner sequencing (§9). CD-079 → W19, CD-092 → W18 |
| 16 | W16-test-corpus-5-1-gated | TD-021, TD-031 (+ the deprioritized CD-007 inline-test giants) | W18-test-corpus-and-build-ci | neutral | S | `testing-architecture.md` §5.1 (homing threshold) ratification — still a Proposal after the 2026-09-19 gate; every other former W16 row moved to W18 |
| 18 | W18-test-corpus-and-build-ci | test-corpus: TD-001, TD-003, TD-006, TD-008, TD-010, TD-011, TD-012, TD-013, TD-015, TD-019, TD-022, TD-026, TD-027, TD-033, TD-035, TD-036, TD-037, TD-038, TD-041, TD-042, TD-043, TD-045, TD-046, TD-047, TD-049, TD-050, TD-052, TD-053, TD-054, TD-055, TD-056, TD-058, TD-059, TD-060, TD-061, TD-062, TD-066, TD-067, TD-068, TD-069, TD-070, TD-071, TD-072, TD-073, TD-074, TD-075, TD-076, TD-077, CD-116 (tree side), CD-117, CD-118 (tree side), OD-051; build-ci: CD-099, CD-100, CD-101, CD-102, CD-103, CD-104, CD-105, CD-106, CD-107, CD-108, CD-109, CD-110, CD-111, CD-112, CD-113, CD-114, CD-115, OD-049, OD-050; plus CD-092 | W12-now-fixes-and-ungated-b | neutral (CD-099 advances-0b, script-only) | L | UNGATED 2026-09-19/20 — the agreed next wave (§9); coverage bar = nextest count parity under the Rule 2 Boundary carve-out; the doc sides of CD-116/CD-118 and the README fold ride E11 |
| 19 | W19-phase-1-data-model | CD-065, CD-070, CD-071, CD-079, TD-040, CD-119, CD-120, CD-122, CD-123, CD-124, CD-126 (CD-044 inherited as FIXED; CD-121 record-only) | W15-phase-0a-0b-structural | serves 1 | L | Phase 1 (`beyond-sddp-generalization.md` V.2), after the 0b carve; adjudicated 2026-09-20, not scheduled |
| 17 | W17-sev-c-opportunistic | CD-075, CD-077, CD-081, OD-034, PD-035, PD-037, PD-043, PD-046, PD-048, OD-036, OD-039, OD-041, OD-042, CD-093, CD-094, CD-097, CD-098, PD-052, PD-053, PD-054, PD-055, OD-044, OD-045, OD-046, OD-047, OD-048 | - | neutral | S | SATISFIED 2026-09-21 (`feat/quality-wave-ungated`; PD-037 dropped, PD-043 deferred) |

Every one of the 104 new ids appears in exactly one wave (verified mechanically when this section
was written; prior ids in parentheses are not double-counted against §5). The 2026-09-21 reconciliation
adds the 36 ids minted by the build-ci, test-corpus and alignment stations (19 + 8 + 9) to W18, W19 and
W15, moves the 14 surviving W9 rows and 30 former W16 rows into W18 (TD-040 → W19), retires TD-017 and
lists already-fixed ids the ledger tagged `advances-*` (CD-044, CD-051, CD-059, CD-089, CD-091, CD-095)
only as inherited; the §5 W9 and
W10 rows are superseded as noted there.

---

## 8. Next steps (agreed with the owner 2026-09-19; supersedes §6) — step status as of 2026-09-21 in bold at the end of each row; §9 holds the live sequence

Owner decisions this section rests on: **(a)** hybrid sequencing — tier the three ratified stations,
fix Tier 6 now, and run the two evaluation stations whose inputs are already in hand alongside it,
before alignment, test-corpus and the unified roadmap; **(b)** no release-scoping cutoff — the wave
plans **everything that is not gated on an unrun station** (Tiers 6, 7, 8, the Sev-C W17 items and
W8 setup-perf) and lands as much as time allows; releases are cut from whatever has merged.

| # | Step | Why in this position | Inputs in hand | Owner call still open |
| --- | --- | --- | --- | --- |
| 0 | **Register housekeeping** — `3356da2d` written into the CD-074 status and ROADMAP step 3; this section and §7; ROADMAP.md 2026-09-19 sequence. | The tracker must describe the merged tree before anything schedules from it. | done in the same change | none |
| 1 | **Plan and execute the ungated quality wave** (`/plan` off `develop`): epics for Tier 6 (§7), then Tier 7, then Tier 8, then W17 and W8 as outline epics. Per-ticket bars: the parity goldens, `tests/common/permute.rs`, `mpiexec -n 1/2`, the CLI-vs-Python value golden and file-set parity, full local CI gates incl. `cargo doc -D warnings` and `check-doc-paths.sh`, cobre-python manifest build, schema regen if any schema-bearing type moves. CHANGELOG describes the two `--json` contract changes (CD-029, CD-091), the enumerated + DCS rejection (CD-086) and the negative-count rejection (OD-040) as behaviour. | User-visible defects are live (validate exit-0 gap, missing Python families/column, silent `--json` gaps); every fix-shape is owner-decided; none waits on E9/E10. | §7 tables; the gate `Owner decision` bullets; `plans/state-canonicalization/RELEASE-CHECKLIST.md` for the release bar | none — **DONE 2026-09-21: `feat/quality-wave-ungated`, 58 tickets (49 authored + 9 splits), mean quality 0.994; PD-037 dropped, PD-043 deferred; the ParquetWriterConfig removal has no CHANGELOG entry yet by owner decision (release curation)** |
| 2 | **build-ci station (E07-2 … E07-6)**, alongside step 1. | Opened at `60d10309` with inventory, gate census and prior register; inputs queued by three gates: NH9 (`invariance-shuffle.yml` without `test-support`), NH40 (hull/ as a second unsafe island vs CLAUDE.md), NH41 (`CutSelectionStrategy::Dynamic` docs reversed), R16 (`backend-testing.md` phantom cite), CD-081's genericity-gate blind spot, CD-075's lints-table checker, the `doc = false` intra-doc-link gap, the parity-script surface. Read-only; touches nothing step 1 edits. | `stations/build-ci/`, `stations/*/handoffs.json`, `tools/verify-station.sh` | none — **DONE 2026-09-19: STATION 7 ratified, 19 ids (scheduled in W18)** |
| 3 | **performance-sweep station (E10)**, alongside step 1, after step 2 or interleaved. | All five perf queues exist now (31 rows: 8 + 7 + 3 + 10 + 3); `CAL`/`CAL-ENUM` re-calibrated at the pin on the re-sanctioned decks; the 15 B rows of W14 cannot schedule until measured. Measure at the pin, not on the step-1 tree, so the claim table matches the register's baseline. | `tools/perf-run.sh`, `measurements/CAL*`, `stations/*/perf-queue.json` | per the standing rule, the owner runs any production-scale benchmark manually — **DONE for every single-process claim 2026-09-20 (22 not-material, 6 case-infeasible, PD-004 not-material); open: the 6 UNMEASURED owner calls (§7 Tier 9), the collective 2x2 rows PD-008/PD-017 (already fixed), E10-6 verify** |
| 4 | **Release** from whatever has merged, whenever the owner chooses: version-bump sites, both lockfiles, license regen, schema check, CHANGELOG cut, back-merge (`plans/state-canonicalization/RELEASE-CHECKLIST.md`). | Not a gate on anything above. | the checklist | owner runs the release — **open; `develop` now carries the wave + the ledger merge** |
| 5 | **generalization-alignment station (E09)** → then the **Phase-0a/0b plan** (W15). | Adjudicates 23 queued alignment rows and the I.3-7 / I.3-8 / I.5 handoffs; unblocks CD-004, CD-005, CD-025, CD-079 and W10. Runs after the wave so it reads the tree Tier 6 leaves. | `stations/{sddp,cli-python}/alignment-queue.json`, `stations/solver-comm/handoffs.json`, `plans/generalizing/beyond-sddp-generalization.md` | the I.3-8 shed timing (R13) — **station DONE 2026-09-20 (R13 resolved: CD-079 sheds at Phase 1); the Phase-0a plan (W15) is unblocked and sequenced after W18 (§9)** |
| 6 | **Ratify `testing-architecture.md` §5.1, then the test-corpus station (E08)** → W9 + W16 as one test-corpus wave. | 56 TD rows wait on it; the owner's "now" items are already pulled into W12. | `stations/*/td-queue.json` | §5.1 ratification — **station DONE 2026-09-19 with §5.1 still a Proposal; W9 + W16 split into the ungated W18 (next wave, §9) and the §5.1-gated W16 remainder (TD-021, TD-031)** |
| 7 | **W8 setup-perf** (PD-009 first) — folded into step 1 as an outline epic. | Ungated since W6. | §Deprioritized | none — **DONE 2026-09-21 (epic 5 of the wave; W8 SATISFIED)** |
| 8 | **reconciliation + unified-roadmap (E11)**: fold §5 and §7's waves into the register's roadmap section in `check-roadmap-dag.py` vocabulary; write the mirror back. | Last, so it lifts a settled schedule. Mirror write-backs owed: the three stations' fixed items (none recorded yet); the stale "Python-binding Rust tests invisible to CI" entry (:347) and every `testing-architecture.md` trace (R5-nh-31); the facade reserved-seam row + CLAUDE.md / ARCHITECTURE.md listings (R5-nh-32); the Legacy cost-scale seam row; the `#[allow]` census clause corrected down to comments.md D4 (R6-nh-29, R5-nh-19); the shared-filesystem deployment assumption (R5-nh-12); CD-025's destination wording; `schemas/policy.fbs` path drift; the README status vocabulary rows; the HiGHS wall-clock retry reproducibility follow-up (R12); `BroadcastNodeGraph` recorded as removed (R5-nh-17); the `LEGACY_COST_SCALE_FACTOR` nit (R5-nh-10). | `tools/check-roadmap-dag.py`, `docs/design/reserved-seams-and-deferred-debt.md` | none — **open; the write-back list grew by the E07/E08/E09 handoffs (§9 step 5)** |

---

## 9. Next steps (2026-09-21, after the wave and the ledger merge; supersedes §8's order)

State this section rests on: `develop` = the ungated wave (`04635b8f..cbc0ca10`) + the
`chore/quality-evaluation` ledger merge (`d2545190`); eight stations ratified; the single-process
perf sweep measured; §5.1 of `testing-architecture.md` still a Proposal. Owner choice recorded
2026-09-21: **reconcile the register first (this change), then plan the test-corpus + build-ci wave
(W18); the Phase-0a plan (W15) follows it; the six UNMEASURED perf rows wait for an explicit owner
call.**

| # | Step | Why in this position | Inputs in hand | Owner call still open |
| --- | --- | --- | --- | --- |
| 0 | **Register reconciliation** — this change: merge the ledger, mark W14 measured, W15 unblocked, split W16 → W16 + W18, add W19, record the 2026-09-21 section in `BACKLOG.md`. | The tracker must describe the merged tree before anything schedules from it. | done in the same change | none |
| 1 | **Plan and execute W18** (`/plan` off `develop`): the §5.1-independent test-corpus slice (44 prior TD rows + TD-074…TD-077, CD-117, OD-051, the tree sides of CD-116/CD-118, CD-092) and the 19 STATION 7 build-ci entries. Suggested epics: build-ci gates and workflows first (CD-099…CD-103, CD-112…CD-115, OD-049/OD-050, TD-074/TD-075 — they change no crate code and tighten the gates the later epics run under), then the doc-drift rows (CD-104…CD-111, TD-076), then one test-corpus epic per crate (cobre-core/io/stochastic/comm/solver/sddp/cli/python) with the owner directions already fixed at the gates (StubComm pair → cobre-sddp `test_support`; TD-054 cfg-gated macro; TD-042 uniform spread + one canary literal; TD-045 behind `test-support`; TD-037 `tests/common` in cobre-solver; TD-047 one `tests.rs` per module; TD-059 pinned 1dtoy numbers retired; TD-072 `__file__`-resolved paths; §5.11 freezes assertions, not fixtures). Per-ticket bars: nextest count parity under the Rule 2 Boundary carve-out (before-count recorded), the parity goldens where a fixture moves, full local CI gates incl. the manifest-scoped cobre-python fmt/clippy/build, `cargo doc -D warnings`, `check-doc-paths.sh`, and — for the ci.yml edits — a dry run of every changed workflow's shell steps. | Ungated since the 2026-09-19 gates; coverage-neutral by construction; every fix-shape owner-decided. | §7 Tier 11 + STATION 7/8 entries and their `Owner decision` bullets; `stations/test-corpus/lens-rules.md` (Rule 2 as restated); `stations/build-ci/gate.md` §3.5 directions | none |
| 2 | **Owner call on the six UNMEASURED perf rows** (PD-032/033/034 CLP, PD-049/050/051 cobre-python): commission the two extra profiles, close not-material by rule, or leave until next touched. | Cheap to decide; decides whether W14 closes. | §7 Tier 9; `measurements/_4t/unmeasured.md` | this is the call |
| 3 | **Plan and execute W15 (Phase 0a)** after W18: CD-004 (+ OD-043), CD-005, CD-025 (+ CD-088), CD-061, CD-125, CD-127; CD-051/CD-059/CD-089/CD-091/CD-095 inherited as fixed seams. One plan; a parity re-baseline naming all three bars for CD-005. | Alignment ratified; the wave left the tree the 0a plan reads; owner sequenced it after W18. | `alignment/gate.md` §4 handoffs; `alignment/lp-classification.md` (42–56% share); Tier 10 table | the D12 split timing for CD-004's `study` block if it collides with the release |
| 4 | **Release** from whatever has merged, whenever the owner chooses (checklist unchanged); add the ParquetWriterConfig `### Removed` entry at curation. | Not a gate on anything above. | `plans/state-canonicalization/RELEASE-CHECKLIST.md` | owner runs the release |
| 5 | **E11 reconciliation + unified-roadmap**: lift §5 + §7 + this section's waves into the register's roadmap section (`check-roadmap-dag.py` vocabulary; the section currently has no table and the checker reports `malformed-table`), then the mirror write-backs — §8 step 8's list plus: the nine dated Part-I amendments and the lp/ share figure (E09), the 10 retired seeds' resolving commits and the Rule 2 restatement (E08), the doc sides of CD-116/CD-118, the README fold and `policy.fbs` path drift (E07), the `slow-tests` twin note, the E08 §5.1/§5.5/§5.6 sequencing notes. | Last, so it lifts a settled schedule. | `tools/check-roadmap-dag.py`, `docs/design/reserved-seams-and-deferred-debt.md`, the three gates' §4 | §5.1 ratification (then W16's two rows and CD-007 schedule) |
| 6 | **W19 (Phase 1)** — not scheduled; opens after the 0b carve. | Adjudicated only. | Tier 10 Phase-1 set | none yet |
