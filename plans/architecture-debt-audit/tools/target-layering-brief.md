# Target layering brief — alignment reference

Baseline: a136840d4f2ea137f685f0af6dac04254b983b60 (v0.15.0-1-ga136840d).
Source of every claim: `plans/generalizing/beyond-sddp-generalization.md`
(cited below by numbered section). Every station tags its `Alignment` field
against this page; the closed vocabulary is `advances-0a | advances-0b |
advances-1 | neutral | conflicts`, followed by the section that justifies it.

## 1. The layers (beyond-sddp-generalization.md IV.1, IV.5)

| Layer | Crates | Forbidden here | Source |
|-------|--------|----------------|--------|
| L0 foundations | `cobre-core` (paradigm-neutral data model), `cobre-solver` (`SolverInterface` + capability traits), `cobre-comm` (MPI / local) | any engine or problem concept; SDDP-, Benders- or multistage-shaped types, fields or vocabulary | beyond-sddp-generalization.md IV.1 |
| L1 shared kernels | `cobre-stochastic` (uncertainty store + scenario generation), `cobre-model` (new: indexer, builder, `VarDomain`, `BuildProblem`), `cobre-network` (new: PTDF/LODF/Ybus + `NetworkFormulation`) | engine concepts; the genericity rule extends to the two new crates only by amending the CI grep's crate list | beyond-sddp-generalization.md IV.1 |
| L2 case I/O | `cobre-io` (multi-vertical case v2, import adapters, shared output orchestration) | engine dispatch; an engine-specific output tree hard-wired in the CLI instead of orchestrated here | beyond-sddp-generalization.md IV.1 |
| L3 engines + composition | `cobre-sddp` (SDDP + SDDiP), `cobre-direct` (new: ED / OPF / deterministic UC / monolithic expansion), `cobre-study` (new, Phase 4: composition orchestrator) | a base engine depending on a sibling engine; only `cobre-study` may name both base engines | beyond-sddp-generalization.md IV.1 |
| L4 entry points | `cobre-cli`, `cobre-python` | nothing engine-shaped is forbidden: this is the only layer that owns the `Engine` enum and the dispatch | beyond-sddp-generalization.md IV.1, IV.4 |

What leaves L0 under the purification (beyond-sddp-generalization.md IV.5):
`InflowModel`, `LoadModel`, `NcsModel`, `CorrelationModel` and the external-scenario
tables leave `System` for the `cobre-stochastic` store; `StageRiskConfig` (CVaR),
`ScenarioSourceConfig` and `inflow_lags` leave `Stage` for an SDDP per-node config;
`training_event.rs` moves to a telemetry module the verticals emit into;
`InitialConditions`' PAR-lag and defluence seeds follow the uncertainty and routing
state; one layer down, `cobre-solver`'s `StageTemplate` sheds
`n_state` / `n_transfer` / `n_dual_relevant` / `n_hydro` / `max_par_order` to the
layer that owns the multistage layout, leaving the L0 container pure CSC.

## 2. The phase split (beyond-sddp-generalization.md V.0, V.1, V.2)

Sequencing principle (beyond-sddp-generalization.md V.0): pull, don't push. The
greenfield data model is the destination; a cheap second consumer (deterministic
economic dispatch on the existing case format) is the vehicle that proves the
seams before the data model breaks. Cheap and safe first; irreversible and
speculative later.

- **0a** (beyond-sddp-generalization.md V.1, split resolved by D12): the `Engine`
  seam and dispatch in `cobre-cli` / `cobre-python`; the `study` config block and
  its typed admission gate (an unsupported combination is a rejection, never a
  silent ignore); shared output orchestration in `cobre-io`; rank-0-executes MPI
  semantics for a direct study (D14); engine-tagged setup stages. Byte-identical
  behaviour for the existing SDDP engine; no second engine ships in 0a.
- **0b** (beyond-sddp-generalization.md V.1 / IV.2, D12): carve `cobre-model` out
  of the engine-neutral part of `crates/cobre-sddp/src/lp/` (indexer, builder,
  `VarDomain`, `BuildProblem`), leaving SDDP geometry (cuts, state coupling,
  Benders rows) in the engine.
- **1** (beyond-sddp-generalization.md V.2): purify the data model — stochastic
  representation off `System` / `Stage` into the `cobre-stochastic` store,
  `training_event` out of `cobre-core`, `StageTemplate` shed, case format v2 with a
  bit-for-bit shim for v1 decks.

Milestone precedence in the register: `0a` < `0b` < `1`; `neutral` advances no
phase; `gnl-import` is a trigger, not a phase (BACKLOG.md `## Milestones`).

## 3. The four `conflicts` triggers

A fix-shape that does any of the following is tagged `Alignment: conflicts` and
held for the owner gate. It is never silently downgraded to `neutral`, and it
enters the roadmap only under a dated `Owner-override:` line.

1. **Engine or paradigm concepts pushed into L0/L1.** Anything SDDP-, Benders-,
   or multistage-decomposition-shaped proposed into `cobre-core`, `cobre-solver`,
   `cobre-comm`, `cobre-stochastic`, or the future `cobre-model` / `cobre-network`.
   (beyond-sddp-generalization.md IV.1, IV.5)
2. **Engine-to-engine dependency.** `cobre-sddp` depending on `cobre-direct` or
   the reverse. Engines are siblings at L3 and share only through L0-L2; only the
   composition orchestrator `cobre-study` may name both. (beyond-sddp-generalization.md IV.1)
3. **The `Engine` enum placed below L4.** It lives in `cobre-cli` and
   `cobre-python`, the only layer that already depends on every engine crate; a
   lower-layer enum reintroduces the dependency cycle the seam design exists to
   avoid. (beyond-sddp-generalization.md IV.4)
4. **A one-consumer abstraction.** Pull, don't push: a seam, trait, or crate
   justified by a single present consumer is speculative generality, not
   architecture. The second consumer precedes the abstraction.
   (beyond-sddp-generalization.md V.0, and the speculative-generalization row of
   the V.10 risk register)

## 4. Part-I items at the pinned baseline (beyond-sddp-generalization.md I.3, I.5)

The `v0.12 anchor` column repeats the anchor Part I printed at its snapshot; the
`baseline symbol` column is the declaration that survives at a136840d and that
`check-anchors.py` can resolve. The alignment epic re-verifies each row.

| # | Claim | v0.12 anchor | Baseline symbol | Source |
|---|-------|--------------|-----------------|--------|
| 1 | `System` carries the stochastic input model (`inflow_models`, `load_models`, `ncs_models`, correlation) as first-class fields | `system/mod.rs:107` | `crates/cobre-core/src/system/mod.rs::System` | beyond-sddp-generalization.md I.3 item 1 |
| 2 | `Stage` bakes in stochastic and risk configuration (`risk_config`, `scenario_config`, `state_config`, `inflow_lags`) | `temporal.rs:281` | `crates/cobre-core/src/model/temporal.rs::Stage` | beyond-sddp-generalization.md I.3 item 2 |
| 3 | `PolicyGraph` is framed entirely around multi-stage-stochastic forward/backward traversal | `anchor-tbd` (I.3 prints no path; no `PolicyGraph` declaration exists under `crates/cobre-core/src/` at the baseline — the nearest surviving surfaces are `crates/cobre-core/src/model/temporal.rs::PolicyGraphType` and `crates/cobre-core/src/model/horizon.rs::HorizonGraph`) | `anchor-tbd` | beyond-sddp-generalization.md I.3 item 3 |
| 4 | `InitialConditions` is warm-start-shaped (`past_defluences`, `past_anticipated_commitments`; `past_inflows` removed in v0.13.0) | `constraints/initial_conditions.rs:183` | `crates/cobre-core/src/constraints/initial_conditions.rs::InitialConditions` | beyond-sddp-generalization.md I.3 item 4 |
| 5 | `training_event.rs` lives in `cobre-core` (`WorkerPhaseTimings`, `StoppingRuleResult`, `StageRowSelectionRecord`, `TrainingEvent`) | `training_event.rs:125` | `crates/cobre-core/src/constraints/training_event.rs::TrainingEvent` | beyond-sddp-generalization.md I.3 item 5 |
| 6 | `cobre-io`'s policy checkpoint format is literally cut records (the genericity gate exempts the four policy files) | `crates/cobre-io/src/output/policy/` | `crates/cobre-io/src/output/policy/mod.rs`, `crates/cobre-io/src/output/policy/records.rs`, `crates/cobre-io/src/output/policy/codec.rs`, `crates/cobre-io/src/output/policy/checkpoint.rs` | beyond-sddp-generalization.md I.3 item 6 |
| 7 | The config type is SDDP-shaped: `Config.training` and the `StudyParams::from_config` conversion | `cobre_io::Config.training` | `crates/cobre-io/src/config/mod.rs::Config`, `crates/cobre-sddp/src/setup/params.rs::from_config` | beyond-sddp-generalization.md I.3 item 7 |
| 8 | The leakage reaches below core: `cobre-solver`'s `StageTemplate` carries `n_state` / `n_transfer` / `n_dual_relevant` / `n_hydro` / `max_par_order` | `crates/cobre-solver/src/types.rs` | `crates/cobre-solver/src/types.rs::StageTemplate` | beyond-sddp-generalization.md I.3 item 8 |
| 9 | The CLI is typed on `cobre_sddp::StudySetup` with no algorithm-selection seam; the output writer lists are hand-mirrored between the CLI and the Python binding | `crates/cobre-cli/src/commands/run/` | `crates/cobre-cli/src/commands/run/outputs.rs`, `crates/cobre-cli/src/commands/run/policy.rs`, `crates/cobre-python/src/run.rs` | beyond-sddp-generalization.md I.5 |
