# Prior register — cobre-stochastic (baseline `a136840d`)

What the register and the committed mirror already say about this crate, so the four
attacker passes cannot re-raise a settled item. Every heading was re-resolved against
the baseline blob (`git show a136840d:<path>`), never a remembered line number.
Dispositions: **keep** (live, ours), **not-ours** (anchored in another crate — named),
**cross-reference** (cite, do not restate), **resolved** (closed at baseline). Each entry
carries a `reraiseKey` token list `check-reraise.py` matches candidate titles and anchors
against.

---

### Stage-calendar crate home

- **Recorded in**: `docs/design/reserved-seams-and-deferred-debt.md` L449 (baseline `a136840d`)
- **Recorded shape**: the season/stage-calendar machinery may belong in cobre-core rather than cobre-stochastic; a relocation is deferred until it is worth its scope.
- **Recorded owner**: the stochastic / temporal owner
- **Recorded trigger**: when the season-machinery relocation is worth its scope
- **Baseline anchors**: `crates/cobre-stochastic/src/season_cast/mod.rs` (672 non-test lines) — resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: keep
- **Evidence at baseline**: `season_cast/mod.rs` still lives in this crate and holds 672 non-test lines (`awk '/^#\[cfg(test)\]/{exit}{n++}END{print n}'`); the relocation has not happened.
- **Attacker note**: a candidate that re-proposes the relocation is a **re-raise** unless it adds the L2→L1 layering evidence recorded in `## Layering evidence` below (i.e. argues homing on measured consumer edges, not a bare "feels misplaced").
- **reraiseKey**: `stage-calendar`, `season_cast`, `StageCalendar`, `crate home`

---

### External-noise take/fill glue duplication

- **Recorded in**: `docs/design/reserved-seams-and-deferred-debt.md` L460 (baseline `a136840d`)
- **Recorded shape**: the take/fill glue that threads external noise through the backward pass is duplicated across the two branching-mode backward drivers.
- **Recorded owner**: the cobre-sddp training owner
- **Baseline anchors**: `crates/cobre-sddp/src/training/backward/by_node.rs`, `crates/cobre-sddp/src/training/backward/by_scenario.rs` — resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: not-ours
- **Evidence at baseline**: both anchors are under `crates/cobre-sddp/`, not `crates/cobre-stochastic/`; the duplication is in the SDDP backward drivers, so this crate's attackers must not raise it.
- **Owning station (if not-ours)**: cobre-sddp
- **Attacker note**: a stochastic candidate touching external-noise take/fill is only in scope if its anchor is inside `crates/cobre-stochastic/src`; anything under `cobre-sddp/training/backward` is this entry and belongs to that station.
- **reraiseKey**: `take/fill`, `external noise glue`, `by_node`, `by_scenario`, `backward`

---

### Deterministic (σ = 0) AR(p > 0) external inflow stays rejected

- **Recorded in**: `docs/design/reserved-seams-and-deferred-debt.md` L985 (baseline `a136840d`)
- **Recorded shape**: an external inflow declared AR(p > 0) with σ = 0 is deliberately rejected rather than silently treated as deterministic; this is a ratified decision, not a validation gap.
- **Recorded owner**: the stochastic / scenario owner
- **Recorded trigger**: n/a — a standing decision, revisited only if the external-scenarios contract changes
- **Baseline anchors**: `docs/design/reserved-seams-and-deferred-debt.md` L985 (the ratified decision), `crates/cobre-stochastic/src/par/fitting` (the AR(p) fitting surface) — resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: keep
- **Evidence at baseline**: the mirror records the rejection as intended behavior with an owner and a trigger; the code path still rejects rather than coerces.
- **Attacker note**: do **not** re-raise as a "missing deterministic fast-path" or a "validation gap" — the rejection is ratified. A sharpen would need a concrete wrong-result case the current rejection causes, not a preference.
- **reraiseKey**: `sigma zero`, `AR(p)`, `deterministic external inflow`, `rejected`

---

### `LoadModel` conflates physical load with its stochastic model

- **Recorded in**: `docs/design/reserved-seams-and-deferred-debt.md` L1000 (baseline `a136840d`)
- **Recorded shape**: `LoadModel` mixes the physical load entity with the parameters of its stochastic model, so the two concerns cannot be reasoned about independently.
- **Recorded owner**: the cobre-core modeling owner
- **Baseline anchors**: `crates/cobre-core/src/model/scenario.rs` (`LoadModel` at line 368) — resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: not-ours
- **Evidence at baseline**: `LoadModel` is declared in `crates/cobre-core/src/model/scenario.rs:368`, an L0 core entity; the conflation lives in cobre-core, not cobre-stochastic.
- **Owning station (if not-ours)**: core+io (cobre-core)
- **Attacker note**: a stochastic candidate may cite `LoadModel` as context but must not raise its conflation as a finding here; that is the core+io station's entry.
- **reraiseKey**: `LoadModel`, `physical load`, `stochastic model`, `conflate`

---

### Cross-path static-RHS contract not yet in `.claude/rules/sddp.md`

- **Recorded in**: `docs/design/reserved-seams-and-deferred-debt.md` L1015 (baseline `a136840d`)
- **Recorded shape**: the contract that the static right-hand side is identical across the two backward paths is enforced in code but not yet written into the SDDP rules doc.
- **Recorded owner**: the `.claude/rules/sddp.md` owner
- **Baseline anchors**: `.claude/rules/sddp.md` (the doc owed the contract) — resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: not-ours
- **Evidence at baseline**: the gap is a documentation-debt item against `.claude/rules/sddp.md`, an SDDP-owned rules file; no cobre-stochastic source is the subject.
- **Owning station (if not-ours)**: SDDP-rules owner (surfaced at E11 documentation consolidation)
- **Attacker note**: not a code finding for this crate; do not raise the RHS-contract doc gap as a stochastic defect.
- **reraiseKey**: `static RHS`, `cross-path contract`, `sddp.md`, `rules doc`

---

### Oracle test-harness duplication

- **Recorded in**: `docs/design/reserved-seams-and-deferred-debt.md` L285 (baseline `a136840d`)
- **Recorded shape**: the extensive-form / branching value oracle test harness is duplicated across SDDP integration tests.
- **Recorded owner**: the cobre-sddp test owner
- **Baseline anchors**: `crates/cobre-sddp/tests/extensive_form_oracle.rs`, `crates/cobre-sddp/tests/branching_value_oracle.rs`, `crates/cobre-sddp/tests/common/mod.rs` — resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: cross-reference
- **Evidence at baseline**: the oracle harness lives under `crates/cobre-sddp/tests`; this crate's test-bloat lens over `tests/{halton,sobol,lhs}_integration.rs` must **cite** this entry, not restate the duplication.
- **Owning station (if not-ours)**: cobre-sddp (test corpus)
- **Attacker note**: the stochastic test-bloat lens references this entry when it touches integration harness sharing; a fresh "oracle harness duplicated" finding anchored in `cobre-sddp/tests` is a re-raise.
- **reraiseKey**: `oracle`, `test harness`, `extensive_form_oracle`, `branching_value_oracle`, `duplication`

---

### CD-001 — setup config-projection sprawl / CLI non-root reconstruction

- **Recorded in**: `plans/architecture-debt-audit/BACKLOG.md` L148 (`CD-001 · Sev A · leaky-boundary + duplication`); mirror partial-closure prose at `docs/design/reserved-seams-and-deferred-debt.md` L540 (baseline `a136840d`)
- **Recorded shape**: the CLI hand-mirrored the rank-0 stochastic pipeline across the crate boundary, risking a silent MPI-vs-local divergence.
- **Recorded owner**: the setup / CLI owner
- **Baseline anchors**: `crates/cobre-sddp/src/setup/stochastic_pipeline.rs` (`build_stochastic_context_for_study` at line 424), `crates/cobre-sddp/src/lib.rs` (re-export at line 165), `crates/cobre-cli/src/commands/run/setup.rs` (single CLI call site at line 409), `crates/cobre-stochastic/src/context.rs` (`build_stochastic_context` L1 sibling at line 601) — all resolved via `git show a136840d:<path>`, exit 0
- **Disposition**: resolved
- **Evidence at baseline**: the CLI no longer hand-mirrors the pipeline — it calls the single shared `build_stochastic_context_for_study` (definition `stochastic_pipeline.rs:424`, re-export `lib.rs:165`, sole caller `run/setup.rs:409`), and the committed mirror already records the closure in prose at `reserved-seams-and-deferred-debt.md:540` ("the silent-MPI-vs-local-divergence hazard is retired"). No live duplication remains.
- **Alignment hint (NOT a finding, tagged for the generalization-alignment epic)**: the only open question is whether an engine-neutral context constructor belongs at L1 beside this crate's own `build_stochastic_context` (`crates/cobre-stochastic/src/context.rs:601`). This travels to Epic 9 as a homing/alignment note. The residual `StudyParams`/`BroadcastConfig` and `StudySetup` god-struct halves of the mirror entry are **out of station** (setup/cli owner). **No attacker may raise CD-001 as a fresh duplication finding** — it is resolved.
- **Attacker note**: closed; any re-raise of the CLI/pipeline duplication is rejected. A stochastic candidate may only cite the L1-homing alignment hint, never a new finding.
- **reraiseKey**: `CD-001`, `build_stochastic_context`, `stochastic_pipeline`, `CLI reconstruction`, `rank-0 pipeline`

---

## Layering evidence

`cobre-io` (L2) consumes `season_cast` / `StageCalendar` from `cobre-stochastic` (L1) at
four sites, re-resolved at baseline with `grep -rn season_cast crates/cobre-io/src`:

- `crates/cobre-io/src/scenarios/estimation.rs:70`
- `crates/cobre-io/src/validation/semantic/inflow_seeding.rs:22`
- `crates/cobre-io/src/validation/semantic/thermal.rs:17`
- `crates/cobre-io/src/validation/semantic/scenarios.rs:12`

The edge is **L2 → L1**, which is **consistent** with the target layering (a higher layer
depending on a lower one). Therefore the recorded stage-calendar relocation is a
**core-vs-stochastic homing question** (should the calendar live in L0 core or stay in L1
stochastic?), **not a layering violation**. Recording this stops an architecture attacker
from raising a phantom inversion off the `cobre-io → cobre-stochastic` dependency.
