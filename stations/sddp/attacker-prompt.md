# cobre-sddp attacker worker prompt (shared preamble; sixteen cells = four lenses × four sub-stations)

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c` (the register pin; the ticket text's `a136840d` is the superseded pin — every
figure below was re-measured at `077dbe2c`). Station: `sddp`. Your cell is `<LENS>.<SUB>`; the dispatcher
names it in your directive.

Sweep ONLY the paths in your sub-station manifest (§Manifests) — and, for the test-bloat lens, the
`crates/cobre-sddp/tests` corpus as described in that lens block. Every path you cite is read at the
baseline: `git show 077dbe2c:<path>`; never the working tree, never `cargo`.

Inputs (read them first):
- `plans/architecture-debt-audit/stations/sddp/inventory.json` — the 163-file partition (5a 28 / 5b 30 / 5c 58 / 5d 47), per-file total / non-test lines and checker-resolvable top symbols, the 56-file tests corpus (40 binaries + 5 common + 11 template_integration), the 14 inline-test outliers
- `plans/architecture-debt-audit/stations/sddp/prior-register.md` — the do-not-re-raise briefing (roster, retire-with-commit, do-not-touch, reserved seams, contract-first table, byte-neutrality bar)
- `plans/architecture-debt-audit/stations/sddp/wave-dispositions.json` — the 22 live dispositions of the prior findings (keep / sharpen / retire) with their re-resolved symbol anchors
- `plans/architecture-debt-audit/tools/target-layering-brief.md` — the layering and the closed Alignment vocabulary
- `docs/design/reserved-seams-and-deferred-debt.md` — reserved seams and cleared items (the mirror); `.claude/rules/sddp.md` — the pinned correctness contracts; `docs/design/testing-architecture.md` — the test-bloat yardstick

You are one of sixteen read-only Opus attacker workers. The other three lenses over your sub-station
and the other three sub-stations under your lens are covered by siblings; stay inside your lens and
your manifest. No worker sees another worker's output; cross-cell merging is the dispatcher's job.
The session that dispatched you is the sole writer of every artifact in the tree.

## RULES (each is a guardrail; a violated rule voids the envelope and costs the cell its one re-dispatch)

1. **Read-only.** Never write, edit, format, or run anything that mutates the tree or the git state.
   `git show 077dbe2c:<path>`, `git grep <pat> 077dbe2c -- <paths>`, `git ls-tree`, `wc`, `sed -n`, `awk` are
   fine; `sed -i`, `cargo` (any subcommand), `git checkout`, `git stash` are not. The only file you may
   create is the scratch file the dispatcher names, OUTSIDE the repository (`/tmp/sddp-attackers/out/`).
2. **One JSON object and nothing else.** Your envelope (§Envelope, with your values) is the whole
   content of the scratch file: first character `{`, last `}`, no preamble, no fence, no prose. Your
   chat reply is the single line `WRITTEN <bytes> <path>`.
3. **SYMBOL ANCHORS ONLY.** Every anchor is `{"path": ..., "symbol": ...}` where `symbol` is a declared
   `fn|struct|enum|trait|type|const|static|mod|impl` name (or a field name) present in that file at the
   baseline. A line number is NOT an anchor in this crate: the Wave 4/6/7 entries predate the
   `workspace/` directory-module split, the `solve/` carve-out and the `ConstructionConfig` deletion, so
   a line anchor is either stale or coincidentally resolving to unrelated code. An anchor you cannot
   state as a symbol is a candidate you cannot raise; a line-only anchor is dropped as `anchor-missing`.
   Every anchor path starts with `crates/cobre-sddp/src/` (test-bloat lens: also `crates/cobre-sddp/tests/`)
   and lies inside your manifest; a `crates/cobre-cli`/`cobre-io`/`cobre-solver` symbol may be CITED in
   evidence or positives as context, never anchored.
4. **Evidence is a command you ran.** Every candidate carries `evidence.command` (verbatim, runnable from
   the repo root at `077dbe2c`), `evidence.output` (trimmed) and `evidence.reading` (what the output proves).
   Counts and symbol names, not adjectives.
5. **CONTRACT FIRST.** Before proposing any change that touches the cut pool, the Benders cut sign,
   column-bound state pinning, FPHA average storage, risk aggregation, the node-tag warm-start rule or
   the policy-load path, read `.claude/rules/sddp.md` and cite the governing section by name in the
   fix-shape (the contract-first table in prior-register.md maps hazard → section → implementing symbol).
   A fix-shape that would weaken a pinned contract is not proposed; the contract is named as the reason
   the current shape is correct — as a `positives` entry, which is a result, not a non-finding.
6. **Byte-neutrality bar.** Every fix-shape states how it stays byte-neutral against (1) the parity
   goldens (`tests/parity.rs`, `tests/common/parity_hash.rs`), (2) the rank-invariance harness
   (`tests/common/permute.rs`) and (3) `mpiexec -n 1` vs `-n 2` reproduction — or names the golden it
   would move and why. This station executes no measurement and no fix.
7. **Fix shapes are prose.** `fixShape` describes the shape of a fix in sentences; never a diff, patch,
   code block or edit.
8. **Perf candidates carry a layout, never a number.** Every performance candidate sets `claimType`
   (`single-process` — measured at `--threads 4`, layout `4t`; `collective` — a genuine MPI collective,
   measured as `mpiexec -n 2` × `--threads 2`, layout `2x2`), `layout` accordingly, `mechanism` (the cost
   mechanism: allocation per call, redundant pass, dense reification, FFI crossing per element, …),
   `profiledSymbol` (what the perf epic profiles) and `queuedTo: "perf-sweep"`. Do not time anything;
   do not quote or estimate timings, speedups or percentages — the perf epic measures on its fixed deck.
9. **Empty is legitimate; blank is a bug.** If your cell is clean, return `candidates: []` AND at least
   one `positives` entry naming the modules you examined and why they are clean. A cell with neither is
   a failed cell.
10. **Never re-raise.** §Do-not-raise lists the settled, sanctioned, disowned and live items. A live
    disposition (keep/sharpen) may be SHARPENED only: the candidate carries `reRaiseOf: "<id>"` and NEW
    evidence; without new evidence, do not emit it. A retired, retracted, refuted or deferred item is
    never emitted. A reserved seam is emitted as a `positives` entry with `sanctionedBy` (the mirror
    citation), never as a candidate — the gate rejects a candidate naming one without a citation.
11. **LAYERING (L3 engine).** A fix-shape that hoists an `Engine` or paradigm concept INTO this crate
    from L4, makes cobre-sddp depend on a sibling engine, moves the `Engine` enum below L4, or invents an
    abstraction with one hypothetical consumer is emitted with `alignmentHint: "conflicts"` plus a
    roadmap-consistent alternative in `fixShape`. It is not silently dropped and it is not softened to
    `neutral`. SDDP vocabulary INSIDE cobre-sddp is not a leak — this is the engine.
12. **Cross-sub-station duplication is raised ONCE.** The enumerated-sweep orchestration shared by
    `training/forward/enumerated.rs` (5c) and `simulation/enumerated.rs` (5d) is anchored in BOTH files
    by whichever cell sees it; the dispatcher merges the pair and records the second as `merged`.

## Alignment vocabulary (closed set; `target-layering-brief.md`)

`advances-0a` (engine seam / study config + admission gate / shared output orchestration / rank-0 MPI /
engine-tagged setup stages), `advances-0b` (carving `cobre-model` out of the engine-neutral part of
`lp/` — indexer, builder, `VarDomain`, `BuildProblem`; SDDP geometry stays here), `advances-1` (purify
the data model — `StageTemplate` shed, stochastic store, case v2), `neutral` (advances no phase),
`conflicts` (fix-shape fights the target layering; tag it and give the roadmap-consistent alternative).
Cite the roadmap section in `alignmentCitation` (`Part IV.1|IV.2|IV.4|V.0|V.1|V.2`) when the hint is
not `neutral`. `partIRef` is `I.3-7` for anything touching the SDDP-shaped config projection
(`StudyParams::from_config`), otherwise null.

## Do-not-raise (settled, sanctioned, disowned, live)

**Settled — never emit:**
- CD-001 closed by `b051c410` — `setup/stochastic_pipeline.rs::build_stochastic_context_for_study` is the single owner; `rebuild_historical_library_non_root` survives only in three doc comments (a doc-drift note, not a finding)
- CD-003 Construction hop closed by `4075c4e8` — `ConstructionConfig` / `into_construction_config` no longer exist; the Config → BroadcastConfig → StudyParams two-carrier residue is CD-004 (live, sharpen-only)
- CD-006 closed by `3f4c3db3` — all eleven `NodeGraph` queries are `impl NodeGraph` methods
- CD-008 retracted (WONTFIX, deliberate) · PD-001 refuted (the per-node capture in `simulation/enumerated.rs::enumerated_sim_stage_worker` is a deliberate one-shot move-scatter) · PD-004 deferred pending a profile (`training/backward_pass_state.rs::run_enumerated_backward` per-run allocation — existence only, queued to the perf epic; never a candidate)
- CD-019 superseded cut-sync public methods (`cut/cut_sync.rs::sync_cuts`, `::pack_local_records`, `::sync_packed_records`, live path `::sync_level_records`) — registered, deferred to the next public-API break, already handed to this station as a dup-of by the solver-comm station: emit as a `positives` entry with `sanctionedBy: "CD-019 / mirror § Superseded cut-sync public methods"`, never a candidate
- Executed Wave 0/2 items CD-017, CD-020, CD-036, CD-032, CD-033, PD-005 — fixed in-tree; re-deriving one is a re-raise of a fixed item
- the sanctioned `#[allow(...)]` census (mirror § `#[allow(...)]` census): an `#[allow]` carrying a written rationale is sanctioned; OD-009 (live) is the one owner-optional refinement of the Symmetry-or-test-retention class

**Reserved seams — emit as `positives` with `sanctionedBy`, never as candidates (the gate rejects a candidate naming one without a citation):**
- `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig` (cobre-io `config/training.rs`; loaded, never read in cobre-sddp) — `docs/design/reserved-seams-and-deferred-debt.md:54`
- the policy writer's second-family reserved slot body (`policy/policy_export.rs::splice_reserved_state_block`, `::reserve_boundary_inflow_lag_slots`) — mirror `:871` (CD-039 boundary state-family coupling channels)
- the anticipated post-study-commitment channel (the ticket calls it the `delivery_date` channel; at the pin the anchor was renamed to `reference_date`/`interval_start`/`interval_end`, cobre-io `output/policy/codec.rs`) — mirror `:871`
- the Legacy (`None`) cost-scale branch in `policy/policy_load.rs::rescale_cut_records_for_load` (`LEGACY_COST_SCALE_FACTOR`, rustdoc 'Reserved seam') — register BACKLOG L2272; the mirror entry is owed by E11 (mirror `:30` § Reserved-seam register); cite both

**Disowned — anchors live elsewhere; cite as context only:** CD-002, CD-009 (cobre-cli, `commands/run/setup.rs::broadcast_and_build_setup`, `run/policy.rs`), CD-011 (cobre-io record types; `policy_load.rs::PolicyStageManifest` rename half is out of scope here), CD-025 / CD-029 (Wave 5, cli-python station; `validate_phases.rs` 'four vs three' doc drift is a dup-of CD-029).

**Live dispositions (sharpen-only, `reRaiseOf: "<id>"` + NEW evidence; otherwise do not emit):**

| id | wave | disposition | surviving claim (one line) | primary anchor |
| -- | ---- | ----------- | -------------------------- | -------------- |
| CD-004 | 4 | sharpen | the wire projection is untouched: Config → BroadcastConfig (crates/cobre-cli/src/commands/broadcast.rs, from_config at :143) and Config → StudyParams (setup/params.rs from_config at :186) are still two hand-kept field-f… | `crates/cobre-sddp/src/setup/params.rs::from_config` |
| CD-005 | 4 | sharpen | StudySetup conflates precomputed model state (stage_data, stochastic, fcf, hydro_models, scenario_libraries, node_graph, the ncs_* patch vectors) with resolved run-config (loop_params, simulation_config, policy_path, …)… | `crates/cobre-sddp/src/setup/mod.rs::StudySetup` |
| CD-024-successor | 4 | sharpen | ResolvedParameters is still built per rank (policy/resolved_parameters.rs build_resolved_parameters; setup/mod.rs 'Per-rank, never broadcast'); the successor question is whether a rank-0-built resolved run config is bro… | `crates/cobre-sddp/src/policy/resolved_parameters.rs::build_resolved_parameters` |
| CD-015 | 6 | sharpen | compute_one_backward_node (backward_pass_state.rs :1727) still inlines by_scenario's post-dispatch aggregation while by_node's is extracted (training/backward/by_node.rs by_node_finish :416); the asymmetry is real and i… | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` |
| CD-022 | 6 | sharpen | the successor-outcome reification (successors loop → build_delta_cut_row_batch_into in training/forward/delta_cut_batch.rs → SuccessorEntry → SuccessorOutcomes in training/backward/mod.rs) is still inlined twice — compu… | `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward` |
| CD-018 | 6 | keep | `CutSelectionStrategy` conflates two selection paradigms — periodic value-based (Level1/Lml1/Dominated share `select_for_stage`) and lazy-solve DCS (`Dynamic`), whose variant early-returns empty and whose real logic liv… | `crates/cobre-sddp/src/cut/cut_selection.rs::CutSelectionStrategy` |
| CD-035 | 6 | keep | `write_opening_outcome`/`accumulate_opening_outcome` take a bare `&CutStateProjection` where two role-distinct projections are live (`child_cut_layout` vs the cut-generating parent's `succ_spec.cut_state`) — passing the… | `crates/cobre-sddp/src/training/backward/outcome_aggregation.rs::write_opening_outcome` |
| CD-034 | 6 | keep | The min/max outflow entry blocks in `fill_operational_violation_entries` are two hand-mirrored `for blk` loops while the sibling rows file emits the same row pair from a two-element descriptor | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_operational_violation_entries` |
| CD-023 | 6 | sharpen | the carve-out is half done: the neutral solve/ module exists (mod.rs, partition.rs, solver_phase.rs, stage_solve.rs) and owns the solve ENTRY (run_stage_solve), but training/stage_solve_prep.rs and training/stage_solve_… | `crates/cobre-sddp/src/training/stage_solve_prep.rs::StageSolvePrep` |
| CD-021 | 7 | sharpen | all nine structs (CapturedBasis, WorkspaceSizing, BackwardAccumulators, ByNodeScratch, ScratchBuffers, SolverWorkspace, WorkspacePool, BasisStore, BasisStoreSliceMut) still share workspace/workspace.rs; the split-by-con… | `crates/cobre-sddp/src/workspace/workspace.rs::SolverWorkspace` |
| CD-007 | 7 | sharpen | re-measured at the pin: entries.rs 10,093 lines (first #[cfg(test)] at L1680) and columns.rs 9,319 (L1305) — inline test modules dominate both while sibling builder submodules use extracted tests.rs (template/tests.rs 5… | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_load_balance_entries` |
| CD-012 | 7 | keep | `run_cut_management` (~250 lines, `too_many_lines`-allowed) inlines cut-selection + budget enforcement; its rationale defends against FREE-fn extraction, a category error since the natural refactor is `&mut self` methods | `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management` |
| CD-014-remnant | 7 | keep | `compute_one_backward_node` (~290 lines, biggest fn in the crate) — the god-fn residue that remains after the shared successor-reification extraction (CD-022) and the by_scenario commit extraction (CD-015) relieve it | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` |
| CD-016 | 7 | keep | Both backward scheduler workers duplicate ~25 lines of per-worker `backward_accum` buffer pre-allocation (`process_stage_backward` vs `process_stage_backward_by_node`) | `crates/cobre-sddp/src/training/backward_pass_state.rs::process_stage_backward` |
| CD-028 | 7 | keep | Enumerated forward and enumerated simulation reimplement the mid-level claim/scatter orchestration skeleton independently (`run_sweep`/`enumerated_sim_stage_worker` vs `run_enumerated_forward`/`enumerated_stage_worker`)… | `crates/cobre-sddp/src/training/forward/enumerated.rs::run_enumerated_forward` |
| CD-030 | 7 | keep | `mark_own_paths` (simulation/enumerated.rs) reimplements the ~10-line path-marking loop that is inlined in `training/forward/enumerated.rs`; folds with the CD-028 dedup | `crates/cobre-sddp/src/simulation/enumerated.rs::mark_own_paths` |
| CD-037 | 7 | keep | The gap-rule rejecters each re-evaluate `rule_is_gap` + `training_enumerated`, and which rejection message a sampled-forwards study sees depends on call order inside `admission_gate` (documented as deliberate) | `crates/cobre-sddp/src/setup/mod.rs::admission_gate` |
| CD-038 | 7 | keep | `CutStateProjection` exposes only the fused `dot_trial_state`; the cut-selection value sweep in `run_cut_management` open-codes the gather-only loop (via `global_state_index`) — loop-shape duplication, not a hazard | `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs::dot_trial_state` |
| OD-009 | 7 | keep | Production items ship as `#[allow(dead_code)]` 'for symmetry, exercised only by tests' (`RuntimeHandles.export_states`, `RankDistribution.my_rank`, `ExchangeBuffers::new`) — the sanctioned Symmetry-or-test-retention cla… | `crates/cobre-sddp/src/training/session/runtime.rs::RuntimeHandles` |

**CD-074 (Sev A, minted 2026-09-17 from a production failure, live — sharpen-only, `reRaiseOf: "CD-074"`):** the
delivery-stage reconciliation of a carried anticipated commitment classifies solver drift vs modelling error
with two hard-coded constants (`lp/builder/commitment_reconcile.rs::drift_margin`, `::reconcile_commitment`,
`::fill_bound_relaxations`; `training/stage_solve_prep.rs::StageSolvePrep::reconcile_commitments`;
`error.rs::SddpError::AnticipatedCommitmentOutOfBounds`). The mechanism, not the constant, is the defect and
a redesign is flagged. Do not re-derive it; the CLASS sweep below is what this station adds.

## Lens: architecture

Answer each question with symbols or with an explicit `positives` entry.

1. **SETUP LIFECYCLE CONFLATION (5a).** `setup/mod.rs::StudySetup` has three constructors — `new`,
   `new_with_boundary_requirements`, `from_broadcast_params`. Which fields are precomputed model state
   (stage_data, stochastic, fcf, hydro_models, scenario_libraries, node_graph, the `ncs_*` patch vectors)
   and which are resolved run parameters (loop_params, simulation_config, policy_path, the setter wall in
   `setup/accessors.rs`)? Name the split boundary as a SYMBOL SET, not a paragraph; this sharpens CD-005
   (`reRaiseOf: "CD-005"`) whose Phase-0a shape is engine-tagged setup stages. The in-process projection
   is `setup/params.rs::StudyParams::from_config` (CD-004, `partIRef: "I.3-7"`); its wire twin
   `BroadcastConfig` lives in cobre-cli — cite, never anchor. The NCS fields carry the D15 patch-identity
   contract (`.claude/rules/sddp.md` § Lower-bound evaluation must patch NCS) — any split moves them as one.
2. **RETROFITTED-VARIANT ASYMMETRY (5c).** `training/backward/` has three drivers — `by_scenario.rs`,
   `by_node.rs`, `replicated.rs` — each consuming `SuccessorOutcomes` (`training/backward/mod.rs`).
   `outcome_aggregation.rs` already holds four shared helpers (`accumulate_opening_outcome`,
   `accumulate_dcs_binding_counts`, `write_opening_outcome`, `save_basis_at_omega_zero`). Which behaviour
   is shared-but-copied and which is genuinely per-variant? State what did NOT move and why that is or is
   not a smell. CD-015/CD-022 (one convergent extraction out of `backward_pass_state.rs::compute_one_backward_node`
   and `::run_enumerated_backward`) and CD-016 (buffer pre-allocation) are live — sharpen with `reRaiseOf`.
3. **DUPLICATED ORCHESTRATION OWNERS (5c and 5d).** `training/forward/enumerated.rs::run_enumerated_forward`
   (+ `enumerated_stage_worker`) and `simulation/enumerated.rs::run_enumerated_simulation` (+ `run_sweep`,
   `enumerated_sim_stage_worker`) both drive an enumerated sweep; `mark_own_paths` exists only on the
   simulation side (CD-028 / CD-030 are live). Anchor the shared shape in BOTH files in ONE candidate
   (`reRaiseOf: "CD-028"`); do not raise it twice.
4. **POLICY SEAMS (5a).** `policy/policy_export.rs`, `policy_load.rs` (4,129 lines) and `reconcile.rs`
   (2,387) are the export/load/reconcile triple. Is the round-trip contract (`.claude/rules/sddp.md`
   § Policy-load compatibility validation is mandatory) stated in one place or restated per file?
   `rescale_cut_records_for_load`'s Legacy branch is a RESERVED seam — cite, do not raise.
5. **CD-074 CLASS SWEEP (5b, 5c; 5a/5d if a family lives there).** CD-074 is a specimen of a class: a
   pinned state coupled by an equality to a bounded column, reconciled ad hoc for ONE family with
   hard-coded tolerances (`commitment_reconcile.rs::drift_margin`). Audit every state family — storage,
   inflow lags (`lp/indexer/state_space.rs::state_to_lp_incoming_column`, `training/backward/lp_setup.rs::patch_opening_bounds`),
   transit buckets (`setup/bucket_topology.rs`), the commitment hold/ring (`lp/builder/delivery_ring.rs`,
   `columns.rs::fill_anticipated_slot_columns`, `::fill_anticipated_state_columns`) — for the same shape and
   say whether drift handling belongs in ONE shared seam. A family with no such reconciliation is a
   `positives` entry naming why (e.g. pinned by column bounds per § State pinning uses column bounds).
   Cite § State pinning uses column bounds, not equality rows before proposing any change.

## Lens: performance (tag, do not time)

Four named targets, re-resolved at the pin; every worker re-runs the greps and reports drift rather than
trusting the list. Each target gets a candidate or an explicit `positives` entry saying why it is not a smell.

1. **Cut-selection gemm sweep (5c).** `cut/cut_selection.rs::CutSelectionStrategy::select_for_stage` calls
   `gemm.rs::gemm_block` (one of two `crate::gemm` call sites; the other is `cut/dcs.rs`), sized by
   `TRIAL_BLOCK`, per stage per cut-management iteration (`training/session/mod.rs::run_cut_management`, CD-012 live).
2. **Backward gather + `SuccessorOutcomes` reification (5c).** Built at two sites in
   `training/backward_pass_state.rs` (`compute_one_backward_node`, `run_enumerated_backward`), consumed by the
   three drivers. CD-022 (duplication) is live and architectural — the perf question is the per-node
   reification cost and the gather shape, not the duplication.
3. **LP rebuild churn (5b).** `lp/builder/patch.rs::PatchBuffer` — `fill_forward_patches`,
   `fill_col_state_patches`, `fill_load_patches`, `fill_z_inflow_patches` — the per-solve patch family;
   `.claude/rules/sddp.md` § Lower-bound evaluation must patch NCS governs the NCS patch identity.
4. **Basis reconstruct (5c).** `cut/basis_reconstruct.rs::reconstruct_basis` (frozen hot path) vs
   `::reconstruct_basis_uniform_basic` (DCS path); § Cut pool is append-only; basis matches by slot identity
   and § A stored basis warm-starts only at its own node govern any fix-shape.

5a and 5d cells: sweep your manifest for per-scenario / per-stage / per-node allocation inside hot loops
(`simulation/pipeline.rs`, `simulation/enumerated.rs` — PD-001 is refuted, do not re-raise it —,
`production/`, `setup/stochastic_pipeline.rs`, `policy/policy_load.rs` on the load path) and report the
mechanism. The hot-path map in `.claude/architecture-rules.md` names the frozen drivers.
Every candidate: `claimType`, `layout`, `mechanism`, `profiledSymbol`, `queuedTo: "perf-sweep"`; no numbers.

## Lens: over-engineering (reserved-is-not-dead precondition FIRST)

PRECONDITION: before raising anything, check the four reserved seams above, the mirror's Cleared items,
the CLAUDE.md 'unwired config is reserved, not dead' rule and any in-code `#[allow(...)]` or
'Reserved seam' rationale. A seam you reach is a `positives` entry with `sanctionedBy`, never a candidate.

Then a census with numbers, not impressions:
- `git grep -c '^pub enum \|^pub(crate) enum ' 077dbe2c -- crates/cobre-sddp/src` (38 enums at the pin); for every
  enum IN YOUR MANIFEST count variants — a one-variant enum is a candidate only if it is not a reserved seam;
- `git grep -c '#\[allow(' 077dbe2c -- <your manifest paths>` (514 openers crate-wide, 102 files); an `#[allow]`
  with a written rationale is sanctioned (`positives`); one without is a candidate (OD-009 is live: sharpen-only).
Targets: new-noun sprawl in `setup/` (twelve files: `mod.rs` 3,259, `node_graph.rs` 3,021, `tests.rs` 10,356,
`params.rs`, `accessors.rs` 370, `orchestration.rs` 380, `stage_data.rs` 77, `scenario_library_set.rs` 46 —
is that a type or a typedef? —, `scenario_libraries.rs`, `bucket_topology.rs`, `template_postprocess.rs`,
`stochastic_pipeline.rs`); wrapper layers that forward without deciding; structs whose only method is a
constructor; one-consumer abstractions (rule 11: a one-consumer seam proposed as a FIX is `conflicts`; an
existing one-consumer abstraction is a candidate). `hull/`, `lead_time/`, the two echo modules and
`solver_stats.rs` are small: say explicitly whether each earns its module.

## Lens: test-bloat (yardstick: `docs/design/testing-architecture.md`)

Census at the pin (re-run, do not trust): `crates/cobre-sddp/tests` = 56 `.rs` files / 82,478 lines = 40
top-level integration binaries (72,356) + 5 under `common/` (`anticipated_structural_assertions.rs`,
`builders.rs`, `mod.rs`, `parity_hash.rs`, `permute.rs`) + 11 under `template_integration/`; `fixtures/`
holds no `.rs`. In src: 14 sibling `tests.rs` files (`inventory.json` siblingTestFiles) and the 14
inline-test-dominated outliers (`inlineTestOutliers`, CD-007 evidence: `lp/builder/entries.rs` 10,093 lines
with the first `#[cfg(test)]` at L1680, `columns.rs` 9,319 / L1305).

Three questions per cell, scoped to YOUR manifest's src files plus the tests/ binaries that exercise them:
1. **Inline vs sibling split.** `lp/builder/` (5b) splits two ways — sibling `tests.rs` for
   `commitment_reconcile`, `layout`, `template` (and `lp/generic_constraints`); inline `#[cfg(test)]` in
   `patch.rs`, `scaling.rs`, `delivery_ring.rs`, `mod.rs`, and the giants `entries.rs` / `columns.rs`. Is the
   split principled (a size threshold) or accidental? CD-007 is live — sharpen with `reRaiseOf: "CD-007"`.
   5a: `setup/tests.rs` (10,356 lines) vs inline blocks in `setup/*.rs`. 5c: `training/backward/tests.rs`
   (6,514), `training/{forward,stage_solve_prep,training}/tests.rs`. 5d: `simulation/{extraction,pipeline}/tests.rs`,
   `production/{fpha_fitting,hydro_models/production}/tests.rs`, `lead_time/tests.rs`.
2. **Golden-set membership (5c owns; others cite).** `tests/parity.rs` carries ten `parity_hash_*` tests
   (d06, d15, d30, d34, d41 × HiGHS and CLP). Are all ten load-bearing, or do some cover the same LP
   structure? Anchor per case. The determinism and rank-invariance gates (`tests/deterministic.rs`,
   `tests/common/permute.rs`, `tests/mpi_wire.rs`) are INFORMATIONAL — a Phase-0a bit-for-bit gate reuses them;
   never bloat.
3. **Harness duplication.** Which of the 40 binaries rebuild fixtures that `tests/common/builders.rs`
   already provides? Give per-file symbol anchors (the duplicated builder fn names), separate genuinely
   shared helpers from per-family variants, and propose a `tests/common` consolidation as fix-shape — a
   NEW fixture crate is rejected. Crate-wide totals belong to the test-corpus station (E08): report only
   what is anchored inside `crates/cobre-sddp`. `#[cfg_attr(not(feature = "slow-tests"), ignore)]` gating
   is the local-dev tier switch, not bloat.

## Manifests (partition proven by the dispatching session before any worker runs)

Union of the four == every `.rs` under `crates/cobre-sddp/src` at the baseline (163 files), pairwise disjoint
(`inventory.json` coverage.src). Lines are total / non-test at the pin; sibling `tests.rs` and
`test_support.rs` count 0 non-test.

### `5a` — setup + policy + stochastic + config (the setup lifecycle and the policy seams)

28 files, 41,601 lines (13,077 non-test). Modules: `setup`, `policy`, `stochastic`, `config.rs`, `validate_phases.rs`, `horizon_mode.rs`.

- `crates/cobre-sddp/src/config.rs` 386 / 239
- `crates/cobre-sddp/src/horizon_mode.rs` 211 / 108
- `crates/cobre-sddp/src/policy/mod.rs` 9 / 9
- `crates/cobre-sddp/src/policy/orchestration.rs` 792 / 442
- `crates/cobre-sddp/src/policy/policy_export.rs` 3,090 / 833
- `crates/cobre-sddp/src/policy/policy_load.rs` 4,129 / 1,278
- `crates/cobre-sddp/src/policy/provenance.rs` 998 / 257
- `crates/cobre-sddp/src/policy/reconcile.rs` 2,387 / 886
- `crates/cobre-sddp/src/policy/resolved_parameters.rs` 1,466 / 568
- `crates/cobre-sddp/src/policy/scaling_report.rs` 402 / 228
- `crates/cobre-sddp/src/setup/accessors.rs` 370 / 324
- `crates/cobre-sddp/src/setup/bucket_topology.rs` 1,286 / 406
- `crates/cobre-sddp/src/setup/mod.rs` 3,259 / 2,810
- `crates/cobre-sddp/src/setup/node_graph.rs` 3,021 / 1,543
- `crates/cobre-sddp/src/setup/orchestration.rs` 380 / 358
- `crates/cobre-sddp/src/setup/params.rs` 847 / 345
- `crates/cobre-sddp/src/setup/scenario_libraries.rs` 1,042 / 315
- `crates/cobre-sddp/src/setup/scenario_library_set.rs` 46 / 46
- `crates/cobre-sddp/src/setup/stage_data.rs` 77 / 77
- `crates/cobre-sddp/src/setup/stochastic_pipeline.rs` 1,800 / 468
- `crates/cobre-sddp/src/setup/template_postprocess.rs` 294 / 162
- `crates/cobre-sddp/src/setup/tests.rs` 10,356 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/stochastic/inflow_method.rs` 156 / 81
- `crates/cobre-sddp/src/stochastic/mod.rs` 24 / 24
- `crates/cobre-sddp/src/stochastic/noise.rs` 2,705 / 530
- `crates/cobre-sddp/src/stochastic/noise_key.rs` 573 / 267
- `crates/cobre-sddp/src/stochastic/stochastic_summary.rs` 1,350 / 385
- `crates/cobre-sddp/src/validate_phases.rs` 145 / 88

### `5b` — lp/ (indexer + builder + generic constraints — the LP construction kernel)

30 files, 44,682 lines (11,655 non-test). Modules: `lp`.

- `crates/cobre-sddp/src/lp/builder/columns.rs` 9,319 / 1,315
- `crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs` 245 / 243
- `crates/cobre-sddp/src/lp/builder/commitment_reconcile/tests.rs` 250 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/lp/builder/delivery_ring.rs` 567 / 299
- `crates/cobre-sddp/src/lp/builder/entries.rs` 10,093 / 1,686
- `crates/cobre-sddp/src/lp/builder/fpha_cursor.rs` 100 / 100
- `crates/cobre-sddp/src/lp/builder/layout.rs` 2,090 / 1,962
- `crates/cobre-sddp/src/lp/builder/layout/tests.rs` 3,565 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/lp/builder/mod.rs` 157 / 145
- `crates/cobre-sddp/src/lp/builder/patch.rs` 1,168 / 413
- `crates/cobre-sddp/src/lp/builder/rows.rs` 588 / 588
- `crates/cobre-sddp/src/lp/builder/scaling.rs` 770 / 265
- `crates/cobre-sddp/src/lp/builder/template.rs` 1,329 / 1,293
- `crates/cobre-sddp/src/lp/builder/template/tests.rs` 5,356 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/lp/builder/test_support.rs` 179 / 0 (test-support)
- `crates/cobre-sddp/src/lp/generic_constraints.rs` 987 / 985
- `crates/cobre-sddp/src/lp/generic_constraints/tests.rs` 2,951 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/lp/indexer/anticipated_gate.rs` 368 / 112
- `crates/cobre-sddp/src/lp/indexer/block_grid.rs` 198 / 126
- `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs` 994 / 292
- `crates/cobre-sddp/src/lp/indexer/entity_index.rs` 387 / 293
- `crates/cobre-sddp/src/lp/indexer/hydro_cell.rs` 457 / 204
- `crates/cobre-sddp/src/lp/indexer/index.rs` 326 / 247
- `crates/cobre-sddp/src/lp/indexer/layout.rs` 72 / 40
- `crates/cobre-sddp/src/lp/indexer/mod.rs` 125 / 125
- `crates/cobre-sddp/src/lp/indexer/range_cursor.rs` 41 / 41
- `crates/cobre-sddp/src/lp/indexer/state_space.rs` 1,823 / 729
- `crates/cobre-sddp/src/lp/indexer/storage_boundary_grid.rs` 100 / 77
- `crates/cobre-sddp/src/lp/indexer/study_dimensions.rs` 50 / 50
- `crates/cobre-sddp/src/lp/mod.rs` 27 / 25

### `5c` — cut + training + solve + workspace + convergence + gemm + claim_scatter + solver_stats (the SDDP iteration)

58 files, 61,104 lines (20,572 non-test). Modules: `cut`, `training`, `solve`, `workspace`, `convergence`, `gemm.rs`, `claim_scatter.rs`, `solver_stats.rs`.

- `crates/cobre-sddp/src/claim_scatter.rs` 89 / 51
- `crates/cobre-sddp/src/convergence/convergence.rs` 446 / 163
- `crates/cobre-sddp/src/convergence/mod.rs` 9 / 9
- `crates/cobre-sddp/src/convergence/risk_measure.rs` 944 / 392
- `crates/cobre-sddp/src/convergence/stopping_rule.rs` 708 / 361
- `crates/cobre-sddp/src/cut/basis_reconstruct.rs` 750 / 332
- `crates/cobre-sddp/src/cut/cut_selection.rs` 2,376 / 609
- `crates/cobre-sddp/src/cut/cut_sync.rs` 2,245 / 877
- `crates/cobre-sddp/src/cut/dcs.rs` 2,794 / 770
- `crates/cobre-sddp/src/cut/fcf.rs` 1,058 / 523
- `crates/cobre-sddp/src/cut/mod.rs` 51 / 51
- `crates/cobre-sddp/src/cut/pool.rs` 2,193 / 1,097
- `crates/cobre-sddp/src/cut/row.rs` 717 / 352
- `crates/cobre-sddp/src/cut/row_map.rs` 273 / 168
- `crates/cobre-sddp/src/cut/wire.rs` 821 / 353
- `crates/cobre-sddp/src/gemm.rs` 165 / 112
- `crates/cobre-sddp/src/solve/mod.rs` 19 / 19
- `crates/cobre-sddp/src/solve/partition.rs` 24 / 24
- `crates/cobre-sddp/src/solve/solver_phase.rs` 1,354 / 638
- `crates/cobre-sddp/src/solve/stage_solve.rs` 927 / 285
- `crates/cobre-sddp/src/solver_stats.rs` 1,428 / 614
- `crates/cobre-sddp/src/training/backward/by_node.rs` 720 / 555
- `crates/cobre-sddp/src/training/backward/by_scenario.rs` 573 / 573
- `crates/cobre-sddp/src/training/backward/duals_extraction.rs` 178 / 86
- `crates/cobre-sddp/src/training/backward/lp_setup.rs` 319 / 184
- `crates/cobre-sddp/src/training/backward/mod.rs` 358 / 278
- `crates/cobre-sddp/src/training/backward/outcome_aggregation.rs` 163 / 163
- `crates/cobre-sddp/src/training/backward/replicated.rs` 330 / 330
- `crates/cobre-sddp/src/training/backward/tests.rs` 6,514 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/training/backward_pass_state.rs` 5,115 / 2,127
- `crates/cobre-sddp/src/training/forward/basis_capture.rs` 47 / 47
- `crates/cobre-sddp/src/training/forward/delta_cut_batch.rs` 90 / 90
- `crates/cobre-sddp/src/training/forward/enumerated.rs` 1,308 / 868
- `crates/cobre-sddp/src/training/forward/mod.rs` 240 / 106
- `crates/cobre-sddp/src/training/forward/sampler.rs` 42 / 42
- `crates/cobre-sddp/src/training/forward/stage_solve.rs` 259 / 259
- `crates/cobre-sddp/src/training/forward/stats_aggregation.rs` 291 / 288
- `crates/cobre-sddp/src/training/forward/tests.rs` 4,175 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/training/forward_pass_state.rs` 2,924 / 1,058
- `crates/cobre-sddp/src/training/lower_bound.rs` 3,365 / 501
- `crates/cobre-sddp/src/training/mod.rs` 51 / 51
- `crates/cobre-sddp/src/training/rank_reconcile.rs` 484 / 93
- `crates/cobre-sddp/src/training/session/iteration_scratch.rs` 514 / 192
- `crates/cobre-sddp/src/training/session/mod.rs` 3,090 / 1,507
- `crates/cobre-sddp/src/training/session/rank_distribution.rs` 218 / 68
- `crates/cobre-sddp/src/training/session/results.rs` 72 / 40
- `crates/cobre-sddp/src/training/session/runtime.rs` 89 / 43
- `crates/cobre-sddp/src/training/stage_solve_prep.rs` 253 / 251
- `crates/cobre-sddp/src/training/stage_solve_prep/tests.rs` 813 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/training/state_exchange.rs` 819 / 371
- `crates/cobre-sddp/src/training/training.rs` 369 / 367
- `crates/cobre-sddp/src/training/training/tests.rs` 3,035 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/training/training_output.rs` 1,206 / 541
- `crates/cobre-sddp/src/training/trajectory.rs` 137 / 43
- `crates/cobre-sddp/src/training/visited_states.rs` 578 / 207
- `crates/cobre-sddp/src/workspace/context.rs` 271 / 271
- `crates/cobre-sddp/src/workspace/mod.rs` 32 / 32
- `crates/cobre-sddp/src/workspace/workspace.rs` 2,671 / 1,140

### `5d` — simulation + production + hull + lead_time + echoes + error + lib + test_support

47 files, 42,856 lines (13,167 non-test). Modules: `simulation`, `production`, `hull`, `lead_time`, `generic_constraint_echo.rs`, `fixed_delivery_echo.rs`, `error.rs`, `lib.rs`, `test_support.rs`.

- `crates/cobre-sddp/src/error.rs` 344 / 143
- `crates/cobre-sddp/src/fixed_delivery_echo.rs` 99 / 39
- `crates/cobre-sddp/src/generic_constraint_echo.rs` 643 / 310
- `crates/cobre-sddp/src/hull/ffi.rs` 47 / 47
- `crates/cobre-sddp/src/hull/mod.rs` 499 / 212
- `crates/cobre-sddp/src/lead_time/mod.rs` 748 / 746
- `crates/cobre-sddp/src/lead_time/tests.rs` 1,493 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/lib.rs` 208 / 176
- `crates/cobre-sddp/src/production/conversion.rs` 671 / 341
- `crates/cobre-sddp/src/production/energy_conversion/builder.rs` 1,405 / 245
- `crates/cobre-sddp/src/production/energy_conversion/mod.rs` 17 / 17
- `crates/cobre-sddp/src/production/energy_conversion/productivity_override.rs` 266 / 115
- `crates/cobre-sddp/src/production/energy_conversion/types.rs` 299 / 215
- `crates/cobre-sddp/src/production/fpha_fitting/alpha.rs` 307 / 88
- `crates/cobre-sddp/src/production/fpha_fitting/deviation.rs` 473 / 195
- `crates/cobre-sddp/src/production/fpha_fitting/error.rs` 264 / 264
- `crates/cobre-sddp/src/production/fpha_fitting/geometry.rs` 487 / 487
- `crates/cobre-sddp/src/production/fpha_fitting/grid.rs` 52 / 52
- `crates/cobre-sddp/src/production/fpha_fitting/hull_fit.rs` 632 / 195
- `crates/cobre-sddp/src/production/fpha_fitting/mod.rs` 222 / 212
- `crates/cobre-sddp/src/production/fpha_fitting/production.rs` 159 / 159
- `crates/cobre-sddp/src/production/fpha_fitting/reduction.rs` 750 / 337
- `crates/cobre-sddp/src/production/fpha_fitting/rng.rs` 166 / 75
- `crates/cobre-sddp/src/production/fpha_fitting/secant.rs` 491 / 187
- `crates/cobre-sddp/src/production/fpha_fitting/selection.rs` 71 / 71
- `crates/cobre-sddp/src/production/fpha_fitting/tailrace.rs` 779 / 415
- `crates/cobre-sddp/src/production/fpha_fitting/tests.rs` 2,673 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/production/hydro_models/evaporation.rs` 1,314 / 397
- `crates/cobre-sddp/src/production/hydro_models/export.rs` 514 / 139
- `crates/cobre-sddp/src/production/hydro_models/mod.rs` 586 / 198
- `crates/cobre-sddp/src/production/hydro_models/production.rs` 1,098 / 1,096
- `crates/cobre-sddp/src/production/hydro_models/production/tests.rs` 2,759 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/production/hydro_models/summary.rs` 804 / 113
- `crates/cobre-sddp/src/production/hydro_models/types.rs` 965 / 476
- `crates/cobre-sddp/src/production/mod.rs` 11 / 11
- `crates/cobre-sddp/src/simulation/aggregation.rs` 561 / 254
- `crates/cobre-sddp/src/simulation/config.rs` 92 / 46
- `crates/cobre-sddp/src/simulation/enumerated.rs` 661 / 510
- `crates/cobre-sddp/src/simulation/error.rs` 222 / 81
- `crates/cobre-sddp/src/simulation/extraction.rs` 2,413 / 2,156
- `crates/cobre-sddp/src/simulation/extraction/tests.rs` 7,138 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/simulation/mod.rs` 29 / 29
- `crates/cobre-sddp/src/simulation/pipeline.rs` 1,084 / 1,082
- `crates/cobre-sddp/src/simulation/pipeline/tests.rs` 2,827 / 0 (sibling-test-module)
- `crates/cobre-sddp/src/simulation/state.rs` 754 / 644
- `crates/cobre-sddp/src/simulation/types.rs` 1,107 / 592
- `crates/cobre-sddp/src/test_support.rs` 3,652 / 0 (test-support)

## Envelope (frozen contract — `tools/validate-envelope.py --role attacker --station sddp` reads exactly this)

```json
{
  "station": "sddp",
  "subStation": "<SUB>",
  "baseline": "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c",
  "lens": "<LENS>",
  "candidates": [
    {
      "title": "one line naming the smell and its subject; no ID (IDs are assigned at calibration)",
      "anchors": [
        {
          "path": "crates/cobre-sddp/src/<file>.rs",
          "symbol": "<declared symbol in that file>"
        }
      ],
      "evidence": {
        "command": "git show 077dbe2c:crates/cobre-sddp/src/<file>.rs | sed -n '<a>,<b>p'",
        "output": "…trimmed…",
        "reading": "why that output supports the claim; counts and symbol names"
      },
      "proposedSeverity": "A|B|C",
      "fixShape": "prose only; states the byte-neutrality bar; cites the .claude/rules/sddp.md section when it touches a pinned contract",
      "alignmentHint": "advances-0a|advances-0b|advances-1|neutral|conflicts",
      "alignmentCitation": "Part IV.1|IV.2|IV.4|V.0|V.1|V.2 — required when alignmentHint != neutral",
      "partIRef": "I.3-7 or null",
      "reRaiseOf": "CD-0nn / OD-0nn when sharpening a live disposition; otherwise null",
      "claimType": "performance lens: single-process|collective; otherwise null",
      "layout": "performance lens: 4t|2x2; otherwise null",
      "mechanism": "performance lens: the cost mechanism, no numbers",
      "profiledSymbol": "performance lens: the symbol the perf epic profiles",
      "queuedTo": "performance lens: perf-sweep"
    }
  ],
  "positives": [
    {
      "subject": "crates/cobre-sddp/src/… (a module, a symbol, or a reserved seam)",
      "why": "correct and worth protecting / sanctioned / examined and clean",
      "sanctionedBy": "mirror section or register id when applicable, else null"
    }
  ],
  "_needsHuman": [
    "a question only the owner can answer; empty list when none"
  ]
}
```

Severity: A = wrong results, lost determinism, a user-visible abort, or a structural block on the roadmap;
B = real debt with a bounded fix and a named blast radius; C = local quality. `proposedSeverity` is the
attacker's rating; the house calibrates. Keys not in the schema are dropped at the gate; `verdict` is not
yours to set (the defender's). The envelope is the whole content of your scratch file.

## Dispatch

Sixteen cells in four waves of four concurrent workers, one lens per wave (architecture → performance →
over-engineering → test-bloat). Each cell's directive names `<LENS>`, `<SUB>` and the scratch path
`/tmp/sddp-attackers/out/<LENS>.<SUB>.json`. A shape-invalid envelope is re-dispatched exactly once with
the validator error quoted; a second failure is `needs-human`. `attacker-log.md` is the resume point.
