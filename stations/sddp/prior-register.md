# Prior register — cobre-sddp (baseline `077dbe2c`)

What the register and the committed mirror already say about this crate, so the four
attacker fan-outs (5a setup+policy+stochastic+config, 5b lp/, 5c cut+training+solve+
workspace+convergence, 5d simulation+production+hull+lead_time+echoes) receive a real
do-not-re-raise list instead of re-discovering ratified work. Every code anchor below is
`path::symbol`, resolved against the baseline blob (`git show 077dbe2c:<path>`) with the
declaration regex `tools/check-anchors.py` uses — never `path:line`, because the prior waves'
line numbers predate the `workspace/` directory-module split, the `solve/` carve-out and the
`ConstructionConfig` deletion. A span that has no declaration is written as unbackticked prose
(file L<n>, re-read before quoting); a historical symbol that resolves to no declaration at the
baseline is written `unresolved` with the reason, never silently refreshed. Mirror and
`.claude/rules/sddp.md` citations are `path:line` at the baseline. Each block carries a
`reraiseKey` token list the attacker screen and `check-reraise.py` match candidate titles and
anchors against.

Five dispositions: a **roster row** is a prior finding the re-verification ticket (E05-3) fills
with keep / retire-with-commit / sharpen — attackers may _sharpen_ a row, never re-derive it as
new; a **retire-with-commit** item is closed and cited, not raised; a **do-not-touch** item is
retracted, refuted or deferred by owner decision; a **reserved seam** is Cleared-with-citation at
ingest and assigns no OD id; a **contract** is named as the reason the current shape is correct
and is never weakened by a fix-shape.

---

## Wave 4/6/7 roster — disposition pending

| ID | Wave | Sev | Claim (one line) | Anchor (path + symbol) | Stale-anchor note | Disposition |
| -- | ---- | --- | ---------------- | ---------------------- | ----------------- | ----------- |
| CD-004 | 4 | B (A-risk) | Two parallel Config → construction projections (in-process `StudyParams::from_config` vs the MPI `BroadcastConfig` hand-assembly) that must agree field-for-field or MPI diverges silently from local | `crates/cobre-sddp/src/setup/params.rs::StudyParams` · `crates/cobre-sddp/src/setup/params.rs::from_config` · `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig` (cross-station corroboration (cobre-cli), not an sddp-station anchor) · `crates/cobre-cli/src/commands/run/setup.rs::build_study_setup` (cross-station corroboration) · `crates/cobre-sddp/src/setup/params.rs` · _`into_construction_config` unresolved — deleted by 4075c4e8 — absence is the successor state, do not refresh_ | `into_construction_config` (params.rs:297) no longer exists — 4075c4e8 merged StudyParams+ConstructionConfig, so the MPI arm is now Config → BroadcastConfig (cobre-cli, broadcast.rs:87) → `StudyParams::from_config` on every rank (broadcast.rs:144). The remaining duplication is the two-carrier BroadcastConfig/StudyParams pair (postcard boundary), not the three-struct trio the entry text describes. | _pending_ |
| CD-005 | 4 | B | `StudySetup` god-struct: half-finished NCS sub-struct extraction plus role conflation of precomputed model state with resolved run-config (the setter wall in accessors.rs exists only because run-config lives in the data struct) | `crates/cobre-sddp/src/setup/mod.rs::StudySetup` · `crates/cobre-sddp/src/setup/mod.rs::from_broadcast_params` · `crates/cobre-sddp/src/setup/accessors.rs::set_budget` · `crates/cobre-sddp/src/setup/accessors.rs::set_risk_measures` · `crates/cobre-sddp/src/setup/accessors.rs::set_scheduler` | Line drift only; the sub-struct map the entry alludes to now lives in .claude/architecture-rules.md ('StudySetup Sub-Structs'). Field count last recorded 35 (2026-08-22) — not re-counted here. | _pending_ |
| CD-024-successor | 4 | — (CD-024 itself RESOLVED 2026-08-18: broadcast pair deleted) | The resolved-parameters broadcast seam CD-024 registered was deleted outright; its successor is the Wave-4 question whether rank 0 builds and broadcasts the resolved run config (`ResolvedParameters` is still built per-rank, 'never broadcast') | `crates/cobre-sddp/src/policy/resolved_parameters.rs::ResolvedParameters` · `crates/cobre-sddp/src/policy/resolved_parameters.rs::build_resolved_parameters` · `crates/cobre-sddp/src/policy/resolved_parameters.rs` · _`serialize_resolved_parameters` unresolved — deleted (register L1187); `deserialize_resolved_parameters` and `RESOLVED_PARAMETERS_WIRE_VERSION` also resolve nowhere — the absence IS the recorded state_ | 'successor' framing: the entry's own anchors (`serialize_/deserialize_resolved_parameters` at resolved_parameters.rs:477/529) were deleted as dead code; the live seam is the per-rank `build_resolved_parameters` derivation (`setup/mod.rs:970`) that a Wave-4 'rank-0 builds ResolvedRunConfig, broadcasts it' redesign would either activate (broadcast) or make permanent (per-rank). File is `policy/resolved_parameters.rs` at the baseline (re-exported as `crate::resolved_parameters`). Spans (no declaration): mod.rs L970 (build_resolved_parameters call site). | _pending_ |
| CD-015 | 6 | B | Inside `compute_one_backward_node`, by_node's post-dispatch aggregation is extracted (`by_node_finish`) while by_scenario's — the ORIGINAL scheduler — is inline (retrofitted-variant class instance #2) | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` · `crates/cobre-sddp/src/training/backward/by_node.rs::by_node_finish` · `crates/cobre-sddp/src/training/backward/by_scenario.rs::process_by_scenario_backward` (natural home for the proposed `commit_by_scenario_cuts`) · `crates/cobre-sddp/src/training/backward_pass_state.rs` · _`commit_by_scenario_cuts` unresolved — proposed extraction; does not exist yet (expected)_ | `by_node_finish` moved out of backward_pass_state.rs into the `training/backward/by_node.rs` directory module; the inline by_scenario span is a line range inside `compute_one_backward_node` — re-read at the baseline before quoting a span. | _pending_ |
| CD-022 | 6 | B | Successor-outcome reification (~66 lines: successors loop → `build_delta_cut_row_batch_into` → `SuccessorEntry` → `SuccessorOutcomes`) is duplicated between the sampled path (`compute_one_backward_node`) and the enumerated path (`run_enumerated_backward`); converges with CD-014 | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` · `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward` · `crates/cobre-sddp/src/training/backward/mod.rs::SuccessorEntry` · `crates/cobre-sddp/src/training/backward/mod.rs::SuccessorOutcomes` · `crates/cobre-sddp/src/training/forward/delta_cut_batch.rs::build_delta_cut_row_batch_into` · `crates/cobre-sddp/src/training/backward_pass_state.rs` · _`reify_successor_outcomes` unresolved — proposed shared helper; does not exist yet (expected)_ | `SuccessorEntry`/`SuccessorOutcomes` now live in `training/backward/mod.rs` and `build_delta_cut_row_batch_into` in `training/forward/delta_cut_batch.rs`; the two copies are line spans inside the two fns above. | _pending_ |
| CD-018 | 6 | C | `CutSelectionStrategy` conflates two selection paradigms — periodic value-based (Level1/Lml1/Dominated share `select_for_stage`) and lazy-solve DCS (`Dynamic`), whose variant early-returns empty and whose real logic lives in `dcs.rs::lazy_solve_preloaded` | `crates/cobre-sddp/src/cut/cut_selection.rs::CutSelectionStrategy` · `crates/cobre-sddp/src/cut/cut_selection.rs::select_for_stage` · `crates/cobre-sddp/src/cut/dcs.rs::DcsParams` · `crates/cobre-sddp/src/cut/dcs.rs::from_strategy` · `crates/cobre-sddp/src/cut/dcs.rs::lazy_solve_preloaded` | enum moved :144 → :132; the early-return moved :320 → :308. Spans (no declaration): cut_selection.rs L308 (Dynamic early-return); cut_selection.rs L492 (unreachable! for Dynamic). | _pending_ |
| CD-035 | 6 | B | `write_opening_outcome`/`accumulate_opening_outcome` take a bare `&CutStateProjection` where two role-distinct projections are live (`child_cut_layout` vs the cut-generating parent's `succ_spec.cut_state`) — passing the wrong one compiles | `crates/cobre-sddp/src/training/backward/outcome_aggregation.rs::accumulate_opening_outcome` · `crates/cobre-sddp/src/training/backward/outcome_aggregation.rs::write_opening_outcome` · `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs::CutStateProjection` · `crates/cobre-sddp/src/training/backward/mod.rs::SuccessorSpec` · `crates/cobre-sddp/src/training/backward/mod.rs` · _`CutGeneratingProjection` unresolved — proposed newtype; does not exist (expected)_ | :854 → :857; everything else holds. Spans (no declaration): backward_pass_state.rs L799 (child_cut_layout binding); backward_pass_state.rs L857 (succ_spec.cut_state read). | _pending_ |
| CD-034 | 6 | B | The min/max outflow entry blocks in `fill_operational_violation_entries` are two hand-mirrored `for blk` loops while the sibling rows file emits the same row pair from a two-element descriptor | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_operational_violation_entries` · `crates/cobre-sddp/src/lp/builder/rows.rs::fill_operational_violation_rows` | entries.rs :1473 → :1487; the rows.rs :461-477 descriptor span is inside `fill_operational_violation_rows` (:443) — re-read before quoting. | _pending_ |
| CD-023 | 6 | C (B-risk: declared-undesirable inversion) | `StageSolvePrep` (the single-owner solve-prep every solve site routes through) still lives under `training/` while `simulation/pipeline.rs` imports it — the sim→training inversion the f527e3f9 hoist set out to eliminate | `crates/cobre-sddp/src/training/stage_solve_prep.rs::StageSolvePrep` · `crates/cobre-sddp/src/training/stage_solve_prep.rs::run` (`StageSolvePrep::run`) · `crates/cobre-sddp/src/solve/stage_solve.rs::run_stage_solve` · `crates/cobre-sddp/src/solve/stage_solve.rs::StageInputs` | Partially addressed and still open at 077dbe2c exactly as ROADMAP L106 says: `solve/` owns the solve ENTRY (`run_stage_solve`), the PREP is still `training/stage_solve_prep.rs:71` and `simulation/pipeline.rs:50` still imports it. Spans (no declaration): pipeline.rs L50 (training::stage_solve_prep import). | _pending_ |
| CD-007 | 7 | C | Inline `#[cfg(test)]` modules dominate `lp/builder/{entries,columns}.rs` (~83–85% test) while sibling builder submodules use extracted `tests.rs` files — two conventions in one module dir; the inline giants are the sole reason these read as god-files | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_load_balance_entries` · `crates/cobre-sddp/src/lp/builder/entries.rs::fill_ncs_load_balance_entries` | At 077dbe2c entries.rs is 10,093 lines (first cfg(test) at 1680) and columns.rs 9,319 (first cfg(test) at 1305) — both grew again; the ticket's 10,079 / 9,026 are the 2026-08-22 figures and are stale. Inventory step 3 must re-measure, not copy. Spans (no declaration): entries.rs L1680 (first #[cfg(test)]); columns.rs L1305 (first #[cfg(test)]). | _pending_ |
| CD-012 | 7 | C | `run_cut_management` (~250 lines, `too_many_lines`-allowed) inlines cut-selection + budget enforcement; its rationale defends against FREE-fn extraction, a category error since the natural refactor is `&mut self` methods | `crates/cobre-sddp/src/training/session/mod.rs::run_cut_management` · `crates/cobre-sddp/src/training/session/mod.rs::run_iteration` (affirmed cohesive, not flagged) · `crates/cobre-sddp/src/training/session/mod.rs` · _`run_cut_selection` unresolved — proposed method; does not exist (expected)_ | :937 → :1015 → :1034 → :1035 across three refreshes; symbol stable. | _pending_ |
| CD-014-remnant | 7 | C | `compute_one_backward_node` (~290 lines, biggest fn in the crate) — the god-fn residue that remains after the shared successor-reification extraction (CD-022) and the by_scenario commit extraction (CD-015) relieve it | `crates/cobre-sddp/src/training/backward_pass_state.rs::compute_one_backward_node` | 'remnant' is a waves-table framing, not a register entry of its own: the register entry is CD-014 (L531). Function length (~289 lines from :1727) must be re-measured at the baseline before citing. | _pending_ |
| CD-016 | 7 | C | Both backward scheduler workers duplicate ~25 lines of per-worker `backward_accum` buffer pre-allocation (`process_stage_backward` vs `process_stage_backward_by_node`) | `crates/cobre-sddp/src/training/backward_pass_state.rs::process_stage_backward` · `crates/cobre-sddp/src/training/backward/by_node.rs::process_stage_backward_by_node` · `crates/cobre-sddp/src/workspace/workspace.rs::BackwardAccumulators` · `crates/cobre-sddp/src/training/backward/mod.rs` · _`prepare_shared_backward_buffers` unresolved — proposed helper; does not exist (expected)_ | `by_node.rs` is `training/backward/by_node.rs`; line drift only. | _pending_ |
| CD-021 | 7 | C | `workspace/workspace.rs` is a flat mega-file holding NINE per-worker arena structs, asymmetric with the crate's directory-module convention (`setup/`, `lp/builder/`, `training/session/`, `cut/`) | `crates/cobre-sddp/src/workspace/workspace.rs::CapturedBasis` · `crates/cobre-sddp/src/workspace/workspace.rs::WorkspaceSizing` · `crates/cobre-sddp/src/workspace/workspace.rs::BackwardAccumulators` · `crates/cobre-sddp/src/workspace/workspace.rs::ByNodeScratch` · `crates/cobre-sddp/src/workspace/workspace.rs::ScratchBuffers` · `crates/cobre-sddp/src/workspace/workspace.rs::SolverWorkspace` · `crates/cobre-sddp/src/workspace/workspace.rs::WorkspacePool` · `crates/cobre-sddp/src/workspace/workspace.rs::BasisStore` · `crates/cobre-sddp/src/workspace/workspace.rs::BasisStoreSliceMut` | The historical anchor named a flat `workspace.rs`; at 077dbe2c the tree has `workspace/{mod,context,workspace}.rs` (32 / 271 / 2671 lines; workspace.rs's first `#[cfg(test)]` at 1141 → ~1140 production lines). All nine structs still live in `workspace/workspace.rs`, so the finding holds against the directory module's one large member — record the directory module, never refresh to the flat path. | _pending_ |
| CD-028 | 7 | B | Enumerated forward and enumerated simulation reimplement the mid-level claim/scatter orchestration skeleton independently (`run_sweep`/`enumerated_sim_stage_worker` vs `run_enumerated_forward`/`enumerated_stage_worker`); only the low-level primitives are shared | `crates/cobre-sddp/src/simulation/enumerated.rs::run_sweep` · `crates/cobre-sddp/src/simulation/enumerated.rs::enumerated_sim_stage_worker` · `crates/cobre-sddp/src/training/forward/enumerated.rs::run_enumerated_forward` · `crates/cobre-sddp/src/training/forward/enumerated.rs::enumerated_stage_worker` | line drift only (292→290, 161→158, 531→472). | _pending_ |
| CD-030 | 7 | C | `mark_own_paths` (simulation/enumerated.rs) reimplements the ~10-line path-marking loop that is inlined in `training/forward/enumerated.rs`; folds with the CD-028 dedup | `crates/cobre-sddp/src/simulation/enumerated.rs::mark_own_paths` · `crates/cobre-sddp/src/training/forward/enumerated.rs::run_enumerated_forward` (the training-side copy is an inline span inside this fn; no symbol of its own) | The training-side copy has no symbol — recorded via its enclosing fn; the :502-513 span predates the −103-line hoist and must be re-located. | _pending_ |
| CD-037 | 7 | C | The gap-rule rejecters each re-evaluate `rule_is_gap` + `training_enumerated`, and which rejection message a sampled-forwards study sees depends on call order inside `admission_gate` (documented as deliberate) | `crates/cobre-sddp/src/setup/mod.rs::admission_gate` · `crates/cobre-sddp/src/setup/mod.rs::reject_gap_under_effective_risk_aversion` · `crates/cobre-sddp/src/setup/mod.rs::reject_gap_under_sampled_selection` · `crates/cobre-sddp/src/setup/mod.rs::rule_is_gap` | — | _pending_ |
| CD-038 | 7 | C | `CutStateProjection` exposes only the fused `dot_trial_state`; the cut-selection value sweep in `run_cut_management` open-codes the gather-only loop (via `global_state_index`) — loop-shape duplication, not a hazard | `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs::dot_trial_state` · `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs::CutStateProjection` · `crates/cobre-sddp/src/lp/indexer/cut_state_projection.rs` · _`gather_trial_state` unresolved — proposed sibling; does not exist (expected)_ | Spans (no declaration): mod.rs L1118 (open-coded gather (global_state_index)). | _pending_ |
| OD-009 | 7 | C | Production items ship as `#[allow(dead_code)]` 'for symmetry, exercised only by tests' (`RuntimeHandles.export_states`, `RankDistribution.my_rank`, `ExchangeBuffers::new`) — the sanctioned Symmetry-or-test-retention class; `#[cfg(test)]`-gating is the cleaner form | `crates/cobre-sddp/src/training/session/runtime.rs::RuntimeHandles` (#[allow(dead_code)] at :16, `export_states` at :17) · `crates/cobre-sddp/src/training/session/rank_distribution.rs::RankDistribution` (`my_rank` at :15) · `crates/cobre-sddp/src/training/state_exchange.rs::ExchangeBuffers` · `crates/cobre-sddp/src/training/state_exchange.rs::new` (`ExchangeBuffers::new`, #[allow(dead_code)] at :121) · `crates/cobre-sddp/src/training/state_exchange.rs::with_actual_counts` | `rank_distribution.rs` is `training/session/rank_distribution.rs` at the baseline. | _pending_ |

**Owned elsewhere — do not raise here:** CD-002, CD-009 (cobre-cli station (E06)) · CD-011 (cobre-io station (E02, ratified) per the ticket).

- **CD-002** (Sev B, register L168) — Positional 10-tuple straddling three concerns in `broadcast_and_build_setup` (cobre-cli run/setup.rs); output now a named `LoadBroadcastResult`, input tuple unchanged. Anchors: `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup` · `crates/cobre-cli/src/commands/run/setup.rs::LoadBroadcastResult`.
- **CD-009** (Sev C, register L312) — `policy_dir` resolve + `!exists()` guard repeated 3× in cobre-cli run/policy.rs (WarmStart, Resume, simulation). Anchors: policy.rs L172 (policy_dir guard (WarmStart)) · policy.rs L201 (policy_dir guard (Resume)) · policy.rs L334 (policy_dir guard (simulation)).
- **CD-011** (Sev C, register L334) — Two conventions for the borrowed/owned duality (`Owned*` vs `*Payload`/`*ReadResult`) plus the stale `PolicyStageManifest` name. Anchors: `crates/cobre-sddp/src/policy/policy_load.rs::PolicyStageManifest` (NOTE: this anchor is in cobre-sddp although the ticket marks CD-011 io-owned — the record types live in cobre-io; the rename half lives here. Record as owned-elsewhere per the ticket, do not raise.).

**Roster count.** The ticket lists 19 rows; the ROADMAP reconciliation says 22 owned dispositions. Waves table (BACKLOG L1929/L1931/L1932) yields exactly the ticket's 19 sddp-owned pending rows: Wave 4 → CD-004, CD-005, CD-024-successor (CD-001/CD-003 are retire-with-commit; CD-002 cli-owned); Wave 6 → CD-015, CD-022, CD-018, CD-035, CD-034, CD-023 (CD-014 relieved, carried as the Wave-7 remnant); Wave 7 → CD-007, CD-012, CD-014-remnant, CD-016, CD-021, CD-028, CD-030, CD-037, CD-038, OD-009 (CD-009 cli-owned, CD-011 io-owned). 19 + the 3 retire-with-commit dispositions E05-3 must also fill (CD-001, CD-003 hop, CD-006) = 22 — the reading consistent with 'owned dispositions'. (19 + the 3 owned-elsewhere ids also = 22, but those are 'do not raise here', not dispositions.)

**Wave 7 also carries, without ids:** the two v0.14.2 residues (no IDs) (register L1374) — (a) test copy of the interior-node filter at `session/mod.rs:2729-2737` can diverge from the cached `interior_cut_nodes`; (b) one-owner nit — a `NodeGraph::interior_cut_nodes()` accessor (sibling of `backward_cut_levels`, node_graph.rs:774) would give constructor + test one owner. Registered in the mirror's perf-section residue; Waves table L1932 carries them in Wave 7; the ticket's 19 omit them. These ride CD-006's one-owner residue below and are not separate roster rows.

**sddp-owned ids outside Waves 4/6/7 — also not raisable here:**

- **CD-019** — deferred to next licensed public-API break (Wave 0 override, L1995); mirror §'Superseded cut-sync public methods' :334. arrives as the solver-comm E5 dup-of handoff (handoffs.json) — dup-of, no new id
- **CD-025** — open, Wave 5. cli+python station (E06-2) owns Wave 5
- **CD-029** — partial 2026-09-17 (Python half fixed; PrepPhase doc/abstraction half open), Wave 5. E06-2 owns; validate_phases.rs is an sddp file — an sddp attacker restating the 'four vs three' doc drift is a dup-of CD-029
- **CD-039** — setup-half RESOLVED (Wave 3); writer channel + transit-bucket remainder open. mirror L871 '#### Boundary state-family coupling channels are per-family bespoke'; the reserved seams 2 and 3 below cite it
- **CD-017 / CD-020 / CD-036 / CD-032 / CD-033 / PD-005** — executed Waves 0/2. not open; a candidate re-deriving them is a re-raise of a fixed item

**Rule:** anchors are path + symbol, never path + line. A historical anchor that resolves to no
symbol at the baseline is recorded `unresolved`, never silently refreshed; an anchor that names a
_proposed_ helper (a fix-shape target that was never built) is unresolved by construction and says
so. Every row is `_pending_` until E05-3 writes keep / retire-with-commit / sharpen; an attacker
candidate that restates a row is merged into that row's sharpening, never minted as a new id.

- **reraiseKey**: `CD-004`, `StudyParams`, `from_config`, `BroadcastConfig`, `build_study_setup`, `CD-005`, `StudySetup`, `from_broadcast_params`, `set_budget`, `set_risk_measures`, `set_scheduler`, `CD-024-successor`, `ResolvedParameters`, `build_resolved_parameters`, `CD-015`, `compute_one_backward_node`, `by_node_finish`, `process_by_scenario_backward`, `CD-022`, `run_enumerated_backward`, `SuccessorEntry`, `SuccessorOutcomes`, `build_delta_cut_row_batch_into`, `CD-018`, `CutSelectionStrategy`, `select_for_stage`, `DcsParams`, `from_strategy`, `lazy_solve_preloaded`, `CD-035`, `accumulate_opening_outcome`, `write_opening_outcome`, `CutStateProjection`, `SuccessorSpec`, `CD-034`, `fill_operational_violation_entries`, `fill_operational_violation_rows`, `CD-023`, `StageSolvePrep`, `run`, `run_stage_solve`, `StageInputs`, `CD-007`, `fill_load_balance_entries`, `fill_ncs_load_balance_entries`, `CD-012`, `run_cut_management`, `run_iteration`, `CD-014-remnant`, `CD-016`, `process_stage_backward`, `process_stage_backward_by_node`, `BackwardAccumulators`, `CD-021`, `CapturedBasis`, `WorkspaceSizing`, `ByNodeScratch`, `ScratchBuffers`, `SolverWorkspace`, `WorkspacePool`, `BasisStore`, `BasisStoreSliceMut`, `CD-028`, `run_sweep`, `enumerated_sim_stage_worker`, `run_enumerated_forward`, `enumerated_stage_worker`, `CD-030`, `mark_own_paths`, `CD-037`, `admission_gate`, `reject_gap_under_effective_risk_aversion`, `reject_gap_under_sampled_selection`, `rule_is_gap`, `CD-038`, `dot_trial_state`, `OD-009`, `RuntimeHandles`, `RankDistribution`, `ExchangeBuffers`, `new`, `with_actual_counts`, `CD-019`, `CD-025`, `CD-029`, `CD-039`, `interior_cut_nodes`

---

## Retire-with-commit — closed, cite, never re-raise

Each claim was confirmed in the baseline tree BEFORE being recorded as retired (commands under
Re-derive). E05-3 fills the `retire-with-commit` disposition from these rows.

### CD-001 — closed by `b051c410`

- **Claim (historical, register L152)**: CLI crate hand-rolled a mirror of the rank-0 stochastic pipeline (`rebuild_historical_library_non_root`)
- **Closing commit**: `b051c410` — refactor(sddp,cli): own the non-root stochastic-context rebuild in cobre-sddp
- **Confirmed at the baseline**: yes.
- **Register trail**: BACKLOG L2193 (Wave 3 continuation, Phase 4)
- **Residue (may be sharpened, never re-derived)**: cobre-cli `reconstruct_stochastic_context_non_root` (run/setup.rs:396) survives as the thin non-root caller of the single owner. Two DOC mentions of the deleted `rebuild_historical_library_non_root` remain (setup/mod.rs:1027, stochastic_pipeline.rs:1298/1301 in a test doc) — stale doc references, a doc-drift note for E07, not a re-open.
- **Anchors**: `crates/cobre-sddp/src/setup/stochastic_pipeline.rs::build_stochastic_context_for_study` · `crates/cobre-cli/src/commands/run/setup.rs::reconstruct_stochastic_context_non_root` (cross-station) · `crates/cobre-sddp/src/setup/mod.rs` · _`rebuild_historical_library_non_root` unresolved — deleted — only doc mentions at setup/mod.rs:1027 and stochastic_pipeline.rs:1298_
- **reraiseKey**: `CD-001`, `build_stochastic_context_for_study`, `reconstruct_stochastic_context_non_root`, `rebuild_historical_library_non_root`

### CD-003 (Construction hop) — closed by `4075c4e8`

- **Claim (historical, register L176)**: `BroadcastConfig` → `ConstructionConfig` ~22-field manual copy in `build_study_setup`; third parallel representation of run config
- **Closing commit**: `4075c4e8` — refactor(sddp): merge StudyParams and ConstructionConfig into one local projection
- **Confirmed at the baseline**: yes.
- **Register trail**: BACKLOG L2186 (Wave 3 continuation, Phase 3): `ConstructionConfig` + `into_construction_config` DELETED; `BroadcastConfig` wire projection UNTOUCHED (= Wave 4)
- **Residue (may be sharpened, never re-derived)**: Config (cobre-io config/mod.rs:67) → BroadcastConfig (cobre-cli commands/broadcast.rs:87; `impl BroadcastConfig` :142 calls `StudyParams::from_config` at :144) → StudyParams (setup/params.rs:114) via `build_study_setup` (cobre-cli run/setup.rs:423). The Broadcast→Construction hop is gone; the Config→Broadcast→StudyParams two-carrier residue is CD-004's Wave-4 business.
- **Anchors**: `crates/cobre-io/src/config/mod.rs::Config` (cross-station) · `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig` (cross-station) · `crates/cobre-sddp/src/setup/params.rs::StudyParams` · `crates/cobre-sddp/src/setup/params.rs::from_config` · `crates/cobre-cli/src/commands/run/setup.rs::build_study_setup` (cross-station)
- **reraiseKey**: `CD-003`, `Config`, `BroadcastConfig`, `StudyParams`, `from_config`, `build_study_setup`

### CD-006 — resolved in-tree, no closing sha registered

- **Claim (historical, register L227)**: `NodeGraph` query surface scattered as free functions (`frontier_node`, `node_parent`, …) with exactly one method
- **Confirmed at the baseline**: yes.
- **Register trail**: RESOLVED in the 2026-08-18 whole-lifecycle reconciliation (BACKLOG L1185; re-confirmed L1676); mirror 'Byte-neutral consolidations — already executed' §662. No closing sha recorded in the register — record 'resolved, sha not registered', do not invent one.
- **Residue (may be sharpened, never re-derived)**: None for the query surface: six `impl NodeGraph` blocks (:369, :748, :880, :1018, :1114, :1192); methods node_pool_ids :375, max_successor_outcome_count :759, backward_cut_levels :774, pool_cut_stride :910, forward_solve_counts :1038, stage_frontier :1120, frontier_node :1133, any_stage_node :1149, node_parent :1198, node_opening_range :1217, node_pinned_scenario :1232, build_parent_map :1265. `assemble_outcome_weights` (:717) stays free by design (algorithm, not a query). The v0.14.2 'one-owner nit' (a `NodeGraph::interior_cut_nodes()` accessor) is the only related open item (see rosterDiscrepancy).
- **Anchors**: `crates/cobre-sddp/src/setup/node_graph.rs::NodeGraph` · `crates/cobre-sddp/src/setup/node_graph.rs::frontier_node` · `crates/cobre-sddp/src/setup/node_graph.rs::node_parent` · `crates/cobre-sddp/src/setup/node_graph.rs::backward_cut_levels` · `crates/cobre-sddp/src/setup/node_graph.rs::stage_frontier` · `crates/cobre-sddp/src/setup/node_graph.rs::max_successor_outcome_count` · `crates/cobre-sddp/src/setup/node_graph.rs::build_parent_map` · `crates/cobre-sddp/src/setup/node_graph.rs::assemble_outcome_weights` (deliberately free)
- **reraiseKey**: `CD-006`, `frontier_node`, `node_parent`, `backward_cut_levels`, `stage_frontier`, `max_successor_outcome_count`, `build_parent_map`, `assemble_outcome_weights`

---

## Do-not-touch — retracted, refuted, deferred, sanctioned

Owner decisions already taken (register do-not-touch list L1942). PD-004 is the one item an
attacker may _mention_: as an existence check queued to the perf epic (E10-2) with no severity.

### CD-008 — RETRACTED (WONTFIX, deliberate-by-design)

- **Claim**: `fill_parallel_water_entries` vs `fill_chronological_water_entries` looked ~70% duplicated; the overlap masks two genuinely different LP formulations (rows, storage variables, noise scale, arc-release helper) (register L273)
- **Anchors**: `crates/cobre-sddp/src/lp/builder/entries.rs::fill_parallel_water_entries` · `crates/cobre-sddp/src/lp/builder/entries.rs::fill_chronological_water_entries`
- **reraiseKey**: `CD-008`, `fill_parallel_water_entries`, `fill_chronological_water_entries`

### PD-001 — REFUTED 2026-08-18 (not debt; do not fix) — re-confirmed L1760

- **Claim**: `enumerated_sim_stage_worker` allocates `out_state`/`AccumSnapshot` per claimed node — deliberate one-shot move-scatter (one allocation per arena buffer, zero copies) per the `worker_captures` rationale (register L1276)
- **Anchors**: `crates/cobre-sddp/src/simulation/enumerated.rs::enumerated_sim_stage_worker` · `crates/cobre-sddp/src/simulation/enumerated.rs::EnumeratedSimScratch` · enumerated.rs L260 (AccumSnapshot::default() / to_vec() moves)
- **reraiseKey**: `PD-001`, `enumerated_sim_stage_worker`, `EnumeratedSimScratch`

### PD-004 — DEFERRED pending a profile (2026-08-18; re-confirmed L1757) — existence check + queue to the perf epic only

- **Claim**: `run_enumerated_backward` builds a fresh `Vec<(usize, Vec<StageWorkerOpeningDelta>)>` + per-stage `SolverStatsDelta::clone()` per run; symmetric with `run_sampled_backward`, so fixing one alone would diverge them (register L1316)
- **Anchors**: `crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward` (register's :713 / :919-931 are stale line refs; symbol resolves) · `crates/cobre-sddp/src/training/backward_pass_state.rs::run_sampled_backward`
- **reraiseKey**: `PD-004`, `run_enumerated_backward`, `run_sampled_backward`

### sanctioned #[allow(...)] census — excluded by construction (BACKLOG L69, L1669; do-not-touch list L1942)

- **Claim**: Every `#[allow(...)]` under crates/*/src falls in a sanctioned class carrying a `// Rationale:`; an OD candidate that is only 'this allow exists' is Cleared, not recorded (OD-009 is the one owner-optional refinement of the Symmetry-or-test-retention class) (register L1942)
- **Mirror**: `docs/design/reserved-seams-and-deferred-debt.md:1272` '### `#[allow(...)]` census' — three classes (Load-bearing, Reserved-seam (Voice 4), Symmetry-or-test-retention) + the narrower `#[allow(deprecated)]` class
- **Census at the baseline**: 102 files carry `#[allow(` (514 openers, 22 single-line `dead_code`); counts include #[cfg(test)] modules and sibling tests.rs; multi-line `#[allow(` openers counted once each; top lints cast_possible_truncation 116, too_many_lines 53, cast_precision_loss 42, too_many_arguments 40 Re-derive: `git grep -c -E '#\[allow\(' 077dbe2c -- crates/cobre-sddp/src | awk -F: '{s+=$NF} END{print NR, s}'`
- **reraiseKey**: `allow`

---

## Reserved seams — Cleared-with-citation, never a live OD finding

`CLAUDE.md` hard rule: unwired config is reserved, not dead. An over-engineering candidate whose
subject is one of these seams is closed as **Cleared (sanctioned)** at ingest with the citation
below and assigns no OD id; a candidate about a _defect inside_ the seam's implementation is not
filtered. Mirror citations are lines of the baseline blob of the mirror.

### `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`

- **Mirror**: `docs/design/reserved-seams-and-deferred-debt.md:54` '### `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`'
- **Anchors**: `crates/cobre-io/src/config/training.rs::UpperBoundEvaluationConfig` · `crates/cobre-io/src/config/training.rs::LipschitzConfig` · `crates/cobre-io/src/config/training.rs::mode`
- **Note**: Lives in cobre-io (core-io station cleared it as sanctioned, BACKLOG L3177); the sddp attackers meet it as 'loaded but unread anywhere in crates/cobre-sddp'. Also a CLAUDE.md hard rule ('Unwired config is reserved, not dead'). Its milestone is a DECISION (implement vertex-based UB or retire), so it is the strongest removal candidate in the register — still Cleared here, the decision is E11's.
- **reraiseKey**: `UpperBoundEvaluationConfig`, `LipschitzConfig`, `mode`, `LipschitzConfig.mode`

### Writer's second-family reserved slot body (boundary state-family coupling channel)

- **Mirror**: `docs/design/reserved-seams-and-deferred-debt.md:871` '#### Boundary state-family coupling channels are per-family bespoke' (under '### Post-release fix-wave findings (2026-08-19)' :828; Trigger at :910)
- **Register**: CD-039 (L1515; setup-half RESOLVED L2140; writer + transit remainder open)
- **Anchors**: `crates/cobre-sddp/src/policy/policy_export.rs::splice_reserved_state_block` (family-independent core, extracted 273f801f) · `crates/cobre-sddp/src/policy/policy_export.rs::reserve_boundary_inflow_lag_slots` (the one authored family's slot-body constructor)
- **Note**: What remains for a second authored family is its own reserved slot-body constructor + keyed per-cut coefficient field; the mirror says the debt is the missing per-family wiring, not the shared mechanism. An OE candidate calling `splice_reserved_state_block`'s generality one-consumer is Cleared with this citation.
- **reraiseKey**: `splice_reserved_state_block`, `reserve_boundary_inflow_lag_slots`

### Anticipated post-study-commitment channel (ticket text: 'the anticipated delivery_date channel')

- **Mirror**: `docs/design/reserved-seams-and-deferred-debt.md:871` (same entry: 'the anticipated post-study-commitment import is the concrete queued case', :910-912); related :440 '### Anticipated-commitment per-block tension' and :451 '### Pre-study-decided, post-study-delivered anticipated commitments (no carrier) — RESOLVED'
- **Register**: CD-039
- **Anchors**: codec.rs L40 (`delivery_date` anchor replaced by `reference_date`/`interval_start`/`interval_end`) · `crates/cobre-sddp/src/policy/policy_export.rs::splice_reserved_state_block` · `crates/cobre-sddp/src/policy/policy_export.rs::reserve_boundary_inflow_lag_slots`
- **Note**: TICKET WORDING STALE: at 077dbe2c no anticipated slot carries a `delivery_date` field — codec.rs:40 records the anchor was replaced by `reference_date`/`interval_start`/`interval_end`; `delivery_date` survives only in simulation output rows (simulation_writer.rs:436, simulation/extraction.rs:302) and production/conversion.rs:259. BACKLOG L2184 ('anticipated slot `delivery_date` is resolver-derived') is the 2026-08-22 wording. Record the seam under its mirror name and note the field rename; do not anchor `delivery_date` in policy_export.rs (it would fail check-anchors).
- **reraiseKey**: `splice_reserved_state_block`, `reserve_boundary_inflow_lag_slots`

### Legacy (`None`) cost-scale branch in `rescale_cut_records_for_load`

- **Mirror**: `docs/design/reserved-seams-and-deferred-debt.md:30` "## Reserved-seam register" — **GAP: no entry for this seam at the baseline.** The register documents it (BACKLOG L2272, "New documented reserved seam", Wave 3 continuation) and the code's own rustdoc sanctions it; the mirror entry is owed by the E11 mirror write-back. Recorded as a discovery on E05-1, not invented here.
- **Anchors**: `crates/cobre-sddp/src/policy/policy_load.rs::rescale_cut_records_for_load` · `crates/cobre-sddp/src/policy/policy_load.rs::LEGACY_COST_SCALE_FACTOR` · policy_load.rs L98 (Legacy branch (`loading_cost_scale_factor == LEGACY_COST_SCALE_FACTOR`)) · policy_load.rs L56 ('Reserved seam' rustdoc)
- **Note**: NO mirror entry at 077dbe2c (grep for cost-scale / LEGACY / rescale_cut_records in docs/design/reserved-seams-and-deferred-debt.md returns nothing; :538 'Legacy' is the retired-input-spellings entry). The sanction is the code's own rustdoc at policy_load.rs:52-60 ('Reserved seam: this repo's own front ends never reach the `None`/Legacy path … reachable only by a direct library caller') plus the CLAUDE.md 'reserved, not dead' rule. The ticket's AC-6 asks for a mirror citation on each of the four seams — for this one E05-1 must record 'code-doc sanctioned, no mirror entry' honestly and hand the mirror gap to E11 rather than invent a citation.
- **reraiseKey**: `rescale_cut_records_for_load`, `LEGACY_COST_SCALE_FACTOR`, `None`

---

## Contract-first rule — `.claude/rules/sddp.md`

A fix-shape in this crate cites the governing contract **before** it proposes touching the cut
pool, the cut sign, state pinning, FPHA storage, risk aggregation or the policy-load path. A
fix-shape that would weaken a pinned contract is not proposed; the contract is named as the
reason the current shape is correct. Headings and lines are those of the baseline blob.

| Hazard area | `.claude/rules/sddp.md` section | Implementing symbol | Also |
| ----------- | ------------------------------- | ------------------- | ---- |
| Benders cut sign / subgradient extraction | `.claude/rules/sddp.md:13` "Benders cut sign & subgradient extraction" | `crates/cobre-sddp/src/training/backward/duals_extraction.rs::extract_duals_from_view` | `crates/cobre-sddp/src/cut/row.rs::push_scaled_coefficient` (:22, negates the raw subgradient); `crates/cobre-sddp/src/training/backward/outcome_aggregation.rs::write_opening_outcome` (:102) |
| State pinning via column bounds (never equality rows) | `.claude/rules/sddp.md:55` "State pinning uses column bounds, not equality rows" | `crates/cobre-sddp/src/lp/indexer/state_space.rs::state_to_lp_incoming_column` | `crates/cobre-sddp/src/training/backward/lp_setup.rs::patch_opening_bounds` (:44); `crates/cobre-sddp/src/training/stage_solve_prep.rs::run` (`StageSolvePrep::run`) (:85; set_col_bounds at :157) |
| FPHA average storage | `.claude/rules/sddp.md:63` "FPHA uses average storage" | `crates/cobre-sddp/src/lp/builder/entries.rs::fill_fpha_entries` | `crates/cobre-sddp/src/lp/builder/rows.rs::fill_fpha_rows` (:351) |
| Append-only cut pool + slot-identity basis matching | `.claude/rules/sddp.md:330` "Cut pool is append-only; basis matches by slot identity" | `crates/cobre-sddp/src/cut/basis_reconstruct.rs::reconstruct_basis` | `crates/cobre-sddp/src/cut/basis_reconstruct.rs::reconstruct_basis_uniform_basic` (:176, DCS path); `crates/cobre-sddp/src/cut/pool.rs::CutPool` (:58; grow :983); `crates/cobre-sddp/src/cut/cut_selection.rs::CutMetadata` (:65; `node` field :77) |
| Stored basis warm-starts only at its own node (node-tag) | `.claude/rules/sddp.md:409` "A stored basis warm-starts only at its own node (node-tag)" | `crates/cobre-sddp/src/solve/stage_solve.rs::run_stage_solve` | `crates/cobre-sddp/src/solve/stage_solve.rs:160` (`.filter(|captured| captured.node_id == inputs.node_id)`); `crates/cobre-sddp/src/workspace/workspace.rs::CapturedBasis` (:44; `node_id` field :57) |
| Joint risk applied once over the flattened successor×opening vector | `.claude/rules/sddp.md:601` "Joint risk is applied once over the flattened successor×opening vector" | `crates/cobre-sddp/src/convergence/risk_measure.rs::aggregate_cut_into` (`RiskMeasure::aggregate_cut_into`) | `crates/cobre-sddp/src/setup/node_graph.rs::assemble_outcome_weights` (:717); `crates/cobre-sddp/src/training/backward/by_scenario.rs::process_by_scenario_backward` (:398); `crates/cobre-sddp/src/training/backward/by_node.rs::by_node_finish` (:416) |
| Nested enumerated CVaR upper bound | `.claude/rules/sddp.md:664` "The enumerated CVaR upper bound is NESTED, not end-of-horizon" | `crates/cobre-sddp/src/training/forward/stats_aggregation.rs::nested_ub_recursion` | `crates/cobre-sddp/src/training/forward/stats_aggregation.rs::ForwardBound` (:29, `NestedRisk` arm); `crates/cobre-sddp/src/training/forward/stats_aggregation.rs::sync_forward` (:105); `crates/cobre-sddp/src/convergence/risk_measure.rs::uniform_effective_measure` (:261) |
| Mandatory policy-load compatibility validation | `.claude/rules/sddp.md:816` "Policy-load compatibility validation is mandatory" | `crates/cobre-sddp/src/policy/policy_load.rs::validate_policy_load` | `crates/cobre-sddp/src/policy/policy_load.rs::PolicyLoadProof` (:232); `crates/cobre-sddp/src/policy/policy_load.rs::load_boundary_cuts` (:1025); `crates/cobre-sddp/src/policy/policy_load.rs::compare_graph_manifest_identity` (:305) |

All `## ` sections of the baseline blob, for completeness (a fix-shape may need to cite one outside the eight hazard areas):

- `.claude/rules/sddp.md:13` Benders cut sign & subgradient extraction
- `.claude/rules/sddp.md:55` State pinning uses column bounds, not equality rows
- `.claude/rules/sddp.md:63` FPHA uses average storage
- `.claude/rules/sddp.md:73` Hydro-cell aggregation assumes one production map per cell
- `.claude/rules/sddp.md:330` Cut pool is append-only; basis matches by slot identity
- `.claude/rules/sddp.md:409` A stored basis warm-starts only at its own node (node-tag)
- `.claude/rules/sddp.md:476` NCS stochastic availability is a dimensionless factor
- `.claude/rules/sddp.md:483` Lower-bound evaluation must patch NCS
- `.claude/rules/sddp.md:493` Per-level exchange in the backward pass
- `.claude/rules/sddp.md:502` Backward opening order is warm-start-only
- `.claude/rules/sddp.md:532` By-node scheduler is warm-start-only
- `.claude/rules/sddp.md:601` Joint risk is applied once over the flattened successor×opening vector
- `.claude/rules/sddp.md:629` The branching backward integrates every successor exhaustively
- `.claude/rules/sddp.md:657` No EWMA upper bound
- `.claude/rules/sddp.md:664` The enumerated CVaR upper bound is NESTED, not end-of-horizon
- `.claude/rules/sddp.md:717` Terminal boundary FCF is booked in the reported total cost
- `.claude/rules/sddp.md:750` Fused terminal slice projects with the parent pool, not the leaf pool
- `.claude/rules/sddp.md:801` Spillage is frozen `[0, 0]` during PreFilling
- `.claude/rules/sddp.md:816` Policy-load compatibility validation is mandatory
- `.claude/rules/sddp.md:1310` Initial-state seeding resolves IDs through a position map, never `binary_search`
- `.claude/rules/sddp.md:1339` Water travel time
- `.claude/rules/sddp.md:1678` Anticipated thermal commitments

- **reraiseKey**: `cut sign`, `subgradient`, `column bounds`, `state pinning`, `average storage`, `append-only`, `slot identity`, `node-tag`, `warm-start`, `joint risk`, `flattened`, `nested CVaR`, `policy-load`, `compatibility validation`, `reconstruct_basis`

---

## Byte-neutrality bar — every proposal states it

Every fix-shape recorded for this crate states how it stays byte-neutral against
1. parity goldens (crates/cobre-sddp/tests/common/parity_hash.rs);
2. rank-invariance harness;
3. mpiexec -n 1/2 reproducibility;

or states which golden it intends to move and why. The harness files are
`crates/cobre-sddp/tests/common/parity_hash.rs` (golden SHA helper), `crates/cobre-sddp/tests/parity.rs`
and `crates/cobre-sddp/tests/common/permute.rs` (order-invariance shuffle); the golden decks live under
`crates/cobre-sddp/tests/fixtures/`. **This station executes no measurement and no fix**: perf claims are queued UNMEASURED with a
layout; fix-shapes are prose.

---

## Do not re-raise

- any Wave 4/6/7 roster row above as a new finding — sharpen the row (E05-3 owns the disposition)
- CD-001 — closed by `b051c410`
- CD-003 (Construction hop) — closed by `4075c4e8`
- CD-006 — resolved in-tree (register L1185)
- CD-008 — RETRACTED (WONTFIX, deliberate-by-design)
- PD-001 — REFUTED 2026-08-18 (not debt; do not fix) — re-confirmed L1760
- PD-004 — DEFERRED pending a profile (2026-08-18; re-confirmed L1757) — existence check + queue to the perf epic only
- sanctioned #[allow(...)] census — excluded by construction (BACKLOG L69, L1669; do-not-touch list L1942)
- `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig` as dead / speculative / unwired — reserved seam (Cleared-with-citation)
- Writer's second-family reserved slot body (boundary state-family coupling channel) as dead / speculative / unwired — reserved seam (Cleared-with-citation)
- Anticipated post-study-commitment channel (ticket text: 'the anticipated delivery_date channel') as dead / speculative / unwired — reserved seam (Cleared-with-citation)
- Legacy (`None`) cost-scale branch in `rescale_cut_records_for_load` as dead / speculative / unwired — reserved seam (Cleared-with-citation)
- CD-002, CD-009 (cobre-cli station) and CD-011 (cobre-io station) — owned elsewhere
- CD-019 (E5 dup-of from solver-comm), CD-025 / CD-029 (Wave 5, cli+python station), the CD-039 remainder (the seams above), and the executed Wave 0/2 ids
- a fix-shape that weakens any contract in the table above — cite the contract instead

## Re-derive

```sh
git ls-tree -r 077dbe2c --name-only -- crates/cobre-sddp/src | grep '\.rs$' | wc -l  # 163
git log --oneline -1 b051c410 ; git log --oneline -1 4075c4e8
git grep -n 'fn build_stochastic_context_for_study' 077dbe2c -- crates/cobre-sddp/src/setup/
git grep -n 'fn rebuild_historical_library_non_root' 077dbe2c -- crates || echo 'CD-001: deleted, confirmed'
git grep -n 'ConstructionConfig\|into_construction_config' 077dbe2c -- crates || echo 'CD-003 hop: deleted, confirmed'
git grep -n '^impl NodeGraph' 077dbe2c -- crates/cobre-sddp/src/setup/node_graph.rs
git grep -n -E '^pub(\(crate\))? fn (frontier_node|node_parent|backward_cut_levels|stage_frontier|max_successor_outcome_count|build_parent_map)\b' 077dbe2c -- crates/cobre-sddp/src || echo 'CD-006: zero free-fn defs'
git grep -n -E 'fn (run_enumerated_backward|run_sampled_backward|compute_one_backward_node|process_stage_backward)\b' 077dbe2c -- crates/cobre-sddp/src/training/
git grep -n -E 'pub(\(crate\))? struct (CapturedBasis|WorkspaceSizing|BackwardAccumulators|ByNodeScratch|ScratchBuffers|SolverWorkspace|WorkspacePool|BasisStore|BasisStoreSliceMut)\b' 077dbe2c -- crates/cobre-sddp/src/workspace/
git ls-tree -r 077dbe2c --name-only -- crates/cobre-sddp/src/workspace/  # context.rs mod.rs workspace.rs
git grep -n 'training::stage_solve_prep' 077dbe2c -- crates/cobre-sddp/src/simulation/pipeline.rs ; git grep -n 'struct StageSolvePrep' 077dbe2c -- crates/cobre-sddp/src/training/stage_solve_prep.rs
git grep -n '^#\[cfg(test)\]' 077dbe2c -- crates/cobre-sddp/src/lp/builder/entries.rs crates/cobre-sddp/src/lp/builder/columns.rs | head -2 ; for f in entries columns; do git show 077dbe2c:crates/cobre-sddp/src/lp/builder/$f.rs | wc -l; done
git grep -n -E 'pub struct (UpperBoundEvaluationConfig|LipschitzConfig)|pub mode' 077dbe2c -- crates/cobre-io/src/config/training.rs
git grep -n -E 'fn (splice_reserved_state_block|reserve_boundary_inflow_lag_slots|rescale_cut_records_for_load)|LEGACY_COST_SCALE_FACTOR|Reserved seam' 077dbe2c -- crates/cobre-sddp/src/policy/
git show 077dbe2c:docs/design/reserved-seams-and-deferred-debt.md | grep -n '^## \|^### \|^#### '
git show 077dbe2c:docs/design/reserved-seams-and-deferred-debt.md | grep -n -i 'cost.scale\|LEGACY\|rescale_cut_records' || echo 'no mirror entry for the Legacy cost-scale branch'
git show 077dbe2c:.claude/rules/sddp.md | grep -n '^## '
git grep -c -E '#\[allow\(' 077dbe2c -- crates/cobre-sddp/src | awk -F: '{s+=$NF} END{print NR" files", s" openers"}'
grep -n '^| \*\*[0-7]\*\* |' plans/architecture-debt-audit/BACKLOG.md  # waves table rows L1925-1932
```
