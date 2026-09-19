# Ingest log — station cobre-sddp

Baseline: `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`  ·  ingested 2026-09-18  ·  sub-stations 5a 5b 5c 5d

Sources screened: the sixteen gated envelopes `candidates.<lens>.<subStation>.json` written by the
attacker ticket, each re-validated here with `tools/validate-envelope.py --role attacker --station sddp`
(exit 0 on all sixteen). candidateRef = `<subStation>-<lens>-<nn>`, nn the zero-based index of the
candidate in its envelope — the envelopes carry no attacker id of their own (the gate records key on
`cell` + index), so this ref is the stable key every screen, the defender dispatch and `verdicts.json`
use. Every candidate is accounted for exactly once in the roster at the end and in `verdicts.json`.
Screens ran in the fixed order anchor → station scope → reserved seams → dup-of / re-raise (roster,
prior register, solver-comm handoff) → contract → cross-cell overlap fold → defender.

## Candidate census

n (positives, needs-human) per cell; sub-stations 5a setup+policy+stochastic+config, 5b lp/, 5c cut+training+solve+workspace, 5d simulation+production+support.

| lens | 5a | 5b | 5c | 5d | total |
|---|---|---|---|---|---|
| architecture | 4 (5 pos, 0 nh) | 5 (10 pos, 2 nh) | 5 (7 pos, 1 nh) | 2 (5 pos, 2 nh) | 16 |
| performance | 3 (14 pos, 2 nh) | 3 (7 pos, 1 nh) | 7 (8 pos, 2 nh) | 8 (18 pos, 2 nh) | 21 |
| over-engineering | 4 (14 pos, 1 nh) | 2 (9 pos, 1 nh) | 2 (13 pos, 1 nh) | 3 (16 pos, 1 nh) | 11 |
| test-bloat | 7 (8 pos, 0 nh) | 8 (5 pos, 0 nh) | 6 (6 pos, 2 nh) | 6 (5 pos, 0 nh) | 27 |
| **total** | 18 | 18 | 20 | 19 | **75** |

No cell returned zero candidates, so there is no no-finding line to record; every cell also carries
at least one `positives` entry, so an empty cell would have been distinguishable from a skipped one.
Needs-human items raised by the attackers (18) travel on their cells and are re-attached to the
candidate they concern in `verdicts.json`.

## Anchor rejections

None (`anchorRejected = 0`). `anchor-probe.md` renders all 339 anchors of the 75 candidates,
every one as `path::symbol`: the attacker gate had already demoted each line-only, struct-field,
out-of-station and out-of-manifest anchor into `citedContext` (a cited context is evidence, not an
anchor), so no `path:line` anchor exists to flag coincidental-until-symbolised and no anchor was
refreshed by hand. Checker run at the pin:

```
plans/architecture-debt-audit/tools/check-anchors.py INGEST ANCHOR PROBE — sddp (2026-09, baseline) --register plans/architecture-debt-audit/stations/sddp/anchor-probe.md --baseline 077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c --json
{
  "baseline": "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c",
  "checked": 339,
  "failures": [],
  "section": "INGEST ANCHOR PROBE \u2014 sddp (2026-09, baseline)"
}
```

exit 0.

| candidateRef | failing anchor | checker diagnostic |
|---|---|---|
| — | — | — |

## Out-of-station hand-offs

None handed off (`handedOff = 0`): every anchor of every candidate sits under `crates/cobre-sddp/`.
Two candidates cite an out-of-station symbol as CONTEXT (rejected as an anchor by the attacker gate,
kept as `citedContext`); they stay in this station on their in-station anchors and the cited half is
recorded here for the owning station:

| candidateRef | cited context (not an anchor) | owning station | note |
|---|---|---|---|
| 5a-architecture-01 | `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig` | cli+python station (E06) | CD-004 wire-projection twin; the I.3-7 handoff already names it |
| 5a-architecture-02 | `crates/cobre-cli/src/commands/validate.rs::reconcile_boundary` | cli+python station (E06) | the `validate` path builds templates against the empty parameter table too; CD-004-adjacent |

## Cleared (sanctioned)

None (`sanctionedCleared = 0`). Sources searched, in order: (1) the mirror `## Reserved-seam register`
(four entries: transit bucket topology, `LipschitzConfig.mode` + `UpperBoundEvaluationConfig`, the
shared-memory communicator hierarchy, the boundary-cut `graph_stage_id`) and `## Verified NOT reserved`;
(2) the mirror entry `#### Boundary state-family coupling channels are per-family bespoke` (the writer's
reserved second-family slot body beside `splice_reserved_state_block` and the anticipated
resolver-derived `delivery_date` channel); (3) the mirror `### #[allow(...)] census` (Load-bearing /
Reserved-seam / Symmetry-or-test-retention); (4) CLAUDE.md `Unwired config is reserved, not dead`;
(5) the Legacy (`None`) cost-scale branch of `rescale_cut_records_for_load` — no mirror entry at this
pin (the inventory ticket routed the missing entry to the E11 mirror write-back), sanctioned by the
register (`CD-039` remainder) and the fn's rustdoc. The four named seams reached the envelopes only as
positives with `sanctionedBy`: `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig` ×4, Boundary state-family coupling channels are per-family bespoke ×5, Legacy (`None`) cost-scale branch of `rescale_cut_records_for_load` ×5, Superseded cut-sync public methods ×3. No candidate names a seam symbol (`LipschitzConfig`, `UpperBoundEvaluationConfig`,
`splice_reserved_state_block`, `delivery_date`, `rescale_cut_records_for_load`, `sync_cuts`,
`pack_local_records`, `sync_packed_records`) in its title or anchors — the attacker prompt applied the
seams as a filter, so nothing proposes their removal and no OD id can exist for them. The one
textual hit of the seam-symbol probe — `5c-test-bloat-00` names `sync_cuts` in its title and test-fn anchors — is the cut-sync adjudication under dup-of below (test-binary
duplication, not the superseded methods).

Candidates whose fix-shape deletes, narrows or gates an unconsumed or test-only item did reach the
screen; each is a miss and passes to its defender with the search recorded so the defender does not
repeat it:

| candidateRef | item the fix-shape removes / narrows | search result |
|---|---|---|
| 5a-architecture-03 | `CutManagementConfig::warm_start_cuts` (pub field, hard-coded 0 at both production sites, no production reader) | miss — not in the mirror register, not in Verified NOT reserved, not a census `dead_code` allow; CLAUDE.md `Unwired config is reserved, not dead` covers user-loaded config sections (loaded/validated/schema-exported), this field is an engine struct literal never fed from cobre-io config; the attacker's own fallback (keep as a reserved seam) needs owner sign-off, so the public-API-removal precedent of CD-019 is noted for calibration |
| 5b-architecture-04 | unused `_state: &StateSpace` parameter of `is_anticipated_decision_active_for_delivery` | miss — `anticipated_gate` absent from the mirror; not a `dead_code` allow |
| 5a-over-engineering-01 | `ncs_stochastic_dormant_for_test` shipped un-gated in the default API | miss — absent from the mirror; not an allow; the candidate gates, it does not delete |
| 5a-over-engineering-02 | `training_ctx` constructor `#[cfg]`-gated with no production consumer | miss — absent from the mirror; the fix UN-gates it (adds a consumer), the opposite of a dead-code sweep |
| 5a-over-engineering-03 | `setup/scenario_library_set.rs` module fold / rename | miss — absent from the mirror; module paths only |
| 5b-over-engineering-00 | `Col` / `Row` LP-index newtypes (zero consumers, pub re-export) | miss — absent from the mirror register and Verified NOT reserved; `lp/indexer/index.rs` carries no `#[allow(dead_code)]` (pub re-export hides the warning), so it is not census-class; public-API removal → CD-019 precedent noted |
| 5b-over-engineering-01 | `FphaRowRange` (zero consumers, doc names a carrier that does not exist) | miss — absent from the mirror; `lp/indexer/layout.rs` has no `#[allow(` at the pin (the census names lp/BUILDER/layout.rs), so it is not Reserved-seam class; public-API removal → CD-019 precedent noted |
| 5c-over-engineering-01 | `RankDistribution::actual_per_rank` forwarding method | miss — the mirror names the symbol once (`:933`, minor-residues note that the split arithmetic gains another implementation beside it) as a duplication residue, not as a reserved seam |
| 5d-over-engineering-01 | five `pub(crate)` items narrowable to private | miss — none in the mirror; visibility only |
| 5d-over-engineering-02 | `SimulationInputs::new` pass-through constructor | miss — absent from the mirror; the suppression it carries is a Load-bearing `too_many_arguments` allow, not a Reserved-seam allow |
| 5d-test-bloat-01 | no-op assertions in `hold_k1_byte_stability_probe.rs` | miss — test-only; not a seam |
| 5c-test-bloat-00 | `test_mpi_allgatherv_nonuniform_workers.rs` binary (delete) | miss — test-only; relation to the superseded cut-sync methods recorded under dup-of / related |
| 5a-over-engineering-00 | four Load-bearing `#[allow(clippy::cast_*)]` sites without `// Rationale:` | miss — the census sanctions the Load-bearing class ON THE PREMISE that each site carries a Rationale comment; the candidate disputes the premise (and `scripts/ci/` has an allow-rationale checker the defender must reconcile) — it does not propose deleting a Reserved-seam `dead_code` allow |
| 5d-over-engineering-00 | 42 of 69 production `#[allow(...)]` openers without rationale (5d manifest) | miss — same premise dispute as 5a-over-engineering-00 over the 5d manifest; sharpen-not-sweep fix; not a Reserved-seam deletion |

## Dup-of merges

**Restatements → `dup-of` (6 + 4 intra-station), no defender, no new id.** A
candidate restating a live roster entry merges into that id; its anchors and evidence are appended to
the prior entry by the calibration ticket rather than recorded twice. Five candidates are per-manifest
instances of CD-007 (the inline-vs-sibling homing finding, disposition `sharpen`, advances-0b); one
restates the OD-009 gating refinement with new instances. Four cross-cell pairs of the test-bloat lens
(handed over by the attacker gate's overlap table) are one finding each and fold onto the 5a survivor,
which takes the union of both anchor sets and both fix-shapes into its defender input.

| candidateRef | merged into | source | note |
|---|---|---|---|
| 5a-test-bloat-00 | CD-007 | wave-dispositions.json (Wave 6) | per-manifest instance of the inline-vs-sibling homing finding (setup/, policy/, stochastic/): three test homes in setup/mod.rs + inverted extraction |
| 5b-test-bloat-00 | CD-007 | wave-dispositions.json (Wave 6) | per-file instance: lp/builder/layout.rs declares `mod tests;` and an inline block back to back |
| 5b-test-bloat-01 | CD-007 | wave-dispositions.json (Wave 6) | the CD-007 claim itself re-measured over lp/builder + lp/indexer (largest four inline, smallest extracted) |
| 5c-test-bloat-04 | CD-007 | wave-dispositions.json (Wave 6) | per-manifest instance over training/, cut/, workspace/, solve/ (three inversions + the cfg-gated per-backend third convention) |
| 5d-test-bloat-05 | CD-007 | wave-dispositions.json (Wave 6) | per-manifest instance over simulation/ + production/ (two-homes file, four inline modules above threshold) |
| 5c-over-engineering-00 | OD-009 | wave-dispositions.json (Wave 7, OD) | new instances of the Symmetry-or-test-retention class with the same `#[cfg(test)]`-gating refinement OD-009 already records (StageWorkerStatsBuffer::get/index, n_workers/n_slots, ci_95_half_width) |
| 5d-test-bloat-00 | 5a-test-bloat-05 | attacker-log.md § Cross-cell overlaps | same clone family (state_layout_for / study_dims(_for) / all_enabled_cut_state_layouts hand-copied into integration binaries under a false doc rationale); survivor carries partIRef I.3-8 and the union of anchors |
| 5d-test-bloat-02 | 5a-test-bloat-03 | attacker-log.md § Cross-cell overlaps | same four right_boundary_* binaries, same md5-identical fixture prelude; survivor takes the union of anchors (incl. leaf_positions / fixture_priced_date adapters) |
| 5d-test-bloat-03 | 5a-test-bloat-04 | attacker-log.md § Cross-cell overlaps | same anticipated_core.rs fixture re-declarations (build_config ×14, build_system ×13, hydro defaults ×6); union of anchors |
| 5c-test-bloat-05 | 5a-test-bloat-02 | attacker-log.md § Cross-cell overlaps | same root: no neutral HydroPenalties constructor on the shared surface, sixteen-field literals everywhere; union of anchors (adds lower_bound.rs + forward_sampler_no_alloc.rs) |

**Sharpenings → defended, merged into the prior id at calibration, no new id (6).** The attacker
prompt allowed live roster rows to be SHARPENED (never re-raised) and marked such candidates
`reRaiseOf <id>`. A sharpening carries a new, checkable claim about the prior finding (a symbol-set
split, a mechanism the register's proposed fix does not match at the pin, a second copied block), so it
is defended like any other candidate; its verdict travels with `priorId` + `priorRelation: sharpens`
and the calibration ticket appends the verified residue to the prior entry instead of allocating a
number. Ticket wording covers restatements (`dup-of`); this is the ingest's reading for sharpenings,
recorded as a policy so the calibration ticket applies one rule.

| candidateRef | sharpens | prior disposition (E05-3) | candidate title |
|---|---|---|---|
| 5a-architecture-00 | CD-005 | sharpen | The run-parameter half of StudySetup uses two incompatible conventions (named projections vs raw config types)… |
| 5a-architecture-01 | CD-004 | sharpen | The MPI wire config is not a superset-projection of StudyParams: one field is computed on params then dropped … |
| 5b-architecture-01 | CD-074 | present-at-pin (register entry, not a Wave 4/6/7 row) | Pin-round-trip exactness for the four state families is handled by three unrelated per-family mechanisms in th… |
| 5c-architecture-01 | CD-015 | sharpen | CD-015 sharpened: by_scenario's risk aggregation HAS moved out into process_by_scenario_backward, leaving exac… |
| 5c-architecture-02 | CD-016 | keep | CD-016 sharpened: the per-worker backward_accum bookkeeping is duplicated at more than the pre-allocation the … |
| 5c-architecture-04 | CD-028 | keep | CD-028 sharpened: the shared claim-and-scatter owner already exists and its module doc declares a boundary nar… |

**Related, kept distinct (defended; the relation travels on the verdict).**

| candidateRef | related prior id(s) | adjudication |
|---|---|---|
| 5a-architecture-02 | CD-004, CD-024-successor | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-architecture-00 | CD-022 | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-performance-00 | CD-018, CD-012 | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-performance-02 | CD-012 | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-performance-04 | CD-023 | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-performance-05 | CD-018, CD-023 | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-performance-06 | CD-016 | the gate's `dupOf CD-016` rests on two shared fn symbols; the CLAIM differs — CD-016 is duplicated backward_accum PRE-ALLOCATION code across the two drivers, this is a per-node owned staged-cut Vec allocated at process_stage_backward's return boundary (a perf mechanism) — kept as a distinct candidate related to CD-016 |
| 5a-over-engineering-00 | OD-009 | the gate's `reRaiseOf OD-009` comes from the census wording; OD-009 is the Symmetry-or-test-retention gating refinement, this candidate is missing `// Rationale:` on Load-bearing cast allows — distinct |
| 5c-over-engineering-01 | OD-009 | the mirror (docs/design/reserved-seams-and-deferred-debt.md:933, deferred-debt entry `Nested risk-adjusted upper bound is an override, not an estimator arm`) names `RankDistribution::actual_per_rank` as the DECLARED OWNER of the rank-partition counts/displs arithmetic and records a second copy in the nested-UB recursion and one in cut_sync.rs as duplicate-owner residue; the candidate proposes deleting the declared owner in favour of `cobre_comm::per_rank_counts` — not a reserved seam (passes the screen), but the defender must weigh the fix against that owner declaration |
| 5b-test-bloat-02 | CD-007 | shared symbol(s) only, distinct claim (gate `relatedTo`) |
| 5c-test-bloat-00 | Superseded cut-sync public methods | names `sync_cuts` only through the two test fns that assert its count-mismatch rejection; the candidate is test-binary duplication, not the superseded public methods the E5 handoff merged — distinct; if the methods retire at the next public-API break the surviving assertion goes with them |

**Cut-sync handoff (solver-comm `handoffs.json` E5).** No candidate targets `CutSyncBuffers::sync_cuts`,
`pack_local_records` or `sync_packed_records` as superseded public API; `5c-test-bloat-00` names
`sync_cuts` only through the two test fns asserting its count-mismatch rejection and is adjudicated
distinct above (test-binary duplication). No new id is allocated on that route (`dupOf` to the mirror
entry `Superseded cut-sync public methods` = 0 here; the two E4 records already own it).

## Re-raise rejections

None (`reRaiseRejected = 0`). Every candidate's title, fix-shape and anchor set was screened against the six closed items:

| closed item | closed by | candidates matching the item's symbols / id |
|---|---|---|
| CD-001 | closed by `b051c410` — `build_stochastic_context_for_study` (setup/stochastic_pipeline.rs) is the single owner | none |
| CD-003 (Construction hop) | closed by `4075c4e8` — `ConstructionConfig` / `into_construction_config` deleted | none |
| CD-006 | resolved in-tree — `NodeGraph` method conversion (register L1185) | none |
| CD-008 | RETRACTED (WONTFIX, deliberate-by-design) | none |
| PD-001 | REFUTED 2026-08-18 (simulation/enumerated.rs perf claim; not debt) | none |
| PD-004 | DEFERRED pending a profile — existence check + perf-epic queue only (`run_enumerated_backward`) | none |

No candidate's title, fix-shape or anchor set matches a closed item's id or symbols (the
`ConstructionConfig` deletion that closed the CD-003 hop is referenced only by the I.3-7 handoff's
`changedSinceV012` note, not by any candidate). File-scoped anchor overlaps with retired or mirror-cited files were already
justified by the attacker gate (`retiredOverlaps`, `NOT a re-raise`): `5c-architecture-00` shares
`backward_pass_state.rs` with the deferred PD-004 but claims the DCS binding-metadata channel, not the
enumerated-backward cost; `5a-performance-01/02` and `5a-over-engineering-00` share `node_graph.rs`
with the mirror's module-map / graph-shape-predicate sections; `5c-performance-00/02/04/05` and
`5c-architecture-00/04` share files with mirror follow-up lists. All distinct; none rejected.

## Contract dismissals

None (`contractDismissed = 0`). The six trap patterns from the ticket were run over every candidate's
title + fixShape against the 52 headings of `.claude/rules/sddp.md`; 10 hits, each adjudicated:

| candidateRef | contract matched | adjudication |
|---|---|---|
| 5b-architecture-00 | State pinning uses column bounds, not equality rows | cites the section as the reason the re-open stays a column-bound write; no equality row proposed — preserved |
| 5b-architecture-01 | State pinning uses column bounds, not equality rows | cites the section: the pin stays a `set_col_bounds` write owned by `fill_col_state_patches` — preserved |
| 5b-architecture-02 | State pinning uses column bounds, not equality rows | cites the section: new typed resolvers route through `state_to_lp_incoming_column`, no equality row — preserved (the section REQUIRES the resolver) |
| 5b-architecture-03 | State pinning uses column bounds, not equality rows | cites the section as the invariant to state once; template bounds inert because pins are column bounds every solve — preserved |
| 5b-performance-00 | State pinning uses column bounds, not equality rows | cites the section as untouched: same incoming columns, same column bounds, only a cache of the resolver — preserved |
| 5b-performance-01 | State pinning uses column bounds, not equality rows | cites three sections and keeps all three (pin stays a column bound at the same columns, NCS patch stays per opening, opening order untouched); what changes is the per-opening RE-SUBMISSION count — a defender question (`contractWatch`), not a weakening |
| 5c-performance-00 | Cut pool is append-only; basis matches by slot identity | cites the section as governing: no slot renumbered or compacted, only the scoring scratch reused — preserved |
| 5c-performance-04 | Cut pool is append-only; basis matches by slot identity | cites the section as preserved exactly (slot→position map unchanged, no compaction; generation stamp replaces the sentinel clear) — preserved |
| 5d-performance-06 | State pinning uses column bounds, not equality rows | regex false positive: `pin.*equality` matched `…keeping retain … predicate stays exact bit equality` (hull facet dedup); no state pinning involved |
| 5c-test-bloat-05 | Lower-bound evaluation must patch NCS | regex false positive: `drop.*availability` matched `…keep the FPHA average-storage and NCS availability contracts … intact`; the candidate keeps them |

**Rule.** A contract is named as the reason the current shape is CORRECT, never as an obstacle. Every
hit above cites its section as PRESERVED (the attacker prompt's contract-first rule required it), or is
a regex false positive; nothing proposes compacting or reordering the append-only cut pool, collapsing
the mirrored outflow rows (CD-034 stays a keep with no candidate touching it), replacing column-bound
pinning with equality rows, folding `apply_nested_cvar_ub` into an end-of-horizon estimator, skipping
the NCS patch, or unifying the two basis-reconstruct entry points. Three candidates carry a
`contractWatch` list into their defender input because their mechanism sits next to a contract
(`5b-performance-01` per-opening re-submission of the pin; `5c-performance-05` demoting the BASIC
recount to a debug assertion; `5b-architecture-00` the anticipated ring axis) — the defender must
either cite the heading verbatim in `contractCited` if it dismisses on it, or state the contract
holds under the fix. No residue had to be re-scoped because no candidate was dismissed.

| candidateRef | fix-shape as filed | contractCited (.claude/rules/sddp.md) | re-scoped residue |
|---|---|---|---|
| — | — | — | — |

## Cross-cell overlaps (attacker-log.md table) — decisions

| pair | decision | reason |
|---|---|---|
| 5a-test-bloat-05 ← 5d-test-bloat-00 | merged (one finding) | same clone family (state_layout_for / study_dims(_for) / all_enabled_cut_state_layouts hand-copied into integration binaries under a false doc rationale); survivor carries partIRef I.3-8 and the union of anchors |
| 5a-test-bloat-03 ← 5d-test-bloat-02 | merged (one finding) | same four right_boundary_* binaries, same md5-identical fixture prelude; survivor takes the union of anchors (incl. leaf_positions / fixture_priced_date adapters) |
| 5a-test-bloat-04 ← 5d-test-bloat-03 | merged (one finding) | same anticipated_core.rs fixture re-declarations (build_config ×14, build_system ×13, hydro defaults ×6); union of anchors |
| 5a-test-bloat-02 ← 5c-test-bloat-05 | merged (one finding) | same root: no neutral HydroPenalties constructor on the shared surface, sixteen-field literals everywhere; union of anchors (adds lower_bound.rs + forward_sampler_no_alloc.rs) |
| 5a-test-bloat-00 / 5d-test-bloat-05 | kept distinct | both dup-of CD-007 (per-manifest instances; folded there, not into each other) |
| 5a-test-bloat-02 / 5d-test-bloat-03 | kept distinct | distinct: crate-wide HydroPenalties literal census vs one binary's fixture re-declaration (the latter merged into 5a-test-bloat-04) |
| 5a-test-bloat-04 / 5c-test-bloat-05 | kept distinct | distinct: anticipated_core.rs re-declarations vs the missing neutral-penalties constructor (the latter merged into 5a-test-bloat-02) |
| 5a-test-bloat-06 / 5d-test-bloat-05 | kept distinct | distinct: a test oracle homed in a production file vs the homing-threshold split (dup-of CD-007) |
| 5b-test-bloat-07 / 5c-test-bloat-01 | kept distinct | distinct binaries (lp_builder.rs case_dir vs cut_basis.rs helper trio); both defended, related |
| 5c-test-bloat-02 / 5d-test-bloat-04 | kept distinct | distinct halves of the E08-2 needs-human (src-side StubComm re-declaration vs tests/common shape); both defended, related, cross-station note |
| 5c-test-bloat-05 / 5d-test-bloat-03 | kept distinct | see the two merges above |
| 5a-over-engineering-00 / 5d-over-engineering-00 | kept distinct | same premise dispute over two manifests (not in the gate table — different lens rows): kept as two defended candidates so each cell's census is verified independently; calibration folds them into one entry with both anchor sets if both survive |


## Defender pass

65 survivors (75 received − 10 dup-of), one read-only Opus `adversarial-defender`
each, dispatched in parallel batches of four (a slot refilled as soon as its defender returned; the
final two candidates were dispatched together, so one fifth worker ran for a few minutes at the tail) with
`defender-prompt.md` read verbatim plus the single candidate input object
(`/tmp/sddp-defenders/in/<candidateRef>.json`: the candidate, its prior context, merged twins,
contract watch-list and the attacker's needs-human items), envelopes handed back through a scratch
file outside the repository (`WRITTEN <bytes> <path>`). Every envelope was checked with
`tools/validate-envelope.py --role defender --station sddp` plus the station clauses (exactly one
verdict keyed by the candidateRef; `survivingClaim` strictly narrower than the title — not equal after
normalisation and token-Jaccard < 0.85; `sanctionedBy` ∈ the closed set; `contractCited` a verbatim
`.claude/rules/sddp.md` heading — a copied `##`/`###` markdown marker is stripped, the heading TEXT
must match; `byteNeutral` and `alignmentHint` in vocabulary; `partIRef` copied;
no diff or code block; no timing number on a performance verdict).

Re-dispatched once: none. Verdict null (needs-human after a second failure): 0.

Result: **49 confirmed, 16 dismissed**; 33 candidates carry a `_needsHuman` note for the owner gate.

| lens | confirmed | dismissed | verdict null |
|---|---|---|---|
| architecture | 12 | 4 | 0 |
| over-engineering | 7 | 3 | 0 |
| performance | 14 | 7 | 0 |
| test-bloat | 16 | 2 | 0 |

**Dismissed:**

- `5b-architecture-01` (architecture) — Pin-round-trip exactness for the four state families is handled by three unrelated per-family mechanisms in three modules, and the commitmen… — `contractCited`: Delivered commitments reconcile against solver drift; exactness is unreachable. The delta this candidate adds over CD-074 is that the commitment family's two exactness mechanisms are mutually unaware and their coupling unstated; both halves are false at the pin. The reconciler's module doc (crates/cobre-sddp/src/lp/builder/commitment_reconcile.rs:8-12) names `apply_commitment_h…

- `5b-architecture-03` (architecture) — transit_buckets_in is the only incoming state family whose template bounds are never written, inheriting them from buffer initialization whi… — `contractCited`: State pinning uses column bounds, not equality rows. The factual half holds (nothing in columns.rs writes state.transit_buckets_in) but the reading does not, on three counts. (1) The asymmetry is a stated rule, not an unexplained omission. The buffer default is (0.0, +INF) at columns.rs:31-32, so a family is written exactly when its open domain differ…

- `5b-architecture-04` (architecture) — A production gate predicate takes an unused _state: &StateSpace parameter justified by uniformity with a sibling that is #[cfg(test)]-gated. The candidate's load-bearing premise fails at the pin: in a release build the gate family has two members, not one. anticipated_gate.rs:123-142 declares anticipated_resolution_for with NO cfg gate, re-exported for production use at indexer/mod.rs:109-111, taking state as its first parameter and read…

- `5c-architecture-03` (architecture) — The replicated backward driver never enters the outcome_aggregation module and open-codes a second owner of the Benders intercept derivation… — `contractCited`: The cut intercept dots the trial state through the projection, never positionally. The named fix target does not do what the candidate says it does. solve_replicated_outcome_slice (replicated.rs:83-166) never touches a BackwardOutcome: at :137-138 and :160-161 it pushes the raw objective followed by state_duals[..n_state] into the flat out Vec<f64> whose per-outcome width is outco…

- `5b-performance-02` (performance) — Every generic-constraint term allocates and immediately drops a one- or two-element heap vector during stage-template construction. The mechanism is real but it is setup-time only, and the allocation policy on this exact path is deliberate and documented in present tense. Frequency: resolve_variable_ref has one production caller, fill_generic_constraint_entries at lp/builder/entries.rs:1363; that function is reached only from lp…

- `5c-performance-01` (performance) — apply_column_rule walks the row-major gemm output panel column-strided twice for every trial column. The mechanism as filed ('redundant strided pass', 'full-panel stride') does not hold at the pin, on five grounds read from the code. (1) Nothing is redundant and the interchange removes no load. The per-column nest at crates/cobre-sddp/src/cut/cut_selection.rs:392-403 issues one max-loop load per pa…

- `5c-performance-05` (performance) — enforce_basic_count_invariant recounts BASIC statuses with two full filter passes over the status vectors the reconstruction just wrote. The mechanism rests on the premise that the reconstruction already knew the BASIC counts; at the pin it does not. reconstruct_col_statuses (basis_reconstruct.rs:209-211) is clear + extend_from_slice + resize, and reconstruct_template_row_statuses (:224-232) is the same pair; BasisStatus is a fieldle…

- `5d-performance-03` (performance) — FPHA fitting rebuilds the shared (V, Q) grid and re-walks the same nodes at four pipeline stages instead of building it once per plant. Setup scope is verified at the pin, and it decides this candidate. The only non-test entry into the fitting pipeline is `fit_fpha_planes` (production/fpha_fitting/mod.rs:137), reached through `fit_planes_for_hydro` (production/hydro_models/production.rs:414) and `fit_computed_planes_per_stage` (:508…

- `5d-performance-04` (performance) — fit_gamma_s_for_planes rebuilds the grid and recomputes the whole min-over-planes envelope once per plane. The mechanism the candidate names is really present: at secant.rs:74 every call to representative_operating_point re-allocates both axis Vecs through build_grid, and at secant.rs:80-88 the min-over-planes fold and pf.evaluate depend only on the node and the immutable snapshot, so both are recomputed…

- `5d-performance-06` (performance) — Hull facet dedup uses a linear Vec::contains membership scan whose cost grows with the number of planes already kept. The linear membership scan is real as read, but it sits entirely in study setup and its operand set is bounded to a handful of keys, so the named mechanism has no measurable region to act on. Reachability: fit_hull_planes has exactly one non-test caller, production/fpha_fitting/mod.rs:165 inside fit…

- `5d-performance-07` (performance) — prepare_hydro_models_from_artifacts groups and per-hydro sorts the same geometry table three times in one call. This is a study-setup path, not a solve path. The only production call sites of prepare_hydro_models_from_artifacts are crates/cobre-cli/src/commands/run/setup.rs:129 and crates/cobre-python/src/run.rs:981, plus the two validate-only entries crates/cobre-cli/src/commands/validate.rs:419 and crates/c…

- `5a-over-engineering-00` (over-engineering) — Four production #[allow(clippy::...)] sites in the 5a manifest carry no written rationale, so the census's sanctioned-by-construction claim … — `contractCited`: Backward opening order is warm-start-only. Dismissed on the code plus the project's own pinned scope for rationale-on-suppression. comments.md directive D4 (.claude/rules/comments.md:365-371) mandates a rationale for a CLOSED list of refactor-decision lints (too_many_arguments, too_many_lines, type_complexity, dead_code, unused_*) plus borro…

- `5a-over-engineering-02` (over-engineering) — train_inner hand-mirrors the 19-field TrainingContext and the StageContext literal that accessors.rs already constructs, and the training_ct…. The filed fix does not compile, and that is why the literals exist. accessors.rs:233 and :274 declare stage_ctx(&self) -> StageContext and training_ctx(&self) -> TrainingContext with elided output lifetimes, so each returned value holds a borrow of all of *self for as long as it lives. train_inner t…

- `5a-over-engineering-03` (over-engineering) — setup/scenario_library_set.rs is a 46-line public module of two impl-less structs whose name collides with the sibling setup/scenario_librar…. Both load-bearing premises fail against the code and the pinned rules at the baseline. (1) The impl-less-public-module shape is the documented layout, not an accident: .claude/architecture-rules.md section 'StudySetup Sub-Structs' (line 80 onward) tabulates ScenarioLibraries and PhaseLibraries by na…

- `5b-test-bloat-05` (test-bloat) — template_integration/generic_constraints.rs silently shadows the parent's `one_hydro_system` builder with a same-named local of a different …. The shadow is real but nothing about it is silent, which is the load-bearing word in the title. The parent declaration at tests/template_integration.rs:251 takes two usize parameters (n_stages, lag_order); the local declaration at tests/template_integration/generic_constraints.rs:1433 takes four (n_…

- `5b-test-bloat-06` (test-bloat) — Three separate Stage fixture surfaces with divergent defaults serve one crate, and the builder-module one is `#[cfg(test)]`-only so no integ…. Dismissed on four independent grounds, all read at the pin. (1) The count is wrong: crate::test_support exposes no Stage constructor at all. geometry_stage is a private fn at crates/cobre-sddp/src/test_support.rs:409 with exactly one caller, :554, which feeds StageLayout::new(...).geometry(BlockMode…

**Sharpenings (merge into the prior id at calibration, no new number):**

| candidateRef | priorId | verdict | survivingClaim |
|---|---|---|---|
| 5a-architecture-00 | CD-005 | confirmed | Delta over CD-005: of the seven setter-wall members, `StudySetup::set_budget` (setup/accessors.rs:62-65) is the only one with zero call sites anywhere in the repository at the pin AND the only one that is `pub` and ungat |
| 5a-architecture-01 | CD-004 | confirmed | Verified delta over CD-004: the two projections are NOT independent Config readers -- BroadcastConfig::from_config calls StudyParams::from_config at broadcast.rs:144 and every wire field except training_enabled, policy_m |
| 5b-architecture-01 | CD-074 | dismissed | — |
| 5c-architecture-01 | CD-015 | confirmed | Delta over CD-015: the live entry mislabels the inline residue. By-scenario's risk aggregation is NOT in compute_one_backward_node — aggregate_cut_into runs inside process_by_scenario_backward (by_scenario.rs:525) and th |
| 5c-architecture-02 | CD-016 | confirmed | Beyond the pre-allocation duplication CD-016 records, the same two backward drivers carry a second copied block -- the slot_increments into metadata_sync_contribution fold at by_scenario.rs:559-566 and again at by_node.r |
| 5c-architecture-04 | CD-028 | confirmed | Delta over CD-028: the shared owner claim_scatter.rs already exists but its module doc's consumer list (:1-4) names only by_node and the enumerated forward while simulation/enumerated.rs:26 is a third importer of both pr |

**Needs-human (33), for the owner gate:**

- `5a-architecture-00` — Owner call: `set_budget` is enumerated in the declared accessor surface at `.claude/architecture-rules.md:117`, so the disposition of a caller-free ungated mutator is either remove it and edit that rules line, or keep it as a deliberate public-library knob and route it through the same validation `StudyParams::from_config` applies - the station cannot pick without the owner.

- `5a-architecture-02` — Whether the fail-loud leg changes the public surface of ResolvedParameters::get (a fallible signature or a panic) or is instead enforced as a construction-time check at the LP-build site, leaving get infallible: get is pub and must_use, so the choice is an owner call on public API shape, not a correctness question.

- `5a-architecture-03` — Owner must choose between removing `CutManagementConfig::warm_start_cuts` (a breaking change to a re-exported public type) and giving it a real reader plus a reserved-seams entry; the verdict establishes the field is inert and its doc line false, not which of the two lands.

- `5b-architecture-02` — Alignment-epic owner call: whether conflicts trigger 4 (one-consumer abstraction) reaches a purpose-named single-consumer StateSpace accessor that completes an existing five-member family, or is bounded to seams, traits and crates as its text states.

- `5c-architecture-00` — Carried over from the attacker and still decisive: is enumerated traversal combined with dynamic cut selection a SUPPORTED configuration? Road (a) rejects the pairing beside the existing enumerated preconditions in `setup/mod.rs`; road (b) wires the binding contribution plus a per-stage metadata reduction into the enumerated driver. Both are byte-neutral on today's goldens, so only owner intent de

- `5c-architecture-00` — Scope and severity call: `CutPool::enforce_budget`'s `(last_active_iter, active_count)` eviction key is a second production reader that degrades the same way under enumerated traversal, and unlike the DCS seed it changes which cuts are deactivated and therefore the bound. The owner must decide whether that reader belongs inside this item's scope and whether it lifts the severity above B; I did not

- `5c-architecture-04` — Owner call: whether the confirmed residue should produce any code change at all, or only the module-doc consumer-list correction -- the identical fragment is a four-line composition of stdlib collect plus an integer sum, and hoisting it behind a generic-over-E helper trades two transparent ? sites for one opaque one.

- `5d-architecture-00` — Whether the public half of the alias block is also retired is an owner call: lib.rs:23-26 declares the pub mod namespaces non-semver-stable and the nested paths already resolve, but removing pub use policy::orchestration would touch cobre-cli, cobre-python, the integration corpus and the literal match in scripts/ci/check_python_parity.py.

- `5d-architecture-01` — Destination is an owner call and decides the hint: re-home inside the engine beside simulation/types.rs (a pure local cleanup, hint degrades to neutral) or leave it until Phase 0a's shared output orchestration in cobre-io absorbs the engine-to-io projection wholesale (target-layering-brief.md section 2, 0a bullet), which keeps advances-0a but makes the intra-engine move wasted work.

- `5a-performance-00` — Owner call: whether the goldens-moving delta reformulation of the 2-opt accept test is worth a parity re-baseline at all, given that the confirmed free win is only the symmetric matrix fill; if not, the perf sweep should carry the symmetric-fill item alone.

- `5a-performance-01` — Placement is an owner call the architecture rules do not disambiguate: a derived study-invariant consumed only by the backward drivers fits decision-tree rule 2 (a `NodeGraph` field beside `n_pools`/`pool_stage`, the candidate's shape) and rule 5 (a `BackwardPassState` scratch field) equally; the training owner should also decide whether this lands together with the register's open one-owner resid

- `5b-performance-01` — Owner call: whether removing a bit-identical repeat submission of already-in-force column bounds counts as a warm-start-chain change that must clear the parity goldens, the rank-invariance harness and the opening_order_determinism gate before the hoist lands, given that the CLP backend's set_row_bounds rustdoc asserts factorization/basis preservation across a bound patch but nothing in this repo p

- `5c-performance-00` — Owner call: whether the inaccurate rustdoc sentence at training/session/mod.rs:1026-1027 ('no heap allocation when the cut pools have not grown') gets its own doc-accuracy item — it is false because of run_cut_management's own per_stage / deactivations / record_by_pool allocations, independent of this candidate's mechanism claim, so it is stripped here rather than fixed by it.

- `5c-performance-03` — Owner call, needed only to scope the LARGER change and not the claim confirmed here: is the level-wide over-inclusion a required correctness margin, or is narrowing each node's archive to its own routed states licensed as a golden-moving change? Only that narrowing reduces the per-node value sweep; the shared-block fix confirmed above does not depend on the answer.

- `5d-performance-00` — Owner picks where the per-stage cache lives (shared `StageContext` vs per-worker simulation scratch); the span-only residue shrinks the memory side of that trade from a full row vector to the load-balance rows.

- `5d-performance-03` — Process call: the 4t sweep skips setup-time-only items, so the owner must say whether a setup-budget track exists to record the two conceded residues (per-plane grid rebuild in the secant; duplicate deviation walk under the opt-in flag) or whether they are dropped.

- `5d-performance-05` — Rule-11 scope call the owner may want to settle once for all similar candidates: does the setup-time per-plant carve-out cover a per-plant recomputation whose cost is sized by a whole shared table (superlinear in plant count), or only one sized by the plant's own data? I read it as the latter and narrowed accordingly.

- `5a-over-engineering-00` — Owner call (prose, not code): reword the allow-census Load-bearing class at docs/design/reserved-seams-and-deferred-debt.md:1282-1288 and the scripts/README.md:36 one-liner so neither reads as an absolute every-site rationale claim, since D4 and the E4 gate never require one of a cast_* allow.

- `5b-over-engineering-00` — Scheduling only: whether the public-API removal of Col and Row ships as its own semver-major break or waits for the 0b cobre-model carve-out of lp/, as the CD-019 deferral precedent suggests — the defect itself is settled.

- `5b-over-engineering-01` — Owner call carried over from the attacker: deleting FphaRowRange is a public-API removal (reachable as cobre_sddp::indexer::FphaRowRange via lib.rs:69) with no in-repo consumer; decide whether it lands now as ordinary cleanup or is batched with the Phase 0b cobre-model carve-out that relocates lp/indexer anyway, per the CD-019 deferral precedent.

- `5b-over-engineering-01` — Scope call: the fix as filed leaves the same per-plant framing live in BlockGrid::advance_fpha_base's rustdoc (block_grid.rs:106-107), so the reader hazard is only half retired; decide whether that sibling doc correction joins this item or gets its own.

- `5d-over-engineering-00` — Doc-owner call: the mirror sentence at reserved-seams-and-deferred-debt.md:1278-1284 asserts every numeric-cast allow on production code carries a `// Rationale:`, which comments.md D4 and the E4 gate's IN_SCOPE_LINTS both contradict; decide whether that sentence is corrected down to the D4 scope or D4 is widened to cover cast_* lints.

- `5d-over-engineering-00` — Routing call: the surviving residue is a missing input-boundary validation owned by cobre-io, not an over-engineering item in cobre-sddp; decide whether it stays on this station's over-engineering ledger or is re-filed against the owning crate.

- `5d-over-engineering-01` — Scope call: whether crate-internal single-file items should be narrowed to module-private at all, or whether pub(crate) is the ratified uniform internal marker for this workspace; if narrowing is wanted it should be one crate-wide pass over the 25 sites with a per-site forcing check, not this five-anchor slice.

- `5a-test-bloat-02` — Whether the corpus should keep at least one literal that still spells all sixteen penalty fields as a canary: spreading `..HydroPenalties::uniform(v)` everywhere means a newly added penalty field silently defaults in every fixture instead of breaking the build, and while `uniform`'s rustdoc endorses the spread and builders.rs places its strictness on the make_<entity> literal, no artifact rules on

- `5a-test-bloat-04` — Scope call for the owner: the default_hydro_penalties body is one md5 across anticipated_core.rs and anticipated_scenarios.rs, and scenarios build_config at 2954 equals core build_config at 3831, so the owner must decide whether the hoist stays per-binary at file scope or lifts the shared class into tests/common (the test-corpus station owns the cross-binary convention).

- `5a-test-bloat-05` — Owner call outside this candidate: invariance-shuffle.yml builds cargo nextest -p cobre-sddp --test parity without the test-support feature, while parity.rs reaches cobre_sddp::test_support ungated at :894 and :1303, so either that workflow cannot build at the pin or the clones never protected a working default-feature configuration; deciding which fixes whether the consolidation must also gate th

- `5a-test-bloat-06` — Scope call for this item: whether relocating the oracle should also re-word the three production doc comments that use `shift_lag_state` as the canonical lag-remap vocabulary (the bracketed reference at `noise.rs:211` plus the code spans at `lp/indexer/state_space.rs:443` and `lp/indexer/cut_state_projection.rs:269`), or leave them naming a test-module-private symbol.

- `5b-test-bloat-02` — Target shape for the split is a convention call CD-007 owns: one extracted tests.rs matching commitment_reconcile/layout/template, or N subject-named files matching columns.rs's 12 inline modules. The unratified section 5.1 threshold cannot settle it.

- `5b-test-bloat-03` — Whether this binary's classical arm should be parameterized in place or instead folded onto the existing two_hydro_par_system helper when template_integration and par_a_lag12_lp_coefficient are grouped under the testing-architecture section 5.1 layout - a test-corpus station call, since that homing layout is still a proposal at the pin.

- `5b-test-bloat-04` — Scope call: whether the ticket is the family collapse as filed or the higher-leverage extension of the O(1) Spec layer to InflowModel, LoadModel and the four bounds/penalties sizing structs, which owns the 20-copy fan-out the collapse leaves at fifteen.

- `5c-test-bloat-00` — Sequencing call for the cut-pool/training owner: delete the redundant binary now, or let it retire together with the superseded sync_cuts family at the next licensed public-API break, since all three copies of this assertion die with the method.

- `5c-test-bloat-02` — Destination owner call for the shared StubComm/Rank0Of2 pair: testing-architecture.md section 5.2 assigns them to cobre-comm, which has no test-support feature at the pin, while section 6 phase 3 collapses tests/common into cobre-sddp's test-support surface; and Rank0Of2's load-bearing forward_passes == 1 caveat names cobre-sddp's RankDistribution, which the infrastructure-genericity hard rule wou

- `5c-test-bloat-03` — Owner call on the shape of the inert-only consolidation: a cfg-gated declarative macro in test_support.rs versus the candidate's field-configured shared struct, since a single struct cannot express the seven-to-one get_basis split and the two strict sites without adding modes no test reads.

- `5d-test-bloat-01` — Owner call on whether the audited k_max partition should survive in any executable form now that the shift-to-hold switchover has landed and there are zero k_max >= 2 tier-1 goldens: retire the checklist to prose, or re-home it as a script-backed gate that resolves each name against a nextest listing, since Rust cannot reflect over another integration binary's tests (the probe's own module doc, li

- `5d-test-bloat-04` — Owner call: whether testing-architecture.md section 5.8 (keep StubComm/Rank0Of2) plus the section 5.2 move into cobre-comm's test-support surface is the final disposition for the harness pair, or whether the test-corpus station's E08-2 needs-human item may still merge them - my dismissal of the merge half rests on section 5.8 being ratified.

- `5d-test-bloat-04` — Owner call: whether conformance.rs:40 LocalComm's no-copy collectives (Ok(()) without writing recv, unlike common::StubComm) are deliberate; I read the divergence as semantics-by-design, but no rustdoc states why.

`contractCited` present on 4 verdict(s): `5b-architecture-01` → Delivered commitments reconcile against solver drift; exactness is unreachable, `5b-architecture-03` → State pinning uses column bounds, not equality rows, `5c-architecture-03` → The cut intercept dots the trial state through the projection, never positionally, `5a-over-engineering-00` → Backward opening order is warm-start-only.

Alignment over the defended set: advances-0a: 3, advances-0b: 4, advances-1: 1, neutral: 57.

## Per-candidate roster

Every one of the 75 candidateRefs, exactly once (mirrors `verdicts.json`; `state` is the station
test vocabulary — accepted / rejected-anchor / rejected-reserved-seam / rejected-defender / merged —
derived from disposition + verdict; a null verdict reads `needs-human`).

| candidateRef | disposition | verdict | state | partIRef | alignment |
|---|---|---|---|---|---|
| 5a-architecture-00 | defended | confirmed (sharpens CD-005) | accepted |  | neutral |
| 5a-architecture-01 | defended | confirmed (sharpens CD-004) | accepted | I.3-7 | advances-0a |
| 5a-architecture-02 | defended | confirmed | accepted | I.3-7 | advances-0a |
| 5a-architecture-03 | defended | confirmed | accepted |  | neutral |
| 5a-over-engineering-00 | defended | dismissed | rejected-defender |  | neutral |
| 5a-over-engineering-01 | defended | confirmed | accepted |  | neutral |
| 5a-over-engineering-02 | defended | dismissed | rejected-defender |  | neutral |
| 5a-over-engineering-03 | defended | dismissed | rejected-defender |  | neutral |
| 5a-performance-00 | defended | confirmed | accepted |  | neutral |
| 5a-performance-01 | defended | confirmed | accepted |  | neutral |
| 5a-performance-02 | defended | confirmed | accepted |  | neutral |
| 5a-test-bloat-00 | dup-of | dup-of → CD-007 | merged |  | neutral |
| 5a-test-bloat-01 | defended | confirmed | accepted |  | neutral |
| 5a-test-bloat-02 | defended | confirmed | accepted |  | neutral |
| 5a-test-bloat-03 | defended | confirmed | accepted |  | neutral |
| 5a-test-bloat-04 | defended | confirmed | accepted |  | neutral |
| 5a-test-bloat-05 | defended | confirmed | accepted | I.3-8 | advances-1 |
| 5a-test-bloat-06 | defended | confirmed | accepted |  | neutral |
| 5b-architecture-00 | defended | confirmed | accepted |  | neutral |
| 5b-architecture-01 | defended | dismissed | rejected-defender |  | neutral |
| 5b-architecture-02 | defended | confirmed | accepted |  | advances-0b |
| 5b-architecture-03 | defended | dismissed | rejected-defender |  | neutral |
| 5b-architecture-04 | defended | dismissed | rejected-defender |  | neutral |
| 5b-over-engineering-00 | defended | confirmed | accepted |  | advances-0b |
| 5b-over-engineering-01 | defended | confirmed | accepted |  | advances-0b |
| 5b-performance-00 | defended | confirmed | accepted |  | neutral |
| 5b-performance-01 | defended | confirmed | accepted |  | neutral |
| 5b-performance-02 | defended | dismissed | rejected-defender |  | neutral |
| 5b-test-bloat-00 | dup-of | dup-of → CD-007 | merged |  | advances-0b |
| 5b-test-bloat-01 | dup-of | dup-of → CD-007 | merged |  | advances-0b |
| 5b-test-bloat-02 | defended | confirmed | accepted |  | neutral |
| 5b-test-bloat-03 | defended | confirmed | accepted |  | neutral |
| 5b-test-bloat-04 | defended | confirmed | accepted |  | neutral |
| 5b-test-bloat-05 | defended | dismissed | rejected-defender |  | neutral |
| 5b-test-bloat-06 | defended | dismissed | rejected-defender |  | neutral |
| 5b-test-bloat-07 | defended | confirmed | accepted |  | neutral |
| 5c-architecture-00 | defended | confirmed | accepted |  | neutral |
| 5c-architecture-01 | defended | confirmed (sharpens CD-015) | accepted |  | neutral |
| 5c-architecture-02 | defended | confirmed (sharpens CD-016) | accepted |  | neutral |
| 5c-architecture-03 | defended | dismissed | rejected-defender |  | neutral |
| 5c-architecture-04 | defended | confirmed (sharpens CD-028) | accepted |  | neutral |
| 5c-over-engineering-00 | dup-of | dup-of → OD-009 | merged |  | neutral |
| 5c-over-engineering-01 | defended | confirmed | accepted |  | neutral |
| 5c-performance-00 | defended | confirmed | accepted |  | neutral |
| 5c-performance-01 | defended | dismissed | rejected-defender |  | neutral |
| 5c-performance-02 | defended | confirmed | accepted |  | neutral |
| 5c-performance-03 | defended | confirmed | accepted |  | neutral |
| 5c-performance-04 | defended | confirmed | accepted |  | neutral |
| 5c-performance-05 | defended | dismissed | rejected-defender |  | neutral |
| 5c-performance-06 | defended | confirmed | accepted |  | neutral |
| 5c-test-bloat-00 | defended | confirmed | accepted |  | neutral |
| 5c-test-bloat-01 | defended | confirmed | accepted |  | neutral |
| 5c-test-bloat-02 | defended | confirmed | accepted |  | neutral |
| 5c-test-bloat-03 | defended | confirmed | accepted |  | neutral |
| 5c-test-bloat-04 | dup-of | dup-of → CD-007 | merged |  | neutral |
| 5c-test-bloat-05 | dup-of | dup-of → 5a-test-bloat-02 | merged |  | neutral |
| 5d-architecture-00 | defended | confirmed | accepted |  | advances-0b |
| 5d-architecture-01 | defended | confirmed | accepted |  | advances-0a |
| 5d-over-engineering-00 | defended | confirmed | accepted |  | neutral |
| 5d-over-engineering-01 | defended | confirmed | accepted |  | neutral |
| 5d-over-engineering-02 | defended | confirmed | accepted |  | neutral |
| 5d-performance-00 | defended | confirmed | accepted |  | neutral |
| 5d-performance-01 | defended | confirmed | accepted |  | neutral |
| 5d-performance-02 | defended | confirmed | accepted |  | neutral |
| 5d-performance-03 | defended | dismissed | rejected-defender |  | neutral |
| 5d-performance-04 | defended | dismissed | rejected-defender |  | neutral |
| 5d-performance-05 | defended | confirmed | accepted |  | neutral |
| 5d-performance-06 | defended | dismissed | rejected-defender |  | neutral |
| 5d-performance-07 | defended | dismissed | rejected-defender |  | neutral |
| 5d-test-bloat-00 | dup-of | dup-of → 5a-test-bloat-05 | merged |  | neutral |
| 5d-test-bloat-01 | defended | confirmed | accepted |  | neutral |
| 5d-test-bloat-02 | dup-of | dup-of → 5a-test-bloat-03 | merged |  | neutral |
| 5d-test-bloat-03 | dup-of | dup-of → 5a-test-bloat-04 | merged |  | neutral |
| 5d-test-bloat-04 | defended | confirmed | accepted |  | neutral |
| 5d-test-bloat-05 | dup-of | dup-of → CD-007 | merged |  | advances-0b |

Counts (`verdicts.json.counts`): received 75 = anchorRejected 0 + sanctionedCleared
0 + contractDismissed 0 + dupOf 10 + reRaiseRejected 0 + handedOff
0 + defended 65; defended 65 = confirmed 49 + dismissed 16 + verdictNull
0; needsHuman 33 (defended candidates carrying at least one owner question).
