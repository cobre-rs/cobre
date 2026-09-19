# Next-step roadmap — 2026-09-17, amended 2026-09-19

Companion to `PRIORITIES.md` §6 (2026-09-17) and §8 (2026-09-19, the live sequence). That section ranks the unrun stations; this file is the
operating sequence the owner agreed to on 2026-09-17, including the user-reported bug
CD-074 (`BACKLOG.md`, section "USER-REPORTED BUG (2026-09-17)") and the SpecForge
reconciliation. Amend in place; do not fork a second copy.

## Constraints fixed by the owner

- **CD-074 cannot be reproduced locally.** The failing study is a large HPC-cluster run;
  the fix is driven by analysis of the mechanism plus synthetic regressions, not by a
  replay of the reporter's deck.
- **No standalone patch release.** CD-074 ships inside **v0.16.0** together with the other
  pending work; no v0.15.1.
- **The SpecForge quality-evaluation spec (46c31993) is stale.** Much of what its
  remaining tickets describe has since been done or rewritten on `develop`; it is
  reconciled against the live tree before any station resumes, not resumed as written.

## Sequence

1. **Commit the register update** (`BACKLOG.md` CD-074 entry, `PRIORITIES.md` step 0,
   this file) on `chore/quality-evaluation`. — DONE in the commit that adds this file.

2. **Reporter follow-up (no deck expected).** Ask only for the risk configuration
   (CVaR or risk-neutral) and, if available, the run summary JSON. That decides whether
   `LB 3.03e12 > UB 1.87e11` at iteration 37 is benign CVaR reporting or a second defect.
   Reproduction of the abort itself is not requested.

3. **Fix CD-074 as one plan off `develop`, released in v0.16.0.** DONE (`3356da2d`) —
   shipped as the state-canonicalization plan (projection onto the admissible box via the
   single read-back seam `assemble_outgoing_state`; runtime verdict retired; over-commitment
   moved to the cobre-io load-time validator `check_committed_value_bounds`).
   AMENDMENT: Ticket A's per-run drift tally (below) was REMOVED as overengineering after
   landing (owner decision, `7fbb3da2`) — sub-tolerance clamps are absorbed silently, with no
   run-summary tally, metadata, or CLI/Python surface. The projection, the load-time reject,
   and the byte-neutrality-for-in-bounds-solves property all stand.
   Do not ship a constant-only hotfix; projection onto the enforced delivery box is the
   redesign's core and is what unblocks users.
   - **Ticket A — solve-time projection.** Replace relax-the-bound with project-the-pin
     onto the intersection over blocks of the delivery stage's enforced per-block
     `[col_lower·scale, col_upper·scale]` for the thermal (both sides: zero floors,
     positive must-run floors, caps, dormant `[0, 0]` stages). Margin sized from the
     box/LP magnitude, never the commitment. Drift tallied (max abs / rel per run) into
     the run summary with one `tracing::warn!` past a diagnostic threshold; nothing on
     this path aborts. Rewrite (never delete) the `.claude/rules/sddp.md` contract
     "Delivered commitments reconcile against solver drift". Regressions: a hair below a
     `0` floor (the report), below a positive must-run floor, above a cap, against a
     dormant delivery stage, and with block-varying bounds. Byte-neutral for every
     in-bounds solve: D34 golden and both backend parity baselines stay bit-identical.
   - **Ticket B — load-time owner of "genuine over-commitment".** Validate declared past
     and post-horizon commitments against the delivery-stage box in cobre-io/setup;
     reject an empty block intersection there; retire or re-scope
     `SddpError::AnticipatedCommitmentOutOfBounds`.
   - Release mechanics: full local CI bar; schema regen if the run summary gains the
     tally field; Python parity for that field; CHANGELOG describes behaviour only.

4. **Reconcile the SpecForge spec against the live tree** before any station resumes — DONE 2026-09-17, per-ticket table below:
   list every remaining ticket, mark each as done-on-develop / still-valid / obsolete,
   and decide per ticket whether to close, rewrite, or keep. Outcome recorded in
   `stations/` and in this file (see "SpecForge reconciliation" below once done).

5. **Re-pin the evaluation baseline.** Original sequencing: after the CD-074 fix merges. Proposed
   amendment (see "Owner decisions" §2 below, owner to confirm): pin once, now, at `077dbe2c`, and
   record CD-074 at the pin with a later fixed-status bullet.

6. **sddp station with one extra lens:** ABSORBED by the state-canonicalization plan — the
   class CD-074 was a specimen of (a pinned state coupled by an equality to a bounded column,
   reconciled ad hoc per family) is now handled in ONE shared seam `assemble_outgoing_state`
   across every state family (storage, inflow lags, transit buckets, commitment hold). No
   separate audit lens remains for this shape.

7. **Remaining stations in the existing order:** cli-python, W8 setup-perf
   (opportunistic), build-ci + test-corpus, generalization-alignment, unified-roadmap.
   — cli-python DONE (ratified 2026-09-18); solver-comm and sddp also ratified 2026-09-18; the
   rest is re-sequenced in the 2026-09-19 section below.

## SpecForge reconciliation (2026-09-17, ticket bodies read)

Spec `46c31993-82c2-4371-9447-b4f5aba7f75a` "Quality evaluation" (project `27f2e2e0`): 11 epics, 76
tickets, 19 done, 1 active (E03-6), 56 pending, 0 actionable. Every pending ticket body was read on
2026-09-17 and its baseline facts re-checked against `origin/develop` @ `077dbe2c`. The earlier
provisional table (written while the backend returned 502) is superseded by this section.

### Cross-cutting staleness (affects many tickets at once)

| # | Spec premise | Live state | Consequence |
| --- | --- | --- | --- |
| X1 | One baseline `a136840d`, cited by every station; scaffold headings and ticket section titles carry it literally | `develop` is 99 commits past it | DONE 2026-09-17: re-pinned to `077dbe2c` (`tools/pin-baseline.sh --repin`; the header keeps a `Previous baselines:` line). Scaffold headings keep their minted sha (checkers resolve sections by slug). Each entry's own `Baseline:` field is now authoritative for its anchors: `check-anchors` resolves an entry's Anchors/Evidence at that field and only `Status`/`Correction` provenance bullets at the register pin; `check-reraise` scans the finding, not its provenance bullets; the drift rule exempts the mirror the evaluation itself writes. Station test modules measure their own baseline through `station_checks.Tree`, never the worktree. No ticket text needs the sha edited; new stations mint entries at the register pin. |
| X2 | "No fixes in this spec"; "the only tracked file written is the mirror"; "BACKLOG.md stays gitignored" | Tier 1–5 + W7 fixes were executed (outside the spec, on fix branches); `plans/architecture-debt-audit/` is tracked on `develop` since 2026-09-06 (owner decision) | The guardrail still binds spec tickets. Every ticket that asserts "zero tracked change because plans/ is gitignored" (E11-5, E11-11, E11-9's documentation-only allowlist) must treat `plans/architecture-debt-audit/**` as a tracked, allowed write path. |
| X3 | `cobre summary` reads a run's phase split | The subcommand was deleted by the CLI plan; `src/summary.rs` only prints the post-run block of `cobre run`; per-iteration timings live in `training/timing/iterations.parquet` (`time_cut_sync_ms`, `time_mpi_allreduce_ms` still present) | E10-2/3/4 take the phase split from the run's own stdout summary (not under `--quiet`) or from the timing parquet; the "one extra untimed replay for `cobre summary`" edge case is obsolete. |
| X4 | cobre-python's 22 Rust `#[test]` are invisible to CI | The Python CI job now builds the CLI (`--require-cli-binary`) and runs the bindings crate's Rust tests with `LD_LIBRARY_PATH` | E06-4/5/6 and E08-2/4 lose that candidate; the mirror item "Python-binding Rust tests invisible to CI" is a retirement for E11-8, not a dup-of target. The new `doc = false` intra-doc-link gap (cobre-python `Cargo.toml:18`) is the replacement seed for E07. |
| X5 | ID floors CD-040/PD-006/OD-010/TD-001 | Live max: CD-074, PD-031, OD-031, TD-034 | Harmless: every calibrate ticket reads the floor at run time. E08-5's title "open the TD class" is moot (opened by core-io). |
| X6 | Test-corpus and perf queues are fresh | 15 TD and several PD rows from core-io/stochastic are already FIXED (Tiers 2–5) | E08-3 and E10-1 filter seeds by register status; a fixed PD is measured as its fix, or dropped with the reason. |
| X7 | CD-074 did not exist | Sev A bug minted 2026-09-17 in `commitment_reconcile` | E05-4 adds the pinned-state-vs-bounded-column lens; E05-5's contract screen must NOT dismiss drift-margin candidates by citing the contract CD-074 re-opened; E05-6 merges any restatement as dup-of CD-074. |

### Per-ticket verdicts

Vocabulary: **run** = valid as written, only the pin and re-measured figures change (step notes);
**deviate** = method valid, the named premises are stale and go into step notes; **rewrite** = a
premise the ticket is built on no longer holds (needs owner text change or an explicit deviation
approval); **supersede** = outcome already produced outside the spec, close with a pointer;
**blocked-owner** = waiting on an owner action.

| Ticket | Verdict | What changed / what to note |
| --- | --- | --- |
| E03-6 gate (stochastic) `01M1SSND5B…` | **blocked-owner** | Work complete & committed (`60fb2204`, `Gate: RETURNED 2026-09-08` in `stations/stochastic/gate.md`). Blocked on the webapp file-delivery step: flip `perf-queue.json`/`td-queue.json` to `reads` (or waive) + approve the two candidate-`reads` deviations, then `complete_work_session` with `worktree=/home/rogerio/git/cobre`. Unblocks all 56. |
| E04-1 inventory | **run** | 23 + 7 src files unchanged; prior-register anchors unchanged (`cut_sync.rs` fns still at 243/400/495/581). Line counts re-measured. |
| E04-2 Part-I item 8 | **run** | `types.rs` fields still at 270–297; `cut_nz_per_col` still in `freeze.rs`; `SCAN_DIRS` (5 crates), `EXCLUDED_FILES=()`, `PATTERN` unchanged. |
| E04-3 attackers | **run** | — |
| E04-4 ingest | **run** | — |
| E04-5 calibrate | **deviate** | X1 (entry `Baseline:` = new pin), X5. |
| E04-6 verify | **run** | `verify-station.sh` exists; pre-station porcelain snapshot instead of `.gitignore` carry-in. |
| E04-7 gate | **run** | — |
| E05-1 inventory | **deviate** | 163 src files still; tests now 56 files / 40 binaries (was 53/37); the four sub-station line totals re-measured; `policy/` grew (boundary-policy plan). |
| E05-2 lp-inventory (done @ `00abd53a`) | **deviate** | Regenerate `measurements/lp-inventory.json` at the new pin (`tools/lp-inventory.py`); record the re-run in E05-1's notes since the ticket is closed. Path set is still 30. |
| E05-3 Wave 4/6/7 re-verify | **run** | All 22 owned dispositions still open in the register; anchors are symbol-only by design so the boundary-policy drift is absorbed. `stage_solve_prep.rs` still under `training/`, `solve/` exists, `workspace/workspace.rs` exists — the CD-021/CD-023 traps hold. |
| E05-4 attackers | **deviate** | X7 lens; figures (187,059 lines, `policy_load.rs` 3,769) stale; CD-074 pre-listed so it is sharpened, not re-derived. |
| E05-5 ingest | **deviate** | X7 contract-screen carve-out. |
| E05-6 calibrate | **deviate** | X1, X5, X7 (CD-074 dup-of); section title `STATION 5 — cobre-sddp (2026-09)`. |
| E05-7 verify | **deviate** | 163 files holds; lp-inventory regenerated; timing-literal grep unchanged. |
| E05-8 gate | **run** | — |
| E06-1 inventory | **rewrite** (figures) | 18 CLI src files (was 20: `commands/report.rs`, `commands/summary.rs` deleted); S6b diagnostics = `validate.rs` + `src/summary.rs` only; CLI test binaries 14 (was 16); pytest files 37 (was 32); `cobre.run.run` rebuilt over `Study`; writer/parity figures all re-measured. The three "supersessions" in the text are themselves superseded. Method (three sub-surfaces, set equality) stands. |
| E06-2 Wave 5 + I.5 | **deviate** | CD-029 Python-parity half FIXED (phase 11 via the shared reconciler) — disposition is sharpen/partial, not keep; `validate_phases.rs:20` still says "four" (drift stands); `StudyParams::from_config` still 5 non-test callers (`broadcast.rs:144`, `validate.rs:384`, `io.rs:254`, `run.rs:994`, `setup/mod.rs:378`); the `report.rs` sub-claim is moot (file gone). |
| E06-3 attackers | **deviate** | Owner decision (PRIORITIES §6): feed `plans/cli-simplification-python-review/{parity-arguments,parity-behaviour,docstring-audit,fix-list}.md` as prior evidence; re-adjudicate the 16 justified-as-is rows, the orphaned `hydro_models`/`model_provenance` sidecars, the `TrainingSummary` Option asymmetry (owner-decided, do not re-raise). |
| E06-4 ingest | **deviate** | X4: the runtime parity test no longer skips (CLI binary required in CI) and the Rust tests run — `enforcement-measurements.json` must record the NEW state; `check_python_parity.py` still present. |
| E06-5 calibrate | **deviate** | X1, X5; CD-029 partial; X4 turns the "re-raise-blocked Cleared xref" into a retirement. |
| E06-6 verify | **deviate** | The "three known stale figures" are stale again; `cargo check` premise fine; CI-invisibility premise gone (X4). |
| E06-7 gate | **deviate** | CD-025 hoist stays Wave 5 (owner); parity routing to E07 unchanged. |
| E07-1 inventory | **deviate** | `scripts/ci` = 18 files (was 17); `check-comment-line-refs.sh` scans `.py/.pyi`; Python job changed (X4); seed `doc = false`. 8 workflows / 14 jobs / 18 schemas / trigger asymmetry unchanged. |
| E07-2 attackers | **run** | Headline facts hold (SCAN_DIRS, PATTERN, EXCLUDED_FILES, MPICH ×8, README rows 26–27); `cobre-cli/Cargo.toml` slow-tests line 56→57. |
| E07-3 ingest | **run** | — |
| E07-4 calibrate | **deviate** | X1, X5. |
| E07-5 verify | **deviate** | Hard-coded "17 gate files" → re-measure (18). |
| E07-6 gate | **run** | — |
| E08-1 re-measure figures | **deviate** | Its job is re-measurement, so mostly run; literal pairs in text (sddp 37/53, io 12/13) are now 40/56 and 12/13; yardstick status line now "Partially adopted (§5.2); the rest Proposal"; `.config/nextest.toml` still absent; shuffle cron still commented; cobre-comm still no `test-support`. |
| E08-2 prior register | **deviate** | X4 retires one of the three mirror items; StubComm census moved (`test_support.rs:3555`, new copies at `training/training/tests.rs:173`, `examples/dhat_baseline.rs:44`); `permute_helpers.rs` exists (TD-034 fixed). Precondition from PRIORITIES §6: ratify `testing-architecture.md` §5.1 first. |
| E08-3 lenses | **deviate** | X6 seed filtering; 15 TD rows already closed. |
| E08-4 ingest | **deviate** | X4; §5.8 census updated. |
| E08-5 calibrate | **deviate** | X1, X5 (title moot). |
| E08-6 verify | **deviate** | Literal pairs and the "13 tracked slow-tests files" figure re-measured. |
| E08-7 gate | **run** | — |
| E09-1 Part-I consolidation | **run** | Item 3 (no `struct PolicyGraph`; `PolicyGraphType` in 4 files), item 5 (`training_event.rs` 938 lines), item 6 (`EXCLUDED_FILES=()`) all hold; item 7 `BroadcastConfig` line drifted (`broadcast.rs:87`→`:32`). Needs the five station handoffs first. |
| E09-2 lp/ classification | **deviate** | 30 files hold; non-test totals (11,675 / 44,374) and trap line numbers re-measured after the boundary-policy plan. |
| E09-3 adjudicate Alignment | **run** | Universe parsed from the register at run time (now includes the reconciliation/W7 sections). |
| E09-4 verify | **deviate** | Literals (11,675/11,396; 79/95 refs; `broadcast.rs:87`) re-measured. |
| E09-5 gate | **run** | — |
| E10-1 binary + claim table | **deviate** | `[profile.profiling]` intact; queues exist only for core-io/stochastic until stations run; X6; CAL/CAL-ENUM re-measured at the new pin (2026-09-17). **Deck deviation (owner decision 2026-09-17):** the sanctioned 4t/2x2 deck `cobre_reduzido_2` no longer exists (cobre-bridge `example/` is gitignored, no history); `~/git/cobre-bridge/example/cobre_reduzido` (113 monthly stages, sampled selection, 4 forward passes, `iteration_limit 5`, bridge 0.12.0) is re-sanctioned in its place — `perf-run.sh`, `verify-harness.sh` and the register header name it; every E10 ticket text naming `cobre_reduzido_2` reads as `cobre_reduzido`. The enumerated deck is unchanged. |
| E10-2 PD-004 | **deviate** | `run_enumerated_backward` present; BACKLOG line refs (`:1255`, `:1696`, `:1881`) drifted; X3. |
| E10-3 4t | **deviate** | X3 (edge case (c) obsolete); all six anchors still resolve; Tier-2-fixed stochastic claims are measured post-fix. |
| E10-4 2x2 | **deviate** | X3; `time_cut_sync_ms`/`time_mpi_allreduce_ms` still in the timing parquet; all anchors resolve. |
| E10-5 write blocks | **run** | — |
| E10-6 verify | **deviate** | `env.txt` sha = new pin. |
| E11-1 Wave 4–7 re-verify | **deviate / partly superseded** | Headings still at BACKLOG `1888`/`1917`; the POST-PLAN section already re-verified CD-025 (open), CD-029 (partial), CD-009, CD-011 (open); PRIORITIES §5 closed W1–W7 of the 2026-09 waves (a different roster). Fold, do not redo; 28-unit count stands. |
| E11-2 dedup + checks | **run** | — |
| E11-3 unified roadmap | **deviate** | Must ingest PRIORITIES.md §5 (W1–W7 closed) and §6 (ranking) and the ROADMAP.md sequence; exactly one Milestones block already exists (header) — repoint, do not duplicate. |
| E11-4 doc inventory | **deviate** | 43 tracked `.md` outside `plans/` holds (85 with the now-tracked register — X2); HEAD≠pin abort → re-pin first. |
| E11-5 fold plans/ | **deviate** | Nothing folded yet (3 loose `plans/*.md`, `refinement-todo.md`, `decomp-program-reconciliation.md`, HTML + `mermaid.min.js` + builder all present); X2 makes BACKLOG edits tracked commits. |
| E11-6 fold-and-delete design docs | **deviate** | Both docs still present with the sixth status; `post-horizon-input-unification.md` still has no README row; NEW: `testing-architecture.md` row reads "Partially adopted (§5.2); the rest Proposal" — a further out-of-vocabulary status to normalize. |
| E11-7 dedup root docs | **run** | All named duplications persist (CONTRIBUTING `### Building` 18, `### Testing cobre-solver` 121, `### Testing cobre-sddp` 174, `### Project Structure` 329, `### Improving Documentation` 424, `### General` 471, `### Python Parity` 485); CLAUDE.md unsafe bullet still names only `gemm.rs`; 4 crates override `unsafe_code`. Lines re-resolved. |
| E11-8 mirror-check + mirror | **deviate** | Mirror already gained 2026-09 `### Fixed —` H3s from the Tier waves (projection rule: still no new H2); the two bare "ticket" lines persist (token layer premise holds); X4 retirement. |
| E11-9 verify | **deviate** | X2: documentation-only allowlist must admit `plans/architecture-debt-audit/**`. |
| E11-10 final gate | **run** | — |
| E11-11 commit | **rewrite** | "Branch off `main` at the pinned baseline" is wrong: `main` = v0.15.0 = `a136840d`, work bases on `develop` @ the new pin; "everything under plans/ staying untracked" is false (X2). No push / no PR / no CHANGELOG rules stand. |

### Owner decisions (SpecForge-side)

1. **Deviation notes and tool-side resolutions, not reopen and not webapp edits.** Every
   `deviate` row is recorded in the ticket's step notes at `start_work_session`; the two `rewrite`
   rows (E06-1 figures, E11-11 branch base) are recorded as heavy deviations through the work
   session itself (a `create_discovery` with `proposedOptions` for anything needing an owner
   decision; a declared file the plan mis-named or that a full-accept gate leaves untouched is
   `waived`, one whose content moved into another declared file is `consolidated`). The owner's
   role in the SpecForge webapp is approval of those records, never editing declarations.
   `reopen_specification` is not proposed (declined once for E03-6's blast radius).
2. **Pin once, now — CONFIRMED and executed 2026-09-17.** Re-pinned to `077dbe2c` rather than
   waiting for the CD-074 fix to merge. The spec assumes one sha for all stations, E04 was
   otherwise idle, and the sddp station *records* CD-074 at the pin (the bug is present there) and
   marks it fixed by status bullet later — the same way every Tier fix was recorded. The two
   ratified stations stay anchored at `a136840d` through their entries' own `Baseline:` fields
   (X1).
3. **CD-074 fix stays outside the spec** (guardrail "no fixes in this spec"); its plan is its own
   SpecForge spec or a plain `plans/` plan, tracked by the register's status bullet.
4. **E11-1/E11-2 fold** the existing PRIORITIES.md and POST-PLAN reconciliation rather than redoing it.

### Order of operations once E03-6 clears

E03-6 finalize (DONE 2026-09-17) → re-pin (DONE) + re-calibrate `CAL`/`CAL-ENUM` (DONE, on the
re-sanctioned deck) → E04 (run) → E05 (with CD-074 lens; regenerate lp-inventory) → E06 (matrices
as prior evidence) → E07 → E08 (after §5.1 ratification) → E09 → E10 → E11.

## Sequence after stations 4–6 (2026-09-19; the live sequence — `PRIORITIES.md` §8 holds the table)

Merged into `develop` at `3356da2d`: the three station ratifications (solver-comm, sddp, cli-python;
`PRIORITIES.md` §7 tiers their 104 new ids and 26 sharpened priors), the opened build-ci station, and
the state-canonicalization plan that fixed CD-074 (drift tally later removed, `7fbb3da2`). No release
has been cut since v0.15.0.

### Owner decisions (2026-09-19)

1. **Hybrid sequencing.** Tier the three ratified stations (done, §7); fix Tier 6 now as one plan;
   run build-ci (E07) and the performance sweep (E10) alongside it because their inputs are already
   in hand and they touch nothing the wave edits; alignment (E09), test-corpus (E08) and the unified
   roadmap (E11) follow, in that order.
2. **No release-scoping cutoff.** The wave plans everything not gated on an unrun station (Tiers 6,
   7, 8 of `PRIORITIES.md` §7 plus the W17 Sev-C items and W8 setup-perf) and lands as much as time
   allows; releases are cut from whatever has merged. (The 2026-09-17 constraint "no standalone patch
   release" stands.)

### Steps

1. Register housekeeping — DONE in this change (`3356da2d` recorded; §7, §8, this section).
2. `/plan` the ungated quality wave off `develop`: Tier 6 → Tier 7 → Tier 8 → W17 and W8 as outline
   epics. Bars and CHANGELOG duties as listed in §8 step 1.
3. build-ci station E07-2 … E07-6 (read-only), alongside step 2.
4. performance-sweep station E10 at the register pin, alongside step 2; the 15 B perf rows of W14
   schedule only from its claim table.
5. Release (owner) from whatever has merged — `plans/state-canonicalization/RELEASE-CHECKLIST.md`.
6. generalization-alignment station E09 → the Phase-0a/0b plan (W15: CD-004, CD-005, CD-025, CD-079,
   OD-043, CD-088, CD-092, W10).
7. Ratify `testing-architecture.md` §5.1 → test-corpus station E08 → W9 + W16 as one wave.
8. W8 setup-perf rides step 2 as an outline epic (PD-009 first).
9. reconciliation + unified-roadmap E11: lift §5 + §7 waves into the register's roadmap section;
   mirror write-backs as enumerated in §8 step 8.

### Open owner calls carried into step 2

- CD-082: the constructor-input half lands in Tier 6; the fail-loud construction-time check was
  decided to ride CD-004's carrier (R6-nh-2) — confirm the split when the ticket is specced.

### Preconditions found while verifying this amendment (2026-09-19, this host)

- **Re-pin before E07/E10 run.** `check-anchors.py` exits 2 with `baseline-drift: HEAD 3356da2d !=
  baseline 077dbe2c` — the evaluated surfaces (crates/, docs/, scripts/, schemas/) changed when the
  state-canonicalization plan merged, so the drift rule now bites. Run `tools/pin-baseline.sh --repin`
  at the tip the stations will evaluate (the five ratified stations keep their entries' own `Baseline:`
  fields, as X1 already established). The step-1 wave changes crates/ again, so pin once, after it
  merges, unless E07/E10 run first at `3356da2d`.
- **The harness needs `python3` ≥ 3.10 on PATH.** The tools use `@dataclass(slots=True)` and the unit
  tests spawn the literal `python3`; this host's `/usr/bin/python3` is 3.8 and every checker and all 20
  harness tests fail on import (pre-existing, unrelated to this change). `~/.local/bin/python3.13`
  exists; put a 3.10+ interpreter first on PATH as `python3` before running `verify-harness.sh` /
  `verify-station.sh`. With 3.13, `fields-check` and `check-roadmap-dag` self-tests pass; `check-anchors`
  fails only on the drift above.
- **`plans/generalizing/` is absent on this host** (untracked, never committed; present on the owner's
  other machine). `check-reraise.py` aborts with `CorpusMissing: plans/generalizing/refinement-todo.md`,
  and the alignment station (E09) reads `plans/generalizing/beyond-sddp-generalization.md`. Copy or
  track the corpus before E09 or any station verify runs here.
