# Next-step roadmap — 2026-09-17

Companion to `PRIORITIES.md` §6. That section ranks the unrun stations; this file is the
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

3. **Fix CD-074 as one plan off `develop`, two tickets, released in v0.16.0.**
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

4. **Reconcile the SpecForge spec against the live tree** before any station resumes (provisional table below; backend was down on 2026-09-17):
   list every remaining ticket, mark each as done-on-develop / still-valid / obsolete,
   and decide per ticket whether to close, rewrite, or keep. Outcome recorded in
   `stations/` and in this file (see "SpecForge reconciliation" below once done).

5. **Re-pin the evaluation baseline after the CD-074 fix merges**, not before — every
   remaining station reads a surface the fix will touch.

6. **sddp station with one extra lens:** CD-074 is a specimen of a class — a pinned state
   coupled by an equality to a bounded column, reconciled ad hoc for one family. Audit
   every state family (storage, inflow lags, transit buckets, commitment hold) for the
   same shape and decide whether drift handling belongs in one shared seam.

7. **Remaining stations in the existing order:** cli-python, W8 setup-perf
   (opportunistic), build-ci + test-corpus, generalization-alignment, unified-roadmap.

## SpecForge reconciliation

Spec `46c31993-82c2-4371-9447-b4f5aba7f75a` "Quality evaluation" (project `27f2e2e0`), 11 epics /
76 tickets, planned 2026-09-05/06 against baseline `a136840d`.

**Status 2026-09-17: PROVISIONAL.** The SpecForge backend answered `502` (`/health` →
`Internal server error`) for the whole session, so the ticket bodies were not re-read; every
row below is derived from the local tree, the register, and the session record, and must be
confirmed against the ticket text before a station is started or a ticket closed.

**What moved under the spec** (`a136840d..077dbe2c` on `develop`: 99 commits, 327 files,
+30,042 / −25,384 across `crates docs scripts .github schemas`; per crate: cobre-io 107 files,
cobre-sddp 70, cobre-stochastic 51, cobre-python 44, cobre-cli 22, cobre-core 19,
cobre-solver 1, cobre-comm 0):

- The spec's premise "evaluation + ranked roadmap only, no fixes" no longer holds: Tier 1–5 and
  W7 fix waves (`PRIORITIES.md` §5, W1–W7 closed) were executed from the two ratified
  stations, plus two unrelated plans (boundary policy by date; CLI simplification + Python
  review) landed. `BACKLOG.md` "POST-PLAN RECONCILIATION (2026-09-17)" is the register-side
  reconciliation of those.
- The evaluation branch `chore/quality-evaluation` is at `develop` (0 commits behind).

| Epic | Spec premise | Live state | Provisional disposition |
| --- | --- | --- | --- |
| E01 harness | tools + calibration at `a136840d` | DONE, ratified. `CAL`/`CAL-ENUM` bounds measured at `a136840d` on the profiling binary. | Closed. Bounds are stale for any station evaluating at the new pin → re-run calibration (E10 or a re-opened E01-5 note), not a new ticket. |
| E02 core+io | station at `a136840d` | DONE, ratified 2026-09-08; most Sev A/B entries since FIXED by Tiers 1–5 / W7. | Closed as-is; the fixes are register-side status bullets, not spec work. |
| E03 stochastic | station at `a136840d` | Work complete & committed; **E03-6 owner gate still blocked** on the webapp file-delivery step (two queue files declared `modifies` but correctly unchanged; owner chose to fix in the webapp). | Owner flips the two queues to `reads` (or waives) + approves the two candidate-`reads` deviations → `complete_work_session` with `worktree=/home/rogerio/git/cobre`. No re-work. |
| E04 solver+comm | inventory at `a136840d` | Surface nearly untouched since the pin (cobre-solver 1 file, cobre-comm 0). | **Still valid as written**, only the pin changes. Cheapest station; can run before the CD-074 fix merges. |
| E05 sddp | inventory at `a136840d`; `lp-inventory.json` (E05-2) already produced | cobre-sddp +8.6k/−3.4k from the boundary-policy plan; CD-074 will touch `commitment_reconcile`/`stage_solve_prep`; `lp-inventory.json` must be regenerated at the new pin. | **Valid in method, stale in inputs.** Re-pin after CD-074 merges; regenerate `lp-inventory.json`; add the CD-074 "pinned-state vs bounded-column drift" lens (roadmap step 6). Ticket text likely needs the baseline sha and the inventory regeneration noted as deviations, not rewrites. |
| E06 cli+python | attacker discovers from scratch | Surface rewritten: `report`/`summary` deleted, `cobre.run.run` over `Study`, 48+30+128-row parity/docstring matrices and a 26-item fix list exist in `plans/cli-simplification-python-review/`. | **Rewrite the attacker inputs**: the station ratifies residuals from the plan's matrices (16 justified-as-is rows, orphaned `hydro_models`/`model_provenance` sidecars, CD-025 owner question, `doc = false` gate gap) instead of discovering. Method/tickets otherwise stand. |
| E07 build/CI/docs/schemas | at `a136840d` | scripts 4 files, `.github` 1, docs 7, schemas 1 changed; Python CI job now builds the CLI, runs the bindings' Rust tests, `check-comment-line-refs.sh` scans `.py`/`.pyi`. | Valid; small. Pin change only, plus the `doc = false` intra-doc-link gap as a seeded candidate. |
| E08 test corpus | yardstick `testing-architecture.md` §5.1 (Proposal) | Tier-4/5 test-support surface work + `permute_helpers.rs` split landed; `testing-architecture.md` now reads "Partially adopted (§5.2); the rest Proposal"; 15 TD findings from core-io/stochastic already closed. | **Valid but re-scoped**: the yardstick section must be ratified first (PRIORITIES §6 step 5 dependency); the station's prior register is the TD- ledger, most of which is closed. |
| E09 generalization alignment | Part-I re-verification + lp/ classification + alignment tags | Not started. Part-I cross-refs exist in both ratified stations; `plans/generalizing/` untouched (HTML + `mermaid.min.js` + build script still present — the E11 consolidation never ran). | **Valid as written.** Inputs: the two stations' `partI-handoff.json`, `tools/target-layering-brief.md`. |
| E10 perf sweep | measure Sev-A/B perf claims at `a136840d`, protocol 4t/2t/2x2 | `perf-queue.json` exists for core-io + stochastic only; Tier-2 hot-path wave changed cobre-stochastic forward sampling (byte-neutral, perf not re-measured); `claim-table.json` absent. | **Valid but must re-calibrate** `CAL`/`CAL-ENUM` at the new pin before any claim is measured; several queued PD- claims are already FIXED (Tier 2) → measure the fix, not the claim. |
| E11 reconciliation / roadmap / doc consolidation / mirror | E11-1 Wave 4–7 re-verify (sddp + cli-python), E11-2 dedup, E11-3/4 doc inventory, E11-10 final gate, E11-5..8 execute deletions, E11-9 verify, E11-11 commit | `PRIORITIES.md` (outside the spec) already ranks and its §5 waves W1–W7 are executed; POST-PLAN section already re-verified CD-025/029; **doc consolidation not done** (`docs/design/anticipated-fixed-post-horizon-commitments.md` and `external-scenarios-are-authoritative.md` still present; `plans/generalizing/*.html`, `mermaid.min.js`, `build_beyond_sddp_html.py` still present). | **Split**: E11-1/2 largely superseded by `PRIORITIES.md` + the POST-PLAN section (fold, don't redo); the roadmap tickets must ingest W1–W7 as closed and `PRIORITIES.md` §6 as the ranking input; E11-3..9/11 (doc consolidation + mirror) **still fully valid**. |

**Decisions the owner must take once SpecForge is reachable** (recorded here so they are not
re-derived): (1) whether the remaining stations run under the existing tickets with
deviation notes (pin sha, regenerated inventories) or the spec is reopened — reopening was
declined once for E03-6's blast radius, so the default is deviation notes; (2) whether E11-1/2
are closed as superseded or executed as a thin fold of `PRIORITIES.md`; (3) whether CD-074's
fix plan lives outside the spec (recommended: it is a fix, and the spec is evaluation-only).

**Next action when the backend is up:** `list tickets --epicId` for E04..E11, diff each
ticket's `filesToBeReferenced`/`filesToBeModified` and baseline sha against this table, and
replace "PROVISIONAL" with per-ticket close / deviate / rewrite verdicts.
