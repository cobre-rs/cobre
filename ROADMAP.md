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

4. **Reconcile the SpecForge spec against the live tree** before any station resumes:
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

_(filled by step 4)_
