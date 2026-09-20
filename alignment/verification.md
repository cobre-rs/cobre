# Alignment verification (register pin 077dbe2c; the tickets' scaffold pin a136840d)

Produced by `bash plans/architecture-debt-audit/alignment/verify-alignment.sh --report plans/architecture-debt-audit/alignment/verification.md`. The three harness checkers run over the register section `★ QUALITY EVALUATION (2026-09, baseline a136840d) — generalization-alignment` (the scaffold heading minted at the register pin; the tickets' `★ GENERALIZATION ALIGNMENT (…)` title never existed). Nothing is fixed; every write of the epic lives under plans/architecture-debt-audit/alignment/, which is tracked in this repository (the tickets' 'gitignored plan artifacts' wording is stale), so the read-only proof is that the committed tree carries no change outside plans/.

| check | detail | result |
| --- | --- | --- |
| anchors | checked 188 anchors, 0 failing | PASS |
| re-raise | checked 9 entries, 0 unjustified re-raise(s) | PASS |
| fields (section) | checked 9 entries in 1 section(s), 0 incomplete | PASS |
| Part-I dispositions | dispositions OK: 9 items (7 sharpen / 2 keep / 0 retire); I.5: 9 figures re-run from their commands, 3 station-quoted with handoff provenance, `use cobre_sddp` = 81 in the working tree | PASS |
| lp/ rows vs find | rows OK: 30 classification rows == 30 find paths == the tree at 077dbe2c | PASS |
| LOC recount | LOC OK: brace-aware 11701 vs naive 11411 (differ) over 44682 gross lines; the ticket's 11675 / 11396 / 44374 were quoted at the scaffold pin and are recorded as a deviation in lp-classification.json | PASS |
| grep proof | grep proof OK: 285 substring hits across the universe; engine-neutral 931 + mixed half 3939-5610 = 4870-6541 of 11701 vs IV.2 band 2340-2925 -> amended (ticket band 2335-2919 over 11675 quoted at the scaffold pin) | PASS |
| Alignment presence | checked 256 entries in 11 section(s), 0 incomplete | PASS |
| ledger integrity | alignment ledger OK: 243 entries, 12 retagged, 0 held for owner override | PASS |
| alignment tests | 46 passed in 12.54s | PASS |
| read-only | read-only OK: porcelain empty (modulo .gitignore); git diff HEAD over crates scripts .github schemas docs Cargo.toml is empty | PASS |

**Overall:** PASS.

Held for the owner gate: none (no station entry violates a Part IV.1 guardrail; see alignment/conflicts-docket.md).

Figures the tickets quote at the scaffold pin and how they read at the register pin (each recorded as a deviation in the producing artifact, never edited into the spec): non-test universe 11,675 → 11,701 (the ticket's own rule does not reproduce its total; 11,685 at a136840d), naive first-marker 11,396 → 11,411, gross 44,374 → 44,682, IV.2 band 2,335-2,919 → 2,340-2,925; I.5 `use cobre_sddp` lines 79 → 81 and `cobre_sddp::` references 95 → 97 (the envelope states both pins' figures with their commands). The engine-neutral share is an amended figure (42-56% measured against the fifth-to-a-quarter estimate), dated 2026-09-19.
