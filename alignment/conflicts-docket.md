# Conflicts docket — alignment adjudication (owner gate input)

Adjudicated 2026-09-19 at the register pin `077dbe2c` over 243 station entries. A `conflicts` decision names the Part IV.1 guardrail it violates and carries a roadmap-consistent alternative (or the owner scoping question); a held row enters no actionable set until the owner overrides or accepts.

## Guardrails (the rule, not the reading)

- **engine-concept-in-L0/L1** — Part IV.1 — L0/L1 name no engine or problem
- **engine-to-engine-dependency** — Part IV.1 — base engines depend on the shared kernels, never on each other
- **Engine-enum-below-L4** — Part IV.1 / Part IV.4 — the Engine enum lives only at L4
- **one-consumer-abstraction** — Part V.0 — pull, don't push: no abstraction with a single consumer
- **output-orchestration-not-at-L2** — Part IV.1 (L2 cobre-io owns shared output orchestration) / Part V.1 (Phase 0a deliverable)
- **phase-0a-gate-substrate-removal** — Part V.1 — the Phase 0a gate: SDDP output bit-for-bit unchanged from the pinned baseline

## Held rows

**None held.** No station entry's recorded fix-shape violates a guardrail: none places a paradigm concept in an L0/L1 crate, none introduces an engine-to-engine edge, none puts the Engine enum below L4, none hoists shared output orchestration into cobre-cli (the Wave-5 CD-025/CD-029 restatements were closed at the cli-python owner gate in favour of the L2 entry point, and no live entry restates them), and none deletes Phase-0a gate substrate (the E08 fix-shape refusals kept the golden parity_hash_* roster, common/parity_hash.rs and invariance-shuffle.yml out of every accepted fix-shape).

Rule hits reviewed and dismissed by hand (recorded so the predicates are shown to be live):

- OD-030 — `phase-0a-gate-substrate-removal` fired on “Collapse the speculative generality: since every production caller sorts largest-key-first, drop the SweepDirection enum and the direction parameter on set_solv” → decided neutral.

The two expected conflict classes, written out so an override request has a template:

### output-orchestration-not-at-L2 (no live instance)

**Shape.** A fix-shape hoisting shared writers into a cobre-cli-local helper (the Wave-5 CD-025/CD-029 restatement).
**Roadmap-consistent alternative.** One cobre-io entry point taking the resolved output set, called by both `crates/cobre-cli/src/commands/run/outputs.rs` and `crates/cobre-python/src/run.rs`; the CLI keeps only argument resolution and path assembly — the Phase-0a output-orchestration deliverable the hint was reaching for.
**Cost of overriding.** A helper in cobre-cli is unreachable from cobre-python, so the hand-mirrored outputs.rs-to-run.rs pair the fix claims to remove is re-created at L4 and Python parity stays a hand-kept rule.

### phase-0a-gate-substrate-removal (no live instance)

**Shape.** A test-corpus TD entry deleting a golden or determinism harness listed in stations/test-corpus/gate-substrate.json (the `parity_hash_highs` / `parity_hash_clp` mods of `crates/cobre-sddp/tests/parity.rs`, `crates/cobre-sddp/tests/common/parity_hash.rs`, `.github/workflows/invariance-shuffle.yml`).
**Roadmap-consistent alternative.** Retire the duplicated harness while keeping the golden: the parity_hash_* mods, common/parity_hash.rs and invariance-shuffle.yml stay; only the second copy of whatever they duplicate goes.
**Cost of overriding.** The Phase-0a gate states SDDP output bit-for-bit unchanged from the pinned baseline, and the substrate is how that is proved; deleting it leaves the gate unprovable.

