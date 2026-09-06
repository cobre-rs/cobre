# core-io station — prior register (do-not-re-raise list)

Baseline `a136840d4f2ea137f685f0af6dac04254b983b60` (pinned 2026-09-05). Every BACKLOG.md and
mirror (`docs/design/reserved-seams-and-deferred-debt.md`) entry whose anchor resolves inside
`crates/cobre-core` or `crates/cobre-io`, each with a status, a KEEP | RETIRE | SHARPEN disposition
and a one-line reason. Attackers read this before the four lens passes; anything listed under
`## Do not re-raise` is rejected at ingest unless the candidate carries a `Re-raise-of:` line with
new evidence.

Entry grammar: `### <ref> · <title> — <KEEP|RETIRE|SHARPEN>` followed by `- **Register ID:**`
(a `CD-|PD-|OD-nnn` id, `reserved-seam-census` for the sanctioned seams, or `mirror:<section>` for
a mirror-only entry), `- **Anchor:**` (every backticked anchor resolves at the baseline),
`- **Status:**`, `- **Disposition:**` and `- **Reason:**`.

## Resolved in tree

### CD-010 · untyped `entity_type: u8` state-family dictionary — RETIRE

- **Register ID:** CD-010
- **Anchor:** `crates/cobre-io/src/output/policy/records.rs:52` (`enum StateFamily`), `crates/cobre-io/src/output/policy/records.rs:29-31` (`EntitySlot.entity_type`, documented as the raw `StateFamily` discriminant), `crates/cobre-io/src/output/policy/records.rs::family` (`EntitySlot::family`, line 89), `crates/cobre-io/src/lib.rs:106` (crate-root re-export), `crates/cobre-io/src/output/policy/checkpoint.rs:56` (consumer)
- **Status:** RESOLVED (Wave 1, 2026-08-22; BACKLOG "Wave 1 — executed", mirror "Untyped state-family primitive + colliding entity dictionaries — RESOLVED")
- **Disposition:** RETIRE
- **Reason:** the typed enum landed — `StateFamily` (HydroStorage=0, HydroInflowLag=1, AnticipatedThermalState=2, HydroTransitBucket=3) mirrors `EntityType` in `schemas/policy.fbs`, `EntitySlot::family()` reads the wire byte, the `checkpoint.rs:30` duplicate const is gone. Do not re-raise the untyped-primitive claim. The spec's `records.rs:29-31` cites the field; the method itself is at line 89.
- **Residue to re-check, not re-raise:** the physical-output dictionary `crates/cobre-io/src/output/dictionary.rs:31-38` (`OUTPUT_ENTITY_*`, `i8`) still shares a code space with the state-family discriminants (0..3 overlap by value, disjoint by use). Grep hazard only; a finding needs new evidence of a value crossing between the two tables.

### CD-026 · `write_scenario` 12× write-partition repeat — RETIRE

- **Register ID:** CD-026
- **Anchor:** `crates/cobre-io/src/output/simulation_writer.rs::write_partition` (line 676)
- **Status:** RESOLVED (BACKLOG "Resolved since the audit", 2026-08-18)
- **Disposition:** RETIRE
- **Reason:** one `write_partition` helper owns the create-dir/write/push tail; the per-entity batch builders (genuinely varying args) remain by design. The BACKLOG line-number `:685` is stale by nine lines; the symbol anchor resolves. The `simulation_writer.rs` flat-file size (BACKLOG Station 5 "noted, not raised") is a separate note below.

### CD-031 (cobre-io half) · `BoundaryPolicy::checkpoint_path` accessor — RETIRE

- **Register ID:** CD-031
- **Anchor:** `crates/cobre-io/src/config/policy.rs::checkpoint_path` (line 61), `crates/cobre-io/src/config/policy.rs::BoundaryPolicy` (line 42)
- **Status:** RESOLVED (Wave 3, commit `44e72b76`)
- **Disposition:** RETIRE
- **Reason:** the seven open-coded `case_dir.join(&bp.path)` joins collapsed onto this accessor; the cobre-sddp half of CD-031 is out of station (below). Do not re-raise "checkpoint path join is open-coded".

## Sanctioned reserved seams (dismissable-with-citation)

### `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig` — KEEP (sanctioned)

- **Register ID:** reserved-seam-census
- **Anchor:** `crates/cobre-io/src/config/training.rs:531` (`struct LipschitzConfig`), `crates/cobre-io/src/config/training.rs::mode` (line 534), `crates/cobre-io/src/config/training.rs::UpperBoundEvaluationConfig` (line 509)
- **Status:** SANCTIONED — mirror section "`LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`" (reserved-seam register) and "Reserved and consumed knobs (snapshot correction)"
- **Disposition:** KEEP (dismissable with that citation; not recordable as over-engineering or dead config)
- **Reason:** the whole struct is a loaded, schema-exported, unconsumed stub for vertex-based inner approximation; its milestone is a decision, not an implementation trigger. Under the "unwired config is reserved, not dead" rule (CLAUDE.md) a candidate naming it is dismissed with the mirror citation. The over-engineering lens may note that the mirror itself calls it the strongest removal candidate — as a pointer to the existing entry, never as a new finding.

### Hydro `storage_violation_below_cost` / `filling_target_violation_cost` penalties — KEEP (consumed, not reserved)

- **Register ID:** reserved-seam-census
- **Anchor:** `crates/cobre-core/src/model/resolved/penalties.rs::storage_violation_below_cost` (line 46), `crates/cobre-core/src/model/resolved/penalties.rs::HydroStagePenalties` (line 37; the mirror names it `HydroPenalties`, a stale spelling)
- **Status:** VERIFIED CONSUMED — mirror section "Verified NOT reserved"
- **Disposition:** KEEP (consumed, not reserved; dismissable with that citation; not recordable as an unwired seam)
- **Reason:** both penalties are consumed for filling-phase hydros (`fill_filling_target_columns` / `fill_filled_min_storage_floor_columns` in `crates/cobre-sddp/src/lp/builder/columns.rs`); they are unconsumed only for ordinary hydros whose storage bounds are hard. A candidate calling them "always 0 / unwired" repeats the over-generalization the mirror already corrected.

### `historical_years` on `ScenarioSource` — KEEP (consumed, not reserved)

- **Register ID:** reserved-seam-census
- **Anchor:** `crates/cobre-core/src/model/scenario.rs::ScenarioSource` (line 109)
- **Status:** VERIFIED CONSUMED — mirror section "Verified NOT reserved"
- **Disposition:** KEEP (dismissable with that citation)
- **Reason:** threaded by `crates/cobre-sddp/src/setup/mod.rs` into `discover_historical_windows` (`crates/cobre-stochastic/src/sampling/window.rs`) and read by the historical sampler; the older "inert field" interface-review note predates that wiring.

## Registered and open — KEEP (cite the existing entry, do not duplicate)

### Horizon-type config field (`graph_type` in `stages.rs`) — KEEP

- **Register ID:** mirror:Horizon-type config field long-term fate
- **Anchor:** `crates/cobre-io/src/stages.rs:134` (`graph_type: RawPolicyGraphType`)
- **Status:** DEFERRED (mirror deferred-debt register; trigger = the next licensed input-format break)
- **Disposition:** KEEP
- **Reason:** the field accepts only the finite-horizon value and is a deletion candidate under the node-native engine. A sub-station B candidate must cite this entry and may only SHARPEN it (e.g. a second reserved-but-rejected value), not re-record it.

### Boundary-policy source-node (`BoundaryPolicy.source_stage`) — KEEP

- **Register ID:** mirror:Boundary-policy source-node
- **Anchor:** `crates/cobre-io/src/config/policy.rs:53` (`source_stage: Option<u32>`)
- **Status:** DEFERRED (trigger = a study-level boundary redesign)
- **Disposition:** KEEP
- **Reason:** addresses its source by stage, no node selector, rejects a multi-node source — all already registered. Sub-station C cites it.

### Stage-configuration engine-scoping (`stages.rs` carries every axis) — KEEP

- **Register ID:** mirror:Stage-configuration engine-scoping
- **Anchor:** `crates/cobre-io/src/stages.rs:212` (`num_openings: Option<u32>`, the conditionally-required first step)
- **Status:** DEFERRED (trigger = the full engine-scoping split of the stage file)
- **Disposition:** KEEP
- **Reason:** the stage file forcing every study kind to declare SDDP's uncertainty/risk axes is the Milestone-1 "stochastic off System/Stage" target; a candidate here is an Alignment hint on the existing entry, tagged `advances-1`, not a new finding.

### Retired input spellings — the reject tests are load-bearing, not bloat — KEEP

- **Register ID:** mirror:Retired input spellings (recorded as retired, not deferred)
- **Anchor:** `crates/cobre-io/src/config/training.rs`, `crates/cobre-io/src/config/simulation.rs`, `crates/cobre-io/tests/flatbuffers_schema_conformance.rs`
- **Status:** RETIRED SPELLINGS, guarded (unknown-field / unknown-variant reject tests + the FlatBuffers conformance test)
- **Disposition:** KEEP
- **Reason:** the test-bloat lens must not flag the deserialize-reject tests in these files as redundant; they are the invariant that stands in for a version snapshot.

### `simulation_writer.rs` flat file (1905 prod lines) — KEEP (noted, not raised)

- **Register ID:** CD-021
- **Anchor:** `crates/cobre-io/src/output/simulation_writer.rs`
- **Status:** NOTED, NOT RAISED (BACKLOG Station 5 "_Minor (not a separate ID)_"; CD-021-neighbour organizational option)
- **Disposition:** KEEP
- **Reason:** correctly homed (one file per output type) and cohesive; a sub-station D candidate needs new evidence (a cohesion break, a second owner inside the file), not size alone.

## Phase-1 purification targets (live anchors for the alignment hints)

### Milestone `1` — `training_event` out of cobre-core — KEEP

- **Register ID:** mirror:Milestones
- **Anchor:** `crates/cobre-core/src/constraints/training_event.rs::TrainingEvent` (line 164)
- **Status:** OPEN roadmap target (BACKLOG Milestones table row `1`; beyond-sddp-generalization.md V.2)
- **Disposition:** KEEP
- **Reason:** an L0 crate carrying a training-loop event vocabulary is the Part-I leakage the station re-checks; sub-station A records the disposition with this anchor rather than re-discovering it.

### Milestone `1` — stochastic off `System` / `Stage` — KEEP

- **Register ID:** mirror:Milestones
- **Anchor:** `crates/cobre-core/src/model/temporal.rs:315` (`scenario_config: ScenarioSourceConfig` on `Stage`), `crates/cobre-core/src/model/temporal.rs::ScenarioSourceConfig` (line 250), `crates/cobre-core/src/model/scenario.rs::ScenarioSource`
- **Status:** OPEN roadmap target (Milestones row `1`)
- **Disposition:** KEEP
- **Reason:** same Part-I item; anchors recorded so the alignment epic can diff them against the tree.

## Out of station (owning epic named; not core/io work)

### CD-031 / CD-039 · `BoundaryStateRequirements` and the per-family boundary channels — OUT OF STATION → Station: cobre-sddp

- **Register ID:** CD-039
- **Anchor:** `crates/cobre-sddp/src/setup/params.rs::BoundaryStateRequirements` (line 63), `crates/cobre-sddp/src/setup/mod.rs:328` (`boundary_requirements` field), `crates/cobre-sddp/src/policy/policy_load.rs::resolve_boundary_state_requirements`
- **Status:** CD-031 RESOLVED; CD-039 setup-half RESOLVED, writer channel + transit widening still open
- **Disposition:** OUT OF STATION (Epic 5, Station: cobre-sddp)
- **Reason:** the resolved value lives in cobre-sddp; cobre-io only hosts the `checkpoint_path` accessor (retired above). Not on this station's do-not-re-raise list as core/io work.

### C17 · dropped-source boundary coupling warning (`warn_dropped_source_couplings`) — OUT OF STATION → Station: cobre-sddp

- **Register ID:** CD-039
- **Anchor:** `crates/cobre-sddp/src/policy/policy_load.rs::warn_dropped_source_couplings` (line 916)
- **Status:** SURFACED (Wave 3 continuation, commit `1702b2a0`); the absent transit widening path is intentional per the mirror
- **Disposition:** OUT OF STATION (Epic 5, Station: cobre-sddp)
- **Reason:** cobre-sddp-anchored; cobre-io's checkpoint manifest already self-describes every slot (`entity_type`/`entity_id`/`subindex`/`delivery_date`), so no cobre-io format work is implied.

### CD-025 · CLI/Python training-output hand-mirror — OUT OF STATION → Station: cobre-cli, cobre-python and facade

- **Register ID:** CD-025
- **Anchor:** `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs`, `crates/cobre-python/src/run.rs`
- **Status:** OPEN (HOLD); Milestone `0a` names "shared output orchestration in cobre-io" as the landing zone
- **Disposition:** OUT OF STATION (Epic 6); sub-station D may SHARPEN with the cobre-io owner shape
- **Reason:** the duplication is at the CLI/Python boundary, not inside cobre-io; the cobre-io writers are the positive writer-family reference (BACKLOG Station 5 verdict). A D-pass candidate about "where the shared output orchestration should live" attaches to CD-025's fix-shape, never as a new io finding.

### CD-029 · `PrepPhase` covers 3 of 4 validation phases, boundary check bypasses Python parity — OUT OF STATION → Station: cobre-cli, cobre-python and facade

- **Register ID:** CD-029
- **Anchor:** `crates/cobre-sddp/src/validate_phases.rs::PrepPhase`, `crates/cobre-cli/src/commands/validate.rs`
- **Status:** OPEN (HOLD)
- **Disposition:** OUT OF STATION (Epic 6)
- **Reason:** the enum and the bypass live in cobre-sddp / cobre-cli; cobre-io's validation modules are the phases' implementation, not the seam at fault.

### CD-001 / CD-004 · three parallel run-config representations (`Config` → `StudyParams` → `BroadcastConfig`) — OUT OF STATION → Station: cobre-sddp

- **Register ID:** CD-004
- **Anchor:** `crates/cobre-io/src/config/mod.rs`, `crates/cobre-sddp/src/setup/params.rs`
- **Status:** OPEN (HOLD; Wave 4 setup redesign)
- **Disposition:** OUT OF STATION (Epic 5); sub-station C keeps `Config` as the source-of-truth end
- **Reason:** cobre-io's `Config` is the first representation, not the duplicate; a C-pass candidate "Config duplicates StudyParams" is CD-004 restated.

### Stage-calendar crate home (`StageCalendar` / `season_cast`) — OUT OF STATION → Station: cobre-stochastic

- **Register ID:** mirror:Stage-calendar crate home
- **Anchor:** `crates/cobre-core/src/model/temporal/stage_key.rs`, `crates/cobre-core/src/model/temporal/overlap.rs`
- **Status:** DEFERRED (relocation into a core temporal module)
- **Disposition:** OUT OF STATION (Epic 3); sub-station A records `model/temporal` as the landing zone only
- **Reason:** the machinery lives in cobre-stochastic; cobre-core is the destination, so an A-pass "temporal module is thin" observation is context for that relocation, not debt.

## Executable oracles (the gate is the register entry)

### Infra-genericity gate scans `output/policy/` — the Part-I item-6 exemption is retired — RETIRE

- **Register ID:** mirror:Audit-evidence
- **Anchor:** `scripts/ci/check-infra-genericity.sh::EXCLUDED_FILES` (line 74, `EXCLUDED_FILES=()`), `crates/cobre-core/tests/infra_genericity.rs`, `crates/cobre-io/tests/genericity_gate.rs`
- **Status:** RETIRED BY THE SCRIPT (the format-version bump landed; the directory module is scanned like every other infra file)
- **Disposition:** RETIRE the exemption claim; KEEP the two in-crate oracles as positives
- **Reason:** Part-I item 6 ("output/policy is exempt from the genericity gate") no longer describes the tree. The station re-checks genericity structurally (paradigm nouns in types and doc comments), not by re-running the lexical gate; any lexical finding is a `dup-of` of the gate run.

## Do not re-raise

- **CD-010** untyped state-family primitive (resolved: `StateFamily`, `crates/cobre-io/src/output/policy/records.rs::StateFamily`, `EntitySlot::family`)
- **CD-010** `checkpoint.rs:30` duplicate `ENTITY_TYPE_HYDRO_TRANSIT_BUCKET` const (deleted; `crates/cobre-io/src/output/policy/checkpoint.rs` uses `slot.family()`)
- **CD-026** `write_scenario` write-partition repetition (resolved: `write_partition`, `crates/cobre-io/src/output/simulation_writer.rs::write_partition`)
- **CD-031** open-coded boundary checkpoint path join (resolved: `crates/cobre-io/src/config/policy.rs::checkpoint_path`)
- `LipschitzConfig.mode` / `UpperBoundEvaluationConfig` as dead config (sanctioned reserved seam, `crates/cobre-io/src/config/training.rs::LipschitzConfig`)
- hydro storage/filling penalties as unwired (consumed at filling stages, `crates/cobre-core/src/model/resolved/penalties.rs::HydroStagePenalties` (line 37; the mirror names it `HydroPenalties`, a stale spelling))
- `historical_years` as inert (consumed by the window and historical samplers, `crates/cobre-core/src/model/scenario.rs::ScenarioSource`)
- cobre-io `output/policy` genericity exemption (retired: `EXCLUDED_FILES=()` in `scripts/ci/check-infra-genericity.sh::EXCLUDED_FILES`)
- `graph_type` finite-only horizon field as a new finding (registered: mirror "Horizon-type config field long-term fate", `crates/cobre-io/src/stages.rs:134`)
- `BoundaryPolicy.source_stage` stage-addressed source as a new finding (registered: mirror "Boundary-policy source-node", `crates/cobre-io/src/config/policy.rs:53`)
- stage file carrying every algorithm axis as a new finding (registered: mirror "Stage-configuration engine-scoping", `crates/cobre-io/src/stages.rs:212`; Alignment hint only)
- config-spelling reject tests as test bloat (load-bearing guard: mirror "Retired input spellings", `crates/cobre-io/src/config/training.rs`, `crates/cobre-io/src/config/simulation.rs`)
- `simulation_writer.rs` size alone (noted-not-raised, BACKLOG Station 5, `crates/cobre-io/src/output/simulation_writer.rs`)
