# Owner ratification gate — cobre-core + cobre-io

Baseline: `a136840d4f2ea137f685f0af6dac04254b983b60`. Source: calibration.json (77 assigned, 4 cleared), verdicts.json, verification.md (7/7 checks green).

Decision line (set at close): **Decision: <ratified|returned>**

## 1. Presentation

Actionable entries: 77 (0 Sev-A · 43 Sev-B · 34 Sev-C). Alignment holds (conflicts): 0.

### Sev B (43)

| ID     | Class | Lens             | Align       | Reviewer | Claim (narrowed)                                                                                                |
| ------ | ----- | ---------------- | ----------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| CD-040 | CD    | architecture     | neutral     | —        | HydroPenalties (entities/hydro.rs:58) and HydroStagePenalties (model/resolved/penalties.rs:37) are structurall… |
| CD-041 | CD    | architecture     | neutral     | —        | DisconnectedBus (error.rs:65) and InvalidPenalty (error.rs:72) are emitted by no production/validation path an… |
| CD-043 | CD    | architecture     | neutral     | —        | SystemBuilder::build re-sorts stages by id (builder.rs:364) but neither reassigns nor validates Stage.index (o… |
| CD-044 | CD    | architecture     | advances-1  | —        | StageLagTransition (temporal.rs:175) has zero cobre-core consumers (all 4 occurrences are its own declaration/… |
| CD-045 | CD    | architecture     | advances-1  | —        | The three *_models accessors (mod.rs:432/438/466) promise canonical order that no write path enforces (setters… |
| CD-046 | CD    | architecture     | neutral     | —        | The wire-reproducibility rationale in the System serde(skip) comment (mod.rs:64) is enforced by three unrelate… |
| CD-048 | CD    | architecture     | neutral     | —        | bounds.rs:24 and penalties.rs:23 state `sorted by ID` for entity families that all carry operational_start_dat… |
| CD-050 | CD    | architecture     | neutral     | —        | Exactly two semantic branches -- block-duration>0 (semantic/stages.rs:88-103) and CVaR alpha/lambda range (:10… |
| CD-051 | CD    | architecture     | advances-0a | —        | The scenario-source admission rules are enforced only lazily inside the accessor and cobre-io's own load pipel… |
| CD-052 | CD    | architecture     | neutral     | —        | Confirmed narrowly: `LoadError::CrossReferenceError` is a dead variant (zero production producers, findings ro… |
| CD-053 | CD    | architecture     | neutral     | —        | Confirmed as-scoped to two concrete divergences: the block-id-range defect is `BusinessRuleViolation` in block… |
| CD-054 | CD    | architecture     | neutral     | —        | Confirmed narrowly: the uniform flat 'field is in id-set' reference blocks and the 20 copies of the `s.id >= 0… |
| CD-056 | CD    | architecture     | neutral     | —        | Confirmed narrowly: the stage-axis out-of-horizon validator is missing for the five block-eligible non-thermal… |
| CD-057 | CD    | architecture     | neutral     | —        | Confirmed narrowly: the positional `FILE_ENTRIES` to `manifest_fields_mut` zip is guarded only by an equal-len… |
| CD-058 | CD    | architecture     | neutral     | —        | The crash-safety hole is specifically the in-place (O_TRUNC) overwrites of manifest.bin at checkpoint.rs:244 a… |
| CD-059 | CD    | architecture     | advances-0a | —        | The confirmed defect is narrowly the false module-doc contract at output/mod.rs:6-8 (write_results does not 'w… |
| CD-060 | CD    | architecture     | neutral     | —        | The load-bearing, concretely-defective residue is the over-broad self-description of the two enumerations vers… |
| CD-061 | CD    | architecture     | advances-0a | —        | The confirmed defect is the register/oracle-coverage gap only: the Parquet convergence/timing/row_selection sc… |
| CD-062 | CD    | architecture     | neutral     | —        | The confirmed defect is the unreconciled duplication plus the contradictory documented contract for the empty-… |
| OD-011 | OD    | over-engineering | neutral     | A        | The single missing wire is at columns.rs:1136: the NCS curtailment objective is sourced from the stage-invaria… |
| OD-013 | OD    | over-engineering | neutral     | —        | The load-bearing residue is the build-time plus broadcast-wire cost of a reader-less structure: the System.net… |
| OD-016 | OD    | over-engineering | advances-1  | —        | The zero-non-test-consumer claim holds exactly (only lib.rs:134/137 re-exports; three in-file tests plus a no_… |
| OD-018 | OD    | over-engineering | neutral     | —        | ParsedData.penalties (schema.rs:75) has zero data.penalties reads workspace-wide and its rationale's saving is… |
| OD-019 | OD    | over-engineering | neutral     | —        | The defect is the ORDER-unguarded positional zip between FILE_ENTRIES and manifest_fields_mut (structural.rs:3… |
| OD-023 | OD    | over-engineering | neutral     | —        | Conceding the struct is not dead (its three fields are read by every parquet writer) and that pinning the enco… |
| PD-007 | PD    | perf             | neutral     | —        | Confirmed narrowly: all five discarding call sites lie on one-time setup/validation paths (PAR lag-transition … |
| PD-008 | PD    | perf             | neutral     | —        | Confirmed, narrowed to a one-time study-setup MPI broadcast payload (not any per-iteration hot-path cost) and … |
| PD-009 | PD    | perf             | neutral     | —        | The projection join at estimation.rs:468-479 is O(occurrences x windows) per hydro — quadratic in unbounded hi… |
| PD-010 | PD    | perf             | neutral     | —        | The nested O(hydros x [inflow_history + recent_observations] rows) filter in merged_windows_for_hydro (181/185… |
| PD-011 | PD    | perf             | neutral     | —        | check_prefix_coherence (846-857) re-walks stages 0..=sn per transition and re-does identical cell comparisons … |
| PD-014 | PD    | perf             | neutral     | —        | slot_occupying_classes (357) full-scans up to three external tables plus a fresh HashSet per call, and check_r… |
| PD-017 | PD    | perf             | neutral     | —        | The partitions_written inventory reaches no output file (SimulationMetadata at manifest.rs:434-462 has no fiel… |
| PD-018 | PD    | perf             | neutral     | —        | On the one-shot output-conversion path (cli outputs.rs / python run.rs, not the hot push loop) delta_to_stats_… |
| TD-001 | TD    | test-bloat       | advances-1  | —        | Narrower than 'every test': the derived-Clone/Debug non-empty assertions (`all_variants_clone`, `all_variants_… |
| TD-002 | TD    | test-bloat       | neutral     | —        | Narrower than 'all-same-value ... one per test module': HydroPenalties has no shared fixture or Default, so fi… |
| TD-004 | TD    | test-bloat       | neutral     | —        | Narrower than 'ad-hoc comparators ... where the yardstick calls for one shared comparator': the defect is not … |
| TD-005 | TD    | test-bloat       | neutral     | —        | Narrower than 'copy-pasted across five test sites': the plain zero-varying builders (make_bus/make_line/make_t… |
| TD-007 | TD    | test-bloat       | neutral     | —        | The input path has no shared test-support module while the validation path does, and the fully-verbatim, fully… |
| TD-014 | TD    | test-bloat       | neutral     | —        | The minimal-case corpus + write_file are duplicated across two same-crate inline modules (validation/schema.rs… |
| TD-016 | TD    | test-bloat       | neutral     | —        | Because validation/semantic/test_support is pub(super)-scoped, four validation/-level phase modules hand-roll … |
| TD-018 | TD    | test-bloat       | neutral     | —        | make_config (43 lines) and make_system (6 lines) are byte-identical between output/convergence_reader.rs and o… |
| TD-021 | TD    | test-bloat       | neutral     | —        | The 21 codec-only and 15 checkpoint-only tests are homed in policy/mod.rs (production ends line 28) while code… |
| TD-023 | TD    | test-bloat       | neutral     | —        | parquet_helpers.rs has zero #[cfg(test)] module (166 lines, all non-test), so the six extractors' missing-colu… |

### Sev C (34)

| ID     | Class | Lens             | Align       | Reviewer | Claim (narrowed)                                                                                                |
| ------ | ----- | ---------------- | ----------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| CD-042 | CD    | architecture     | neutral     | —        | The uniform-stride flat-index arithmetic is repeated at 14 sites across hydro/line/pumping/contract with no sh… |
| CD-047 | CD    | architecture     | neutral     | —        | Six numeric extractors in extensions/{hydro_geometry,hydro_energy_productivity,tailrace_curves}.rs re-implemen… |
| CD-049 | CD    | architecture     | neutral     | —        | The two resolvers share an identical algorithm skeleton and five parallel tests collapsible to one generic rou… |
| CD-055 | CD    | architecture     | neutral     | —        | Confirmed narrowly on the concrete drift: registry row 26 asserts a live semantic rule for `simulation.samplin… |
| CD-063 | CD    | architecture     | neutral     | —        | The concrete defensible defect is the cross-domain mis-homing of ensure_parent_dir inside stochastic.rs (impor… |
| OD-010 | OD    | over-engineering | neutral     | —        | InvalidPenalty is constructed nowhere at all (not even a test) and carries no reserving TODO, and both it and … |
| OD-012 | OD    | over-engineering | neutral     | —        | The genuinely unconsumed public surface is the population-statistics arm: population ci_95_half_width and the … |
| OD-014 | OD    | over-engineering | neutral     | —        | The four helper FUNCTIONS (not the Broadcast* mirror types, which setup.rs:288 genuinely consumes) are absent … |
| OD-015 | OD    | over-engineering | neutral     | —        | Both are pub speculative surfaces with no non-test consumer (only crate-root re-exports plus their own #[cfg(t… |
| OD-017 | OD    | over-engineering | neutral     | —        | default_severity (validation/mod.rs:95) has no caller outside its own unit test AND its BusinessRuleViolation-… |
| OD-020 | OD    | over-engineering | advances-0a | —        | The removable defect is precisely the unread third parameter `_config: &Config` on write_dictionaries (diction… |
| OD-021 | OD    | over-engineering | neutral     | —        | The narrower defect is needless pub visibility: both functions back only a same-module serde default-path attr… |
| OD-022 | OD    | over-engineering | neutral     | —        | The defect is exactly the two fields IterationRecord.time_bwd_setup_ms (mod.rs:141) and time_fwd_setup_ms (mod… |
| OD-024 | OD    | over-engineering | neutral     | —        | Conceding the byte-parsing body itself is correct and harmless (a faithful mirror of read_f64_vector), the nar… |
| OD-025 | OD    | over-engineering | neutral     | —        | Conceding run_pipeline_with_artifacts is the legitimate working function and the four public lib.rs entry poin… |
| PD-006 | PD    | perf             | neutral     | —        | Confirmed as a loader/study-setup-path inefficiency only (not any hot path in training/forward, training/backw… |
| PD-012 | PD    | perf             | neutral     | —        | On a chain-dialect deck (data.stages.policy_graph.nodes empty, guard at 620) extract_class's cells map retains… |
| PD-013 | PD    | perf             | neutral     | —        | The identical stage_index build (season.rs:141, season.rs:264, scenarios.rs:1012) and the identical partition_… |
| PD-015 | PD    | perf             | neutral     | —        | The internal checkpoint write path pays an avoidable full-buffer memcpy: each serializer's finished_data().to_… |
| PD-016 | PD    | perf             | neutral     | —        | Each build_*_batch reconstructs its run-invariant Arrow Field list (with fresh column-name Strings) once per s… |
| PD-019 | PD    | perf             | neutral     | —        | build_iterations_columns (solver_stats_writer.rs:88-176) alone builds 18 scalar columns via <Array>::from(iter… |
| TD-003 | TD    | test-bloat       | neutral     | —        | Narrower than 'all eight exercise only derives': the pure clone/eq/hash tests (`test_equality`, `test_hash_con… |
| TD-006 | TD    | test-bloat       | neutral     | —        | The < 1024 bound is an undocumented magic literal asserted only against a single-bus System, giving it too muc… |
| TD-008 | TD    | test-bloat       | neutral     | —        | The durable, threshold-independent residue is the intra-directory homing inconsistency -- scenarios/estimation… |
| TD-009 | TD    | test-bloat       | neutral     | —        | The genuinely redundant triplicated surface is make_batch plus the five non-determinism common cases (valid-so… |
| TD-010 | TD    | test-bloat       | neutral     | —        | The six binaries do verbatim re-declare 'mod helpers' and re-link arrow/parquet + the rlib six times, but per … |
| TD-011 | TD    | test-bloat       | neutral     | —        | test_bus_ordering_invariance is redundant because its bus2.name assertion -- the one surface not textually pre… |
| TD-012 | TD    | test-bloat       | neutral     | —        | The two IoError tests (test_load_error_io_display @152 and test_load_error_io_helper @240) are near-duplicates… |
| TD-013 | TD    | test-bloat       | neutral     | —        | The >= 17 floor in both the doctest (schema.rs:84) and test_generate_schemas_returns_expected_count (166) is l… |
| TD-015 | TD    | test-bloat       | neutral     | —        | test_filling_guard_no_exit_no_error (hydro.rs:1795) is identical to test_filling_guard_entry_below_horizon_no_… |
| TD-017 | TD    | test-bloat       | neutral     | —        | Three boundary_tests mirror earlier tests with identical fixture arguments, and the stage_id=5 mirror (thermal… |
| TD-019 | TD    | test-bloat       | neutral     | —        | every_hydros_schema_column_has_description (3226) and every_hydro_bus_generation_schema_column_has_description… |
| TD-020 | TD    | test-bloat       | neutral     | —        | The defensible core is the three named helpers — fixed_delivery::read_batch and generic_constraints_echo::read… |
| TD-022 | TD    | test-bloat       | neutral     | —        | The redundant surface is the count-only per-schema tests (thermals_schema_field_count and its count-only sibli… |

### Alignment holds (conflicts)

_None — no calibrated entry's fix-shape conflicts with the L0-L4 target layering._

### Cleared (dismissed — do not re-raise) (4)

| Ref                   | Reason / sanctionedBy                                                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| B-perf-00             | At baseline `tokenize` (generic.rs:629) is reached only via `parse_relation` <- `convert`'s per-constraint load-time loop (line 489) — the c |
| B-perf-01             | The evidence resolves at baseline: `inline` (named_expression_inline.rs:57) builds its `HashMap<&str,&ParsedExpression>` index per top-level |
| B-perf-03             | Mechanism confirmed at baseline: `parse_inflow_history` (inflow_history.rs:144-152) maps every parsed `InflowHistoryRow` into a parallel `Ve |
| C-over-engineering-00 | At the baseline all three enums derive serde::Serialize plus schemars::JsonSchema under #[serde(rename_all="snake_case")] (estimation.rs:15- |

### Needs-human (1)

1. Owner must decide whether byte-level wire-payload reproducibility is an actual contract (making the six unguarded HashMaps a latent bug) or a non-goal (making the System serde(skip) comment's reproducibility rationale over-stated); the narrowed defect holds either way but its severity depends on this call.

## 2. Round plan (AskUserQuestion)

- Round NH: each needs-human item, its own question, before the tiers.
- Round B: Sev-B tier (43) — accept all as calibrated / single out entries to downgrade / reject / defer.
- Round C: Sev-C tier (34) — same four options.
- No conflicts holds → no override questions.

## 3. Decision record

Owner decisions taken 2026-09-08 via AskUserQuestion (main session).

| ID / item                                           | Decision                 | Severity        | Rationale (owner)                                                                                                                             | Trigger / override |
| --------------------------------------------------- | ------------------------ | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| Needs-human: wire-payload reproducibility contract? | resolved: **non-goal**   | —               | Cobre determinism is result-level (bit-for-bit results, run-to-run, declaration-order invariance); byte-level wire layout is not a contract   | —                  |
| CD-046                                              | downgrade                | C (reviewer: B) | follows the non-goal ruling: the six unguarded HashMaps are not a latent bug; residue is a doc-accuracy nit on the System serde(skip) comment | —                  |
| Sev-B tier (42 remaining)                           | accept all as calibrated | B               | ratified as the attacker→defender→calibration pipeline recorded them                                                                          | —                  |
| Sev-C tier (34)                                     | accept all as calibrated | C               | ratified as recorded                                                                                                                          | —                  |

Counts: accepted 76 · downgraded 1 (CD-046) · rejected 0 · deferred 0 · overridden 0.
No conflicts holds, so nothing held or overridden; no rejects, so the Cleared list is unchanged.

## Decision

**Decision: ratified**
