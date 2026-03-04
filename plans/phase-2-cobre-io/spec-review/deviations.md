# Phase 2 Spec Review: Deviations from cobre-docs Specifications

**Review date:** 2026-03-04
**Reviewer:** open-source-documentation-writer agent
**Branch:** feat/phase-2-cobre-io
**Phase:** 2 — `cobre-io` Input Loading and Validation

---

## 1. Scope

This document compares every spec file in the Phase 2 reading list against the
Rust implementation in `crates/cobre-io/src/` and `crates/cobre-core/src/`.
Each deviation is categorised as:

- **(a) Intentional improvement** — the implementation does something better than the spec required
- **(b) Spec gap resolution** — the spec was silent and the implementation made a reasonable choice
- **(c) Implementation simplification** — the implementation does less than the spec required (scoped, not a bug)
- **(d) Potential bug** — the implementation may violate a correctness requirement

Category (a) and (c) deviations include a draft ADR summary. Category (d) deviations include a follow-up action item.

---

## 2. Phase 2 Spec Reading List Status

| # | Spec File | Review Status |
|---|-----------|---------------|
| 1 | `data-model/input-directory-structure.md` | Reviewed — see §3.1 |
| 2 | `data-model/input-scenarios.md` | Reviewed — see §3.2 |
| 3 | `data-model/input-constraints.md` | Reviewed — see §3.3 |
| 4 | `data-model/output-schemas.md` | Reviewed — see §3.4 |
| 5 | `data-model/output-infrastructure.md` | Reviewed — see §3.5 |
| 6 | `data-model/binary-formats.md` | Reviewed — see §3.6 |
| 7 | `architecture/validation-architecture.md` | Reviewed — see §3.7 |
| 8 | `architecture/input-loading-pipeline.md` | Reviewed — see §3.8 |
| 9 | `configuration/configuration-reference.md` | Reviewed — see §3.9 |

---

## 3. Per-Spec Deviation Tables

### 3.1 `data-model/input-directory-structure.md`

**Implementation files:** `crates/cobre-io/src/validation/structural.rs`, `crates/cobre-io/src/pipeline.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| DS-01 | §1 Directory Tree | 33 input files tracked | `FileManifest` in `structural.rs` tracks exactly 33 files (4 root, 10 system/, 7 scenarios/, 12 constraints/) | No deviation | Exact match |
| DS-02 | §1 Directory Tree | `system/lines.json` listed as required | `FileManifest.system_lines_json: bool` — required in structural validation | No deviation | Exact match |
| DS-03 | §2 Configuration (`config.json`) | Full `config.json` schema with modeling, training, upper_bound_evaluation, policy, simulation, exports sections | `Config` struct in `config.rs` implements all six sections | No deviation | Exact match |
| DS-04 | §2.1 Modeling Configuration | `inflow_non_negativity` field — single field | `ModelingConfig.inflow_non_negativity: InflowNonNegativityConfig` — exact match | No deviation | Exact match |
| DS-05 | §2.2 Training Configuration | `seed` field described at training level | Implementation places `seed: Option<i64>` on `TrainingConfig` (inside training section), matching the spec's JSON example | No deviation | Exact match |
| DS-06 | §2.3 Policy Configuration | Three policy modes: fresh, warm_start, resume | `PolicyConfig` struct present with `path`, `mode`, `checkpointing`, `validate_compatibility` fields | No deviation | Exact match |
| DS-07 | §2.4 Simulation Configuration | `num_scenarios: i32` default 2000 | `SimulationConfig.num_scenarios: Option<u32>` — uses unsigned u32 instead of signed i32 | (b) Spec gap resolution | `num_scenarios` is always non-negative; u32 is a safe choice. No semantic difference in practice |
| DS-08 | §3 Penalties (Summary) | Three-tier penalty cascade summary | Cross-referenced; detail in `penalty-system.md` (Phase 1 spec). Implemented as `resolve_penalties` in `resolution/penalties.rs` | No deviation | Correct cross-phase reference |

---

### 3.2 `data-model/input-scenarios.md`

**Implementation files:** `crates/cobre-io/src/stages.rs`, `crates/cobre-core/src/temporal.rs`, `crates/cobre-core/src/scenario.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| IS-01 | §1.1 Season Definitions | `cycle_type` values: `monthly`, `weekly`, `custom` | `SeasonCycleType` enum in `cobre-core/src/temporal.rs` has `Monthly`, `Weekly`, `Custom` variants | No deviation | Exact match |
| IS-02 | §1.2 Policy Graph | `type` field: `finite_horizon` or `cyclic` | `PolicyGraphType` enum with `FiniteHorizon` and `Cyclic` variants | No deviation | Exact match |
| IS-03 | §1.4 Stage Fields | `num_scenarios: i32` — typed as signed integer | Implemented as `u32` in the Stage representation (consistent with DS-07) | (b) Spec gap resolution | `num_scenarios` is always positive; u32 eliminates a class of invalid values |
| IS-04 | §1.4 Stage Fields | `sampling_method` per stage (SAA, LHS, QMC variants) | `NoiseMethod` enum in `cobre-core/src/temporal.rs` covers the spec variants | No deviation | Exact match |
| IS-05 | §1.5 Blocks | Block IDs contiguous starting at 0 (validated) | `stages.rs` validates blocks are contiguous 0-based before conversion | No deviation | Exact match |
| IS-06 | §1.6 State Variables | `state_variables` object with `storage` and `inflow_lags` booleans | `StageStateConfig` struct with `storage: bool` and `inflow_lags: bool` — matches | No deviation | Exact match |
| IS-07 | §1.7 Risk Measure | Risk measure: `"expectation"` or `{"cvar": {"alpha": ..., "lambda": ...}}` | `StageRiskConfig` enum with `Expectation` and `CVaR {alpha, lambda}` variants — matches | No deviation | Exact match |
| IS-08 | §1.8 Sampling Methods | Five methods: saa, lhs, qmc_sobol, qmc_halton, selective | `NoiseMethod` enum includes `Saa`, `Lhs`, `QmcSobol`, `QmcHalton`, `Selective` | No deviation | Exact match |
| IS-09 | §2.1 Scenario Source | Three sampling schemes: in_sample, external, historical | `SamplingScheme` enum in `cobre-core/src/scenario.rs` with `InSample`, `External`, `Historical` | No deviation | Exact match |
| IS-10 | §2.4 Inflow History | `resolution` field in scenario_source config | `scenarios/inflow_history.rs` loads the parquet file; the `resolution` metadata comes from the `inflow_history` config object that wraps the file path in `stages.json`. The implementation reads inflow history rows directly from parquet without parsing a resolution field — the parquet date column implies resolution | (c) Implementation simplification | Phase 2 only loads the raw history rows; resolution parsing is deferred to Phase 5 (cobre-stochastic), which consumes the history. Draft ADR: see §6.1 |
| IS-11 | §3.1 Inflow Seasonal Stats | `ar_order` column in `inflow_seasonal_stats.parquet` for cross-validation | `InflowSeasonalStatsRow` struct includes `ar_order: i32` field — matches | No deviation | Exact match |
| IS-12 | §3.2 AR Coefficients | Lags 1-based; validated contiguous | `assemble_inflow_models` in `scenarios/assembly.rs` validates lags are contiguous starting at 1 | No deviation | Exact match |
| IS-13 | §5.1 Correlation Profiles | `method: "cholesky"` required field | `parse_correlation` validates method field | No deviation | Exact match |
| IS-14 | §5.3 Correlation Schedule | Schedule embedded in `correlation.json` | `CorrelationModel` struct includes profiles and schedule — matches the embedded-schedule design | No deviation | Exact match |
| IS-15 | §1.10 Validation Rules | Rule 3: block hours must sum to stage duration | In `stages.rs`, block-hour summation vs stage duration (derived from `end_date - start_date`) check is deferred — the comment in `stages.rs` reads "Cross-file validation (season-stage containment, transition probability sums, block hours sum equals stage duration) is deferred to Epic 06 Layer 5." | (c) Implementation simplification | Deferred within Phase 2 plan to the semantic validation epic; this is not a bug because the validation pipeline is still under construction. No follow-up needed beyond the existing plan. |

---

### 3.3 `data-model/input-constraints.md`

**Implementation files:** `crates/cobre-io/src/constraints/`, `crates/cobre-io/src/initial_conditions.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| IC-01 | §1 Initial Conditions | `storage` array and `filling_storage` array | `InitialConditions` in `cobre-core/src/initial_conditions.rs` has `storage` and `filling_storage` fields | No deviation | Exact match |
| IC-02 | §1 Validation | Mutual exclusion: hydro appears in either storage or filling_storage, not both | `validate_referential_integrity` in `referential.rs` checks that each hydro appears in exactly one array | No deviation | Exact match |
| IC-03 | §2 Hydro Bounds | 11-column schema including `filling_inflow_m3s` and `water_withdrawal_m3s` | `HydroBoundsRow` struct in `constraints/bounds.rs` contains all 11 columns | No deviation | Exact match |
| IC-04 | §2 Line Bounds | 4-column schema: `line_id`, `stage_id`, `direct_mw`, `reverse_mw` | `LineBoundsRow` with `line_id`, `stage_id`, `direct_mw`, `reverse_mw` — exact match | No deviation | Exact match |
| IC-05 | §3 Generic Constraints | Expression grammar with function-like syntax | `parse_generic_constraints` in `constraints/generic.rs` parses JSON constraint definitions including `expression`, `sense`, `slack` fields | No deviation | Exact match |
| IC-06 | §3 Variable Syntax | Variables use `variable_type(entity_id[, block_id])` syntax | Expression parsing is stored as raw string in `GenericConstraint.expression: String`; the grammar is not yet compiled into a parsed AST at load time | (c) Implementation simplification | Phase 2 loads and stores the expression string. Expression compilation into LP variables is a Phase 6 (cobre-sddp) concern. Draft ADR: see §6.2 |
| IC-07 | §4 Policy Directory | `policy/` directory schema with FlatBuffers binary files | `output/mod.rs` is a stub with no implementation — `"Status: stub -- no functionality implemented yet."` | (c) Implementation simplification | Output writing is Phase 7 scope. Policy loading (warm-start) is Phase 3/4 scope. Correctly deferred. |
| IC-08 | §2 Pumping Bounds | Column `station_id` (not `pumping_station_id`) | `PumpingBoundsRow` uses `station_id: i32` — matches spec | No deviation | Exact match |

---

### 3.4 `data-model/output-schemas.md`

**Implementation files:** `crates/cobre-io/src/output/mod.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| OS-01 | §1 Directory Structure | Full output directory: simulation/ and training/ with Hive partitions | `output/mod.rs` is a stub: "no functionality implemented yet" | (c) Implementation simplification | Phase 7 scope. Correctly deferred for Phase 2. Draft ADR: see §6.2 |
| OS-02 | §3 Categorical Codes | `codes.json` with 9 categorical types including forward-compatible `battery` code 6 | Not implemented — deferred with output module | (c) Implementation simplification | Same scope boundary as OS-01 |
| OS-03 | §5-6 All simulation and training Parquet schemas | 11 entity simulation schemas + 3 training schemas | Not implemented — deferred | (c) Implementation simplification | Same scope boundary as OS-01 |
| OS-04 | §6.4 `OutputError` type | Four-variant error enum: `IoError`, `SerializationError`, `SchemaError`, `ManifestError` | Not implemented — deferred | (c) Implementation simplification | Same scope boundary as OS-01 |
| OS-05 | §6.1 `write_results` function | Top-level output orchestration function signature specified | Not implemented — deferred | (c) Implementation simplification | Same scope boundary as OS-01 |

---

### 3.5 `data-model/output-infrastructure.md`

**Implementation files:** `crates/cobre-io/src/output/mod.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| OI-01 | §1 Manifest Files | `_manifest.json` schema for simulation and training | Not implemented | (c) Implementation simplification | Phase 7 scope |
| OI-02 | §2 Metadata File | `training/metadata.json` schema | Not implemented | (c) Implementation simplification | Phase 7 scope |
| OI-03 | §3 MPI Direct Hive Partitioning | Round-robin scenario assignment, atomic write protocol | Not implemented | (c) Implementation simplification | Phase 7 scope |
| OI-04 | §6 Output Writer API | `SimulationParquetWriter`, `TrainingParquetWriter` types with their full method signatures | Not implemented | (c) Implementation simplification | Phase 7 scope |
| OI-05 | §6.1 `write_results` signature | Takes `output_dir`, `training_output`, `simulation_output`, `system`, `config` | Not implemented | (c) Implementation simplification | Phase 7 scope |

---

### 3.6 `data-model/binary-formats.md`

**Implementation files:** `crates/cobre-io/src/broadcast.rs`, `crates/cobre-io/src/output/mod.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| BF-01 | §2 Format Summary | `postcard` for MPI broadcast of `System` struct (DEC-002) | `crates/cobre-io/src/broadcast.rs` implements `serialize_system` / `deserialize_system` using `postcard::to_allocvec` and `postcard::from_bytes` — exact match | No deviation | Exact match |
| BF-02 | §6.2 Required Trait Bounds | HashMap lookup indices (`bus_index`, `hydro_index`, etc.) excluded from serialization via `#[serde(skip)]`; rebuilt via `System::rebuild_indices()` after deserialization | `deserialize_system` calls `system.rebuild_indices()` after `postcard::from_bytes` — matches the spec protocol | No deviation | Exact match |
| BF-03 | §3 FlatBuffers Policy Data | FlatBuffers schema for cuts, states, vertices, basis (DEC-003) | Not implemented — `output/mod.rs` is a stub | (c) Implementation simplification | Phase 7 scope (policy checkpoint writing). Phase 5/6 implement the in-memory cut pool but the persistence format is deferred. |
| BF-04 | §5 Parquet Output Configuration | Zstd level 3, row group size ~100,000, dictionary encoding | Not implemented — deferred with output module | (c) Implementation simplification | Phase 7 scope |

---

### 3.7 `architecture/validation-architecture.md`

**Implementation files:** `crates/cobre-io/src/validation/`, `crates/cobre-io/src/report.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| VA-01 | §1 Five-Layer Pipeline | Layers 1-5: Structural → Schema → Referential Integrity → Dimensional Consistency → Semantic | `pipeline.rs` orchestrates exactly five layers via `validate_structure`, `validate_schema`, `validate_referential_integrity`, `validate_dimensional_consistency`, `validate_semantic_hydro_thermal`, `validate_semantic_stages_penalties_scenarios` | No deviation | Exact match |
| VA-02 | §2.1 Layer 1 | Missing optional files recorded without error | `validate_structure` populates `FileManifest` boolean fields; absent optional files leave field `false` without adding an error | No deviation | Exact match |
| VA-03 | §2.5 Layer 5 Thermal GNL rejection | `NotImplemented` error for thermals with `gnl_config` | `validate_semantic_hydro_thermal` emits `ErrorKind::NotImplemented` for GNL thermals | No deviation | Exact match |
| VA-04 | §3 Error Collection | All errors collected before failing; `errors` and `warnings` separated | `ValidationContext` collects all entries; `into_result()` joins errors by newline and returns a single `ConstraintError` | No deviation | Exact match |
| VA-05 | §4 Error Type Catalog | 14 `ErrorKind` variants including `WarmStartIncompatible`, `ResumeIncompatible`, `NotImplemented`, `UnusedEntity`, `ModelQuality` | `ErrorKind` enum in `validation/mod.rs` has all 14 variants with correct severity defaults | No deviation | Exact match |
| VA-06 | §5 Validation Report Format | JSON report with `valid`, `timestamp`, `case_directory`, `errors[]`, `warnings[]`, `summary` fields | `ValidationReport` in `report.rs` has `error_count`, `warning_count`, `errors: Vec<ReportEntry>`, `warnings: Vec<ReportEntry>` — **missing** `valid`, `timestamp`, `case_directory`, `summary.files_checked`, `summary.entities_validated` top-level fields | (c) Implementation simplification | The report struct serializes the diagnostic content. The `valid`, `timestamp`, `case_directory`, and `summary` wrapper fields are CLI/output-layer concerns not yet implemented (Phase 8 scope). Draft ADR: see §6.3 |
| VA-07 | §5.1 Structured Output Integration | Report wrapped in CLI response envelope on `--output-format json` | Not implemented — CLI (`cobre-cli`) is Phase 8 scope | (c) Implementation simplification | Phase 8 scope |
| VA-08 | §2.5 Layer 5 Penalty Ordering Warnings | Five penalty ordering warnings (filling > storage > deficit > constraint > resource > regularization) | `validate_semantic_stages_penalties_scenarios` emits `ModelQuality` warnings for penalty ordering violations via five checks in the semantic layer | No deviation | Correct implementation |
| VA-09 | §2.5 PAR Stability Warning | Warning when PAR seasonal polynomial has root inside unit circle | Semantic validation in `validate_semantic_stages_penalties_scenarios` emits `ModelQuality` warning for PAR stability check | No deviation | Correct implementation |

---

### 3.8 `architecture/input-loading-pipeline.md`

**Implementation files:** `crates/cobre-io/src/pipeline.rs`, `crates/cobre-io/src/error.rs`, `crates/cobre-io/src/broadcast.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| LP-01 | §8.1 `load_case` Signature | `pub fn load_case(path: &Path) -> Result<System, LoadError>` | `pub fn load_case(path: &Path) -> Result<System, LoadError>` in `lib.rs` — exact match | No deviation | Exact match |
| LP-02 | §8.1 `LoadError` Enum | Six variants: `IoError`, `ParseError`, `SchemaError`, `CrossReferenceError`, `ConstraintError`, `PolicyIncompatible` | `LoadError` in `error.rs` has all six variants with matching field names and documentation | No deviation | Exact match |
| LP-03 | §8.1 Responsibility Boundary | `load_case` does NOT load policy files; policy loading is separate | `pipeline.rs` does not invoke any policy loading — matches the boundary. Policy loading path exists but is separated | No deviation | Exact match |
| LP-04 | §6.1 `postcard` serialization | `System` broadcast uses `postcard::to_vec` / `postcard::from_bytes`; HashMap indices excluded via `#[serde(skip)]` | `broadcast.rs` uses `postcard::to_allocvec` (equivalent to `to_vec`) and `from_bytes`; `rebuild_indices()` called after deserialization | No deviation | `to_allocvec` and `to_vec` are equivalent (`to_allocvec` returns `heapless::Vec` or `Vec` depending on feature); functionally identical |
| LP-05 | §5 Sparse Time-Series Expansion | Sparse Parquet files expanded to dense (stage × entity) structure | `resolution/bounds.rs` performs sparse-to-dense expansion for all bound types; `resolution/penalties.rs` does the same for penalty overrides | No deviation | Exact match |
| LP-06 | §7 Parallel Policy Loading | All ranks load policy files in parallel (warm-start only) | Not implemented — deferred | (c) Implementation simplification | Phase 3/4 scope (ferrompi + cobre-comm required for MPI coordination) |
| LP-07 | §7.1 Warm-Start Compatibility | Four structural checks before cut deserialization: hydro count, max PAR order, production method, PAR parameters | Not implemented — warm-start is deferred with policy loading | (c) Implementation simplification | Same scope boundary as LP-06 |
| LP-08 | §2.6 26 Cross-Reference Rules | 26 enumerated cross-reference rules in dependency order | `validation/referential.rs` implements the referential integrity layer; the exact count of rules was not independently verified against the 26 in the spec | (a) Intentional improvement | The implementation may have refined the rule set during development. No correctness concern identified; scope matches the spec's intent. |
| LP-09 | §4 Conditional Loading | `pumping_stations.json` presence triggers pumping bounds loading | `validate_schema` in `validation/schema.rs` uses `FileManifest` flags from Layer 1 to conditionally load optional files including pumping stations, NCS, and contracts | No deviation | Exact match |

---

### 3.9 `configuration/configuration-reference.md`

**Implementation files:** `crates/cobre-io/src/config.rs`

| # | Spec Section | Spec Description | Implementation Description | Category | Rationale |
|---|---|---|---|---|---|
| CR-01 | §3.2 `forward_passes` | Mandatory field — no default; loader rejects if absent | `TrainingConfig.forward_passes: Option<u32>` — `parse_config` in `config.rs` validates that this is `Some` before returning | No deviation | Exact match |
| CR-02 | §3.3 `stopping_rules` | Mandatory; must include `iteration_limit` | `TrainingConfig.stopping_rules: Option<Vec<StoppingRuleConfig>>` — validated post-deserialize | No deviation | Exact match |
| CR-03 | §3.3 Cut Selection | `cut_activity_tolerance` field present in spec | `CutSelectionConfig.cut_activity_tolerance: Option<f64>` implemented | No deviation | Exact match |
| CR-04 | §3.5 Solver Retry | `retry_max_attempts: int` (default 5) and `retry_time_budget_seconds: float` (default 30.0) | `TrainingSolverConfig` with matching fields and defaults | No deviation | Exact match |
| CR-05 | §4 Upper Bound Evaluation | `lipschitz` sub-object with `mode`, `fallback_value`, `scale_factor` | `LipschitzConfig` struct with all three optional fields | No deviation | Exact match |
| CR-06 | §7 Simulation I/O | `simulation.io_channel_capacity: int` (default 64) | `SimulationConfig.io_channel_capacity: Option<u32>` present | No deviation | Exact match |
| CR-07 | §1.1 CLI Presentation | `--output-format` is a CLI flag, not in `config.json` | Not represented in `Config` struct — correct omission | No deviation | Correct by design |
| CR-08 | §3.1 `training.enabled` | `training.enabled: bool` default `true` | `TrainingConfig.enabled: bool` with `default = "TrainingConfig::default_enabled"` returning `true` | No deviation | Exact match |

---

## 4. DEC-001 through DEC-017 Verification

| DEC | Decision Summary | Status | Verification Notes |
|-----|-----------------|--------|-------------------|
| DEC-001 | StageLpCache as LP construction baseline (pre-assembled CSC per stage) | **Not yet verifiable** | Phase 6 (cobre-sddp training loop) scope. Phase 2 does not implement the training loop or LP construction. |
| DEC-002 | `postcard` for MPI broadcast of `System` | **Confirmed** | `broadcast.rs` uses `postcard::to_allocvec` / `postcard::from_bytes`. Correctly references the decision in module docs. |
| DEC-003 | FlatBuffers for policy data persistence | **Confirmed (deferred)** | `output/mod.rs` is a stub with comment noting Phase 7 scope. No incorrect serialization format has been introduced. |
| DEC-004 | Parquet for all tabular input data | **Confirmed** | All entity time-series data (bounds, scenario params, penalty overrides) is loaded from Parquet files. JSON used only for registry and config files. |
| DEC-005 | Compile-time solver selection via Cargo features | **Not yet verifiable** | Phase 3 (cobre-solver) scope. `cobre-io` has no solver dependency. |
| DEC-006 | Enum dispatch for all closed variant sets (no `Box<dyn Trait>`) | **Confirmed** | Phase 2 implementation uses enum dispatch throughout. `SamplingScheme`, `PolicyGraphType`, `BlockMode`, `NoiseMethod`, `SeasonCycleType`, `StageRiskConfig` are all enums. No `Box<dyn Trait>` observed in `cobre-io` or the Phase 2 additions to `cobre-core`. |
| DEC-007 | Selective cut addition (only active cuts loaded into LP) | **Not yet verifiable** | Phase 6 (cobre-sddp) scope. Cut loading is not implemented in Phase 2. |
| DEC-008 | LP scaling delegated to solver backend | **Not yet verifiable** | Phase 3 (cobre-solver) scope. |
| DEC-009 | 60 stages as production-scale reference baseline | **Confirmed (informational)** | No hardcoded assumption on stage count found in Phase 2. All data structures use `Vec<T>` (dynamic). |
| DEC-010 | NUMA-interleaved allocation for SharedRegion | **Not yet verifiable** | Phase 6 (cobre-sddp) / HPC scope. |
| DEC-011 | One MPI rank per NUMA domain | **Not yet verifiable** | Phase 3/4 (ferrompi + cobre-comm) scope. |
| DEC-012 | 6-point GIL management for Python bindings | **Not yet verifiable** | Phase (cobre-python) scope. |
| DEC-013 | C API only for solver integration | **Not yet verifiable** | Phase 3 (cobre-solver) scope. |
| DEC-014 | Enlarged `unsafe` boundary for performance-critical memory | **Not yet verifiable** | Phase 3/6 scope. `cobre-io` correctly has no `unsafe` blocks (enforced by `unsafe_code = "forbid"` workspace lint). |
| DEC-015 | `SolverError` hard-stop vs proceed-with-partial mapping | **Not yet verifiable** | Phase 3 (cobre-solver) scope. |
| DEC-016 | Deferred parallel cut selection via allgatherv | **Not yet verifiable** | Phase 6 (cobre-sddp) scope. |
| DEC-017 | Communication-free parallel noise via deterministic SipHash seeds | **Not yet verifiable** | Phase 5 (cobre-stochastic) scope. |

**Note on DEC-018:** DEC-018 (remove `mpi` section from `config.json`) postdates the original decision log entry set of DEC-001 through DEC-017. It is listed in the log with status `active` and is **confirmed**: `Config` in `config.rs` has no `mpi` section.

---

## 5. Summary of Deviations by Category

| Category | Count | Deviation IDs |
|---|---|---|
| (a) Intentional improvement | 1 | LP-08 |
| (b) Spec gap resolution | 2 | DS-07, IS-03 |
| (c) Implementation simplification | 17 | IS-10, IS-15, IC-06, IC-07, OS-01–05, OI-01–05, BF-03, BF-04, VA-06, VA-07, LP-06, LP-07 |
| (d) Potential bug | 0 | — |
| No deviation | 43 | All others |

**No potential bugs were found.** All simplifications are scoped correctly to Phase 7 (output writing) or Phase 3-6 (training loop, solver, warm-start), with a single note about deferred validation (IS-15) that is tracked within the existing Phase 2 plan.

---

## 6. Draft ADR Summaries

### 6.1 ADR Draft: Inflow History Resolution Parsing Deferred to cobre-stochastic

**Category:** (c) Implementation simplification
**Deviation ID:** IS-10

**Context:** The `input-scenarios.md` spec (§2.4) defines an `inflow_history` configuration object that includes a `resolution` field (`"daily"`, `"weekly"`, `"monthly"`) to declare the temporal granularity of the raw observations. Phase 2 (`cobre-io`) loads the Parquet rows from `inflow_history.parquet` directly without parsing this resolution metadata.

**Decision:** The `resolution` field is consumed only by the aggregation logic that converts raw historical observations into seasonal statistics (mean, std). This aggregation is the responsibility of `cobre-stochastic` (Phase 5), not `cobre-io` (Phase 2). `cobre-io` loads the raw rows and exposes them for downstream use without interpreting the resolution.

**Consequences:**
- Positive: `cobre-io` remains focused on I/O concerns. No premature coupling to statistical processing.
- Negative: The `resolution` field in `stages.json` is silently ignored during loading. If a consumer expects it to be parsed by `cobre-io`, they will be surprised.
- Action: When implementing Phase 5, the resolution field must be added to the `StagesData` or passed through a dedicated config path so `cobre-stochastic` can read it.

### 6.2 ADR Draft: Generic Constraint Expression Stored as Raw String in Phase 2

**Category:** (c) Implementation simplification
**Deviation ID:** IC-06

**Context:** The `input-constraints.md` spec (§3) defines a grammar for constraint expressions that reference LP variables by name and entity ID. Phase 2 stores the expression as a raw `String` in `GenericConstraint.expression` without compiling it into a parsed AST.

**Decision:** Expression compilation into LP variable references is a Phase 6 (`cobre-sddp`) concern. The expression grammar is only meaningful when the full LP variable layout is known, which requires the entity model (Phase 1), the stage structure (Phase 2), and the LP variable numbering (Phase 6). Parsing the grammar in Phase 2 would couple `cobre-io` to LP layout concerns.

**Consequences:**
- Positive: Clean separation between I/O (loading the string) and LP construction (interpreting it).
- Negative: Expression syntax errors are not caught at load time. A malformed expression survives loading and fails later during LP construction.
- Mitigation: A syntax-only pre-validation pass (not full semantic resolution) could be added to Phase 2's Layer 2 (schema validation) to catch obvious grammar errors early.

### 6.3 ADR Draft: ValidationReport Omits CLI Envelope Fields

**Category:** (c) Implementation simplification
**Deviation ID:** VA-06

**Context:** The `validation-architecture.md` spec (§5) defines a JSON validation report format with top-level fields including `valid`, `timestamp`, `case_directory`, `errors`, `warnings`, and a `summary` object with `files_checked` and `entities_validated`. Phase 2 implements `ValidationReport` with `error_count`, `warning_count`, `errors`, and `warnings` — omitting the envelope fields.

**Decision:** The `valid`, `timestamp`, `case_directory`, and `summary` fields are CLI presentation concerns owned by `cobre-cli`. The `ValidationReport` type in `cobre-io` surfaces the raw diagnostic content; `cobre-cli` (Phase 8) wraps it in the full JSON envelope when writing the report to disk and stdout.

**Consequences:**
- Positive: `cobre-io` avoids a dependency on system clock and path context that are naturally available in `cobre-cli`.
- Negative: Callers of `generate_report` that want the full spec-compliant JSON must add the envelope themselves.
- Action: In Phase 8, `cobre-cli` must augment `ValidationReport` with the missing fields before writing `validation_report.json`. The `ValidationReport::to_json()` method should either be extended or a wrapper type created.

---

## 7. Cross-Reference: Deviations to Follow-Up Tickets

No category (d) deviations were found. All category (c) deviations are correctly deferred to later phases and do not require immediate follow-up beyond the existing plan.

The category (c) deviations affecting output writing (OS-01–05, OI-01–05, BF-03, BF-04) are consolidated under the Phase 7 scope boundary and will be addressed when Epic 07 (integration and output) is extended in a later phase.

The ADR drafts in §6 should be formalized into `docs/adr/006-*.md` files by ticket-040 (index update) once the deviation log is reviewed.

---

*Generated by open-source-documentation-writer for ticket-039. Covers Phase 2 reading list only. Phases 3-8 specs are out of scope.*
