## INGEST ANCHOR PROBE — cli-python (2026-09, baseline)

Register-shaped stub: one block per candidate of the four merged lens files, every anchor rendered so
`check-anchors.py "INGEST ANCHOR PROBE — cli-python (2026-09, baseline)" --register <this file> --baseline 077dbe2c` resolves it with the register's own parser.
Ingest ref `<subSurface>-<lens>-<nn>` (nn = zero-based index among the lens file's candidates of that sub-surface).

**CD-900 · probe · S6a-architecture-00**

The run lifecycle is hand-mirrored across both L4 entry points with no Engine seam: two three-branch phase plans, each reading enablement from a different source

- **Anchors:** `crates/cobre-cli/src/commands/run/mod.rs::execute_inner` `crates/cobre-python/src/run.rs::run_via_study`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-901 · probe · S6a-architecture-01**

The four simulation writes are emitted in exactly reversed order by the two front ends, and the manifest writer is first on one side and last on the other

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::write_simulation_outputs` `crates/cobre-python/src/run.rs::run_simulation_phase_py`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-902 · probe · S6a-architecture-02**

The training write set has no common composition unit: nine inline writes on the CLI against a helper plus five if-any helpers in the bindings, with the results writer at a different position and OutputContext owned by a different side

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs` `crates/cobre-python/src/run.rs::write_training_artifacts`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-903 · probe · S6a-architecture-03**

The positional carrier is nested: a six-slot LoadedCase alias feeds the ten-slot destructure, so one new loaded artifact edits four positional sites

- **Anchors:** `crates/cobre-cli/src/commands/run/setup.rs::LoadedCase` `crates/cobre-cli/src/commands/run/setup.rs::load_case_and_config` `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-904 · probe · S6a-architecture-04**

The CLI run path reaches the engine's parameter projection only through the MPI wire type, unconditionally, and both front ends then enter the engine through a constructor named after that transport

- **Anchors:** `crates/cobre-cli/src/commands/run/setup.rs::load_case_and_config` `crates/cobre-cli/src/commands/run/setup.rs::build_study_setup` `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig` `crates/cobre-python/src/run.rs::build_study_setup`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-905 · probe · S6a-architecture-05**

Training solver-stats aggregation is a third hand-mirror outside CD-025's anchor list: two otherwise identical folds differing only by the rank filter, with a bit-for-bit contract living as prose in one copy

- **Anchors:** `crates/cobre-cli/src/commands/run/training.rs::aggregate_solver_stats` `crates/cobre-python/src/run.rs::aggregate_training_solve_stats`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-906 · probe · S6a-architecture-06**

The mirror's CI gate has leaked into call-site syntax: two writers are called fully qualified so the checker can see them, a reason the pin's import-resolving checker no longer has

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-907 · probe · S6b-architecture-00**

cobre validate --json emits no error object for any boundary-phase failure, because the boundary error formatter is human-first while the prep-phase formatter is render-agnostic

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::run_boundary_check` `crates/cobre-cli/src/commands/validate.rs::format_boundary_error` `crates/cobre-cli/src/commands/validate.rs::emit_validate_json`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-908 · probe · S6b-architecture-01**

The validate --json error contract and cobre.io.validate's error dict disagree on kind and message for the same input, and six CLI failure paths emit no JSON at all

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::execute` `crates/cobre-cli/src/commands/validate.rs::ValidateErrorOutput` `crates/cobre-cli/src/commands/validate.rs::ValidateBoundaryOutput`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-909 · probe · S6b-architecture-02**

The pre-solver phase driver is written twice over one engine-owned PrepPhase: the CLI extracts the (kind, message) pair into a helper, the bindings re-inline the identical two lines three times

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::describe_prep_error` `crates/cobre-cli/src/commands/validate.rs::run_prep_phase` `crates/cobre-python/src/io.rs::validate` `crates/cobre-python/src/io.rs::load_error_kind` `crates/cobre-python/src/io.rs::load_validate_config` `crates/cobre-python/src/io.rs::convert_load_error` `crates/cobre-python/src/io.rs::build_warnings_list`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-910 · probe · S6b-architecture-03**

cobre validate re-runs engine setup work it has already done: the boundary checkpoint is read twice per invocation, where the Python mirror reads it once and reuses the value

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::reconcile_boundary` `crates/cobre-cli/src/commands/validate.rs::execute`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-911 · probe · S6b-architecture-04**

Seven doc examples across four CLI modules import crate paths that cannot resolve, and the crate has no lib target so no doctest ever compiles them

- **Anchors:** `crates/cobre-cli/src/templates.rs::find_template` `crates/cobre-cli/src/banner.rs::render_banner_string` `crates/cobre-cli/src/error.rs::CliError` `crates/cobre-cli/src/progress.rs::run_progress_thread`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-912 · probe · S6b-architecture-05**

summary.rs imports three cobre_sddp provenance types into production module scope, with pub and allow(unused_imports), only so its own test block's use super resolves

- **Anchors:** `crates/cobre-cli/src/summary.rs::print_hydro_model_summary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-913 · probe · S6b-architecture-06**

The two L4 classifications of cobre_sddp::SddpError are not jointly owned and only the CLI's is compiler-enforced, so a new engine variant is silently absorbed on the Python side

- **Anchors:** `crates/cobre-cli/src/error.rs::CliError`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-914 · probe · S6c-architecture-00**

The bindings carry no Engine seam: Study and Policy are typed directly on six cobre_sddp types, and two separately registered public entry points reach the one engine with no dispatch between them

- **Anchors:** `crates/cobre-python/src/study.rs::Study` `crates/cobre-python/src/study.rs::Policy` `crates/cobre-python/src/study.rs::new_native` `crates/cobre-python/src/study.rs::train_native` `crates/cobre-python/src/study.rs::simulate_native` `crates/cobre-python/src/lib.rs::run_module` `crates/cobre-python/src/lib.rs::register_submodule`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-915 · probe · S6c-architecture-01**

results.rs re-enumerates the simulation entity-family list as a ten-name constant against the fourteen families cobre-io writes, so the documented load-everything path silently omits four written families

- **Anchors:** `crates/cobre-python/src/results.rs::ENTITY_TYPES` `crates/cobre-python/src/results.rs::load_simulation` `crates/cobre-python/src/results.rs::load_simulation_arrow`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-916 · probe · S6c-architecture-02**

load_convergence hand-lists fourteen of the fifteen columns cobre-io declares for convergence.parquet, and its sibling load_convergence_arrow documents the same fourteen while passing all fifteen through

- **Anchors:** `crates/cobre-python/src/results.rs::load_convergence` `crates/cobre-python/src/results.rs::load_convergence_arrow`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-917 · probe · S6c-architecture-03**

Both front ends bypass cobre-io's two typed metadata readers in production and read the same files untyped, leaving the L2 reader exercised only by test code

- **Anchors:** `crates/cobre-python/src/results.rs::load_results`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-918 · probe · S6c-architecture-04**

The Python exception class of a failed run is recovered by string-prefix matching on an error message that twenty-five hand-written literals in the run path mint, and five of the eight recognized prefixes are pinned by no test

- **Anchors:** `crates/cobre-python/src/errors.rs::message_prefix_to_pyerr` `crates/cobre-python/src/errors.rs::ErrorSource`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-919 · probe · S6c-architecture-05**

The schema-export write loop is duplicated verbatim between the bindings and the CLI, down to three identical message strings, and only the CLI copy is covered by the schema drift job

- **Anchors:** `crates/cobre-python/src/schema.rs::export`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-920 · probe · S6c-architecture-06**

write_policy_checkpoint in the bindings re-declares the engine's cost-scale default as a bare literal while cobre-sddp exports the named constant the bindings already depend on

- **Anchors:** `crates/cobre-python/src/policy.rs::write_policy_checkpoint`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-921 · probe · S6a-performance-00**

broadcast_value postcard-encodes and discards every payload on rank 0 even when the world has a single rank

- **Anchors:** `crates/cobre-cli/src/commands/broadcast.rs::broadcast_value` `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup` `crates/cobre-cli/src/commands/run/policy.rs::apply_training_policy`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-922 · probe · S6a-performance-01**

Both simulation gathers use allgatherv although only rank 0 consumes the gathered rows

- **Anchors:** `crates/cobre-cli/src/commands/run/simulation.rs::aggregate_simulation_paths` `crates/cobre-cli/src/commands/run/simulation.rs::aggregate_simulation_solver_stats` `crates/cobre-cli/src/commands/run/simulation.rs::write_sim_outputs_on_root`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-923 · probe · S6a-performance-02**

The cobre-io path-row writer takes ownership, so both front ends clone the full simulation path-row set at the write call

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::write_simulation_outputs` `crates/cobre-cli/src/commands/run/simulation.rs::run_simulation_phase` `crates/cobre-python/src/run.rs::run_simulation_phase_py`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-924 · probe · S6a-performance-03**

Both front ends allocate a per-scenario vector only to reorder two tuple fields before the scenario-summary write

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::write_simulation_outputs` `crates/cobre-python/src/run.rs::run_simulation_phase_py`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-925 · probe · S6a-performance-04**

Every per-scenario solver-stats delta is cloned, heap histogram included, only to drop one tuple field before packing

- **Anchors:** `crates/cobre-cli/src/commands/run/simulation.rs::aggregate_simulation_solver_stats`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-926 · probe · S6a-performance-05**

Non-root ranks re-read the case directory and redo the hydro-model fit rank 0 already holds

- **Anchors:** `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup` `crates/cobre-cli/src/commands/run/setup.rs::reconstruct_stochastic_context_non_root`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-927 · probe · S6a-performance-06**

In simulation-only mode every rank reads and rebuilds the whole policy checkpoint, while the sibling boundary-cut path in the same module broadcasts

- **Anchors:** `crates/cobre-cli/src/commands/run/policy.rs::load_policy_for_simulation` `crates/cobre-cli/src/commands/run/policy.rs::apply_training_policy` `crates/cobre-cli/src/commands/run/mod.rs::execute_inner`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-928 · probe · S6a-performance-07**

The serial rank-0 training-output write chain sits on every rank's critical path with nothing overlapping it

- **Anchors:** `crates/cobre-cli/src/commands/run/mod.rs::execute_inner` `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs` `crates/cobre-cli/src/commands/run/simulation.rs::run_simulation_phase`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-929 · probe · S6b-performance-00**

cobre validate resolves the boundary-state requirements twice per invocation, fully deserializing the boundary policy checkpoint on each resolve

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::execute` `crates/cobre-cli/src/commands/validate.rs::reconcile_boundary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-930 · probe · S6b-performance-01**

The progress thread retains every received training event in a vector that the simulation caller discards

- **Anchors:** `crates/cobre-cli/src/progress.rs::run_progress_thread` `crates/cobre-cli/src/progress.rs::ProgressHandle`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-931 · probe · S6c-performance-00**

load_convergence and load_policy hold the GIL across their entire Rust-side decode, while three sibling readers in the same file release it with py.detach

- **Anchors:** `crates/cobre-python/src/results.rs::load_convergence` `crates/cobre-python/src/results.rs::load_policy`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-932 · probe · S6c-performance-01**

read_parquet_partition_into re-runs the Arrow data-type match and the array downcast once per cell, while the sibling load_convergence hoists the identical downcast to once per column per batch

- **Anchors:** `crates/cobre-python/src/results.rs::read_parquet_partition_into` `crates/cobre-python/src/results.rs::arrow_value_to_py`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-933 · probe · S6c-performance-02**

Bulk f64 arrays reach NumPy through a per-element Python list: reshape_f64 and Policy::cut_matrix materialise every coefficient as a Python float that numpy then re-parses

- **Anchors:** `crates/cobre-python/src/results.rs::reshape_f64` `crates/cobre-python/src/results.rs::opening_tree` `crates/cobre-python/src/study.rs::cut_matrix`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-934 · probe · S6c-performance-03**

load_results converts the same training metadata JSON tree into Python twice, once for the manifest key and once for the metadata alias

- **Anchors:** `crates/cobre-python/src/results.rs::load_results` `crates/cobre-python/src/results.rs::json_value_to_py`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-935 · probe · S6c-performance-04**

The seven PySystem entity-list getters deep-clone the whole entity vector on every attribute access, in a file whose from_arc constructor already establishes the share-by-refcount idiom

- **Anchors:** `crates/cobre-python/src/model.rs::buses` `crates/cobre-python/src/model.rs::from_arc` `crates/cobre-python/src/model.rs::deficit_segments`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-936 · probe · S6c-performance-05**

build_stage_cuts_data clones every cut coefficient vector of every stage even on the branch that needs no reservation, and does it under the GIL before the detached write

- **Anchors:** `crates/cobre-python/src/policy.rs::build_stage_cuts_data` `crates/cobre-python/src/policy.rs::StageCutsData`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-937 · probe · S6c-performance-06**

Stochastic::opening_tree scans every opening row of every stage to locate one stage's block, although the reader already rejects unsorted rows and the code comments that the block is contiguous

- **Anchors:** `crates/cobre-python/src/results.rs::opening_tree` `crates/cobre-python/src/results.rs::is_opening_order_sorted`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-938 · probe · S6c-performance-07**

cobre.io.validate and cobre.io.load_case hold the GIL across the whole case load, stochastic preparation and hydro-model preparation, while Study in the same crate detaches at all four of its heavy entry points

- **Anchors:** `crates/cobre-python/src/io.rs::validate` `crates/cobre-python/src/io.rs::load_case`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-939 · probe · S6a-over-engineering-00**

commands/broadcast.rs is a 1,088-line postcard wire-type module filed under commands/, whose own parent doc says every module there is a clap subcommand

- **Anchors:** `crates/cobre-cli/src/commands/broadcast.rs::BroadcastConfig` `crates/cobre-cli/src/commands/run/setup.rs::broadcast_and_build_setup`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-940 · probe · S6a-over-engineering-01**

WriteTrainingArgs carries hydro_models next to setup, and at its one construction site the field is literally &setup.hydro_models

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::WriteTrainingArgs` `crates/cobre-cli/src/commands/run/mod.rs::execute_inner`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-941 · probe · S6a-over-engineering-02**

write_sim_outputs_on_root is a decision-free pass-through layer, and the root guard its name asserts lives at the call site instead

- **Anchors:** `crates/cobre-cli/src/commands/run/simulation.rs::write_sim_outputs_on_root` `crates/cobre-cli/src/commands/run/outputs.rs::write_simulation_outputs`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-942 · probe · S6a-over-engineering-03**

The per-artifact write guards are mirrored across the CLI/Python boundary by four hand-written comments naming the twin symbol, a granularity the name-matching parity gate cannot check

- **Anchors:** `crates/cobre-cli/src/commands/run/outputs.rs::write_training_outputs` `crates/cobre-python/src/run.rs::write_fixed_delivery_if_any` `crates/cobre-python/src/run.rs::write_training_artifacts`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-943 · probe · S6a-over-engineering-04**

Thirteen production #[allow] suppressions in the cell carry no written rationale, including a verbatim-duplicated cast in the two phase files and one whose identical twin in the same file is documented

- **Anchors:** `crates/cobre-cli/src/commands/run/training.rs::run_training_phase` `crates/cobre-cli/src/commands/run/simulation.rs::run_simulation_phase` `crates/cobre-python/src/run.rs::run_via_study`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-944 · probe · S6a-over-engineering-05**

RunSummary is declared pub(crate) but has no consumer outside its own file, unlike the SimSummary declared 14 lines below it

- **Anchors:** `crates/cobre-python/src/run.rs::RunSummary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-945 · probe · S6b-over-engineering-00**

The prep-phase error ladder threads a redundant `json: bool` beside `stdout_sink` while the sibling boundary check never receives it, so the documented fourth validation phase emits no `--json` error object

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::prep_error_to_cli_error` `crates/cobre-cli/src/commands/validate.rs::run_prep_phase` `crates/cobre-cli/src/commands/validate.rs::run_boundary_check` `crates/cobre-cli/src/commands/validate.rs::execute`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-946 · probe · S6b-over-engineering-01**

Four `#[cfg(test)] pub fn format_*_string` renderers duplicate their `print_*` twins line for line, in the same module whose doc prescribes the `*_lines` idiom already used for the other three printers

- **Anchors:** `crates/cobre-cli/src/summary.rs::format_setup_summary_string` `crates/cobre-cli/src/summary.rs::print_setup_summary` `crates/cobre-cli/src/summary.rs::format_hydro_model_summary_string` `crates/cobre-cli/src/summary.rs::format_provenance_summary_string` `crates/cobre-cli/src/summary.rs::format_boundary_summary_string` `crates/cobre-cli/src/summary.rs::training_summary_lines`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-947 · probe · S6b-over-engineering-02**

Three subcommand entry points take their clap args by value and suppress `needless_pass_by_value` without a rationale, while the same `match` binds the run args by reference

- **Anchors:** `crates/cobre-cli/src/commands/schema.rs::execute` `crates/cobre-cli/src/commands/validate.rs::execute` `crates/cobre-cli/src/commands/init.rs::execute` `crates/cobre-cli/src/main.rs::Command`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-948 · probe · S6b-over-engineering-03**

Seven doc examples in the CLI shell are written against two library roots that resolve nowhere, in a crate with no library target and no doctest step anywhere in CI

- **Anchors:** `crates/cobre-cli/src/banner.rs::render_banner_string` `crates/cobre-cli/src/templates.rs::find_template` `crates/cobre-cli/src/progress.rs::run_progress_thread` `crates/cobre-cli/src/error.rs::CliError`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-949 · probe · S6b-over-engineering-04**

Nine production cast suppressions in the summary module carry no `// Rationale:` line, against the mirror's own Load-bearing class definition and the rationalized twin in the sibling progress module

- **Anchors:** `crates/cobre-cli/src/summary.rs::format_split_duration` `crates/cobre-cli/src/summary.rs::time_split_training_walls` `crates/cobre-cli/src/summary.rs::format_time_split_training` `crates/cobre-cli/src/summary.rs::simulation_summary_lines` `crates/cobre-cli/src/progress.rs::fmt_avg_lp_time`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-950 · probe · S6b-over-engineering-05**

The summary module publicly re-exports six engine types under a rationale whose final claim is false, and three of the six are used nowhere outside the module's own test block

- **Anchors:** `crates/cobre-cli/src/summary.rs::print_provenance_summary` `crates/cobre-cli/src/summary.rs::print_hydro_model_summary` `crates/cobre-cli/src/summary.rs::print_boundary_summary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-951 · probe · S6c-over-engineering-00**

Three cobre.model entity twins are self-labelled minimal stubs: full pyclass scaffolding per class surfaces only id and name, and nothing outside a hasattr presence check ever reads them

- **Anchors:** `crates/cobre-python/src/model.rs::PyEnergyContract` `crates/cobre-python/src/model.rs::PyPumpingStation` `crates/cobre-python/src/model.rs::PyNonControllableSource` `crates/cobre-python/src/model.rs::PySystem`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-952 · probe · S6c-over-engineering-01**

Allow-attribute proliferation in the bindings crate: the errors test module re-declares the exact crate-level test allow lib.rs already sets, and 14 of the 17 needless_pass_by_value allows are bare while the single shared PyO3 reason is written out three times

- **Anchors:** `crates/cobre-python/src/errors.rs::tests` `crates/cobre-python/src/results.rs::load_results` `crates/cobre-python/src/schema.rs::export`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-953 · probe · S6a-test-bloat-00**

output_metadata_active_backend.rs and setup_timings_metadata.rs are 53-line twins: two solver-linked binaries run the same d01 CLI training to assert two field groups of the same training/metadata.json

- **Anchors:** `crates/cobre-cli/tests/output_metadata_active_backend.rs::training_metadata_solver_matches_active_backend` `crates/cobre-cli/tests/setup_timings_metadata.rs::training_metadata_carries_well_formed_setup_timings` `crates/cobre-cli/tests/setup_timings_metadata.rs::d01_case_dir`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-954 · probe · S6a-test-bloat-01**

Seven of the nine run-path test binaries hold exactly one #[test], and cobre-cli has no tests/common/: fn cobre() is redefined in 13 of 14 binaries, the repo-root case-dir body in 5 corpus files under 3 names, and a 102-line parquet-reader block is byte-identical across the two anticipated binaries

- **Anchors:** `crates/cobre-cli/tests/cli_run_anticipated.rs::read_thermals_parquet` `crates/cobre-cli/tests/cli_run_anticipated_k2.rs::read_thermals_parquet` `crates/cobre-cli/tests/cli_run_evaporation.rs::d08_case_dir` `crates/cobre-cli/tests/cli_run_generic_echo.rs::case_dir` `crates/cobre-cli/tests/python_parity_check.rs::python_parity_script_passes` `crates/cobre-cli/tests/cli_run.rs::cobre`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-955 · probe · S6a-test-bloat-02**

test_study.py and test_run.py declare no fixtures and re-train or re-run examples/1dtoy per test function, including for pure argument-validation claims, while test_outputs.py in the same directory runs each deck once behind a module-scoped fixture

- **Anchors:** `crates/cobre-python/tests/test_study.py::test_policy_evaluate_stage_out_of_range_raises_indexerror` `crates/cobre-python/tests/test_study.py::test_policy_evaluate_bad_state_length_raises_valueerror` `crates/cobre-python/tests/test_run.py::test_run_1dtoy_succeeds` `crates/cobre-python/tests/test_run.py::test_run_1dtoy_creates_output` `crates/cobre-python/tests/test_outputs.py::run_output` `crates/cobre-python/tests/conftest.py::cli_binary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-956 · probe · S6a-test-bloat-03**

The bindings' inline test module runs seven full run_via_study studies and hand-pins the CLI's 1dtoy numbers, and cli_e2e_run_end_block.rs pins the same mean cost again at different precision in a second crate, while the pytest parity suite derives the CLI/Python agreement live

- **Anchors:** `crates/cobre-python/src/run.rs::python_run_1dtoy_metadata_matches_cli_golden_values` `crates/cobre-python/src/run.rs::copy_dir_all` `crates/cobre-cli/tests/cli_e2e_run_end_block.rs::EXPECTED_MEAN_COST` `crates/cobre-python/tests/test_cli_python_determinism_parity.py::test_cli_python_json_files_match`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-957 · probe · S6a-test-bloat-04**

The pytest parity files each rebuild the same harness: _make_case_with_simulation is declared four times with a byte-identical body, plus a fifth copy inside _seed_case and a sixth in Rust, alongside six one-off parquet collectors, two identical comparators and two _relative_files, while conftest.py and _cobre_cli.py are the existing shared seam

- **Anchors:** `crates/cobre-python/tests/test_contract_output_parity.py::_make_case_with_simulation` `crates/cobre-python/tests/test_anticipated_lanes_output_parity.py::_make_case_with_simulation` `crates/cobre-python/tests/test_anticipated_output_parity.py::_make_case_with_simulation` `crates/cobre-python/tests/test_parity_hydros.py::_make_case_with_simulation` `crates/cobre-python/tests/test_parity_scalar_parameters.py::_seed_case` `crates/cobre-python/tests/test_cli_python_file_set_parity.py::_relative_files` `crates/cobre-python/tests/test_cli_python_determinism_parity.py::_relative_files` `crates/cobre-python/tests/conftest.py::cli_binary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-958 · probe · S6a-test-bloat-05**

test_run_via_study_emits_contract_output asserts a strict subset of test_cli_python_contract_row_parity on the same Python surface and the same derived d41 deck, costing one extra full train-and-simulate

- **Anchors:** `crates/cobre-python/tests/test_contract_output_parity.py::test_run_via_study_emits_contract_output` `crates/cobre-python/tests/test_contract_output_parity.py::test_cli_python_contract_row_parity` `crates/cobre-python/tests/test_contract_output_parity.py::CONTRACT_SCHEMA_FIELDS`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-959 · probe · S6a-test-bloat-06**

run/mod.rs's inline test module is the wrong home for its unit tests: it tests resolve_thread_count, declared in setup.rs which has its own test module, and pins three literal-argument cases of the engine's delta_to_stats_row, which has no unit test in the crate that declares it

- **Anchors:** `crates/cobre-cli/src/commands/run/mod.rs::test_resolve_thread_count_cli_value` `crates/cobre-cli/src/commands/run/mod.rs::test_resolve_thread_count_default` `crates/cobre-cli/src/commands/run/mod.rs::test_delta_to_stats_row_backward_carries_opening_rank_worker` `crates/cobre-cli/src/commands/run/setup.rs::resolve_thread_count`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-960 · probe · S6a-test-bloat-07**

simulation_weighting_census_underivable_from_sampled_traversal is declared in three crates, and both front-end copies test only cobre-sddp node-graph behaviour without touching any CLI or binding code, sharpening CD-025 with a third copy the register's own note does not have

- **Anchors:** `crates/cobre-cli/src/commands/run/simulation.rs::simulation_weighting_census_underivable_from_sampled_traversal` `crates/cobre-cli/src/commands/run/simulation.rs::one_node_graph` `crates/cobre-python/src/run.rs::simulation_weighting_census_underivable_from_sampled_traversal`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-961 · probe · S6b-test-bloat-00**

The seven inline tests of the validate subcommand assert a test-local reimplementation of the report renderer, not the shipped one, and the copy has already drifted

- **Anchors:** `crates/cobre-cli/src/commands/validate.rs::format_report_to_string` `crates/cobre-cli/src/commands/validate.rs::format_entry` `crates/cobre-cli/src/commands/validate.rs::execute` `crates/cobre-cli/src/commands/validate.rs::print_prep_error`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-962 · probe · S6b-test-bloat-01**

Four cfg(test)-gated format_*_string twins in summary.rs let the shipped print_* renderers drift while the tests stay green; the shipped path is asserted only for absence of panic

- **Anchors:** `crates/cobre-cli/src/summary.rs::format_hydro_model_summary_string` `crates/cobre-cli/src/summary.rs::format_setup_summary_string` `crates/cobre-cli/src/summary.rs::format_provenance_summary_string` `crates/cobre-cli/src/summary.rs::format_boundary_summary_string` `crates/cobre-cli/src/summary.rs::print_hydro_model_summary` `crates/cobre-cli/src/summary.rs::training_summary_lines`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-963 · probe · S6b-test-bloat-02**

cobre-cli has no tests/common at all: the binary invocation helper is redefined in 13 of 14 integration binaries and the valid-case fixture in three, against the verbatim text of the standing testing contract

- **Anchors:** `crates/cobre-cli/tests/cli_color.rs::make_valid_case` `crates/cobre-cli/tests/cli_validate.rs::make_valid_case` `crates/cobre-cli/tests/cli_smoke.rs::cobre` `crates/cobre-cli/tests/cli_schema.rs::cobre` `crates/cobre-cli/tests/init.rs::cobre` `crates/cobre-cli/tests/cli_color.rs::write_file`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-964 · probe · S6b-test-bloat-03**

Ten full cobre run trainings execute inside the validate binary and three more inside the color binary, to assert claims the unit tier already covers

- **Anchors:** `crates/cobre-cli/tests/cli_validate.rs::run_case` `crates/cobre-cli/tests/cli_validate.rs::boundary_superset_source_under_strict_exits_nonzero_and_names_the_family` `crates/cobre-cli/tests/cli_color.rs::color_always_flag_forces_ansi_in_banner` `crates/cobre-cli/tests/cli_color.rs::color_never_flag_suppresses_ansi_in_banner` `crates/cobre-cli/src/banner.rs::render_banner_string` `crates/cobre-cli/src/commands/init.rs::execute`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-965 · probe · S6b-test-bloat-04**

Seven executable-marked doctests in cobre-cli never compile or run, and every one names a crate path that does not exist at the pin

- **Anchors:** `crates/cobre-cli/src/banner.rs::print_banner` `crates/cobre-cli/src/templates.rs::find_template` `crates/cobre-cli/src/templates.rs::available_templates` `crates/cobre-cli/src/error.rs::exit_code` `crates/cobre-cli/src/error.rs::CliError` `crates/cobre-cli/src/progress.rs::run_progress_thread`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-966 · probe · S6b-test-bloat-05**

Fifteen of the twenty-two inline progress tests spawn a real renderer thread to re-assert one channel round-trip contract, five of them in cases the widest test already subsumes

- **Anchors:** `crates/cobre-cli/src/progress.rs::run_progress_thread` `crates/cobre-cli/src/progress.rs::ProgressHandle` `crates/cobre-cli/src/progress.rs::RenderMode` `crates/cobre-cli/src/progress.rs::fmt_hms`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-967 · probe · S6b-test-bloat-06**

One assertion per subprocess: several integration tests re-run the same invocation to check a second substring of the same output

- **Anchors:** `crates/cobre-cli/tests/cli_validate.rs::missing_buses_json_exits_1` `crates/cobre-cli/tests/cli_validate.rs::missing_buses_json_stdout_contains_error` `crates/cobre-cli/tests/cli_validate.rs::missing_buses_json_stdout_mentions_file` `crates/cobre-cli/tests/cli_validate.rs::valid_case_exits_0` `crates/cobre-cli/tests/cli_smoke.rs::version_exits_0_and_contains_version_string` `crates/cobre-cli/tests/cli_smoke.rs::version_stdout_contains_active_solver`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-968 · probe · S6b-test-bloat-07**

Six of the seven template tests assert compile-time literals or restate an inert doc example; the one real guard sits next to a frozen file count the sibling schema suite deliberately refuses

- **Anchors:** `crates/cobre-cli/src/templates.rs::tests` `crates/cobre-cli/src/templates.rs::TemplateFile` `crates/cobre-cli/src/templates.rs::DTOY1_FILES` `crates/cobre-cli/tests/cli_schema.rs::test_schema_export_writes_files`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-969 · probe · S6b-test-bloat-08**

Three of the six engine types the summary module re-exports have zero production uses and exist only so its inline test module's imports resolve, with a comment that says so

- **Anchors:** `crates/cobre-cli/src/summary.rs::tests` `crates/cobre-cli/src/summary.rs::format_provenance_summary_string` `crates/cobre-cli/src/summary.rs::print_provenance_summary`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-970 · probe · S6b-test-bloat-09**

fmt_sci is defined byte-identically in two S6b modules and neither copy has a direct test, so a divergence between the two renderers is unasserted

- **Anchors:** `crates/cobre-cli/src/progress.rs::fmt_sci` `crates/cobre-cli/src/summary.rs::fmt_sci` `crates/cobre-cli/src/summary.rs::training_summary_lines`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-971 · probe · S6c-test-bloat-00**

Seven py_to_json_value unit claims are routed through the eleven-phase cobre.io.validate on a premise the crate's own Rust tests falsify

- **Anchors:** `crates/cobre-python/src/convert.rs::py_to_json_value` `crates/cobre-python/src/convert.rs::pydict_to_json_map` `crates/cobre-python/src/errors.rs::tests` `crates/cobre-python/src/io.rs::validate` `crates/cobre-python/tests/test_convert.py::_validate`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-972 · probe · S6c-test-bloat-01**

Schema export is asserted twice in two languages, with one test name byte-identical across the Rust and pytest suites

- **Anchors:** `crates/cobre-python/src/schema.rs::test_export_writes_all_schemas_as_valid_json` `crates/cobre-python/src/schema.rs::test_export_creates_missing_directory` `crates/cobre-python/src/schema.rs::export` `crates/cobre-python/tests/test_schema.py::test_export_creates_missing_directory`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-973 · probe · S6c-test-bloat-02**

conftest.py owns one fixture, so copy-the-case and find-the-case are each re-implemented per module in two incompatible conventions and borrowed across modules by a private import

- **Anchors:** `crates/cobre-python/src/run.rs::copy_dir_all` `crates/cobre-python/tests/conftest.py::cli_binary` `crates/cobre-python/tests/test_validate.py::copy_case_to_tempdir` `crates/cobre-python/tests/test_policy_load_validation.py::_copy_case_with_renamed_hydro` `crates/cobre-python/tests/test_boundary_load.py::_set_boundary_policy` `crates/cobre-python/tests/test_model.py::test_system_entity_counts`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-974 · probe · S6c-test-bloat-03**

Four redundant full solver runs pay for preconditions a module-scoped fixture already establishes in the same directory

- **Anchors:** `crates/cobre-python/src/study.rs::train` `crates/cobre-python/tests/test_boundary_load.py::test_boundary_load_reports_source_date_and_reconciliation` `crates/cobre-python/tests/test_boundary_load.py::test_boundary_load_strict_accepts_a_faithful_self_boundary` `crates/cobre-python/tests/test_boundary_load.py::test_boundary_load_rejects_a_mismatched_source` `crates/cobre-python/tests/test_results.py::run_output` `crates/cobre-python/tests/test_results.py::test_load_stochastic_missing_artifacts_raises`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-975 · probe · S6c-test-bloat-04**

testing-architecture section 5.11 still asks for the bindings crate's Rust tests to be wired into CI and section 5.2 still lists an unadopted dev-dependency, so the doc understates and misdirects the only invocation that runs them

- **Anchors:** `crates/cobre-python/src/errors.rs::tests` `crates/cobre-python/src/schema.rs::tests` `crates/cobre-python/src/policy.rs::tests`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

