## INGEST ANCHOR PROBE — stochastic (2026-09, baseline)

Register-shaped stub: one block per candidate, every anchor rendered so it resolves at the pinned baseline (symbol declaration, or line where the symbol is a descriptive reference). Keyed by the ingest `<sub>-<lens>-<nn>` scheme.

### par-architecture-00 · Decomposition-engine phase vocabulary in par L1-kernel doc comments (forward/backward passes, training/simulation, LP RHS patching) — paradigm leakage the genericity gate does not catch

**Anchors:** `crates/cobre-stochastic/src/par/precompute.rs::PrecomputedPar` `crates/cobre-stochastic/src/par/precompute.rs::mod precompute (module doc)` `crates/cobre-stochastic/src/par/lag_transition.rs::derive_downstream_par_order` `crates/cobre-stochastic/src/par/lag_transition.rs::precompute_stage_lag_transitions` `crates/cobre-stochastic/src/par/lag_transition.rs::resolve_stage_lag_transition (referenced from UNIFORM_MONTHLY_TRANSITION doc)`

### par-over-engineering-00 · Bare `estimate_ar_coefficients` / `estimate_correlation` / `estimate_seasonal_stats` are pub forwarding shims over their `_with_season_map` twins with zero production consumers (only the crate's own tests + doctests)

**Anchors:** `crates/cobre-stochastic/src/par/fitting/ar_coefficients.rs::estimate_ar_coefficients` `crates/cobre-stochastic/src/par/fitting/correlation.rs::estimate_correlation` `crates/cobre-stochastic/src/par/fitting/seasonal_stats.rs::estimate_seasonal_stats`

### par-over-engineering-01 · PAR validation warning apparatus (single-variant `ParWarning` enum + single-field `ParValidationReport`) is built then discarded at its only production caller

**Anchors:** `crates/cobre-stochastic/src/par/validation.rs::ParWarning` `crates/cobre-stochastic/src/par/validation.rs::ParValidationReport` `crates/cobre-stochastic/src/par/validation.rs::validate_par_parameters`

### par-performance-00 · Per-season correlation matrices recompute every hydro-pair date-set intersection twice and clone the full residual map per (hydro, season)

**Anchors:** `crates/cobre-stochastic/src/par/fitting/correlation.rs::compute_seasonal_matrices` `crates/cobre-stochastic/src/par/fitting/correlation.rs::compute_pearson_correlation_matrix` `crates/cobre-stochastic/src/par/fitting/correlation.rs::compute_hydro_residuals`

### par-performance-01 · PAR-A initial per-hydro AR estimation runs single-threaded while the byte-identical classical path already parallelizes the same per-hydro shape

**Anchors:** `crates/cobre-stochastic/src/par/fitting/estimation.rs::estimate_ar_with_pacf_annual` `crates/cobre-stochastic/src/par/fitting/estimation.rs::estimate_all_hydro_ar_coefficients`

### par-performance-02 · Per-hydro observation vectors are deep-copied out of group_obs on every estimate/reduce pass (2x classical, 4x PAR-A) though the consumers only read them

**Anchors:** `crates/cobre-stochastic/src/par/fitting/estimation.rs::estimate_all_hydro_ar_coefficients` `crates/cobre-stochastic/src/par/fitting/estimation.rs::reduce_entity_orders` `crates/cobre-stochastic/src/par/fitting/estimation.rs::reduce_entity_orders_annual` `crates/cobre-stochastic/src/par/fitting/estimation.rs::estimate_ar_with_pacf_annual`

### par-test-bloat-00 · pop_mean_std and pop_mean_std_ann are identical population mean/std helpers declared twice in the same file (par/fitting/tests.rs)

**Anchors:** `crates/cobre-stochastic/src/par/fitting/tests.rs::pop_mean_std` `crates/cobre-stochastic/src/par/fitting/tests.rs::pop_mean_std_ann`

### par-test-bloat-01 · simulate_two_season_par2 PAR(2) Box-Muller scenario simulator duplicated byte-for-byte across the two par/fitting sibling test files

**Anchors:** `crates/cobre-stochastic/src/par/fitting/tests.rs::simulate_two_season_par2` `crates/cobre-stochastic/src/par/fitting/estimation/tests.rs::simulate_two_season_par2`

### par-test-bloat-02 · Inline/sibling unit-test fixture builders (Stage / InflowModel) re-authored across ~9 par test sites with make_model byte-identical between precompute.rs and evaluate.rs

**Anchors:** `crates/cobre-stochastic/src/par/precompute.rs::make_model` `crates/cobre-stochastic/src/par/evaluate.rs::make_model` `crates/cobre-stochastic/src/par/precompute.rs::make_stage` `crates/cobre-stochastic/src/par/evaluate.rs::make_stage` `crates/cobre-stochastic/src/par/validation.rs::make_model` `crates/cobre-stochastic/src/par/aggregate.rs::make_stage`

### par-test-bloat-03 · Integration-binary fixture prelude (make_bus / make_hydro / identity_correlation) re-authored byte-for-byte across the par integration binaries with no shared tests/common

**Anchors:** `crates/cobre-stochastic/tests/forward_sampler.rs::make_hydro` `crates/cobre-stochastic/tests/forward_sampler.rs::make_bus` `crates/cobre-stochastic/tests/forward_sampler.rs::identity_correlation` `crates/cobre-stochastic/tests/conformance.rs::make_hydro` `crates/cobre-stochastic/tests/conformance.rs::make_bus` `crates/cobre-stochastic/tests/conformance.rs::identity_correlation`

### sampling-architecture-00 · Entity-class identity degrades to a stringly-typed &str at the build_class_sampler seam, where a correctness rule is gated by string equality

**Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs::build_class_sampler` `crates/cobre-stochastic/src/sampling/mod.rs::ClassSamplerParams`

### sampling-architecture-01 · ForwardSamplerConfig is both a store handle and a parameter bag, and is the point where the generation-side and realized-value halves straddle

**Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs::ForwardSamplerConfig` `crates/cobre-stochastic/src/sampling/mod.rs::build_forward_sampler`

### sampling-over-engineering-00 · standardize_external_inflow's #[allow(too_many_arguments)] rationale ('no natural sub-grouping exists') is contradicted by the derived-seed 4-tuple being re-threaded verbatim through four signatures across two crates

**Anchors:** `crates/cobre-stochastic/src/sampling/external.rs::standardize_external_inflow` `crates/cobre-stochastic/src/sampling/eta_inversion.rs::run_eta_inversion` `crates/cobre-stochastic/src/sampling/historical.rs::standardize_historical_windows`

### sampling-over-engineering-01 · ForwardSamplerConfig encodes a per-class scheme x library validity matrix as four independent Option<&Library> fields, so invalid scheme/library pairings are representable and caught only at runtime

**Anchors:** `crates/cobre-stochastic/src/sampling/mod.rs::ForwardSamplerConfig` `crates/cobre-stochastic/src/sampling/mod.rs::build_class_sampler`

### sampling-performance-00 · OutOfSample QmcSobol forward draws re-derive the Sobol direction matrix + scramble params per scenario; the dormant `SobolPrecomputed` hoist is never wired (its `Some` branch is dead in production)

**Anchors:** `crates/cobre-stochastic/src/sampling/class_sampler.rs::ClassSampler::fill` `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated` `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated`

### sampling-performance-01 · OutOfSample LHS forward draws regenerate a full Fisher-Yates permutation of `total_scenarios` per dimension per scenario, giving O(dim * total_scenarios^2) work per stage across the scenario axis

**Anchors:** `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated` `crates/cobre-stochastic/src/sampling/class_sampler.rs::ClassSampler::fill`

### sampling-performance-02 · OutOfSample QmcHalton forward draws rebuild the prime sieve and a `total_scenarios`-sized Vec<Vec<Vec<u32>>> scramble table on every scenario; no precompute hoist exists

**Anchors:** `crates/cobre-stochastic/src/sampling/out_of_sample.rs::fill_uncorrelated` `crates/cobre-stochastic/src/sampling/class_sampler.rs::ClassSampler::fill`

### sampling-test-bloat-00 · Per-file test fixtures re-declared across the sampling inline test modules: make_hydro x3, uniform_tree x3 (byte-identical), monthly_season_map x3 (2 byte-identical), dated_stage x2 (byte-identical), plus ~9 hand-rolled Stage struct literals

**Anchors:** `crates/cobre-stochastic/src/sampling/external.rs::make_hydro` `crates/cobre-stochastic/src/sampling/historical.rs::make_hydro` `crates/cobre-stochastic/src/sampling/mod.rs::make_hydro` `crates/cobre-stochastic/src/sampling/class_sampler.rs::uniform_tree` `crates/cobre-stochastic/src/sampling/mod.rs::uniform_tree` `crates/cobre-stochastic/src/sampling/insample.rs::uniform_tree`

### sampling-test-bloat-01 · saa_golden_value.rs re-declares the integration-binary fixture prelude: identity_correlation is byte-identical to the halton/sobol/lhs copies and make_stage is a fourth hand-rolled Stage builder

**Anchors:** `crates/cobre-stochastic/tests/saa_golden_value.rs::identity_correlation` `crates/cobre-stochastic/tests/saa_golden_value.rs::make_stage`

### seam-architecture-00 · StochasticContext is both a computed store and a config/layout echo: 5 precomputed components + 5 derived-layout scalars + 2 pass-through seeds + 1 metadata field, each exposed via a 1:1 &-getter (the nascent Switchable<T> uncertainty store)

**Anchors:** `crates/cobre-stochastic/src/context.rs::StochasticContext` `crates/cobre-stochastic/src/context.rs::set_solve_order`

### seam-architecture-01 · generation seam ingests realized external-scenario tables: external_ar0_inflow_models/external_derived_load_models synthesize fitted models from System::external_*_scenarios() and re-derive cobre-io's stage-id resolution inside the L1 store constructor module

**Anchors:** `crates/cobre-stochastic/src/context.rs::stage_id_to_index` `crates/cobre-stochastic/src/context.rs::resolve_row_stages` `crates/cobre-stochastic/src/context.rs::external_derived_load_models` `crates/cobre-stochastic/src/context.rs::external_ar0_inflow_models`

### seam-over-engineering-00 · Three StochasticError variants have no production producer — dead public error taxonomy (SpectralDecompositionFailed, SeedDerivationError, UnsupportedSamplingScheme)

**Anchors:** `crates/cobre-stochastic/src/error.rs::StochasticError` `crates/cobre-stochastic/src/error.rs::SpectralDecompositionFailed` `crates/cobre-stochastic/src/error.rs::SeedDerivationError` `crates/cobre-stochastic/src/error.rs::UnsupportedSamplingScheme` `crates/cobre-stochastic/src/context.rs::build_stochastic_context`

### seam-performance-00 · Backward season-occurrence walk is quadratic in l_state: derive_inflow_seeds restarts nth_previous_occurrence from the anchor for every k

**Anchors:** `crates/cobre-stochastic/src/seeds.rs::derive_inflow_seeds` `crates/cobre-stochastic/src/season_cast/mod.rs::season_occurrence` `crates/cobre-stochastic/src/season_cast/mod.rs::nth_previous_occurrence`

### seam-performance-01 · derive_inflow_seeds re-scans the entire record and conditioning slices once per hydro instead of grouping by hydro_id in one pass

**Anchors:** `crates/cobre-stochastic/src/seeds.rs::derive_inflow_seeds`

### seam-test-bloat-00 · Entity-fixture prelude (make_bus/make_stage/make_hydro/make_inflow_model/identity_correlation) duplicated near-verbatim across two inline seam test modules and the seam integration binary

**Anchors:** `crates/cobre-stochastic/src/context.rs::make_hydro` `crates/cobre-stochastic/src/provenance.rs::make_hydro` `crates/cobre-stochastic/tests/reproducibility.rs::make_hydro` `crates/cobre-stochastic/src/seeds.rs::make_hydro`

### seam-test-bloat-01 · Tautological equivalence test: test_season_occurrence_matches_pre_refactor_helper_sequence recomputes the same delegating arithmetic on both sides

**Anchors:** `crates/cobre-stochastic/src/season_cast/mod.rs::test_season_occurrence_matches_pre_refactor_helper_sequence` `crates/cobre-stochastic/src/season_cast/mod.rs::season_occurrence`

### tree-noise-architecture-00 · DecomposedCorrelation ships a superseded full-vector correlation seam (test-only) beside the live per-class seam, and both position-precompute fast paths are unwired in production

**Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs::GroupFactor` `crates/cobre-stochastic/src/correlation/resolve.rs::resolve_positions` `crates/cobre-stochastic/src/correlation/resolve.rs::resolve_class_positions` `crates/cobre-stochastic/src/correlation/resolve.rs::apply_correlation`

### tree-noise-architecture-01 · Entity class (inflow/load/ncs) is a stringly-typed closed set matched by `==` across generate.rs and resolve.rs, with no compiler-checked enum

**Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs::GroupFactor` `crates/cobre-stochastic/src/correlation/resolve.rs::apply_correlation_for_class` `crates/cobre-stochastic/src/tree/generate.rs::generate_opening_tree`

### tree-noise-architecture-02 · Three structurally identical point-sample parameter bundles (LhsPointSpec / HaltonPointSpec / SobolPointSpec) with no shared type

**Anchors:** `crates/cobre-stochastic/src/tree/lhs.rs::LhsPointSpec` `crates/cobre-stochastic/src/tree/qmc_halton/mod.rs::HaltonPointSpec` `crates/cobre-stochastic/src/tree/qmc_sobol/mod.rs::SobolPointSpec`

### tree-noise-over-engineering-00 · SweepDirection is effectively one-valued in production: Ascending is constructed only inside opening_tree.rs's own #[cfg(test)] module; every production caller passes Descending

**Anchors:** `crates/cobre-stochastic/src/tree/opening_tree.rs::SweepDirection` `crates/cobre-stochastic/src/tree/opening_tree.rs::set_solve_order`

### tree-noise-performance-00 · apply_correlation_for_class re-resolves the active profile (HashMap<i32,String> + BTreeMap<String,_> string-keyed lookup) on every one of its three per-opening calls, though the profile is constant across a stage's openings

**Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs::apply_correlation_for_class` `crates/cobre-stochastic/src/tree/generate.rs::generate_opening_tree`

### tree-noise-performance-01 · apply_group_precomputed and apply_group_scan heap-allocate gathered/correlated (and positions) scratch per call for correlation groups larger than MAX_STACK_DIM=64, invoked once per opening from the tree loop

**Anchors:** `crates/cobre-stochastic/src/correlation/resolve.rs::apply_group_precomputed` `crates/cobre-stochastic/src/correlation/resolve.rs::apply_group_scan` `crates/cobre-stochastic/src/tree/generate.rs::generate_opening_tree`

### tree-noise-performance-02 · generate_opening_tree pays a per-opening O(group*class) linear position scan because the per-class precomputed fast path (resolve_class_positions) has zero call sites

**Anchors:** `crates/cobre-stochastic/src/tree/generate.rs::generate_opening_tree` `crates/cobre-stochastic/src/correlation/resolve.rs::apply_group_scan` `crates/cobre-stochastic/src/correlation/resolve.rs::resolve_class_positions`

### tree-noise-test-bloat-00 · Inline-giant vs extracted-sibling unit-test homing coin-flip inside cobre-stochastic: tree/generate.rs carries ~2030 inline test-LOC while par/fitting uses extracted tests.rs siblings

**Anchors:** `crates/cobre-stochastic/src/tree/generate.rs::tests` `crates/cobre-stochastic/src/correlation/resolve.rs::tests` `crates/cobre-stochastic/src/normal/precompute.rs::tests` `crates/cobre-stochastic/src/tree/opening_tree.rs::tests` `crates/cobre-stochastic/src/correlation/spectral.rs::tests`

### tree-noise-test-bloat-01 · Near-verbatim ~230-line fixture prelude copied across the QMC integration binaries (halton/sobol/lhs) with eight byte-identical helpers and no test_support home; the copy is already drifting

**Anchors:** `crates/cobre-stochastic/tests/halton_integration.rs::make_hydro` `crates/cobre-stochastic/tests/sobol_integration.rs::make_hydro` `crates/cobre-stochastic/tests/lhs_integration.rs::make_hydro` `crates/cobre-stochastic/tests/halton_integration.rs::approx_erf` `crates/cobre-stochastic/tests/halton_integration.rs::identity_correlation`
