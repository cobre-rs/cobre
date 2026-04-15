//! Per-class sampler building block for the composite forward sampler.
//!
//! [`ClassSampler`] is a per-entity-class noise source that writes uncorrelated
//! (or pre-correlated, for tree/library variants) noise into a caller-provided
//! buffer. Three instances of this type -- one for inflow, one for load, one
//! for NCS -- are combined by the composite `ForwardSampler` (ticket-027).
//!
//! ## Correlation contract
//!
//! - `InSample`, `Historical`, `External`: noise is already correlated (the
//!   tree and libraries store pre-standardized eta values). No further
//!   correlation step is needed at the composite level for these variants.
//! - `OutOfSample`: noise is independent N(0,1). Spatial correlation is
//!   applied at the composite `ForwardSampler` level, not here.
//!
//! ## Window / scenario selection
//!
//! For `Historical` and `External`, the selection is deterministic given
//! `(iteration, scenario)` and does NOT depend on `stage`. This ensures
//! that the same window or external scenario is used for all stages within
//! a single forward trajectory.

use cobre_core::temporal::NoiseMethod;

use crate::{
    StochasticError,
    noise::seed::derive_forward_seed,
    sampling::{
        ExternalScenarioLibrary, HistoricalScenarioLibrary,
        out_of_sample::{FreshNoiseSpec, fill_uncorrelated},
    },
    tree::opening_tree::OpeningTreeView,
};

use super::insample;

// ---------------------------------------------------------------------------
// ClassSampleRequest
// ---------------------------------------------------------------------------

/// Per-call arguments for [`ClassSampler::fill`].
///
/// Bundles arguments to keep the `fill()` signature within budget.
/// All fields are small integers and the struct is `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct ClassSampleRequest {
    /// Training iteration counter (0-based).
    pub iteration: u32,
    /// Global scenario index (includes MPI offset).
    pub scenario: u32,
    /// Stage domain ID used for seed derivation.
    pub stage: u32,
    /// Stage array index used for tree/method lookup.
    pub stage_idx: usize,
    /// Total scenario count across all ranks (for LHS stratification).
    pub total_scenarios: u32,
    /// Noise group identifier for seed derivation (Pattern C sharing).
    ///
    /// Stages within the same `(season_id, year)` bucket share the same
    /// `noise_group_id` so that their `OutOfSample` noise draws are identical.
    /// Until ticket-005 wires actual group IDs, callers supply `stage.id as u32`
    /// to preserve current per-stage seed behaviour.
    pub noise_group_id: u32,
}

// ---------------------------------------------------------------------------
// ClassSampler
// ---------------------------------------------------------------------------

/// Per-class noise source for one entity class (inflow, load, or NCS).
///
/// Each variant draws noise from a different source. The `fill()` method
/// writes exactly `output.len()` f64 values into the caller-provided buffer.
///
/// Constructed by the composite `ForwardSampler` factory (ticket-027). Reused
/// across all `(iteration, scenario, stage)` calls without per-call allocation.
pub enum ClassSampler<'a> {
    /// In-sample scheme: copies a segment from the pre-generated opening tree.
    InSample {
        /// View into the pre-generated opening tree.
        tree: OpeningTreeView<'a>,
        /// Base seed for deterministic opening selection.
        base_seed: u64,
        /// Start offset within the full noise vector for this class.
        offset: usize,
        /// Number of entities in this class (segment length).
        len: usize,
    },
    /// Out-of-sample scheme: generates fresh independent N(0,1) noise on-the-fly.
    ///
    /// Correlation is NOT applied here; it is applied at the composite
    /// `ForwardSampler` level after all classes have filled their segments.
    OutOfSample {
        /// Seed for the forward-pass noise generator.
        forward_seed: u64,
        /// Per-class dimension (number of entities in this class).
        dim: usize,
        /// Per-stage noise generation method for this class.
        noise_methods: Box<[NoiseMethod]>,
    },
    /// Historical replay scheme: replays a pre-standardized window from the library.
    Historical {
        /// Borrowed pre-standardized historical library.
        library: &'a HistoricalScenarioLibrary,
    },
    /// External scenario file scheme: reads from a pre-standardized external library.
    External {
        /// Borrowed pre-standardized external library.
        library: &'a ExternalScenarioLibrary,
    },
}

impl std::fmt::Debug for ClassSampler<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassSampler::InSample {
                base_seed,
                offset,
                len,
                ..
            } => f
                .debug_struct("ClassSampler::InSample")
                .field("base_seed", base_seed)
                .field("offset", offset)
                .field("len", len)
                .finish_non_exhaustive(),
            ClassSampler::OutOfSample {
                forward_seed,
                dim,
                noise_methods,
            } => f
                .debug_struct("ClassSampler::OutOfSample")
                .field("forward_seed", forward_seed)
                .field("dim", dim)
                .field("noise_methods", noise_methods)
                .finish_non_exhaustive(),
            ClassSampler::Historical { .. } => write!(f, "ClassSampler::Historical(..)"),
            ClassSampler::External { .. } => write!(f, "ClassSampler::External(..)"),
        }
    }
}

/// A constant seed offset for historical window selection.
///
/// Using a non-zero constant distinguishes the historical selection hash domain
/// from the forward-pass noise domain (which uses the caller-provided
/// `forward_seed`).
const HISTORICAL_SELECTION_BASE_SEED: u64 = 0x6869_7374_6f72_6963; // b"historic" as u64 LE

/// A constant seed offset for external scenario selection.
///
/// Using a distinct constant distinguishes the external selection hash domain
/// from both the forward-pass noise and historical selection domains.
const EXTERNAL_SELECTION_BASE_SEED: u64 = 0x6578_7465_726e_616c; // b"external" as u64 LE

impl ClassSampler<'_> {
    /// Compute the deterministic historical window index for the given request.
    ///
    /// Uses the same hash domain as [`ClassSampler::fill`] for `Historical` so
    /// that `apply_initial_state` and `fill` always select the same window.
    ///
    /// The result is `derive_forward_seed(HISTORICAL_SELECTION_BASE_SEED,
    /// req.iteration, req.scenario, 0) % n_windows`.
    #[allow(clippy::cast_possible_truncation)]
    fn select_historical_window(req: &ClassSampleRequest, n_windows: usize) -> usize {
        let hash = derive_forward_seed(
            HISTORICAL_SELECTION_BASE_SEED,
            req.iteration,
            req.scenario,
            0,
        );
        (hash as usize) % n_windows
    }

    /// Inject the pre-study lag values for the selected historical window into
    /// the solver state vector.
    ///
    /// For `Historical`: writes `library.lag_slice(window_idx)` into
    /// `state[lag_offset..lag_offset + max_order * n_hydros]`. The window index
    /// is derived by the same deterministic hash as [`ClassSampler::fill`] so
    /// that both methods always refer to the same window for a given
    /// `(iteration, scenario)`.
    ///
    /// For `InSample`, `OutOfSample`, and `External`: this is a no-op — the
    /// initial state is already correct for these schemes (no lag injection
    /// needed).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if
    /// `state.len() < lag_offset + library.max_order() * library.n_hydros()`
    /// for the `Historical` variant.
    pub fn apply_initial_state(
        &self,
        req: &ClassSampleRequest,
        state: &mut [f64],
        lag_offset: usize,
    ) {
        match self {
            ClassSampler::Historical { library } => {
                let window_idx = Self::select_historical_window(req, library.n_windows());
                let lag_data = library.lag_slice(window_idx);
                debug_assert!(
                    state.len() >= lag_offset + lag_data.len(),
                    "state too short for lag injection: state.len()={}, \
                     lag_offset={lag_offset}, lag_data.len()={}",
                    state.len(),
                    lag_data.len(),
                );
                state[lag_offset..lag_offset + lag_data.len()].copy_from_slice(lag_data);
            }
            ClassSampler::InSample { .. }
            | ClassSampler::OutOfSample { .. }
            | ClassSampler::External { .. } => {}
        }
    }

    /// Fill `output` with noise for the given `(iteration, scenario, stage)` triple.
    ///
    /// Writes exactly `output.len()` f64 values into the provided buffer.
    ///
    /// For `InSample`, `Historical`, and `External` variants the noise is
    /// pre-correlated (sourced from the tree or library); no further correlation
    /// is needed. For `OutOfSample`, the noise is independent N(0,1); the
    /// composite `ForwardSampler` applies spatial correlation after all classes
    /// have been filled.
    ///
    /// # Errors
    ///
    /// - [`StochasticError::InsufficientData`] — when `OutOfSample` and
    ///   `req.stage_idx` is out of bounds for the per-stage noise methods.
    /// - [`StochasticError::DimensionExceedsCapacity`] — when `OutOfSample`
    ///   uses `QmcSobol` and `dim > MAX_SOBOL_DIM`.
    ///
    /// # Panics
    ///
    /// `InSample`: panics in debug builds if `output.len() != self.len`
    /// (programming error — caller is responsible for buffer sizing).
    pub fn fill(
        &self,
        req: &ClassSampleRequest,
        output: &mut [f64],
        perm_scratch: &mut [usize],
    ) -> Result<(), StochasticError> {
        match self {
            ClassSampler::InSample {
                tree,
                base_seed,
                offset,
                len,
            } => {
                debug_assert_eq!(
                    output.len(),
                    *len,
                    "ClassSampler::InSample::fill: output.len() ({}) != self.len ({})",
                    output.len(),
                    len,
                );
                let (_idx, slice) = insample::sample_forward(
                    tree,
                    *base_seed,
                    req.iteration,
                    req.scenario,
                    req.stage,
                    req.stage_idx,
                );
                output.copy_from_slice(&slice[*offset..*offset + *len]);
                Ok(())
            }

            ClassSampler::OutOfSample {
                forward_seed,
                dim,
                noise_methods,
            } => {
                let noise_method = noise_methods.get(req.stage_idx).copied().ok_or_else(|| {
                    StochasticError::InsufficientData {
                        context: format!(
                            "stage_idx {} out of bounds for {} noise methods",
                            req.stage_idx,
                            noise_methods.len(),
                        ),
                    }
                })?;
                let spec = FreshNoiseSpec {
                    forward_seed: *forward_seed,
                    noise_method,
                    iteration: req.iteration,
                    scenario: req.scenario,
                    stage_id: req.stage,
                    noise_group_id: req.noise_group_id,
                    dim: *dim,
                    total_scenarios: req.total_scenarios,
                };
                // TODO(ticket-028): replace with fill_uncorrelated call once
                // ticket-028 extracts this into a dedicated function.
                fill_uncorrelated(spec, None, output, perm_scratch)?;
                Ok(())
            }

            ClassSampler::Historical { library } => {
                // Deterministic window selection: hash of (iteration, scenario)
                // using a domain-specific base seed. Stage is NOT included so
                // the same window is used for all stages within a trajectory.
                let window_idx = Self::select_historical_window(req, library.n_windows());
                output.copy_from_slice(library.eta_slice(window_idx, req.stage_idx));
                Ok(())
            }

            ClassSampler::External { library } => {
                let n_scenarios = library.n_scenarios();
                // Deterministic scenario selection: hash of (iteration, scenario)
                // using a domain-specific base seed. Stage is NOT included.
                let hash = derive_forward_seed(
                    EXTERNAL_SELECTION_BASE_SEED,
                    req.iteration,
                    req.scenario,
                    0,
                );
                #[allow(clippy::cast_possible_truncation)]
                let scenario_idx = (hash as usize) % n_scenarios;
                output.copy_from_slice(library.eta_slice(req.stage_idx, scenario_idx));
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod tests {
    use cobre_core::temporal::NoiseMethod;

    use crate::{
        sampling::{ExternalScenarioLibrary, HistoricalScenarioLibrary},
        tree::opening_tree::OpeningTree,
    };

    use super::{ClassSampleRequest, ClassSampler};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn uniform_tree(n_stages: usize, openings: usize, dim: usize) -> OpeningTree {
        let total = n_stages * openings * dim;
        let data: Vec<f64> = (0_u32..u32::try_from(total).unwrap())
            .map(f64::from)
            .collect();
        OpeningTree::from_parts(data, vec![openings; n_stages], dim)
    }

    fn base_req() -> ClassSampleRequest {
        ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
        }
    }

    // -----------------------------------------------------------------------
    // InSample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_sample_fill_copies_correct_segment() {
        // Tree: 1 stage, 3 openings, dim=5. Noise values are 0..14.
        let tree = uniform_tree(1, 3, 5);
        let view = tree.view();

        let sampler = ClassSampler::InSample {
            tree: view,
            base_seed: 42,
            offset: 2,
            len: 3,
        };

        let mut output = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];
        let req = base_req();

        sampler.fill(&req, &mut output, &mut perm).unwrap();

        // The fill must give exactly 3 elements from the chosen opening at
        // indices [2, 3, 4] of the full dim=5 noise vector.
        assert_eq!(output.len(), 3);
        for v in &output {
            assert!(v.is_finite(), "output value {v} is not finite");
        }

        // Verify the extracted segment matches the tree's opening_data at offset..offset+len.
        let (opening_idx, full_slice) = crate::sampling::insample::sample_forward(
            &tree.view(),
            42,
            req.iteration,
            req.scenario,
            req.stage,
            req.stage_idx,
        );
        let _ = opening_idx; // used only to compute full_slice
        assert_eq!(&output, &full_slice[2..5]);
    }

    #[test]
    fn test_in_sample_fill_deterministic() {
        let tree = uniform_tree(2, 5, 5);
        let view = tree.view();

        let sampler = ClassSampler::InSample {
            tree: view,
            base_seed: 99,
            offset: 1,
            len: 2,
        };

        let req = ClassSampleRequest {
            iteration: 3,
            scenario: 7,
            stage: 1,
            stage_idx: 1,
            total_scenarios: 5,
            noise_group_id: 0,
        };

        let mut out_a = vec![0.0f64; 2];
        let mut out_b = vec![0.0f64; 2];
        let mut perm = vec![0usize; 5];

        sampler.fill(&req, &mut out_a, &mut perm).unwrap();
        sampler.fill(&req, &mut out_b, &mut perm).unwrap();

        assert_eq!(out_a, out_b, "InSample::fill must be deterministic");
    }

    // -----------------------------------------------------------------------
    // OutOfSample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_out_of_sample_fill_deterministic() {
        let noise_methods: Box<[NoiseMethod]> =
            vec![NoiseMethod::Saa, NoiseMethod::Saa].into_boxed_slice();

        let sampler = ClassSampler::OutOfSample {
            forward_seed: 42,
            dim: 3,
            noise_methods,
        };

        let req = ClassSampleRequest {
            iteration: 1,
            scenario: 2,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
        };

        let mut out_a = vec![0.0f64; 3];
        let mut out_b = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req, &mut out_a, &mut perm).unwrap();
        sampler.fill(&req, &mut out_b, &mut perm).unwrap();

        assert_eq!(
            out_a, out_b,
            "OutOfSample::fill must produce bit-identical output for same inputs"
        );
    }

    #[test]
    fn test_out_of_sample_fill_stage_idx_out_of_bounds() {
        let noise_methods: Box<[NoiseMethod]> = vec![NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 1,
            dim: 2,
            noise_methods,
        };

        let req = ClassSampleRequest {
            stage_idx: 5, // out of bounds: only 1 method
            ..base_req()
        };

        let mut output = vec![0.0f64; 2];
        let mut perm = vec![0usize; 10];

        let result = sampler.fill(&req, &mut output, &mut perm);
        assert!(
            matches!(result, Err(crate::StochasticError::InsufficientData { .. })),
            "expected InsufficientData error, got: {result:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Historical tests
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_precision_loss)]
    fn make_historical_library() -> HistoricalScenarioLibrary {
        // 3 windows, 4 stages, 2 hydros.
        let mut lib = HistoricalScenarioLibrary::new(3, 4, 2, 1, vec![1990, 1995, 2000]);
        // Fill each (window, stage) with recognizable values.
        for w in 0..3 {
            for s in 0..4 {
                let base = (w * 100 + s * 10) as f64;
                lib.eta_slice_mut(w, s).copy_from_slice(&[base, base + 1.0]);
            }
        }
        lib
    }

    #[test]
    fn test_historical_fill_deterministic() {
        let lib = make_historical_library();
        let sampler = ClassSampler::Historical { library: &lib };

        let req = ClassSampleRequest {
            iteration: 5,
            scenario: 3,
            stage: 0,
            stage_idx: 1,
            total_scenarios: 10,
            noise_group_id: 0,
        };

        let mut out_a = vec![0.0f64; 2];
        let mut out_b = vec![0.0f64; 2];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req, &mut out_a, &mut perm).unwrap();
        sampler.fill(&req, &mut out_b, &mut perm).unwrap();

        assert_eq!(
            out_a, out_b,
            "Historical::fill must be deterministic for same (iteration, scenario)"
        );
    }

    #[test]
    fn test_historical_fill_different_scenarios_may_differ() {
        let lib = make_historical_library();
        let sampler = ClassSampler::Historical { library: &lib };
        let mut perm = vec![0usize; 10];

        let mut outputs: Vec<Vec<f64>> = (0_u32..20)
            .map(|scenario| {
                let req = ClassSampleRequest {
                    iteration: 0,
                    scenario,
                    stage: 0,
                    stage_idx: 0,
                    total_scenarios: 20,
                    noise_group_id: 0,
                };
                let mut out = vec![0.0f64; 2];
                sampler.fill(&req, &mut out, &mut perm).unwrap();
                out
            })
            .collect();

        // At least some outputs should differ (different windows selected).
        outputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        outputs.dedup();
        assert!(
            outputs.len() > 1,
            "expected at least 2 distinct outputs across 20 scenarios, got only 1"
        );
    }

    #[test]
    fn test_historical_window_stable_across_stages() {
        // The same (iteration, scenario) must always produce the same window
        // regardless of stage_idx. We verify this by checking that the selected
        // window gives consistent eta values across stage_idx changes.
        let lib = make_historical_library();
        let sampler = ClassSampler::Historical { library: &lib };

        let req_stage0 = ClassSampleRequest {
            iteration: 2,
            scenario: 7,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
        };
        let req_stage1 = ClassSampleRequest {
            stage_idx: 1,
            ..req_stage0
        };

        let mut out0 = vec![0.0f64; 2];
        let mut out1 = vec![0.0f64; 2];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req_stage0, &mut out0, &mut perm).unwrap();
        sampler.fill(&req_stage1, &mut out1, &mut perm).unwrap();

        // Both must come from the same window — verify using the known library layout.
        // With our layout: eta[w, s] = [w*100 + s*10, w*100 + s*10 + 1].
        // If stage 0 gives base_s0 and stage 1 gives base_s1, then
        // base_s1 - base_s0 == 10 (i.e. same window, adjacent stage).
        assert_eq!(
            out1[0] - out0[0],
            10.0,
            "Expected stage_idx difference of 10.0, got {}",
            out1[0] - out0[0]
        );
    }

    // -----------------------------------------------------------------------
    // External tests
    // -----------------------------------------------------------------------

    #[allow(clippy::cast_precision_loss)]
    fn make_external_library() -> ExternalScenarioLibrary {
        // 4 stages, 50 scenarios, 3 entities.
        let mut lib = ExternalScenarioLibrary::new(4, 50, 3, "inflow", vec![50usize; 4]);
        for s in 0..4_usize {
            for sc in 0..50_usize {
                let base = (s * 1000 + sc * 10) as f64;
                lib.eta_slice_mut(s, sc)
                    .copy_from_slice(&[base, base + 1.0, base + 2.0]);
            }
        }
        lib
    }

    #[test]
    fn test_external_fill_copies_eta_slice() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let req = ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 2,
            total_scenarios: 10,
            noise_group_id: 0,
        };

        let mut output = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req, &mut output, &mut perm).unwrap();

        // Compute the expected scenario_idx using the same hash.
        let hash = crate::noise::seed::derive_forward_seed(
            super::EXTERNAL_SELECTION_BASE_SEED,
            req.iteration,
            req.scenario,
            0,
        );
        #[allow(clippy::cast_possible_truncation)]
        let scenario_idx = (hash as usize) % 50;
        let expected = lib.eta_slice(req.stage_idx, scenario_idx);

        assert_eq!(
            &output, expected,
            "External::fill must match library.eta_slice(stage_idx, scenario_idx)"
        );
    }

    #[test]
    fn test_external_fill_deterministic() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let req = ClassSampleRequest {
            iteration: 3,
            scenario: 17,
            stage: 1,
            stage_idx: 1,
            total_scenarios: 10,
            noise_group_id: 0,
        };

        let mut out_a = vec![0.0f64; 3];
        let mut out_b = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req, &mut out_a, &mut perm).unwrap();
        sampler.fill(&req, &mut out_b, &mut perm).unwrap();

        assert_eq!(
            out_a, out_b,
            "External::fill must be deterministic for same (iteration, scenario)"
        );
    }

    #[test]
    fn test_external_scenario_stable_across_stages() {
        // Same (iteration, scenario) must select the same external scenario index
        // regardless of stage_idx. We verify via the known eta layout.
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let req_stage0 = ClassSampleRequest {
            iteration: 1,
            scenario: 5,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
        };
        let req_stage1 = ClassSampleRequest {
            stage_idx: 1,
            ..req_stage0
        };

        let mut out0 = vec![0.0f64; 3];
        let mut out1 = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req_stage0, &mut out0, &mut perm).unwrap();
        sampler.fill(&req_stage1, &mut out1, &mut perm).unwrap();

        // With our layout: eta[s, sc] = [s*1000 + sc*10, ...].
        // If same sc is selected, out1[0] - out0[0] = 1000.0.
        assert_eq!(
            out1[0] - out0[0],
            1000.0,
            "Expected stage difference of 1000.0 (same scenario, adjacent stage), got {}",
            out1[0] - out0[0]
        );
    }

    // -----------------------------------------------------------------------
    // apply_initial_state tests
    // -----------------------------------------------------------------------

    /// Construct a library with known lag values for `apply_initial_state` tests.
    ///
    /// 3 windows, 4 stages, 2 hydros, `max_order`=2.
    /// Lag layout per window: `lag[lag * n_hydros + hydro]`.
    /// Window 0: lag0=[1.0, 2.0], lag1=[3.0, 4.0]
    /// Window 1: lag0=[10.0, 20.0], lag1=[30.0, 40.0]
    /// Window 2: lag0=[100.0, 200.0], lag1=[300.0, 400.0]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn make_historical_library_with_lags() -> HistoricalScenarioLibrary {
        let mut lib = HistoricalScenarioLibrary::new(3, 4, 2, 2, vec![1990, 1995, 2000]);
        // Populate eta values so fill() is usable too.
        for w in 0..3 {
            for s in 0..4 {
                let base = (w * 100 + s * 10) as f64;
                lib.eta_slice_mut(w, s).copy_from_slice(&[base, base + 1.0]);
            }
        }
        // Populate lag values with recognizable per-window patterns.
        // lag_slice_mut(w) has length max_order * n_hydros = 2 * 2 = 4.
        let factor = 10.0_f64;
        for w in 0..3 {
            let base = factor.powi(w as i32);
            let lags = [base, base * 2.0, base * 3.0, base * 4.0];
            lib.lag_slice_mut(w).copy_from_slice(&lags);
        }
        lib
    }

    #[test]
    fn test_historical_apply_initial_state_copies_lags() {
        let lib = make_historical_library_with_lags();
        let sampler = ClassSampler::Historical { library: &lib };

        // Use scenario=0 and find the expected window via the same hash.
        let req = ClassSampleRequest {
            iteration: 0,
            scenario: 0,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
        };

        let lag_offset = 5;
        let lag_len = lib.max_order() * lib.n_hydros(); // 4
        let mut state = vec![0.0f64; lag_offset + lag_len + 3];
        state[0] = 99.0; // sentinel: should be untouched
        state[lag_offset + lag_len] = 77.0; // sentinel: should be untouched

        sampler.apply_initial_state(&req, &mut state, lag_offset);

        // Compute expected window using the shared helper path.
        let window_idx = ClassSampler::select_historical_window(&req, lib.n_windows());
        let expected_lags = lib.lag_slice(window_idx);

        assert_eq!(
            &state[lag_offset..lag_offset + lag_len],
            expected_lags,
            "apply_initial_state must copy lag_slice for the selected window"
        );
        // Sentinels must be untouched.
        assert_eq!(state[0], 99.0, "bytes before lag_offset must be untouched");
        assert_eq!(
            state[lag_offset + lag_len],
            77.0,
            "bytes after lag region must be untouched"
        );
    }

    #[test]
    fn test_historical_apply_initial_state_consistent_with_fill() {
        // Both fill() and apply_initial_state() must select the same window for
        // a given (iteration, scenario).
        let lib = make_historical_library_with_lags();
        let sampler = ClassSampler::Historical { library: &lib };

        for scenario in 0..20_u32 {
            let req = ClassSampleRequest {
                iteration: 3,
                scenario,
                stage: 0,
                stage_idx: 0,
                total_scenarios: 20,
                noise_group_id: 0,
            };

            // Derive the window index via the shared helper.
            let window_via_helper = ClassSampler::select_historical_window(&req, lib.n_windows());

            // fill() uses the same helper internally.
            let mut output = vec![0.0f64; lib.n_hydros()];
            let mut perm = vec![0usize; 20];
            sampler.fill(&req, &mut output, &mut perm).unwrap();
            let expected_eta = lib.eta_slice(window_via_helper, req.stage_idx);
            assert_eq!(
                &output, expected_eta,
                "fill() must use the same window as select_historical_window for scenario={scenario}"
            );

            // apply_initial_state() must inject lags from the same window.
            let lag_len = lib.max_order() * lib.n_hydros();
            let mut state = vec![0.0f64; lag_len];
            sampler.apply_initial_state(&req, &mut state, 0);
            let expected_lags = lib.lag_slice(window_via_helper);
            assert_eq!(
                &state, expected_lags,
                "apply_initial_state() must use same window as fill() for scenario={scenario}"
            );
        }
    }

    #[test]
    fn test_in_sample_apply_initial_state_noop() {
        let tree = uniform_tree(1, 3, 5);
        let sampler = ClassSampler::InSample {
            tree: tree.view(),
            base_seed: 42,
            offset: 0,
            len: 3,
        };

        let original = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let mut state = original.clone();
        let req = base_req();

        sampler.apply_initial_state(&req, &mut state, 0);

        assert_eq!(
            state, original,
            "InSample::apply_initial_state must be a no-op"
        );
    }

    #[test]
    fn test_out_of_sample_apply_initial_state_noop() {
        let noise_methods: Box<[cobre_core::temporal::NoiseMethod]> =
            vec![cobre_core::temporal::NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 7,
            dim: 3,
            noise_methods,
        };

        let original = vec![5.0f64, 6.0, 7.0];
        let mut state = original.clone();
        let req = base_req();

        sampler.apply_initial_state(&req, &mut state, 0);

        assert_eq!(
            state, original,
            "OutOfSample::apply_initial_state must be a no-op"
        );
    }

    #[test]
    fn test_external_apply_initial_state_noop() {
        let lib = make_external_library();
        let sampler = ClassSampler::External { library: &lib };

        let original = vec![8.0f64, 9.0, 10.0];
        let mut state = original.clone();
        let req = base_req();

        sampler.apply_initial_state(&req, &mut state, 0);

        assert_eq!(
            state, original,
            "External::apply_initial_state must be a no-op"
        );
    }

    // -----------------------------------------------------------------------
    // Debug impl test
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_all_variants() {
        let tree = uniform_tree(1, 2, 3);
        let variants: Vec<Box<dyn Fn() -> String>> = vec![
            Box::new(|| {
                format!(
                    "{:?}",
                    ClassSampler::InSample {
                        tree: tree.view(),
                        base_seed: 1,
                        offset: 0,
                        len: 2,
                    }
                )
            }),
            Box::new(|| {
                format!(
                    "{:?}",
                    ClassSampler::OutOfSample {
                        forward_seed: 2,
                        dim: 3,
                        noise_methods: vec![NoiseMethod::Saa].into_boxed_slice(),
                    }
                )
            }),
        ];

        for fmt_fn in &variants {
            let s = fmt_fn();
            assert!(!s.is_empty(), "Debug output must not be empty");
        }

        // Historical and External debug format (library refs are large).
        let hist_lib = make_historical_library();
        let ext_lib = make_external_library();
        let hist_debug = format!("{:?}", ClassSampler::Historical { library: &hist_lib });
        let ext_debug = format!("{:?}", ClassSampler::External { library: &ext_lib });
        assert!(hist_debug.contains("Historical"));
        assert!(ext_debug.contains("External"));
    }
    // -----------------------------------------------------------------------
    // AC (ticket-003): noise_group_id propagation tests
    // -----------------------------------------------------------------------

    /// AC: Two `OutOfSample::fill()` calls with the same `noise_group_id` but
    /// different `stage` must produce identical noise.
    #[test]
    fn test_out_of_sample_same_group_produces_identical_noise() {
        let noise_methods: Box<[NoiseMethod]> =
            vec![NoiseMethod::Saa, NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 42,
            dim: 3,
            noise_methods,
        };

        // Same noise_group_id=5, different stage (0 vs 1).
        let req_stage0 = ClassSampleRequest {
            iteration: 1,
            scenario: 2,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 5,
        };
        let req_stage1 = ClassSampleRequest {
            stage: 1,
            stage_idx: 1,
            ..req_stage0
        };

        let mut out0 = vec![0.0f64; 3];
        let mut out1 = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req_stage0, &mut out0, &mut perm).unwrap();
        sampler.fill(&req_stage1, &mut out1, &mut perm).unwrap();

        assert_eq!(
            out0, out1,
            "OutOfSample::fill with same noise_group_id must produce identical noise              regardless of stage"
        );
    }

    /// AC: Two `OutOfSample::fill()` calls with different `noise_group_id`
    /// values must produce different noise.
    #[test]
    fn test_out_of_sample_different_group_produces_different_noise() {
        let noise_methods: Box<[NoiseMethod]> =
            vec![NoiseMethod::Saa, NoiseMethod::Saa].into_boxed_slice();
        let sampler = ClassSampler::OutOfSample {
            forward_seed: 42,
            dim: 3,
            noise_methods,
        };

        let req_group0 = ClassSampleRequest {
            iteration: 1,
            scenario: 2,
            stage: 0,
            stage_idx: 0,
            total_scenarios: 10,
            noise_group_id: 0,
        };
        let req_group1 = ClassSampleRequest {
            noise_group_id: 1,
            ..req_group0
        };

        let mut out0 = vec![0.0f64; 3];
        let mut out1 = vec![0.0f64; 3];
        let mut perm = vec![0usize; 10];

        sampler.fill(&req_group0, &mut out0, &mut perm).unwrap();
        sampler.fill(&req_group1, &mut out1, &mut perm).unwrap();

        let any_differ = out0.iter().zip(&out1).any(|(a, b)| a != b);
        assert!(
            any_differ,
            "OutOfSample::fill with different noise_group_id must produce different noise"
        );
    }
}
