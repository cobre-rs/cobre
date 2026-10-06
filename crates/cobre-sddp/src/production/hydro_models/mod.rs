//! Hydro model preprocessing pipeline for the production function and evaporation.
//!
//! Resolves per-`(hydro, stage)` production models (constant productivity or FPHA
//! hyperplanes) and per-hydro evaporation models from the case directory, bundles
//! them into a `PrepareHydroModelsResult`, and produces a display summary.
//!
//! These types and functions live in `cobre-sddp` because they are
//! algorithm-specific (FPHA hyperplane approximation is an SDDP concept). They
//! must not be placed in `cobre-core`.
//!
//! The orchestration entry points (`prepare_hydro_models`,
//! `prepare_hydro_models_from_artifacts`) and the private
//! `load_artifacts_for_hydro_models` reader live here; the `types`, `production`,
//! `evaporation`, `summary`, and `export` submodules own the rest.
//!
//! Every public symbol is re-exported here so both the curated flat surface in
//! `lib.rs` and the `cobre_sddp::hydro_models::Symbol` module path resolve to the
//! same item regardless of which submodule owns it.

use std::path::Path;

use cobre_core::EntityId;
use cobre_core::System;
use cobre_io::CaseArtifacts;
use cobre_io::HydroGeometryRow;
use cobre_io::InputFile;
use cobre_io::ValidationContext;
use cobre_io::extensions::load_tailrace_curves;
use cobre_io::load_fpha_hyperplanes;
use cobre_io::load_hydro_energy_productivity;
use cobre_io::load_hydro_geometry;
use cobre_io::load_production_models;
use cobre_io::validate_structure;

use crate::SddpError;

mod evaporation;
mod export;
mod production;
mod summary;
mod types;

pub use evaporation::resolve_evaporation_models_from_artifacts;
pub use export::{build_deviation_summary, build_evaporation_model_rows};
pub use production::resolve_production_models_from_artifacts;
pub use summary::build_hydro_model_summary;

/// Wall-clock split of the two hydro-model fitting steps run by
/// [`prepare_hydro_models_from_artifacts`]. Populated only when a caller passes a
/// `&mut` borrow; the resolver logic is identical whether or not it is measured.
#[derive(Debug, Clone, Copy, Default)]
pub struct HydroFitTimings {
    /// Wall-clock seconds spent in [`resolve_production_models_from_artifacts`].
    pub production_fit_seconds: f64,
    /// Wall-clock seconds spent in [`resolve_evaporation_models_from_artifacts`].
    pub evaporation_fit_seconds: f64,
}
pub use types::{
    EvaporationModel, EvaporationModelSet, EvaporationReferenceSource, EvaporationSource,
    FphaFitDeviationEntry, FphaHydroDetail, FphaPlane, HydroModelProvenance, HydroModelSummary,
    LinearizedEvaporation, NoTurbineCapacityHydro, PrepareHydroModelsResult, ProductionModelSet,
    ProductionModelSource, ResolvedProductionModel,
};
// ── Top-level pipeline function ───────────────────────────────────────────────

/// Run the full hydro model preprocessing pipeline for a case directory.
///
/// Runs on all MPI ranks independently (each rank loads the optional files from a
/// shared filesystem). `collect_deviation_points` is the
/// `config.exports.fpha_deviation_points` opt-in: `false` leaves
/// [`PrepareHydroModelsResult::fpha_deviation_point_rows`] empty and the fit
/// bit-identical.
///
/// # Errors
///
/// Propagates errors from [`resolve_production_models_from_artifacts`] and
/// [`resolve_evaporation_models_from_artifacts`].
pub fn prepare_hydro_models(
    system: &System,
    case_dir: &Path,
    collect_deviation_points: bool,
) -> Result<PrepareHydroModelsResult, SddpError> {
    let artifacts = load_artifacts_for_hydro_models(case_dir)?;
    prepare_hydro_models_from_artifacts(system, &artifacts, collect_deviation_points, None)
}

/// Variant of [`prepare_hydro_models`] consuming a pre-parsed
/// [`cobre_io::CaseArtifacts`] bundle.
///
/// Use this from any pipeline that has already called
/// [`cobre_io::load_case_with_artifacts`]; it avoids re-parsing and re-validating.
/// `timings`, when `Some`, records each fitting step's wall time.
///
/// # Errors
///
/// Same conditions as [`prepare_hydro_models`].
pub fn prepare_hydro_models_from_artifacts(
    system: &System,
    artifacts: &CaseArtifacts,
    collect_deviation_points: bool,
    timings: Option<&mut HydroFitTimings>,
) -> Result<PrepareHydroModelsResult, SddpError> {
    let production_start = std::time::Instant::now();
    let (
        production,
        productivity_override,
        production_sources,
        fpha_export_rows,
        reference_volumes_hm3,
        fpha_fit_deviations,
        fpha_deviation_point_rows,
    ) = resolve_production_models_from_artifacts(system, artifacts, collect_deviation_points)?;
    let production_fit_seconds = production_start.elapsed().as_secs_f64();

    let evaporation_start = std::time::Instant::now();
    let (evaporation, evaporation_sources, evaporation_reference_sources) =
        resolve_evaporation_models_from_artifacts(system, artifacts)?;
    let evaporation_fit_seconds = evaporation_start.elapsed().as_secs_f64();

    if let Some(timings) = timings {
        timings.production_fit_seconds = production_fit_seconds;
        timings.evaporation_fit_seconds = evaporation_fit_seconds;
    }

    // Sorted by ascending volume, as `ForebayTable::new` expects.
    let mut vha_geometry_by_hydro: std::collections::HashMap<EntityId, Vec<HydroGeometryRow>> =
        std::collections::HashMap::new();
    for row in &artifacts.hydro_geometry {
        vha_geometry_by_hydro
            .entry(row.hydro_id)
            .or_default()
            .push(row.clone());
    }
    for rows in vha_geometry_by_hydro.values_mut() {
        rows.sort_by(|a, b| a.volume_hm3.total_cmp(&b.volume_hm3));
    }

    Ok(PrepareHydroModelsResult {
        production,
        productivity_override,
        evaporation,
        provenance: HydroModelProvenance {
            production_sources,
            evaporation_sources,
            evaporation_reference_sources,
        },
        fpha_export_rows,
        reference_volumes_hm3,
        vha_geometry_by_hydro,
        fpha_fit_deviations,
        fpha_deviation_point_rows,
    })
}

/// Build a [`cobre_io::CaseArtifacts`] by reading the case directory directly,
/// backing the legacy [`prepare_hydro_models`] signature; production pipelines
/// should call [`cobre_io::load_case_with_artifacts`] so the full validation runs
/// once.
fn load_artifacts_for_hydro_models(case_dir: &Path) -> Result<CaseArtifacts, SddpError> {
    let mut ctx = ValidationContext::new();
    let manifest = validate_structure(case_dir, &mut ctx);
    // Fail on a malformed layout here, before any file load, so it does not
    // surface as a confusing downstream parse error or silent default.
    ctx.into_result().map_err(SddpError::from)?;

    let prod_path = manifest
        .present(InputFile::SystemHydroProductionModelsJson)
        .then(|| case_dir.join("system").join("hydro_production_models.json"));
    let geom_path = manifest
        .present(InputFile::SystemHydroGeometryParquet)
        .then(|| case_dir.join("system").join("hydro_geometry.parquet"));
    let fpha_path = manifest
        .present(InputFile::SystemFphaHyperplanesParquet)
        .then(|| case_dir.join("system").join("fpha_hyperplanes.parquet"));
    let prod_eff_path = case_dir
        .join("system")
        .join("hydro_energy_productivity.parquet");
    let prod_eff_path_opt = prod_eff_path.exists().then_some(prod_eff_path.as_path());
    let tailrace_path = manifest
        .present(InputFile::SystemTailraceCurvesParquet)
        .then(|| case_dir.join("system").join("tailrace_curves.parquet"));

    let production_file = load_production_models(prod_path.as_deref())?;

    Ok(CaseArtifacts {
        file_manifest: manifest,
        hydro_geometry: load_hydro_geometry(geom_path.as_deref())?,
        production_models: production_file.configs,
        plane_reduction: production_file.plane_reduction,
        hydro_energy_productivity: load_hydro_energy_productivity(prod_eff_path_opt)?,
        fpha_hyperplanes: load_fpha_hyperplanes(fpha_path.as_deref())?,
        scalar_parameters: Vec::new(),
        tailrace_curves: load_tailrace_curves(tailrace_path.as_deref())?,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::match_wildcard_for_single_variants,
    clippy::cast_precision_loss,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic
)]
mod tests {
    // ── cross-path parity test ────────────────────────────────────────────────

    /// `cobre-cli` builds hydro models two ways: rank 0 from the already-parsed
    /// artifacts (`prepare_hydro_models_from_artifacts`), non-root ranks by
    /// re-reading the case directory (`prepare_hydro_models`). This pins that the
    /// two agree on a computed-FPHA case, where geometry-derived hyperplane fitting
    /// — not a precomputed passthrough — gives the loaders the most room to diverge.
    #[test]
    fn prepare_hydro_models_from_artifacts_matches_prepare_hydro_models_for_computed_fpha() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let cobre_io::LoadedCase { system, artifacts } =
            cobre_io::load_case_with_artifacts(&case_dir)
                .expect("d07-fpha-computed must load successfully");

        let from_artifacts =
            super::prepare_hydro_models_from_artifacts(&system, &artifacts, false, None)
                .expect("prepare_hydro_models_from_artifacts must succeed");
        let from_disk = super::prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        assert!(
            !from_artifacts.fpha_export_rows.is_empty(),
            "rank 0 path: fpha_export_rows must be non-empty for a computed-FPHA case"
        );
        assert_eq!(
            from_artifacts.fpha_export_rows, from_disk.fpha_export_rows,
            "fpha_export_rows must be bit-identical between prepare_hydro_models_from_artifacts \
             (rank 0's artifact-reuse path) and prepare_hydro_models (the non-root disk-reread path)"
        );

        let summary_from_artifacts = super::build_hydro_model_summary(&from_artifacts, &system);
        let summary_from_disk = super::build_hydro_model_summary(&from_disk, &system);
        assert_eq!(
            summary_from_artifacts.total_planes, summary_from_disk.total_planes,
            "total FPHA plane count must match between the two hydro-model construction paths"
        );
    }

    /// `prepare_hydro_models` carries the per-hydro VHA geometry on the result,
    /// grouped by plant and sorted by ascending volume, so the energy-conversion
    /// build can derive ρ_eq from VHA + ρ_esp for an FPHA plant with no parquet
    /// override. A computed-FPHA case ships geometry, so the map must be populated.
    #[test]
    fn prepare_hydro_models_carries_sorted_vha_geometry() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let system =
            cobre_io::load_case(&case_dir).expect("d07-fpha-computed must load successfully");
        let result = super::prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        assert!(
            !result.vha_geometry_by_hydro.is_empty(),
            "a computed-FPHA case ships VHA geometry; the map must be populated"
        );
        for rows in result.vha_geometry_by_hydro.values() {
            assert!(!rows.is_empty(), "no plant entry may be empty");
            assert!(
                rows.windows(2).all(|w| w[0].volume_hm3 <= w[1].volume_hm3),
                "each plant's VHA rows must be sorted by ascending volume"
            );
        }
    }

    // ── thread-count determinism gate ─────────────────────────────────────────

    /// The per-hydro FPHA fit loop in `resolve_production_models_from_artifacts`
    /// is a `par_iter().collect()`, so its output must not depend on the rayon
    /// pool size. This gate fits the computed-FPHA case
    /// d31-backwater-reference-volume under pools of 1, 2, and 4 threads and
    /// asserts the resolved `fpha_export_rows` are `to_bits`-identical across all
    /// three — the `(hydro_id, stage_id, plane_id)` ordering and every gamma
    /// coefficient must match bit-for-bit regardless of thread scheduling. The
    /// case has TWO hydros on purpose: a single-hydro case leaves the
    /// `par_iter()` with one element, which never schedules concurrently and so
    /// cannot exercise the cross-hydro `collect()` reassembly this gate exists to
    /// protect. A regression that collected into a shared `Mutex<Vec>` or pushed
    /// rows from worker threads would reorder the stream and fail here.
    #[test]
    fn fit_is_thread_count_invariant() {
        // The FPHA export rows AND the per-fit deviations both ride the same
        // parallel fit, so both must be pool-size invariant; this alias carries
        // them out of each fixed-size-pool resolve together.
        type FitOutputs = (
            Vec<cobre_io::FphaHyperplaneRow>,
            Vec<super::FphaFitDeviationEntry>,
        );

        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d31-backwater-reference-volume");

        let system = cobre_io::load_case(&case_dir)
            .expect("d31-backwater-reference-volume must load successfully");

        // Resolve the FPHA export rows and the per-fit deviations inside a
        // fixed-size rayon pool so the fit loop runs under exactly `n` worker
        // threads.
        let resolve_under_pool = |n: usize| -> FitOutputs {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("rayon pool must build")
                .install(|| {
                    let result = super::prepare_hydro_models(&system, &case_dir, false)
                        .expect("prepare_hydro_models must succeed");
                    (result.fpha_export_rows, result.fpha_fit_deviations)
                })
        };

        let thread_counts = [1usize, 2, 4];
        let outputs: Vec<FitOutputs> = thread_counts
            .iter()
            .map(|&n| resolve_under_pool(n))
            .collect();
        let rows: Vec<&Vec<cobre_io::FphaHyperplaneRow>> = outputs.iter().map(|(r, _)| r).collect();
        let deviations: Vec<&Vec<super::FphaFitDeviationEntry>> =
            outputs.iter().map(|(_, d)| d).collect();

        assert!(
            !rows[0].is_empty(),
            "the computed-FPHA case must export at least one plane for the fit path to run"
        );
        assert!(
            !deviations[0].is_empty(),
            "the computed-FPHA case must record at least one fit deviation"
        );

        // Bit-exact equality across pool sizes: ids, stage/plane ordering, and the
        // gamma coefficients (compared via `to_bits`, not float `==`).
        let assert_bit_identical = |a: &[cobre_io::FphaHyperplaneRow],
                                    b: &[cobre_io::FphaHyperplaneRow],
                                    threads_a: usize,
                                    threads_b: usize| {
            assert_eq!(
                a.len(),
                b.len(),
                "row count must match across {threads_a}- and {threads_b}-thread pools"
            );
            for (ra, rb) in a.iter().zip(b) {
                assert_eq!(ra.hydro_id, rb.hydro_id, "hydro_id must match");
                assert_eq!(ra.stage_id, rb.stage_id, "stage_id must match");
                assert_eq!(ra.plane_id, rb.plane_id, "plane_id must match");
                assert_eq!(
                    ra.gamma_0.to_bits(),
                    rb.gamma_0.to_bits(),
                    "gamma_0 must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.gamma_v.to_bits(),
                    rb.gamma_v.to_bits(),
                    "gamma_v must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.gamma_q.to_bits(),
                    rb.gamma_q.to_bits(),
                    "gamma_q must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.gamma_s.to_bits(),
                    rb.gamma_s.to_bits(),
                    "gamma_s must be bit-identical across pool sizes"
                );
                assert_eq!(
                    ra.kappa.to_bits(),
                    rb.kappa.to_bits(),
                    "kappa must be bit-identical across pool sizes"
                );
            }
        };

        // Bit-exact equality of the carried per-fit deviations across pool sizes:
        // ids, stage tags, and the four magnitudes compared via `to_bits`, never
        // float `==`. The carry-up flatten is sequential and canonical, so a
        // regression that reordered it (or summed deviations in a thread-scheduled
        // order) would fail here.
        let assert_deviations_bit_identical =
            |a: &[super::FphaFitDeviationEntry],
             b: &[super::FphaFitDeviationEntry],
             threads_a: usize,
             threads_b: usize| {
                assert_eq!(
                    a.len(),
                    b.len(),
                    "deviation count must match across {threads_a}- and {threads_b}-thread pools"
                );
                for (da, db) in a.iter().zip(b) {
                    assert_eq!(da.hydro_id, db.hydro_id, "deviation hydro_id must match");
                    assert_eq!(da.stage_id, db.stage_id, "deviation stage_id must match");
                    assert_eq!(
                        da.mean_abs_mw.to_bits(),
                        db.mean_abs_mw.to_bits(),
                        "mean_abs_mw must be bit-identical across pool sizes"
                    );
                    assert_eq!(
                        da.max_abs_mw.to_bits(),
                        db.max_abs_mw.to_bits(),
                        "max_abs_mw must be bit-identical across pool sizes"
                    );
                    assert_eq!(
                        da.mean_signed_mw.to_bits(),
                        db.mean_signed_mw.to_bits(),
                        "mean_signed_mw must be bit-identical across pool sizes"
                    );
                    assert_eq!(
                        da.relative.to_bits(),
                        db.relative.to_bits(),
                        "relative must be bit-identical across pool sizes"
                    );
                }
            };

        // Compare every pool size against the single-thread baseline.
        for ((rows_n, deviations_n), &n) in rows.iter().zip(&deviations).zip(&thread_counts).skip(1)
        {
            assert_bit_identical(rows[0], rows_n, thread_counts[0], n);
            assert_deviations_bit_identical(deviations[0], deviations_n, thread_counts[0], n);
        }
    }

    // ── per-sampled-point deviation table: opt-in + determinism ───────────────

    /// With `collect_deviation_points = true`, the computed-FPHA case
    /// d07-fpha-computed yields non-empty `fpha_deviation_point_rows` that are
    /// `to_bits`-identical across rayon pools of 1, 2, and 4 threads — mirroring
    /// `fit_is_thread_count_invariant`. The point rows ride the same sequential
    /// canonical-order flatten as `fpha_export_rows`, so a regression that emitted
    /// them from a parallel worker (or in a thread-scheduled order) would fail here.
    #[test]
    fn deviation_points_are_thread_count_invariant_when_on() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let system =
            cobre_io::load_case(&case_dir).expect("d07-fpha-computed must load successfully");

        let resolve_under_pool = |n: usize| -> Vec<cobre_io::FphaDeviationPointRow> {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("rayon pool must build")
                .install(|| {
                    super::prepare_hydro_models(&system, &case_dir, true)
                        .expect("prepare_hydro_models must succeed")
                        .fpha_deviation_point_rows
                })
        };

        let thread_counts = [1usize, 2, 4];
        let outputs: Vec<Vec<cobre_io::FphaDeviationPointRow>> = thread_counts
            .iter()
            .map(|&n| resolve_under_pool(n))
            .collect();

        assert!(
            !outputs[0].is_empty(),
            "flag-on computed-FPHA case must emit at least one deviation point row"
        );

        for (rows_n, &n) in outputs.iter().zip(&thread_counts).skip(1) {
            assert_eq!(
                outputs[0].len(),
                rows_n.len(),
                "deviation-point row count must match across {}- and {n}-thread pools",
                thread_counts[0]
            );
            for (a, b) in outputs[0].iter().zip(rows_n) {
                assert_eq!(a.hydro_id, b.hydro_id, "hydro_id must match");
                assert_eq!(a.stage_id, b.stage_id, "stage_id must match");
                assert_eq!(a.v.to_bits(), b.v.to_bits(), "v must be bit-identical");
                assert_eq!(a.q.to_bits(), b.q.to_bits(), "q must be bit-identical");
                assert_eq!(
                    a.fph_exact.to_bits(),
                    b.fph_exact.to_bits(),
                    "fph_exact must be bit-identical"
                );
                assert_eq!(
                    a.fpha_fitted.to_bits(),
                    b.fpha_fitted.to_bits(),
                    "fpha_fitted must be bit-identical"
                );
                assert_eq!(
                    a.deviation.to_bits(),
                    b.deviation.to_bits(),
                    "deviation must be bit-identical"
                );
                assert_eq!(
                    a.relative.to_bits(),
                    b.relative.to_bits(),
                    "relative must be bit-identical"
                );
            }
        }
    }

    /// With `collect_deviation_points = false`, the same case yields EMPTY
    /// `fpha_deviation_point_rows`, and its `fpha_export_rows` are `to_bits`-
    /// identical to a flag-on run — the gated collection adds no rows and does not
    /// perturb the fit (zero overhead, bit-identical planes regardless of flag).
    #[test]
    fn deviation_points_off_is_empty_and_does_not_perturb_export_rows() {
        let case_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("cobre-sddp parent dir must exist")
            .parent()
            .expect("crates parent dir must exist")
            .join("examples/deterministic/d07-fpha-computed");

        let system =
            cobre_io::load_case(&case_dir).expect("d07-fpha-computed must load successfully");

        let off = super::prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models (off) must succeed");
        let on = super::prepare_hydro_models(&system, &case_dir, true)
            .expect("prepare_hydro_models (on) must succeed");

        assert!(
            off.fpha_deviation_point_rows.is_empty(),
            "flag-off must collect no deviation point rows"
        );
        assert!(
            !on.fpha_deviation_point_rows.is_empty(),
            "flag-on must collect deviation point rows for the computed-FPHA case"
        );

        // The fit (and thus the export rows) must be bit-identical regardless of
        // the opt-in: the collection is read-only over the emitted planes.
        assert_eq!(
            off.fpha_export_rows.len(),
            on.fpha_export_rows.len(),
            "export-row count must not depend on the deviation-points opt-in"
        );
        for (a, b) in off.fpha_export_rows.iter().zip(&on.fpha_export_rows) {
            assert_eq!(a.hydro_id, b.hydro_id, "hydro_id must match");
            assert_eq!(a.stage_id, b.stage_id, "stage_id must match");
            assert_eq!(a.plane_id, b.plane_id, "plane_id must match");
            assert_eq!(
                a.gamma_0.to_bits(),
                b.gamma_0.to_bits(),
                "gamma_0 must be bit-identical with the flag on vs off"
            );
            assert_eq!(
                a.gamma_v.to_bits(),
                b.gamma_v.to_bits(),
                "gamma_v must match"
            );
            assert_eq!(
                a.gamma_q.to_bits(),
                b.gamma_q.to_bits(),
                "gamma_q must match"
            );
            assert_eq!(
                a.gamma_s.to_bits(),
                b.gamma_s.to_bits(),
                "gamma_s must match"
            );
            assert_eq!(a.kappa.to_bits(), b.kappa.to_bits(), "kappa must match");
        }
    }
}
