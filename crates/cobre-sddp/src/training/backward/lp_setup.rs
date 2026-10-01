//! LP load and bound-patch seam for the backward pass: reset the stage LP to its
//! structural template, append delta cuts, and patch the noise-dependent
//! row/column bounds for one opening.

use cobre_solver::SolverInterface;
use cobre_stochastic::{ClassDimensions, ExternalScenarioLibrary};

use crate::{
    context::{StageContext, TrainingContext},
    error::SddpError,
    setup::{NodeId, NodePos, StageIdx},
    training::stage_solve_prep::{InflowNoise, StageSolvePrep, StageSolvePrepParams, StateSource},
    workspace::{BasisStoreSliceMut, CapturedBasis, SolverWorkspace},
};

use super::SuccessorChild;

/// Load one successor child's frozen LP template and append its delta cuts.
///
/// The LP structure is identical across a child's openings, so only bound patching
/// runs per opening. Each child loads ITS OWN pool's template and delta batch, so a
/// fan's blocks never reuse child 0's LP (the child-0 collapse).
pub(crate) fn load_backward_lp<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    child: &SuccessorChild<'_>,
) {
    ws.solver.load_model(child.frozen_template);
    if child.cut_batch.num_rows > 0 {
        ws.solver.add_rows(child.cut_batch);
    }
}

/// Transform opening noise and patch LP bounds for one backward opening.
///
/// The LP structure is already loaded by [`load_backward_lp`]; this delegates to
/// [`StageSolvePrep::run`], pinning `x_hat` as the incoming state.
pub(crate) fn patch_opening_bounds<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    raw_noise: &[f64],
    x_hat: &[f64],
    s: StageIdx,
) {
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(x_hat),
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        s,
        &prep_params,
    );
}

/// Assemble the `[hydro | load-bus | NCS]` opening-noise vector for an
/// `External` successor node's declared scenario column `k` at successor stage
/// `stage`, into `buf`.
///
/// This reproduces the multi-class vector a generated `OpeningTreeView::opening`
/// yields and the forward `ClassSampler::fill` assembles, reading `eta_slice(stage,
/// k)` from each present external library into that class's segment. The segment
/// lengths and offsets are the same `stochastic`'s counts the noise transforms
/// consume, so forward and backward read identical bytes for the pinned column.
///
/// # Errors
///
/// [`SddpError::Validation`] when a class has a nonempty noise segment but no
/// external library — a genuine setup defect, never a silent fallback to the
/// generated opening tree (which is exactly the inert-hash bug this removes).
pub(crate) fn fill_external_opening_noise(
    training_ctx: &TrainingContext<'_>,
    stage: StageIdx,
    k: usize,
    node_id: NodeId,
    buf: &mut Vec<f64>,
) -> Result<(), SddpError> {
    let stochastic = training_ctx.stochastic;
    assemble_external_opening_noise(
        stochastic.class_dimensions(),
        [
            training_ctx.external_inflow_library,
            training_ctx.external_load_library,
            training_ctx.external_ncs_library,
        ],
        stage.0,
        k,
        node_id,
        buf,
    )
}

/// Per-class order the assembled noise vector and its dims/libraries follow —
/// the `[hydro | load-bus | NCS]` layout every noise transform consumes.
const EXTERNAL_CLASS_NAMES: [&str; 3] = ["inflow", "load", "ncs"];

/// The context-free core of [`fill_external_opening_noise`]: given each class's
/// noise dimension and its optional external library, write the concatenated
/// `[hydro | load-bus | NCS]` `eta_slice(stage, k)` columns into `buf` at the
/// cumulative per-class offsets. Split out from the context read so it is
/// directly unit-testable with stub libraries.
///
/// # Errors
///
/// [`SddpError::Validation`] when a class has a nonempty segment but no library.
fn assemble_external_opening_noise(
    dims: ClassDimensions,
    libraries: [Option<&ExternalScenarioLibrary>; 3],
    stage: usize,
    k: usize,
    node_id: NodeId,
    buf: &mut Vec<f64>,
) -> Result<(), SddpError> {
    buf.clear();
    buf.resize(dims.total(), 0.0);
    let (hydro, load, ncs) = dims.split_segments_mut(buf);
    for ((segment, library), class) in [hydro, load, ncs]
        .into_iter()
        .zip(libraries)
        .zip(EXTERNAL_CLASS_NAMES)
    {
        fill_external_class(segment, library, stage, k, class, node_id)?;
    }
    Ok(())
}

/// Copy one class's `eta_slice(stage, k)` into `segment`; a nonempty segment with
/// no library present is a [`SddpError::Validation`] naming the node/stage/class.
fn fill_external_class(
    segment: &mut [f64],
    library: Option<&ExternalScenarioLibrary>,
    stage: usize,
    k: usize,
    class: &str,
    node_id: NodeId,
) -> Result<(), SddpError> {
    if segment.is_empty() {
        return Ok(());
    }
    let library = library.ok_or_else(|| {
        SddpError::Validation(format!(
            "enumerated external node {node_id} at stage {stage} realizes scenario column {k} for \
             the {class} class, but no external {class} scenario library is present"
        ))
    })?;
    segment.copy_from_slice(library.eta_slice(stage, k));
    Ok(())
}

/// Resolve the ω=0 warm-start basis from the worker's `BasisStoreSliceMut`,
/// keyed by the trial point `m` and the successor's canonical node position.
///
/// Returns `None` when the slot is empty (cold start or no prior capture).
#[inline]
pub(crate) fn resolve_backward_basis<'a>(
    basis_slice: &'a BasisStoreSliceMut<'_>,
    m: usize,
    successor_node: NodePos,
) -> Option<&'a CapturedBasis> {
    basis_slice.get(m, successor_node)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use cobre_stochastic::{ClassDimensions, ExternalScenarioLibrary};

    use super::{NodeId, assemble_external_opening_noise, fill_external_class};
    use crate::SddpError;

    /// A single-`(stage, k)` external library carrying `values` for its class.
    fn stub_library(
        class: &'static str,
        n_entities: usize,
        stage: usize,
        k: usize,
        values: &[f64],
    ) -> ExternalScenarioLibrary {
        let n_stages = stage + 1;
        let n_scenarios = k + 1;
        let mut lib = ExternalScenarioLibrary::new(
            n_stages,
            n_scenarios,
            n_entities,
            class,
            vec![n_scenarios; n_stages],
        );
        lib.eta_slice_mut(stage, k).copy_from_slice(values);
        lib
    }

    #[test]
    fn assemble_concatenates_per_class_columns_at_expected_offsets() {
        // dims [hydro=2 | load=1 | ncs=2]; each class reads eta_slice(stage=1, k=2).
        let (stage, k) = (1_usize, 2_usize);
        let inflow = stub_library("inflow", 2, stage, k, &[1.0, 2.0]);
        let load = stub_library("load", 1, stage, k, &[3.0]);
        let ncs = stub_library("ncs", 2, stage, k, &[4.0, 5.0]);

        let mut buf = vec![f64::NAN; 1]; // deliberately wrong-length; resize must fix it
        assemble_external_opening_noise(
            ClassDimensions {
                n_hydros: 2,
                n_load_buses: 1,
                n_ncs: 2,
            },
            [Some(&inflow), Some(&load), Some(&ncs)],
            stage,
            k,
            NodeId(7),
            &mut buf,
        )
        .expect("assembly with all libraries present must succeed");

        // The assembled vector is the concatenation of the per-class columns in
        // [hydro | load-bus | ncs] order at cumulative offsets 0, 2, 3.
        assert_eq!(&buf[0..2], inflow.eta_slice(stage, k), "inflow segment");
        assert_eq!(&buf[2..3], load.eta_slice(stage, k), "load segment");
        assert_eq!(&buf[3..5], ncs.eta_slice(stage, k), "ncs segment");
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn assemble_skips_zero_dimension_classes_without_a_library() {
        // Inflow-only external study: load/ncs carry no noise dimension, so their
        // absent libraries are skipped, not an error.
        let (stage, k) = (0_usize, 0_usize);
        let inflow = stub_library("inflow", 3, stage, k, &[7.0, 8.0, 9.0]);

        let mut buf = Vec::new();
        assemble_external_opening_noise(
            ClassDimensions {
                n_hydros: 3,
                n_load_buses: 0,
                n_ncs: 0,
            },
            [Some(&inflow), None, None],
            stage,
            k,
            NodeId(0),
            &mut buf,
        )
        .expect("zero-dimension load/ncs classes must be skipped");
        assert_eq!(buf, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn assemble_missing_library_for_nonempty_class_is_validation_error() {
        // n_hydros = 2 but no inflow library present ⇒ a named Validation error,
        // never a silent fallback to the generated opening tree.
        let mut buf = Vec::new();
        let err = assemble_external_opening_noise(
            ClassDimensions {
                n_hydros: 2,
                n_load_buses: 0,
                n_ncs: 0,
            },
            [None, None, None],
            1,
            2,
            NodeId(7),
            &mut buf,
        )
        .expect_err("a nonempty class with no library must error");
        match err {
            SddpError::Validation(msg) => {
                assert!(msg.contains("node 7"), "message names the node: {msg}");
                assert!(msg.contains("stage 1"), "message names the stage: {msg}");
                assert!(msg.contains("column 2"), "message names the column: {msg}");
                assert!(msg.contains("inflow"), "message names the class: {msg}");
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    #[test]
    fn fill_external_class_empty_segment_no_library_is_ok() {
        // A zero-length segment needs no library — the class carries no noise.
        fill_external_class(&mut [], None, 0, 0, "load", NodeId(3))
            .expect("an empty segment must be a no-op regardless of the library");
    }

    #[test]
    fn fill_external_class_nonempty_no_library_names_node_stage_class() {
        let mut segment = [0.0_f64; 2];
        let err = fill_external_class(&mut segment, None, 4, 5, "ncs", NodeId(9))
            .expect_err("a nonempty segment with no library must error");
        match err {
            SddpError::Validation(msg) => {
                assert!(msg.contains("node 9"), "message names the node: {msg}");
                assert!(msg.contains("stage 4"), "message names the stage: {msg}");
                assert!(msg.contains("column 5"), "message names the column: {msg}");
                assert!(msg.contains("ncs"), "message names the class: {msg}");
            }
            other => panic!("expected SddpError::Validation, got {other:?}"),
        }
    }

    #[test]
    fn fill_external_class_copies_the_declared_column() {
        let (stage, k) = (2_usize, 1_usize);
        let lib = stub_library("inflow", 3, stage, k, &[10.0, 11.0, 12.0]);
        let mut segment = [0.0_f64; 3];
        fill_external_class(&mut segment, Some(&lib), stage, k, "inflow", NodeId(0))
            .expect("copy from a present library must succeed");
        assert_eq!(segment, [10.0, 11.0, 12.0]);
        assert_eq!(&segment, lib.eta_slice(stage, k));
    }
}
