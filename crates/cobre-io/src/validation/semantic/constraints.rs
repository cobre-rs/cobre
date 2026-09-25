//! Layer 5a — generic-constraint-vs-stage-mode semantic validation.
//!
//! Rejects per-block variable references that cannot resolve to a real column on a
//! stage: interior storage boundaries (`HydroStorageInitial` / `HydroStorageFinal`)
//! on parallel stages, and out-of-range block selectors on any per-block reference.

use std::collections::{HashMap, HashSet};

use cobre_core::{
    AffineBound, CoefficientRef, ComputedParameter, EntityId, GenericConstraint, ParameterKind,
    VariableRef, temporal::BlockMode,
};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

/// Rule 20. Layer 5a — rejects per-block variable references that address a column a stage
/// cannot expose, honoring per-`(constraint, stage)` activation.
///
/// `HydroStorageInitial{Some(b)}` references boundary `b` (start of block `b`);
/// `HydroStorageFinal{Some(b)}` references boundary `b + 1` (end of block `b`); a
/// boundary is interior iff `0 < k < K`. On a `Parallel` stage with `K > 1` only
/// the two endpoints (`0`, `K`) exist — the `storage_internal` family is empty — so
/// an interior `Some(..)` reference resolves outside the storage family and is
/// rejected. A `None` selector is the stage endpoint (`S⁰` / `Sᴷ`), which always
/// exists, so it always passes. On a `Parallel` stage with `K > 1`,
/// `HydroEvaporation` is one stage-level quantity: `None` and `Some(0)` resolve to
/// it and pass, while `Some(1..K)` names a block the collapse discards and is
/// rejected. On a `Chronological` stage each block keeps its own evaporation
/// column, so any in-range `Some(b)` passes, but a bare `HydroEvaporation{None}`
/// with `K > 1` is ambiguous (the blocks differ) and is rejected — a block must
/// be named.
///
/// An out-of-range `block_id` (`b >= K`) is rejected on every stage: it would
/// otherwise resolve to no column and silently drop the term.
///
/// A `(constraint, stage)` pair the constraint does not activate (no bound row for
/// that stage) is skipped: no LP row is emitted there, so validating it would
/// false-reject a legitimate mixed-mode study.
pub(super) fn check_per_block_storage_interior_reference(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    for stage in data.stages.stages.iter().filter(|s| s.id >= 0) {
        let k = stage.blocks.len();
        let stage_id = stage.id;
        for constraint in &data.generic_constraints {
            if !constraint_active_on_stage(data, constraint.id.0, stage_id) {
                continue;
            }
            for term in &constraint.expression.terms {
                validate_block_ref(
                    constraint,
                    &term.variable,
                    k,
                    stage_id,
                    stage.block_mode,
                    ctx,
                );
            }
        }
    }
}

/// Whether `constraint` has a bound row at `stage_id` — mirrors
/// `ResolvedGenericConstraintBounds::is_active`; only an active `(constraint, stage)`
/// pair contributes an LP row.
fn constraint_active_on_stage(data: &ParsedData, constraint_id: i32, stage_id: i32) -> bool {
    data.generic_constraint_bounds
        .iter()
        .any(|r| r.constraint_id == constraint_id && r.stage_id == stage_id)
}

/// Rule 51. Layer 5a — warns when a constraint pairs the `max_stored_energy(h)`
/// computed parameter with the mismatched `accumulated_productivity(h)` coefficient
/// for the same hydro `h`: the two ride different evaluators and would not cancel.
/// The matching (cancelling) coefficient for `max_stored_energy` is
/// `integrated_accumulated_productivity`. Warning only — never rejected.
pub(super) fn check_productivity_tag_pairing(data: &ParsedData, ctx: &mut ValidationContext) {
    let computed: HashMap<EntityId, ComputedParameter> = data
        .scalar_parameters
        .iter()
        .filter_map(|p| match p.kind {
            ParameterKind::Computed { computed_spec } => Some((p.id, computed_spec)),
            _ => None,
        })
        .collect();

    for constraint in &data.generic_constraints {
        let mut max_stored_energy: HashSet<EntityId> = HashSet::new();
        let mut accumulated_productivity: HashSet<EntityId> = HashSet::new();

        for id in constraint_parameter_ids(constraint) {
            match computed.get(&id) {
                Some(ComputedParameter::MaxStoredEnergy { hydro_id }) => {
                    max_stored_energy.insert(*hydro_id);
                }
                Some(ComputedParameter::AccumulatedProductivity { hydro_id }) => {
                    accumulated_productivity.insert(*hydro_id);
                }
                _ => {}
            }
        }

        let mut mismatched: Vec<EntityId> = max_stored_energy
            .intersection(&accumulated_productivity)
            .copied()
            .collect();
        mismatched.sort_unstable();

        for hydro_id in mismatched {
            ctx.add_warning(
                ErrorKind::SemanticAmbiguity,
                "constraints/generic_constraints.json",
                Some(&constraint.name),
                format!(
                    "Constraint \"{}\": pairs max_stored_energy({hid}) with \
                     accumulated_productivity({hid}) for hydro {hid}; the two ride \
                     different evaluators and would not cancel. The coefficient \
                     matching max_stored_energy is integrated_accumulated_productivity, \
                     not accumulated_productivity.",
                    constraint.name,
                    hid = hydro_id.0,
                ),
            );
        }
    }
}

/// Every scalar-parameter id a constraint's expression coefficients or affine
/// bounds reference.
fn constraint_parameter_ids(constraint: &GenericConstraint) -> impl Iterator<Item = EntityId> + '_ {
    let term_ids = constraint
        .expression
        .terms
        .iter()
        .filter_map(|term| match term.coefficient {
            CoefficientRef::Parameter(id) => Some(id),
            CoefficientRef::Literal(_) => None,
        });
    let lower_ids = constraint
        .bound_lower_affine
        .iter()
        .flat_map(AffineBound::params);
    let upper_ids = constraint
        .bound_upper_affine
        .iter()
        .flat_map(AffineBound::params);
    term_ids.chain(lower_ids).chain(upper_ids)
}

/// Which per-block storage boundary a term references, and the block selector.
///
/// `Initial` references boundary `block_id`; `Final` references `block_id + 1`.
#[derive(Clone, Copy)]
enum StorageRef {
    Initial(Option<usize>),
    Final(Option<usize>),
}

fn add_constraint_error(
    ctx: &mut ValidationContext,
    constraint: &GenericConstraint,
    message: String,
) {
    ctx.add_error(
        ErrorKind::BusinessRuleViolation,
        "constraints/generic_constraints.json",
        Some(format!("constraint[id={}]", constraint.id.0)),
        message,
    );
}

/// Dispatch a term to its per-block validity check. Storage boundaries get the
/// full interior + out-of-range check; evaporation gets the out-of-range check
/// plus, on a `Parallel` stage with `K > 1`, a reject for any block past the
/// stage-level slot (`Some(1..K)`) and, on a `Chronological` stage with `K > 1`,
/// a reject for the ambiguous bare reference; every other variant is
/// unrestricted.
fn validate_block_ref(
    constraint: &GenericConstraint,
    variable: &VariableRef,
    k: usize,
    stage_id: i32,
    block_mode: BlockMode,
    ctx: &mut ValidationContext,
) {
    if let Some((accessor, storage)) = storage_boundary_ref(variable) {
        validate_storage_ref(constraint, accessor, storage, k, stage_id, block_mode, ctx);
        return;
    }
    match variable {
        VariableRef::HydroEvaporation {
            block_id: Some(b), ..
        } if *b >= k => {
            add_constraint_error(
                ctx,
                constraint,
                format!(
                    "Constraint \"{}\": per-block evaporation reference \
                     `hydro_evaporation({b})` at stage {stage_id} references block {b} \
                     which does not exist at stage {stage_id} (K = {k})",
                    constraint.name
                ),
            );
        }
        VariableRef::HydroEvaporation {
            block_id: Some(b), ..
        } if block_mode == BlockMode::Parallel && k > 1 && *b >= 1 => {
            add_constraint_error(
                ctx,
                constraint,
                format!(
                    "Constraint \"{}\": per-block evaporation reference \
                     `hydro_evaporation({b})` at stage {stage_id} names a block past \
                     the stage-level evaporation, which requires chronological block \
                     mode (stage {stage_id} is parallel with {k} blocks); use block 0 \
                     or no block",
                    constraint.name
                ),
            );
        }
        VariableRef::HydroEvaporation { block_id: None, .. }
            if block_mode == BlockMode::Chronological && k > 1 =>
        {
            add_constraint_error(
                ctx,
                constraint,
                format!(
                    "Constraint \"{}\": stage-level `hydro_evaporation` at chronological \
                     stage {stage_id} is ambiguous — evaporation is per-block there \
                     (K = {k}); name a block, e.g. `hydro_evaporation(<hydro>, 0)`",
                    constraint.name
                ),
            );
        }
        _ => {}
    }
}

/// The accessor name and boundary of a storage-boundary term, `None` for every
/// other variant.
fn storage_boundary_ref(variable: &VariableRef) -> Option<(&'static str, StorageRef)> {
    match variable {
        VariableRef::HydroStorageInitial { block_id, .. } => {
            Some(("hydro_storage_initial", StorageRef::Initial(*block_id)))
        }
        VariableRef::HydroStorageFinal { block_id, .. } => {
            Some(("hydro_storage_final", StorageRef::Final(*block_id)))
        }
        VariableRef::HydroUsefulVolumeInitial { block_id, .. } => Some((
            "hydro_useful_volume_initial",
            StorageRef::Initial(*block_id),
        )),
        VariableRef::HydroUsefulVolumeFinal { block_id, .. } => {
            Some(("hydro_useful_volume_final", StorageRef::Final(*block_id)))
        }
        _ => None,
    }
}

fn validate_storage_ref(
    constraint: &GenericConstraint,
    accessor: &str,
    storage: StorageRef,
    k: usize,
    stage_id: i32,
    block_mode: BlockMode,
    ctx: &mut ValidationContext,
) {
    let (StorageRef::Initial(block_id) | StorageRef::Final(block_id)) = storage;

    if let Some(b) = block_id
        && b >= k
    {
        add_constraint_error(
            ctx,
            constraint,
            format!(
                "Constraint \"{}\": per-block storage reference `{accessor}({b})` at \
                 stage {stage_id} references block {b} which does not exist at \
                 stage {stage_id} (K = {k})",
                constraint.name
            ),
        );
        return;
    }

    match block_mode {
        BlockMode::Chronological => {}
        BlockMode::Parallel => {
            if k <= 1 {
                return;
            }
            let interior = match (storage, block_id) {
                (_, None) => false,
                (StorageRef::Initial(_), Some(b)) => boundary_is_interior(b, k),
                (StorageRef::Final(_), Some(b)) => boundary_is_interior(b + 1, k),
            };
            if interior {
                let block_label = match block_id {
                    Some(b) => format!("{accessor}({b})"),
                    None => format!("{accessor}(all blocks)"),
                };
                add_constraint_error(
                    ctx,
                    constraint,
                    format!(
                        "Constraint \"{}\": per-block storage reference `{block_label}` at \
                         stage {stage_id} resolves to an interior boundary, which requires \
                         chronological block mode (stage {stage_id} is parallel with {k} blocks)",
                        constraint.name
                    ),
                );
            }
        }
    }
}

/// A boundary `k` is interior iff it is neither the stage-initial anchor
/// (`k == 0`) nor the stage-final boundary (`k == K`).
fn boundary_is_interior(k: usize, num_blocks: usize) -> bool {
    k > 0 && k < num_blocks
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use cobre_core::{
        AffineBound, CoefficientRef, ComputedParameter, ConstraintExpression, EntityId,
        GenericConstraint, LinearTerm, ParameterKind, ScalarParameter, SlackConfig, VariableRef,
        temporal::{Block, BlockMode},
    };

    use super::super::validate_semantic_hydro_thermal;
    use super::check_productivity_tag_pairing;
    use crate::ValidationEntry;
    use crate::constraints::GenericConstraintBoundsRow;
    use crate::test_support::*;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    fn make_blocks(k: usize) -> Vec<Block> {
        (0..k)
            .map(|i| Block {
                index: i,
                name: format!("B{i}"),
                duration_hours: 168.0,
            })
            .collect()
    }

    /// Build `ParsedData` with a single stage of `k` blocks in `block_mode` and a
    /// generic constraint whose sole term references the given variant, active on
    /// stage 0 (a bound row exists, so the per-block check runs).
    fn make_data_storage_ref(block_mode: BlockMode, k: usize, variable: VariableRef) -> ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.stages.stages[0].block_mode = block_mode;
        data.stages.stages[0].blocks = make_blocks(k);
        data.generic_constraints = vec![GenericConstraint {
            id: EntityId::from(1),
            name: "storage_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(1.0, variable)],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(0.0),
            bound_upper: None,
        }];
        data
    }

    fn evaporation(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroEvaporation {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn interior_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/generic_constraints.json")
            })
            .map(|e| (*e).clone())
            .collect()
    }

    fn initial(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn final_(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn useful_initial(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn useful_final(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    /// The single error a reference raises, with `accessor` rewritten to the
    /// storage sibling's name so the two messages can be compared verbatim.
    fn sole_error_as_storage(
        block_mode: BlockMode,
        variable: VariableRef,
        accessor: &str,
        storage_accessor: &str,
    ) -> String {
        let errors = interior_errors(&make_data_storage_ref(block_mode, 3, variable));
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains(accessor),
            "message should name `{accessor}`, got: {msg}"
        );
        msg.replace(accessor, storage_accessor)
    }

    #[test]
    fn useful_volume_out_of_range_block_matches_storage_sibling() {
        for block_mode in [BlockMode::Parallel, BlockMode::Chronological] {
            for (useful, storage, accessor, storage_accessor) in [
                (
                    useful_initial(Some(5)),
                    initial(Some(5)),
                    "hydro_useful_volume_initial",
                    "hydro_storage_initial",
                ),
                (
                    useful_final(Some(5)),
                    final_(Some(5)),
                    "hydro_useful_volume_final",
                    "hydro_storage_final",
                ),
            ] {
                assert_eq!(
                    sole_error_as_storage(block_mode, useful, accessor, storage_accessor),
                    sole_error_as_storage(block_mode, storage, storage_accessor, storage_accessor),
                );
            }
        }
    }

    #[test]
    fn useful_volume_parallel_interior_boundary_matches_storage_sibling() {
        // Initial{1} and Final{0} both reference boundary k=1, interior for K=3.
        for (useful, storage, accessor, storage_accessor) in [
            (
                useful_initial(Some(1)),
                initial(Some(1)),
                "hydro_useful_volume_initial",
                "hydro_storage_initial",
            ),
            (
                useful_final(Some(0)),
                final_(Some(0)),
                "hydro_useful_volume_final",
                "hydro_storage_final",
            ),
        ] {
            assert_eq!(
                sole_error_as_storage(BlockMode::Parallel, useful, accessor, storage_accessor),
                sole_error_as_storage(
                    BlockMode::Parallel,
                    storage,
                    storage_accessor,
                    storage_accessor
                ),
            );
        }
    }

    #[test]
    fn non_storage_per_block_reference_is_unrestricted() {
        // An interior block on a parallel stage would be rejected for a storage boundary.
        for block_mode in [BlockMode::Parallel, BlockMode::Chronological] {
            let turbined = VariableRef::HydroTurbined {
                hydro_id: EntityId::from(1),
                block_id: Some(1),
                bus_id: None,
            };
            let data = make_data_storage_ref(block_mode, 3, turbined);
            assert!(interior_errors(&data).is_empty());
        }
    }

    #[test]
    fn useful_volume_valid_references_accepted() {
        for (block_mode, variable) in [
            (BlockMode::Parallel, useful_initial(Some(0))),
            (BlockMode::Parallel, useful_final(Some(2))),
            (BlockMode::Parallel, useful_initial(None)),
            (BlockMode::Parallel, useful_final(None)),
            (BlockMode::Chronological, useful_initial(Some(1))),
            (BlockMode::Chronological, useful_final(Some(1))),
        ] {
            let data = make_data_storage_ref(block_mode, 3, variable);
            assert!(interior_errors(&data).is_empty());
        }
    }

    #[test]
    fn parallel_k3_interior_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(1)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("storage_constraint") && msg.contains("interior boundary"),
            "message should name the constraint and the interior requirement, got: {msg}"
        );
        assert!(
            msg.contains("parallel with 3 blocks"),
            "message should state the parallel mode and block count, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_interior_final_rejected() {
        // Final{0} references boundary k=1, interior for K=3.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(0)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
    }

    #[test]
    fn parallel_k3_endpoint_initial_accepted() {
        // Initial{0} references boundary k=0, the S⁰ endpoint.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_endpoint_final_accepted() {
        // Final{2} references boundary k=3=K, the Sᴷ endpoint.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(2)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_none_initial_accepted() {
        // Initial{None} is the S⁰ endpoint, which exists on a parallel stage.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_none_final_accepted() {
        // Final{None} is the Sᴷ endpoint, which exists on a parallel stage.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k1_all_references_accepted() {
        for variable in [
            initial(Some(0)),
            final_(Some(0)),
            initial(None),
            final_(None),
        ] {
            let data = make_data_storage_ref(BlockMode::Parallel, 1, variable);
            assert!(
                interior_errors(&data).is_empty(),
                "K=1 parallel reference must be accepted"
            );
        }
    }

    #[test]
    fn chronological_k3_all_references_accepted() {
        for variable in [
            initial(Some(0)),
            initial(Some(1)),
            initial(Some(2)),
            final_(Some(0)),
            final_(Some(1)),
            final_(Some(2)),
            initial(None),
            final_(None),
        ] {
            let data = make_data_storage_ref(BlockMode::Chronological, 3, variable);
            assert!(
                interior_errors(&data).is_empty(),
                "chronological reference must be accepted"
            );
        }
    }

    #[test]
    fn parallel_k3_out_of_range_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("block 5 which does not exist") && msg.contains("K = 3"),
            "message should be the out-of-range message, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_out_of_range_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("block 5 which does not exist") && msg.contains("K = 3"),
            "message should be the out-of-range message, got: {msg}"
        );
    }

    #[test]
    fn chronological_k3_out_of_range_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, initial(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(errors[0].message.contains("block 5 which does not exist"));
    }

    #[test]
    fn chronological_k3_out_of_range_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, final_(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(errors[0].message.contains("block 5 which does not exist"));
    }

    #[test]
    fn parallel_k3_interior_ref_on_inactive_constraint_accepted() {
        // An interior reference the constraint never activates on this stage
        // (no bound row) emits no LP row, so it must not be rejected.
        let mut data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(1)));
        data.generic_constraint_bounds.clear();
        assert!(
            interior_errors(&data).is_empty(),
            "inactive (constraint, stage) pair must be skipped"
        );
    }

    #[test]
    fn mixed_mode_interior_ref_active_only_on_chronological_stage_accepted() {
        // Stage 0 chronological (K=3), stage 1 parallel (K=3); the constraint
        // references an interior boundary but is active only on the chronological
        // stage, so the parallel stage must not be flagged.
        let mut data = make_data_storage_ref(BlockMode::Chronological, 3, initial(Some(1)));
        data.stages.stages.push({
            let mut s = data.stages.stages[0].clone();
            s.id = 1;
            s.block_mode = BlockMode::Parallel;
            s
        });
        assert!(
            interior_errors(&data).is_empty(),
            "constraint inactive on the parallel stage must not be flagged"
        );
    }

    #[test]
    fn parallel_k3_evaporation_block_past_the_stage_slot_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(2)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("hydro_evaporation(2)")
                && msg.contains("parallel with 3 blocks")
                && msg.contains("block 0"),
            "message should name the block, the parallel mode and block count, \
             and the fix, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_evaporation_block_zero_accepted() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_evaporation_block_reference_skipped_where_inactive() {
        // A block-past-the-slot reference the constraint never activates on this
        // stage (no bound row) emits no LP row, so it must not be rejected.
        let mut data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(2)));
        data.generic_constraint_bounds.clear();
        assert!(
            interior_errors(&data).is_empty(),
            "inactive (constraint, stage) pair must be skipped"
        );
    }

    #[test]
    fn evaporation_bare_accepted_in_parallel() {
        // In parallel mode every block shares the stage endpoints, so bare
        // evaporation is the well-defined stage rate.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_accepted_in_chronological_k1() {
        // K == 1: the single block is the whole stage, so bare is unambiguous.
        let data = make_data_storage_ref(BlockMode::Chronological, 1, evaporation(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_block_accepted_in_chronological() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, evaporation(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_rejected_in_chronological_multiblock() {
        // K > 1 chronological: blocks differ, so a bare (stage-level) reference is
        // ambiguous and must name a block.
        let data = make_data_storage_ref(BlockMode::Chronological, 3, evaporation(None));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(
            errors[0].message.contains("ambiguous") && errors[0].message.contains("name a block"),
            "message should ask for an explicit block, got: {}",
            errors[0].message
        );
    }

    #[test]
    fn evaporation_out_of_range_block_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("hydro_evaporation(5)") && msg.contains("block 5 which does not exist"),
            "message should be the evaporation out-of-range message, got: {msg}"
        );
    }

    // ── check_productivity_tag_pairing ──────────────────────────────────────

    fn computed_scalar_param(id: i32, name: &str, spec: ComputedParameter) -> ScalarParameter {
        ScalarParameter {
            id: EntityId::from(id),
            name: name.to_string(),
            kind: ParameterKind::Computed {
                computed_spec: spec,
            },
        }
    }

    fn param_term(coefficient_id: EntityId) -> LinearTerm {
        LinearTerm {
            coefficient: CoefficientRef::Parameter(coefficient_id),
            scale: 1.0,
            variable: VariableRef::HydroTurbined {
                hydro_id: EntityId::from(1),
                block_id: None,
                bus_id: None,
            },
        }
    }

    fn constraint_with_lhs_and_upper_bound(
        lhs_param: EntityId,
        upper_bound_param: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: EntityId::from(1),
            name: "energy_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![param_term(lhs_param)],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: Some(AffineBound::single(upper_bound_param)),
        }
    }

    fn constraint_with_two_lhs_terms(param_a: EntityId, param_b: EntityId) -> GenericConstraint {
        GenericConstraint {
            id: EntityId::from(1),
            name: "energy_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![param_term(param_a), param_term(param_b)],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    fn constraint_with_literal_term() -> GenericConstraint {
        GenericConstraint {
            id: EntityId::from(1),
            name: "no_computed_params".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(
                    1.0,
                    VariableRef::HydroTurbined {
                        hydro_id: EntityId::from(1),
                        block_id: None,
                        bus_id: None,
                    },
                )],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    /// Build `ParsedData` with a single hydro/stage, the given scalar parameters,
    /// and a single generic constraint.
    fn make_data_with_constraint(
        scalar_parameters: Vec<ScalarParameter>,
        constraint: GenericConstraint,
    ) -> ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.scalar_parameters = scalar_parameters;
        data.generic_constraints = vec![constraint];
        data
    }

    #[test]
    fn mismatched_pairing_same_hydro_warns_once() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        let messages: Vec<String> = ctx
            .warnings()
            .iter()
            .map(|e| {
                assert_eq!(e.kind, ErrorKind::SemanticAmbiguity);
                e.message.clone()
            })
            .collect();
        assert_eq!(messages.len(), 1, "expected one warning, got: {messages:?}");
        assert!(
            messages[0].contains('7'),
            "message should name hydro 7, got: {}",
            messages[0]
        );
        assert!(
            messages[0].contains("integrated_accumulated_productivity"),
            "message should name the matching coefficient, got: {}",
            messages[0]
        );

        assert!(
            ctx.into_result().is_ok(),
            "a warning must not fail into_result"
        );
    }

    #[test]
    fn mismatched_pairing_both_as_lhs_coefficients_warns_once() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_two_lhs_terms(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        let warnings = ctx.warnings();
        assert_eq!(warnings.len(), 1, "expected one warning, got: {warnings:?}");
        assert_eq!(warnings[0].kind, ErrorKind::SemanticAmbiguity);
    }

    #[test]
    fn matching_integrated_coefficient_no_warning() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum_int",
                    ComputedParameter::IntegratedAccumulatedProductivity {
                        hydro_id: EntityId::from(7),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(
            ctx.warnings().is_empty(),
            "the matching integrated_accumulated_productivity coefficient must not warn"
        );
    }

    #[test]
    fn different_hydro_pairing_no_warning() {
        let data = make_data_with_constraint(
            vec![
                computed_scalar_param(
                    1,
                    "rho_acum_h9",
                    ComputedParameter::AccumulatedProductivity {
                        hydro_id: EntityId::from(9),
                    },
                ),
                computed_scalar_param(
                    2,
                    "e_max_h7",
                    ComputedParameter::MaxStoredEnergy {
                        hydro_id: EntityId::from(7),
                    },
                ),
            ],
            constraint_with_lhs_and_upper_bound(EntityId::from(1), EntityId::from(2)),
        );
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(
            ctx.warnings().is_empty(),
            "a different-hydro pairing must not warn"
        );
    }

    #[test]
    fn no_computed_scalar_parameters_no_warning_no_panic() {
        let data = make_data_with_constraint(vec![], constraint_with_literal_term());
        let mut ctx = ValidationContext::new();
        check_productivity_tag_pairing(&data, &mut ctx);

        assert!(ctx.warnings().is_empty());
    }
}
