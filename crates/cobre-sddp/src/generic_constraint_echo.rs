//! Assemble the resolved generic-constraint echo from the built stage templates.
//!
//! Projects `Vec<GenericConstraintEchoRow>` from the data the LP builder actually
//! consumed: the per-stage row entries (bounds, block, slack), the resolved
//! parameter table (coefficients), and the system's generic constraints (LHS
//! terms). One row per resolved LHS term per active `(stage, block)`; a term-less
//! constraint contributes a single placeholder row. Output is in canonical
//! `(constraint, stage, block, term)` order and declaration-order invariant.

use cobre_core::{CoefficientRef, EntityId, GenericConstraint, LinearTerm, System, VariableRef};
use cobre_io::GenericConstraintEchoRow;

use crate::ResolvedParameters;
use crate::StudySetup;
use crate::lp::builder::GenericConstraintRowEntry;

/// Build the resolved generic-constraint echo rows for `setup`/`system` in
/// canonical `(constraint, stage, block, term)` order.
#[must_use]
pub fn build_generic_constraint_echo_rows(
    setup: &StudySetup,
    system: &System,
) -> Vec<GenericConstraintEchoRow> {
    build_echo_rows_from_parts(
        &setup
            .inputs
            .stage_data
            .stage_templates
            .generic_constraint_row_entries,
        &setup.inputs.study_stage_ids,
        &setup.resolved_parameters,
        system.generic_constraints(),
    )
}

/// Assemble the echo rows from the raw parts (slices in, canonically-ordered `Vec`
/// out) — decoupled from `StudySetup` so the assembly is unit-testable without a
/// full setup. `entries_per_stage` is indexed `[stage][row]`; `stage_ids[s]` is
/// the study stage id; `constraints` is joined by `entry.constraint_idx`.
// Rationale (cast lints): block and term indices are stage-local counts far below
// i32::MAX; the echo's i32 columns are Parquet Int32.
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn build_echo_rows_from_parts(
    entries_per_stage: &[Vec<GenericConstraintRowEntry>],
    stage_ids: &[i32],
    resolved: &ResolvedParameters,
    constraints: &[GenericConstraint],
) -> Vec<GenericConstraintEchoRow> {
    let mut rows: Vec<GenericConstraintEchoRow> = Vec::new();

    for (stage_idx, entries) in entries_per_stage.iter().enumerate() {
        let stage_id = stage_ids[stage_idx];
        for entry in entries {
            let constraint = &constraints[entry.constraint_idx];
            let block_id = if entry.is_stage_level {
                None
            } else {
                Some(entry.block_idx as i32)
            };
            let shape = derived_shape(entry.bound_lower, entry.bound_upper);
            let slack_penalty = if entry.slack_enabled {
                Some(entry.slack_penalty)
            } else {
                None
            };

            if constraint.expression.terms.is_empty() {
                rows.push(GenericConstraintEchoRow {
                    stage_id,
                    block_id,
                    constraint_id: constraint.id.0,
                    constraint_name: constraint.name.clone(),
                    term_index: None,
                    variable_kind: None,
                    variable: None,
                    coefficient: None,
                    bound_lower: entry.bound_lower,
                    bound_upper: entry.bound_upper,
                    derived_shape: shape.to_string(),
                    slack_enabled: entry.slack_enabled,
                    slack_penalty,
                });
                continue;
            }

            for (term_idx, term) in constraint.expression.terms.iter().enumerate() {
                let (kind, rendered) = render_variable(&term.variable);
                rows.push(GenericConstraintEchoRow {
                    stage_id,
                    block_id,
                    constraint_id: constraint.id.0,
                    constraint_name: constraint.name.clone(),
                    term_index: Some(term_idx as i32),
                    variable_kind: Some(kind.to_string()),
                    variable: Some(rendered),
                    coefficient: Some(resolve_coef(term, resolved, stage_idx, entry.block_idx)),
                    bound_lower: entry.bound_lower,
                    bound_upper: entry.bound_upper,
                    derived_shape: shape.to_string(),
                    slack_enabled: entry.slack_enabled,
                    slack_penalty,
                });
            }
        }
    }

    // Canonical order is a function of the resolved model, so it does not depend on
    // the stage-major walk order — the declaration-order-invariance guarantee.
    rows.sort_by_key(|r| {
        (
            r.constraint_id,
            r.stage_id,
            r.block_id.unwrap_or(-1),
            r.term_index.unwrap_or(-1),
        )
    });
    rows
}

/// Resolve a term's coefficient exactly as `fill_generic_constraint_entries`:
/// `resolve(coefficient, stage, block) * scale`, reading the parameter at the
/// row's own block so a block-varying coefficient echoes its per-block value, not
/// a stage-level one. The per-column `multiplier` from `resolve_variable_ref` is a
/// column-expansion detail deliberately absent here — the echo is the term-level
/// flat form, not the per-cell LP-column expansion.
fn resolve_coef(
    term: &LinearTerm,
    resolved: &ResolvedParameters,
    stage_idx: usize,
    block_idx: usize,
) -> f64 {
    let base = match term.coefficient {
        CoefficientRef::Literal(v) => v,
        CoefficientRef::Parameter(id) => resolved.get(id, stage_idx, block_idx),
    };
    base * term.scale
}

/// Shape label from the bound null-pattern (never an authored sense).
fn derived_shape(bound_lower: Option<f64>, bound_upper: Option<f64>) -> &'static str {
    match (bound_lower, bound_upper) {
        (Some(l), Some(u)) => {
            // Exact (bit-identical) equality, never a tolerance — a narrow band
            // must stay "band".
            if l.to_bits() == u.to_bits() {
                "equality"
            } else {
                "band"
            }
        }
        (Some(_), None) => "floor",
        (None, Some(_)) => "cap",
        (None, None) => {
            debug_assert!(
                false,
                "generic-constraint row has neither bound endpoint; referential validation guarantees at least one"
            );
            "open"
        }
    }
}

/// Return `(kind_tag, rendered_label)` for a variable reference — the `"kind"`
/// discriminant and the `"kind(args)"` display string. The exhaustive `match`
/// makes a new [`VariableRef`] variant a compile error.
fn render_variable(v: &VariableRef) -> (&'static str, String) {
    let (kind, args) = match *v {
        VariableRef::HydroStorage { hydro_id } => ("hydro_storage", format!("id={}", hydro_id.0)),
        VariableRef::HydroTurbined {
            hydro_id,
            block_id,
            bus_id,
        } => (
            "hydro_turbined",
            format!(
                "id={}, bus={}, block={}",
                hydro_id.0,
                bus_label(bus_id),
                block_label(block_id)
            ),
        ),
        VariableRef::HydroSpillage { hydro_id, block_id } => (
            "hydro_spillage",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroDiversion { hydro_id, block_id } => (
            "hydro_diversion",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroOutflow { hydro_id, block_id } => (
            "hydro_outflow",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroGeneration {
            hydro_id,
            block_id,
            bus_id,
        } => (
            "hydro_generation",
            format!(
                "id={}, bus={}, block={}",
                hydro_id.0,
                bus_label(bus_id),
                block_label(block_id)
            ),
        ),
        VariableRef::HydroEvaporation { hydro_id, block_id } => (
            "hydro_evaporation",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroWithdrawal { hydro_id } => {
            ("hydro_withdrawal", format!("id={}", hydro_id.0))
        }
        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => (
            "thermal_generation",
            format!("id={}, block={}", thermal_id.0, block_label(block_id)),
        ),
        VariableRef::LineDirect { line_id, block_id } => (
            "line_direct",
            format!("id={}, block={}", line_id.0, block_label(block_id)),
        ),
        VariableRef::LineReverse { line_id, block_id } => (
            "line_reverse",
            format!("id={}, block={}", line_id.0, block_label(block_id)),
        ),
        VariableRef::LineExchange { line_id, block_id } => (
            "line_exchange",
            format!("id={}, block={}", line_id.0, block_label(block_id)),
        ),
        VariableRef::BusDeficit { bus_id, block_id } => (
            "bus_deficit",
            format!("id={}, block={}", bus_id.0, block_label(block_id)),
        ),
        VariableRef::BusExcess { bus_id, block_id } => (
            "bus_excess",
            format!("id={}, block={}", bus_id.0, block_label(block_id)),
        ),
        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => (
            "pumping_flow",
            format!("id={}, block={}", station_id.0, block_label(block_id)),
        ),
        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => (
            "pumping_power",
            format!("id={}, block={}", station_id.0, block_label(block_id)),
        ),
        VariableRef::ContractImport {
            contract_id,
            block_id,
        } => (
            "contract_import",
            format!("id={}, block={}", contract_id.0, block_label(block_id)),
        ),
        VariableRef::ContractExport {
            contract_id,
            block_id,
        } => (
            "contract_export",
            format!("id={}, block={}", contract_id.0, block_label(block_id)),
        ),
        VariableRef::NonControllableGeneration {
            source_id,
            block_id,
        } => (
            "non_controllable_generation",
            format!("id={}, block={}", source_id.0, block_label(block_id)),
        ),
        VariableRef::NonControllableCurtailment {
            source_id,
            block_id,
        } => (
            "non_controllable_curtailment",
            format!("id={}, block={}", source_id.0, block_label(block_id)),
        ),
        VariableRef::AnticipatedDecision { thermal_id } => {
            ("anticipated_decision", format!("id={}", thermal_id.0))
        }
        VariableRef::HydroInflow { hydro_id, block_id } => (
            "hydro_inflow",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroStorageInitial { hydro_id, block_id } => (
            "hydro_storage_initial",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroStorageFinal { hydro_id, block_id } => (
            "hydro_storage_final",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroUsefulVolumeInitial { hydro_id, block_id } => (
            "hydro_useful_volume_initial",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
        VariableRef::HydroUsefulVolumeFinal { hydro_id, block_id } => (
            "hydro_useful_volume_final",
            format!("id={}, block={}", hydro_id.0, block_label(block_id)),
        ),
    };
    (kind, format!("{kind}({args})"))
}

/// `Some(k)` renders the block index; `None` renders `all`.
fn block_label(block_id: Option<usize>) -> String {
    block_id.map_or_else(|| "all".to_string(), |k| k.to_string())
}

/// `Some(b)` renders the bus id; `None` renders `all` (the plant's cells summed).
fn bus_label(bus_id: Option<EntityId>) -> String {
    bus_id.map_or_else(|| "all".to_string(), |b| b.0.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cobre_core::{ConstraintExpression, SlackConfig};

    fn entry(
        constraint_idx: usize,
        entity_id: i32,
        block_idx: usize,
        is_stage_level: bool,
        bound_lower: Option<f64>,
        bound_upper: Option<f64>,
        slack_enabled: bool,
        slack_penalty: f64,
    ) -> GenericConstraintRowEntry {
        GenericConstraintRowEntry {
            constraint_idx,
            entity_id,
            block_idx,
            is_stage_level,
            bound_lower,
            bound_upper,
            slack_enabled,
            slack_penalty,
            slack_plus_col: None,
            slack_minus_col: None,
        }
    }

    fn constraint(id: i32, name: &str, terms: Vec<LinearTerm>) -> GenericConstraint {
        GenericConstraint {
            id: EntityId(id),
            name: name.to_string(),
            description: None,
            expression: ConstraintExpression { terms },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_affine: None,
            bound_upper_affine: None,
        }
    }

    fn thermal(id: i32) -> VariableRef {
        VariableRef::ThermalGeneration {
            thermal_id: EntityId(id),
            block_id: None,
        }
    }

    /// An upper-only stage-level entry (the d13 shape) renders a `cap` row with
    /// `block_id = None`, one resolved term, and the entry's slack config.
    #[test]
    fn cap_shape_stage_level_thermal_row() {
        let constraints = vec![constraint(
            13,
            "d13",
            vec![LinearTerm::literal(1.0, thermal(0))],
        )];
        let entries = vec![vec![entry(0, 13, 0, true, None, Some(10.0), true, 5000.0)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(&entries, &[7], &resolved, &constraints);

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.stage_id, 7);
        assert_eq!(r.block_id, None);
        assert_eq!(r.constraint_id, 13);
        assert_eq!(r.constraint_name, "d13");
        assert_eq!(r.derived_shape, "cap");
        assert_eq!(r.bound_lower, None);
        assert_eq!(r.bound_upper, Some(10.0));
        assert_eq!(r.term_index, Some(0));
        assert_eq!(r.variable_kind.as_deref(), Some("thermal_generation"));
        assert_eq!(
            r.variable.as_deref(),
            Some("thermal_generation(id=0, block=all)")
        );
        assert_eq!(r.coefficient, Some(1.0));
        assert!(r.slack_enabled);
        assert_eq!(r.slack_penalty, Some(5000.0));
    }

    /// The echo shows the POST-NORMALIZATION flat form, not the authored
    /// `LHS op RHS` string: `thermal_generation(0) <= 73` echoes one term and
    /// the resolved `bound_upper` — there is no authored-expression field in
    /// [`GenericConstraintEchoRow`] to diverge from.
    #[test]
    fn constant_rhs_echoes_resolved_flat_form_not_authored_string() {
        let constraints = vec![constraint(
            1,
            "cap73",
            vec![LinearTerm::literal(1.0, thermal(0))],
        )];
        let entries = vec![vec![entry(0, 1, 0, true, None, Some(73.0), false, 0.0)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(&entries, &[0], &resolved, &constraints);

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.term_index, Some(0));
        assert_eq!(r.variable_kind.as_deref(), Some("thermal_generation"));
        assert_eq!(r.bound_upper, Some(73.0));
        assert_eq!(r.bound_lower, None);
        assert_eq!(r.derived_shape, "cap");
    }

    /// A folded endpoint (a parquet base composed with an inline affine
    /// remainder) echoes the already-composed `f64` verbatim — the echo reads
    /// the resolved bound, with no separate base/remainder breakdown.
    #[test]
    fn folded_endpoint_echoes_composed_value() {
        let constraints = vec![constraint(
            2,
            "folded",
            vec![LinearTerm::literal(1.0, thermal(0))],
        )];
        let entries = vec![vec![entry(0, 2, 0, true, None, Some(95.0), false, 0.0)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(&entries, &[0], &resolved, &constraints);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_upper, Some(95.0));
        assert_eq!(rows[0].derived_shape, "cap");
    }

    /// A `hydro_useful_volume_final` constraint's `V_lo`-folded lower bound
    /// (already composed by the layout fold before the row entry is built)
    /// reaches `GenericConstraintEchoRow.bound_lower` verbatim, not only the
    /// internal `generic_constraint_rows` layout row it is read from.
    #[test]
    fn useful_volume_folded_lower_bound_reaches_the_echo_row() {
        let hydro_ref = VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(9),
            block_id: None,
        };
        let constraints = vec![constraint(
            9,
            "useful_volume_floor",
            vec![LinearTerm::literal(1.0, hydro_ref)],
        )];
        // 20.0 (raw) + 1.0 * V_lo(9) = 32.5, already folded by the time it reaches
        // this row entry.
        let entries = vec![vec![entry(0, 9, 0, true, Some(32.5), None, false, 0.0)]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(&entries, &[0], &resolved, &constraints);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].bound_lower, Some(32.5));
        assert_eq!(rows[0].derived_shape, "floor");
        assert_eq!(
            rows[0].variable_kind.as_deref(),
            Some("hydro_useful_volume_final")
        );
    }

    /// A two-endpoint per-block entry (the d54 shape) renders a `band` row with
    /// `block_id = Some(block_idx)` and both bounds carried through.
    #[test]
    fn band_shape_per_block_row() {
        let constraints = vec![constraint(
            54,
            "d54",
            vec![LinearTerm::literal(1.0, thermal(3))],
        )];
        let entries = vec![vec![entry(
            0,
            54,
            1,
            false,
            Some(35.0),
            Some(100.0),
            false,
            0.0,
        )]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(&entries, &[2], &resolved, &constraints);

        assert_eq!(rows.len(), 1);
        let r = &rows[0];
        assert_eq!(r.derived_shape, "band");
        assert_eq!(r.block_id, Some(1));
        assert_eq!(r.bound_lower, Some(35.0));
        assert_eq!(r.bound_upper, Some(100.0));
        assert!(!r.slack_enabled);
        assert_eq!(r.slack_penalty, None);
    }

    /// A `Parameter(id)` term resolves to `resolved.get(id, stage, block) * scale`,
    /// matching `fill_generic_constraint_entries`, at each stage.
    #[test]
    fn parameter_coefficient_resolves_like_lp() {
        let scale = 2.5;
        let constraints = vec![constraint(
            1,
            "p",
            vec![LinearTerm::parameter(EntityId(42), scale, thermal(0))],
        )];
        let entries = vec![
            vec![entry(0, 1, 0, true, Some(0.0), None, false, 0.0)],
            vec![entry(0, 1, 0, true, Some(0.0), None, false, 0.0)],
        ];
        // Two stages, one block each (length-1 inner), so any block broadcasts.
        let resolved = ResolvedParameters {
            per_param: vec![vec![vec![2.0], vec![3.0]]],
            id_to_slot: vec![(42, 0)],
            ..Default::default()
        };

        let rows = build_echo_rows_from_parts(&entries, &[0, 1], &resolved, &constraints);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].stage_id, 0);
        assert_eq!(
            rows[0].coefficient,
            Some(resolved.get(EntityId(42), 0, 0) * scale)
        );
        assert_eq!(rows[1].stage_id, 1);
        assert_eq!(
            rows[1].coefficient,
            Some(resolved.get(EntityId(42), 1, 0) * scale)
        );
    }

    /// A block-varying (`PerStageBlock`) coefficient echoes its own block's value:
    /// the block-0 and block-1 rows carry the distinct per-block coefficients, not
    /// a single stage-level one.
    #[test]
    fn parameter_coefficient_reads_the_rows_own_block() {
        let scale = 1.0;
        let constraints = vec![constraint(
            1,
            "per_block",
            vec![LinearTerm::parameter(EntityId(42), scale, thermal(0))],
        )];
        // One stage, two per-block rows (is_stage_level = false) at blocks 0 and 1.
        let entries = vec![vec![
            entry(0, 1, 0, false, Some(0.0), None, false, 0.0),
            entry(0, 1, 1, false, Some(0.0), None, false, 0.0),
        ]];
        // Slot 0, stage 0 carries two blocks: [10.0, 20.0].
        let resolved = ResolvedParameters {
            per_param: vec![vec![vec![10.0, 20.0]]],
            id_to_slot: vec![(42, 0)],
            ..Default::default()
        };

        let rows = build_echo_rows_from_parts(&entries, &[0], &resolved, &constraints);

        assert_eq!(rows.len(), 2);
        let block0 = rows.iter().find(|r| r.block_id == Some(0)).unwrap();
        let block1 = rows.iter().find(|r| r.block_id == Some(1)).unwrap();
        assert_eq!(block0.coefficient, Some(10.0));
        assert_eq!(block1.coefficient, Some(20.0));
        assert_ne!(
            block0.coefficient, block1.coefficient,
            "a block-varying coefficient must differ between the block-0 and block-1 rows"
        );
    }

    /// Two `constraints` slices differing only in element order yield
    /// element-for-element identical rows (declaration-order invariance).
    #[test]
    fn declaration_order_invariance_two_orderings() {
        let c1 = constraint(1, "one", vec![LinearTerm::literal(1.0, thermal(0))]);
        let c2 = constraint(
            2,
            "two",
            vec![LinearTerm::literal(
                2.0,
                VariableRef::HydroStorage {
                    hydro_id: EntityId(5),
                },
            )],
        );
        let resolved = ResolvedParameters::default();

        let constraints_a = vec![c1.clone(), c2.clone()];
        let entries_a = vec![vec![
            entry(0, 1, 0, true, None, Some(10.0), false, 0.0),
            entry(1, 2, 0, true, Some(5.0), None, false, 0.0),
        ]];

        // Reverse the constraint order AND the entry order, with each entry's
        // `constraint_idx` re-pointed so it still names its own logical constraint.
        let constraints_b = vec![c2, c1];
        let entries_b = vec![vec![
            entry(0, 2, 0, true, Some(5.0), None, false, 0.0),
            entry(1, 1, 0, true, None, Some(10.0), false, 0.0),
        ]];

        let rows_a = build_echo_rows_from_parts(&entries_a, &[0], &resolved, &constraints_a);
        let rows_b = build_echo_rows_from_parts(&entries_b, &[0], &resolved, &constraints_b);

        assert_eq!(
            format!("{rows_a:?}"),
            format!("{rows_b:?}"),
            "reordering the constraints must not change the emitted rows"
        );
    }

    /// A constant-only (empty-`terms`) constraint yields exactly one placeholder
    /// row per `(stage, block)`, with every per-term column `None`.
    #[test]
    fn empty_terms_constraint_emits_one_placeholder_per_block() {
        let constraints = vec![constraint(7, "const_only", vec![])];
        let entries = vec![vec![
            entry(0, 7, 0, false, Some(5.0), None, false, 0.0),
            entry(0, 7, 1, false, Some(5.0), None, false, 0.0),
        ]];
        let resolved = ResolvedParameters::default();

        let rows = build_echo_rows_from_parts(&entries, &[0], &resolved, &constraints);

        assert_eq!(rows.len(), 2, "one placeholder row per (stage, block)");
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.block_id, Some(i as i32));
            assert_eq!(r.term_index, None);
            assert_eq!(r.variable_kind, None);
            assert_eq!(r.variable, None);
            assert_eq!(r.coefficient, None);
            assert_eq!(r.derived_shape, "floor");
        }
    }

    /// The four reachable null-patterns map to their shape labels.
    #[test]
    fn derived_shape_covers_each_null_pattern() {
        assert_eq!(derived_shape(Some(1.0), None), "floor");
        assert_eq!(derived_shape(None, Some(1.0)), "cap");
        assert_eq!(derived_shape(Some(35.0), Some(100.0)), "band");
        assert_eq!(derived_shape(Some(50.0), Some(50.0)), "equality");
    }

    /// Rendered labels match the canonical `kind(args)` form, including `block=all`
    /// for a `None` block and a bus selector on a split plant.
    #[test]
    fn render_variable_labels_are_canonical() {
        let (kind, r) = render_variable(&VariableRef::ThermalGeneration {
            thermal_id: EntityId(0),
            block_id: None,
        });
        assert_eq!(kind, "thermal_generation");
        assert_eq!(r, "thermal_generation(id=0, block=all)");

        let (_, r) = render_variable(&VariableRef::HydroStorage {
            hydro_id: EntityId(66),
        });
        assert_eq!(r, "hydro_storage(id=66)");

        let (_, r) = render_variable(&VariableRef::HydroGeneration {
            hydro_id: EntityId(66),
            block_id: Some(2),
            bus_id: Some(EntityId(1)),
        });
        assert_eq!(r, "hydro_generation(id=66, bus=1, block=2)");

        let (kind, r) = render_variable(&VariableRef::HydroUsefulVolumeInitial {
            hydro_id: EntityId(66),
            block_id: None,
        });
        assert_eq!(kind, "hydro_useful_volume_initial");
        assert_eq!(r, "hydro_useful_volume_initial(id=66, block=all)");

        let (kind, r) = render_variable(&VariableRef::HydroUsefulVolumeFinal {
            hydro_id: EntityId(66),
            block_id: Some(2),
        });
        assert_eq!(kind, "hydro_useful_volume_final");
        assert_eq!(r, "hydro_useful_volume_final(id=66, block=2)");
    }
}
