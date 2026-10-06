//! Scalar parameter cross-validation against the parsed hydro slice.
//!
//! Provides [`validate_scalar_parameters`], which checks a [`Vec<ScalarParameter>`] that has
//! already passed the per-file structural and schema validations from
//! `constraints/generic_parameters.json`. Its rules are the `scalar_parameters.*` entries of
//! [`RULES`](super::rules::RULES); the uniqueness rule is defense in depth against callers
//! that bypass the loader.
//!
//! The function never short-circuits.  Every violation is appended to `ctx`
//! before the function returns, so the caller receives a complete diagnostic
//! list in a single pass.

use std::collections::HashSet;

use cobre_core::{ComputedParameter, EntityId, Hydro, ParameterKind, ScalarParameter};

use super::{ValidationContext, rules};

// ── Public entry point ────────────────────────────────────────────────────────

/// Cross-validates a parameter list against the parsed hydros and stage count
/// (the `scalar_parameters.*` rules).
///
/// Appends one [`super::ValidationEntry`] to `ctx` per violation found; the
/// function is infallible — all results flow through `ctx`. `n_stages` counts
/// study stages (`stage.id >= 0`).
pub fn validate_scalar_parameters(
    parameters: &[ScalarParameter],
    hydros: &[Hydro],
    n_stages: usize,
    ctx: &mut ValidationContext,
) {
    let hydro_ids: HashSet<EntityId> = hydros.iter().map(|h| h.id).collect();

    check_computed_hydro_references(parameters, &hydro_ids, ctx);
    check_per_stage_lengths(parameters, n_stages, ctx);
    check_global_uniqueness(parameters, ctx);
}

// ── Private helpers ───────────────────────────────────────────────────────────

fn check_computed_hydro_references(
    parameters: &[ScalarParameter],
    hydro_ids: &HashSet<EntityId>,
    ctx: &mut ValidationContext,
) {
    for param in parameters {
        if let ParameterKind::Computed { computed_spec: c } = param.kind {
            let hid = hydro_id_of(c);
            if !hydro_ids.contains(&hid) {
                ctx.emit(
                    &rules::SCALAR_PARAMETER_UNDECLARED_HYDRO,
                    "constraints/generic_parameters.json",
                    Some(format!("{}.computed_spec.hydro_id", param.name)),
                    format!(
                        "parameter '{}' references non-existent hydro id {}",
                        param.name, hid.0
                    ),
                );
            }
        }
    }
}

fn check_per_stage_lengths(
    parameters: &[ScalarParameter],
    n_stages: usize,
    ctx: &mut ValidationContext,
) {
    for param in parameters {
        if let ParameterKind::PerStage { ref values } = param.kind {
            let actual = values.len();
            if actual != n_stages {
                ctx.emit(
                    &rules::SCALAR_PARAMETER_STAGE_COUNT,
                    "constraints/generic_parameters.json",
                    Some(param.name.as_str()),
                    format!(
                        "parameter '{}' has {} values but expected {} (n_stages)",
                        param.name, actual, n_stages
                    ),
                );
            }
        }
    }
}

fn check_global_uniqueness(parameters: &[ScalarParameter], ctx: &mut ValidationContext) {
    let mut seen_ids: HashSet<EntityId> = HashSet::with_capacity(parameters.len());
    let mut seen_names: HashSet<&str> = HashSet::with_capacity(parameters.len());

    for param in parameters {
        if !seen_ids.insert(param.id) {
            ctx.emit(
                &rules::SCALAR_PARAMETER_DUPLICATE,
                "constraints/generic_parameters.json",
                Some(format!("id={}", param.id.0)),
                format!(
                    "duplicate parameter id {} (name: '{}')",
                    param.id.0, param.name
                ),
            );
        }

        if !seen_names.insert(param.name.as_str()) {
            ctx.emit(
                &rules::SCALAR_PARAMETER_DUPLICATE,
                "constraints/generic_parameters.json",
                Some(param.name.as_str()),
                format!("duplicate parameter name '{}'", param.name),
            );
        }
    }
}

/// Extracts the `hydro_id` from any [`ComputedParameter`] variant.
///
/// Keep the match exhaustive with no `_` arm — a new variant must then fail to
/// compile here rather than silently skip its hydro-id check.
fn hydro_id_of(c: ComputedParameter) -> EntityId {
    match c {
        ComputedParameter::EquivalentProductivity { hydro_id }
        | ComputedParameter::AccumulatedProductivity { hydro_id }
        | ComputedParameter::ReferenceVolume { hydro_id }
        | ComputedParameter::ReferenceTurbine { hydro_id }
        | ComputedParameter::MinStorage { hydro_id }
        | ComputedParameter::MaxStorage { hydro_id }
        | ComputedParameter::SpecificProductivity { hydro_id }
        | ComputedParameter::IntegratedEquivalentProductivity { hydro_id }
        | ComputedParameter::IntegratedAccumulatedProductivity { hydro_id }
        | ComputedParameter::MaxStoredEnergy { hydro_id } => hydro_id,
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Bus, ComputedParameter, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
        HydroPenalties, ParameterKind, ScalarParameter, SystemBuilder,
    };

    use super::*;
    use crate::validation::ErrorKind;

    // ── Fixture helpers ───────────────────────────────────────────────────────

    /// Build a minimal [`System`] containing hydros with the given ids.
    ///
    /// A single bus (id=1) is added so `SystemBuilder` does not reject the input.
    fn system_with_hydros(ids: &[i32]) -> cobre_core::System {
        let bus = Bus {
            id: EntityId(1),
            name: "Bus 1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydros: Vec<Hydro> = ids.iter().map(|&id| minimal_hydro(id)).collect();

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .build()
            .unwrap()
    }

    fn minimal_hydro(id: i32) -> Hydro {
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: format!("Hydro {id}"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1000.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 200.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 200.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        };
        hydro.declare_mirror_unit_group(EntityId(1));
        hydro
    }

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    fn computed_param(id: i32, name: &str, c: ComputedParameter) -> ScalarParameter {
        ScalarParameter {
            id: EntityId(id),
            name: name.to_string(),
            kind: ParameterKind::Computed { computed_spec: c },
        }
    }

    fn constant_param(id: i32, name: &str, value: f64) -> ScalarParameter {
        ScalarParameter {
            id: EntityId(id),
            name: name.to_string(),
            kind: ParameterKind::Constant { value },
        }
    }

    fn per_stage_param(id: i32, name: &str, values: Vec<f64>) -> ScalarParameter {
        ScalarParameter {
            id: EntityId(id),
            name: name.to_string(),
            kind: ParameterKind::PerStage { values },
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_computed_unknown_hydro_id_appends_cross_reference_entry() {
        let system = system_with_hydros(&[1, 2]);
        let params = vec![computed_param(
            1,
            "rho_eq_h99",
            ComputedParameter::EquivalentProductivity {
                hydro_id: EntityId(99),
            },
        )];
        let mut ctx = ValidationContext::new();

        validate_scalar_parameters(&params, system.hydros(), 3, &mut ctx);

        assert!(ctx.has_errors(), "should have at least one error");
        let errors = ctx.errors();
        assert_eq!(errors.len(), 1);
        let entry = errors[0];
        assert_eq!(entry.kind, ErrorKind::InvalidReference);
        assert_eq!(
            entry.file.to_str().unwrap(),
            "constraints/generic_parameters.json"
        );
        assert!(
            entry.message.contains("99"),
            "message should name the missing hydro id, got: {}",
            entry.message
        );
    }

    #[test]
    fn test_per_stage_length_mismatch_appends_schema_entry() {
        let system = system_with_hydros(&[1]);
        let params = vec![per_stage_param(1, "alpha", vec![1.0, 2.0])]; // len=2, expected 4
        let mut ctx = ValidationContext::new();

        validate_scalar_parameters(&params, system.hydros(), 4, &mut ctx);

        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert_eq!(errors.len(), 1);
        let entry = errors[0];
        assert_eq!(entry.kind, ErrorKind::SchemaViolation);
        assert_eq!(
            entry.file.to_str().unwrap(),
            "constraints/generic_parameters.json"
        );
        assert!(
            entry.message.contains('2'),
            "message should name actual length 2, got: {}",
            entry.message
        );
        assert!(
            entry.message.contains('4'),
            "message should name expected length 4, got: {}",
            entry.message
        );
    }

    #[test]
    fn test_duplicate_id_appends_entry() {
        let system = system_with_hydros(&[]);
        let params = vec![
            constant_param(42, "alpha", 1.0),
            constant_param(42, "beta", 2.0), // duplicate id
        ];
        let mut ctx = ValidationContext::new();

        validate_scalar_parameters(&params, system.hydros(), 3, &mut ctx);

        assert!(ctx.has_errors());
        let errors = ctx.errors();
        assert!(
            errors
                .iter()
                .any(|e| e.message.to_lowercase().contains("duplicate")
                    && e.message.contains("42")),
            "expected a duplicate-id error naming id 42, got: {:?}",
            errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_duplicate_name_case_sensitive_appends_entry() {
        let system = system_with_hydros(&[]);

        // Exact-same name "rho" — should produce one error.
        let params_dup = vec![constant_param(1, "rho", 1.0), constant_param(2, "rho", 2.0)];
        let mut ctx_dup = ValidationContext::new();
        validate_scalar_parameters(&params_dup, system.hydros(), 3, &mut ctx_dup);
        assert!(
            ctx_dup.has_errors(),
            "duplicate name 'rho' should produce an error"
        );
        assert!(
            ctx_dup.errors().iter().any(|e| e.message.contains("rho")),
            "error message should name 'rho'"
        );

        // Different case "rho" vs "Rho" — should produce NO error.
        let params_case = vec![constant_param(3, "rho", 1.0), constant_param(4, "Rho", 2.0)];
        let mut ctx_case = ValidationContext::new();
        validate_scalar_parameters(&params_case, system.hydros(), 3, &mut ctx_case);
        assert!(
            !ctx_case.has_errors(),
            "'rho' and 'Rho' are different names; should produce no error"
        );
    }

    #[test]
    fn test_fully_valid_inputs_appends_nothing() {
        let system = system_with_hydros(&[1, 2, 3]);
        let params = vec![
            computed_param(
                1,
                "rho_eq_h1",
                ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(1),
                },
            ),
            per_stage_param(2, "load_factor", vec![1.0, 1.1, 0.9]),
            constant_param(3, "penalty_coeff", 500.0),
        ];
        let mut ctx = ValidationContext::new();

        validate_scalar_parameters(&params, system.hydros(), 3, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "fully valid inputs should append no errors"
        );
    }

    #[test]
    fn test_no_short_circuit_collects_all_violations() {
        let system = system_with_hydros(&[1]);

        // Computed param referencing missing hydro id 99 (Check A miss)
        // AND per-stage param with wrong length (Check B miss)
        let params = vec![
            computed_param(
                1,
                "rho_eq_h99",
                ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(99),
                },
            ),
            per_stage_param(2, "alpha", vec![1.0, 2.0]), // len=2, expected 4
        ];
        let mut ctx = ValidationContext::new();

        validate_scalar_parameters(&params, system.hydros(), 4, &mut ctx);

        assert!(ctx.has_errors());
        assert!(
            ctx.errors().len() >= 2,
            "both violations must be collected; got {} errors",
            ctx.errors().len()
        );
    }

    #[test]
    fn test_exhaustive_hydro_id_extraction_for_every_variant() {
        // Every ComputedParameter variant points at hydro id=1, which exists.
        let system = system_with_hydros(&[1]);
        let params = vec![
            computed_param(
                1,
                "rho_eq",
                ComputedParameter::EquivalentProductivity {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                2,
                "rho_acum",
                ComputedParameter::AccumulatedProductivity {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                3,
                "v_ref",
                ComputedParameter::ReferenceVolume {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                4,
                "q_ref",
                ComputedParameter::ReferenceTurbine {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                5,
                "v_min",
                ComputedParameter::MinStorage {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                6,
                "v_max",
                ComputedParameter::MaxStorage {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                7,
                "rho_esp",
                ComputedParameter::SpecificProductivity {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                8,
                "rho_eq_int",
                ComputedParameter::IntegratedEquivalentProductivity {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                9,
                "rho_acum_int",
                ComputedParameter::IntegratedAccumulatedProductivity {
                    hydro_id: EntityId(1),
                },
            ),
            computed_param(
                10,
                "e_max",
                ComputedParameter::MaxStoredEnergy {
                    hydro_id: EntityId(1),
                },
            ),
        ];
        let mut ctx = ValidationContext::new();

        validate_scalar_parameters(&params, system.hydros(), 0, &mut ctx);

        assert!(
            !ctx.has_errors(),
            "every ComputedParameter variant with a valid hydro id should produce no errors; \
             got: {:?}",
            ctx.errors().iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }
}
