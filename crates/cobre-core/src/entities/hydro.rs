//! Hydro plant entity — reservoir, turbine, spillage, and cascade topology.

use crate::EntityId;
use chrono::NaiveDate;

/// A single point on the piecewise tailrace curve.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TailracePoint {
    /// Total outflow at this point \[m³/s\].
    pub outflow_m3s: f64,
    /// Downstream water level (tailrace height) at this outflow \[m\].
    pub height_m: f64,
}

/// A diversion channel that routes water from this plant to a downstream plant.
///
/// Diverted flow bypasses turbines and spillways.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiversionChannel {
    /// Identifier of the downstream hydro plant receiving diverted water.
    pub downstream_id: EntityId,
    /// Maximum diversion flow capacity \[m³/s\].
    pub max_flow_m3s: f64,
}

/// Configuration for a commissioned reservoir that impounds water toward its
/// dead volume before it begins generating.
///
/// A hydro carrying a `FillingConfig` (paired with [`Hydro::entry_stage_id`])
/// passes through the [`crate::commissioning::Phase`] lifecycle keyed on the
/// study `stage.id` being evaluated (the stage's own id, not [`Hydro::id`]).
///
/// The filling target is the dead volume `min_storage_hm3`; there is no separate
/// target field. A hydro with no `FillingConfig` is Operating at every stage.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FillingConfig {
    /// Stage id (inclusive) at which the Filling phase begins.
    ///
    /// `== 0`: no `PreFilling`; the stage-0 level is seeded by
    /// `InitialConditions::filling_storage` (a partial level in
    /// `[0, min_storage_hm3)`). `> 0`: `PreFilling` seeds an empty pit (`0`),
    /// frozen until this stage.
    pub start_stage_id: i32,
    /// Per-stage minimum accumulation rate during Filling \[m³/s\]. Validated `>= 0`.
    ///
    /// A storage floor the reservoir must clear, **not** an inflow and **not** a
    /// cap: it neither alters the natural-inflow RHS nor bounds what is impounded.
    pub filling_min_rate_m3s: f64,
}

/// Penalty costs for a hydro plant, pre-resolved from the global → entity → stage
/// cascade to final, ready-to-use values.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroPenalties {
    /// Penalty on water spilled over the spillway ($/(m³/s·h)).
    pub spillage_cost: f64,
    /// Penalty on water diverted through the diversion channel ($/(m³/s·h)).
    pub diversion_cost: f64,
    /// Regularization cost on turbined flow ($/(m³/s·h)). For FPHA hydros,
    /// should exceed `spillage_cost` to avoid interior solutions; not enforced
    /// by validation.
    pub turbined_cost: f64,
    /// Penalty per hm³ of storage below minimum bound ($/hm³).
    pub storage_violation_below_cost: f64,
    /// Penalty per hm³ of storage below filling target ($/hm³).
    pub filling_target_violation_cost: f64,
    /// Penalty on turbined flow below minimum bound ($/(m³/s·h)).
    pub turbined_violation_below_cost: f64,
    /// Penalty on non-diverted outflow below minimum bound ($/(m³/s·h)).
    pub outflow_violation_below_cost: f64,
    /// Penalty on non-diverted outflow above maximum bound ($/(m³/s·h)).
    pub outflow_violation_above_cost: f64,
    /// Penalty per `MWh` of generation below minimum bound ($/`MWh`).
    pub generation_violation_below_cost: f64,
    /// Symmetric evaporation violation penalty, the default of both directional
    /// evaporation costs ($/(m³/s·h)).
    pub evaporation_violation_cost: f64,
    /// Symmetric water withdrawal violation penalty, the default of both
    /// directional withdrawal costs ($/(m³/s·h)).
    pub water_withdrawal_violation_cost: f64,
    /// Penalty on over-withdrawal, withdrawing more than the target ($/(m³/s·h)).
    pub water_withdrawal_violation_pos_cost: f64,
    /// Penalty on under-withdrawal, withdrawing less than the target ($/(m³/s·h)).
    pub water_withdrawal_violation_neg_cost: f64,
    /// Penalty on over-evaporation ($/(m³/s·h)).
    pub evaporation_violation_pos_cost: f64,
    /// Penalty on under-evaporation ($/(m³/s·h)).
    pub evaporation_violation_neg_cost: f64,
    /// Penalty on the inflow non-negativity slack ($/(m³/s·h)).
    /// Used by the LP builder when the inflow non-negativity method is `Penalty`.
    pub inflow_nonnegativity_cost: f64,
}

#[cfg(any(test, feature = "test-support"))]
impl HydroPenalties {
    /// Every penalty field set to `v`; a fixture needing a different value for
    /// one field overrides it with a functional update.
    #[must_use]
    pub fn uniform(v: f64) -> Self {
        Self {
            spillage_cost: v,
            diversion_cost: v,
            turbined_cost: v,
            storage_violation_below_cost: v,
            filling_target_violation_cost: v,
            turbined_violation_below_cost: v,
            outflow_violation_below_cost: v,
            outflow_violation_above_cost: v,
            generation_violation_below_cost: v,
            evaporation_violation_cost: v,
            water_withdrawal_violation_cost: v,
            water_withdrawal_violation_pos_cost: v,
            water_withdrawal_violation_neg_cost: v,
            evaporation_violation_pos_cost: v,
            evaporation_violation_neg_cost: v,
            inflow_nonnegativity_cost: v,
        }
    }
}

/// Production function model selector for a hydro plant.
///
/// A pure selector carrying no numeric coefficients: the productivity coefficient
/// is supplied externally via `system/hydro_production_models.json`, never by the
/// variant itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum HydroGenerationModel {
    /// Constant power per unit flow, independent of reservoir head.
    ConstantProductivity,
    /// Head-dependent productivity linearized around the head at each time step.
    LinearizedHead,
    /// Full production function with head-area-productivity tables (FPHA model).
    ///
    /// Requires forebay and tailrace elevation tables for high-fidelity head effects.
    Fpha,
}

/// Downstream water level (tailrace elevation) as a function of total outflow.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TailraceModel {
    /// Polynomial tailrace curve: `height = a₀ + a₁·Q + a₂·Q² + …`
    Polynomial {
        /// Coefficients for `Q^i` in ascending power order; must be non-empty.
        coefficients: Vec<f64>,
    },
    /// Piecewise-linear tailrace curve defined by (outflow, height) breakpoints.
    Piecewise {
        /// Breakpoints, sorted by ascending `outflow_m3s`.
        points: Vec<TailracePoint>,
    },
}

/// Model for hydraulic losses in the penstock and draft tube.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HydraulicLossesModel {
    /// Losses as a fraction of net head: `loss = factor * head`.
    Factor {
        /// Dimensionless loss factor (e.g., 0.03 = 3% of net head).
        value: f64,
    },
    /// Constant head loss independent of flow or head conditions.
    Constant {
        /// Fixed head loss \[m\].
        value_m: f64,
    },
}

/// Turbine efficiency model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EfficiencyModel {
    /// Constant efficiency across all operating points.
    Constant {
        /// Turbine efficiency as a fraction in (0, 1\] (e.g., 0.92 = 92%).
        value: f64,
    },
}

/// A turbine group within a hydro plant: the plant owns the water (storage,
/// inflow, spillage, cascade topology), each group owns a share of the power
/// (its own bus and its own generation/turbining envelope).
///
/// The four bound fields mirror `Hydro`'s own four bound fields, both required.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HydroUnitGroup {
    /// Identifier, unique within the owning plant (not globally).
    pub id: EntityId,
    /// Human-readable group name.
    pub name: String,
    /// Bus to which this group's generation is injected.
    pub bus_id: EntityId,
    /// Minimum electrical generation \[MW\].
    pub min_generation_mw: f64,
    /// Maximum electrical generation \[MW\].
    pub max_generation_mw: f64,
    /// Minimum turbined flow \[m³/s\].
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow \[m³/s\].
    pub max_turbined_m3s: f64,
}

/// Hydroelectric power plant with reservoir storage and cascade topology.
///
/// Plants form a cascade via `downstream_id`: water released (turbined + spilled)
/// from an upstream plant flows into the downstream plant's reservoir.
///
/// See Input System Entities SS3 and Internal Structures §1.9.4.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Hydro {
    /// Unique hydro plant identifier.
    pub id: EntityId,
    /// Human-readable plant name.
    pub name: String,
    /// Date the entity enters service (ISO 8601).
    pub operational_start_date: NaiveDate,
    /// Identifier of the downstream hydro plant in the cascade.
    /// None = run-of-river (outflow leaves the system) or final plant.
    pub downstream_id: Option<EntityId>,
    /// Travel time on the cascade arc to `downstream_id` \[hours\]. None =
    /// instantaneous (v1 excludes diversion and pumping arcs).
    #[cfg_attr(feature = "serde", serde(default))]
    pub travel_time_hours: Option<f64>,
    /// Stage index when the plant enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Physical (cadastro) minimum storage \[hm³\], stage-invariant — not the per-stage operative bound.
    pub min_storage_hm3: f64,
    /// Physical (cadastro) maximum storage \[hm³\], stage-invariant — not the per-stage operative bound.
    pub max_storage_hm3: f64,
    /// Minimum outflow down the natural river reach (turbined + spilled)
    /// required at all times \[m³/s\]. Excludes any `diversion` flow, which
    /// routes to a separate downstream target — the floor binds the water that
    /// stays in the plant's own channel.
    pub min_outflow_m3s: f64,
    /// Maximum outflow down the natural river reach (turbined + spilled) \[m³/s\].
    /// None = no upper bound. Symmetric with `min_outflow_m3s`: both bound the
    /// non-diverted channel and exclude `diversion`, which is capped separately by
    /// its own channel limit.
    pub max_outflow_m3s: Option<f64>,
    /// Production function model for this plant.
    pub generation_model: HydroGenerationModel,
    /// Minimum turbined flow \[m³/s\].
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow (installed turbine capacity) \[m³/s\].
    pub max_turbined_m3s: f64,
    /// Specific productivity `ρ_esp` \[MW / ((m³/s) · m)\].
    ///
    /// FPHA hydros derive `ρ_eq` = `ρ_esp` · `h_eq(V_ref, Q_ref)` from this, or may
    /// supply `ρ_eq` directly via `system/hydro_energy_productivity.parquet`; if
    /// neither is supplied the case is rejected. Non-FPHA hydros ignore this field.
    #[cfg_attr(feature = "serde", serde(default))]
    pub specific_productivity_mw_per_m3s_per_m: Option<f64>,
    /// Minimum electrical generation \[MW\].
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Turbine groups partitioning this plant's generation envelope, each with
    /// its own bus and bounds.
    pub unit_groups: Vec<HydroUnitGroup>,
    /// Tailrace elevation model. None = constant zero tailrace height.
    pub tailrace: Option<TailraceModel>,
    /// Penstock hydraulic loss model. None = lossless penstock.
    pub hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine efficiency model. None = 100% efficiency (lossless turbine).
    pub efficiency: Option<EfficiencyModel>,
    /// Monthly evaporation coefficients, one per calendar month \[mm/month\].
    /// Index 0 = January, index 11 = December. None = no evaporation modelled.
    pub evaporation_coefficients_mm: Option<[f64; 12]>,
    /// Monthly reference storage volumes for evaporation linearization \[hm3\].
    /// Index 0 = January, index 11 = December. When `Some`, each entry is the
    /// reservoir volume at which the evaporation area-volume curve is linearized
    /// for that month. None = use default midpoint `(min_storage + max_storage) / 2`.
    pub evaporation_reference_volumes_hm3: Option<[f64; 12]>,
    /// Diversion channel configuration. None = no diversion channel.
    pub diversion: Option<DiversionChannel>,
    /// Reservoir filling configuration. None = no filling operation.
    pub filling: Option<FillingConfig>,
    /// Entity-level penalty costs, resolved from the global → entity cascade.
    /// Always populated — falls back to global defaults when no entity override exists.
    pub penalties: HydroPenalties,
}

impl Hydro {
    /// Puts `unit_groups` in canonical id order so results do not depend on
    /// declaration order.
    pub fn sort_unit_groups(&mut self) {
        self.unit_groups.sort_by_key(|g| g.id.0);
    }

    /// Whether the plant can turbine any flow: `max_turbined_m3s` above 1e-9 m³/s.
    ///
    /// At or below that threshold an FPHA fit's flow axis collapses onto `q = 0`
    /// and no plane survives, so the plant is modeled at zero productivity. A zero
    /// `max_generation_mw` alone does not make a plant capacity-less: fitting
    /// drops a non-positive ceiling and the generation column's own bound holds
    /// output at zero.
    #[must_use]
    pub fn has_turbine_capacity(&self) -> bool {
        self.max_turbined_m3s > 1e-9
    }

    /// Test-only fixture helper: mirrors this plant into a single unit group
    /// when none is declared, matching the construction production code no
    /// longer performs — a production caller is a contract violation, not a
    /// convenience. The caller supplies `bus_id`: a hydro's bus association is
    /// owned by its unit groups, not the plant, so the mirror group's bus is a
    /// caller decision rather than a copy of a plant field.
    #[cfg(any(test, feature = "test-support"))]
    pub fn declare_mirror_unit_group(&mut self, bus_id: EntityId) {
        if self.unit_groups.is_empty() {
            self.unit_groups.push(HydroUnitGroup {
                id: EntityId(0),
                name: self.name.clone(),
                bus_id,
                min_generation_mw: self.min_generation_mw,
                max_generation_mw: self.max_generation_mw,
                min_turbined_m3s: self.min_turbined_m3s,
                max_turbined_m3s: self.max_turbined_m3s,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn penalties_all(v: f64) -> HydroPenalties {
        HydroPenalties {
            inflow_nonnegativity_cost: 1000.0,
            ..HydroPenalties::uniform(v)
        }
    }
    fn minimal_hydro(model: HydroGenerationModel) -> Hydro {
        let mut hydro = Hydro {
            id: EntityId::from(1),
            name: String::from("Itaipu"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 500.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 200.0,
            max_turbined_m3s: 12_600.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 14_000.0,
            unit_groups: Vec::new(),
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: penalties_all(0.0),
        };
        hydro.declare_mirror_unit_group(EntityId::from(10));
        hydro
    }

    #[test]
    fn test_hydro_constant_productivity() {
        let hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        assert_eq!(
            hydro.generation_model,
            HydroGenerationModel::ConstantProductivity
        );
    }

    #[test]
    fn test_hydro_fpha() {
        let hydro = minimal_hydro(HydroGenerationModel::Fpha);
        assert_eq!(hydro.generation_model, HydroGenerationModel::Fpha);
    }

    #[test]
    fn test_hydro_optional_fields_none() {
        let hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);

        assert_eq!(hydro.downstream_id, None);
        assert_eq!(hydro.travel_time_hours, None);
        assert_eq!(hydro.entry_stage_id, None);
        assert_eq!(hydro.exit_stage_id, None);
        assert_eq!(hydro.max_outflow_m3s, None);
        assert!(hydro.tailrace.is_none());
        assert!(hydro.hydraulic_losses.is_none());
        assert!(hydro.efficiency.is_none());
        assert_eq!(hydro.evaporation_coefficients_mm, None);
        assert_eq!(hydro.evaporation_reference_volumes_hm3, None);
        assert!(hydro.diversion.is_none());
        assert!(hydro.filling.is_none());
    }

    #[test]
    fn test_hydro_optional_fields_some() {
        let mut hydro = Hydro {
            id: EntityId::from(2),
            name: String::from("Tucuruí"),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: Some(EntityId::from(3)),
            travel_time_hours: Some(360.0),
            entry_stage_id: Some(1),
            exit_stage_id: Some(600),
            min_storage_hm3: 50.0,
            max_storage_hm3: 45_000.0,
            min_outflow_m3s: 1000.0,
            max_outflow_m3s: Some(100_000.0),
            generation_model: HydroGenerationModel::LinearizedHead,
            min_turbined_m3s: 500.0,
            max_turbined_m3s: 22_500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 8370.0,
            unit_groups: Vec::new(),
            tailrace: Some(TailraceModel::Polynomial {
                coefficients: vec![5.0, 0.001],
            }),
            hydraulic_losses: Some(HydraulicLossesModel::Factor { value: 0.03 }),
            efficiency: Some(EfficiencyModel::Constant { value: 0.93 }),
            evaporation_coefficients_mm: Some([
                80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0,
            ]),
            evaporation_reference_volumes_hm3: Some([
                12_000.0, 11_500.0, 11_000.0, 10_500.0, 10_000.0, 9_500.0, 10_000.0, 10_500.0,
                11_000.0, 11_500.0, 12_000.0, 12_500.0,
            ]),
            diversion: Some(DiversionChannel {
                downstream_id: EntityId::from(4),
                max_flow_m3s: 200.0,
            }),
            filling: Some(FillingConfig {
                start_stage_id: 48,
                filling_min_rate_m3s: 100.0,
            }),
            penalties: penalties_all(1.0),
        };
        hydro.declare_mirror_unit_group(EntityId::from(20));

        assert_eq!(hydro.downstream_id, Some(EntityId::from(3)));
        assert_eq!(hydro.travel_time_hours, Some(360.0));
        assert_eq!(hydro.entry_stage_id, Some(1));
        assert_eq!(hydro.exit_stage_id, Some(600));
        assert_eq!(hydro.max_outflow_m3s, Some(100_000.0));
        assert!(hydro.tailrace.is_some());
        assert!(hydro.hydraulic_losses.is_some());
        assert!(hydro.efficiency.is_some());
        assert!(hydro.evaporation_coefficients_mm.is_some());
        assert_eq!(hydro.evaporation_coefficients_mm.map(|a| a.len()), Some(12));
        assert!(hydro.evaporation_reference_volumes_hm3.is_some());
        assert_eq!(
            hydro.evaporation_reference_volumes_hm3.map(|a| a.len()),
            Some(12)
        );
        assert!(hydro.diversion.is_some());
        assert!(hydro.filling.is_some());
    }

    #[test]
    fn test_tailrace_polynomial() {
        let model = TailraceModel::Polynomial {
            coefficients: vec![3.5, 0.0012, -0.000_001],
        };

        let TailraceModel::Polynomial { coefficients } = model else {
            panic!("expected Polynomial variant");
        };
        assert_eq!(coefficients.len(), 3);
        assert!((coefficients[0] - 3.5).abs() < f64::EPSILON);
        assert!((coefficients[1] - 0.0012).abs() < f64::EPSILON);
        assert!((coefficients[2] - -0.000_001_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tailrace_piecewise() {
        let model = TailraceModel::Piecewise {
            points: vec![
                TailracePoint {
                    outflow_m3s: 0.0,
                    height_m: 3.0,
                },
                TailracePoint {
                    outflow_m3s: 5000.0,
                    height_m: 4.5,
                },
                TailracePoint {
                    outflow_m3s: 15_000.0,
                    height_m: 6.2,
                },
            ],
        };

        let TailraceModel::Piecewise { points } = model else {
            panic!("expected Piecewise variant");
        };
        assert_eq!(points.len(), 3);
        assert!((points[0].outflow_m3s - 0.0).abs() < f64::EPSILON);
        assert!((points[1].height_m - 4.5).abs() < f64::EPSILON);
        assert!((points[2].outflow_m3s - 15_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hydraulic_losses_factor() {
        let model = HydraulicLossesModel::Factor { value: 0.03 };

        let HydraulicLossesModel::Factor { value } = model else {
            panic!("expected Factor variant");
        };
        assert!((value - 0.03).abs() < f64::EPSILON);
    }

    #[test]
    fn test_filling_config() {
        let config = FillingConfig {
            start_stage_id: 48,
            filling_min_rate_m3s: 100.0,
        };

        assert_eq!(config.start_stage_id, 48);
        assert!((config.filling_min_rate_m3s - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hydro_penalties_all_fields() {
        let p = HydroPenalties {
            spillage_cost: 1.0,
            diversion_cost: 2.0,
            turbined_cost: 3.0,
            storage_violation_below_cost: 4.0,
            filling_target_violation_cost: 5.0,
            turbined_violation_below_cost: 6.0,
            outflow_violation_below_cost: 7.0,
            outflow_violation_above_cost: 8.0,
            generation_violation_below_cost: 9.0,
            evaporation_violation_cost: 10.0,
            water_withdrawal_violation_cost: 11.0,
            water_withdrawal_violation_pos_cost: 11.0,
            water_withdrawal_violation_neg_cost: 11.0,
            evaporation_violation_pos_cost: 10.0,
            evaporation_violation_neg_cost: 10.0,
            inflow_nonnegativity_cost: 1000.0,
        };

        assert!((p.spillage_cost - 1.0).abs() < f64::EPSILON);
        assert!((p.diversion_cost - 2.0).abs() < f64::EPSILON);
        assert!((p.turbined_cost - 3.0).abs() < f64::EPSILON);
        assert!((p.storage_violation_below_cost - 4.0).abs() < f64::EPSILON);
        assert!((p.filling_target_violation_cost - 5.0).abs() < f64::EPSILON);
        assert!((p.turbined_violation_below_cost - 6.0).abs() < f64::EPSILON);
        assert!((p.outflow_violation_below_cost - 7.0).abs() < f64::EPSILON);
        assert!((p.outflow_violation_above_cost - 8.0).abs() < f64::EPSILON);
        assert!((p.generation_violation_below_cost - 9.0).abs() < f64::EPSILON);
        assert!((p.evaporation_violation_cost - 10.0).abs() < f64::EPSILON);
        assert!((p.water_withdrawal_violation_cost - 11.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_diversion_channel() {
        let channel = DiversionChannel {
            downstream_id: EntityId::from(7),
            max_flow_m3s: 350.0,
        };

        assert_eq!(channel.downstream_id, EntityId::from(7));
        assert!((channel.max_flow_m3s - 350.0).abs() < f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hydro_serde_roundtrip() {
        let mut hydro = Hydro {
            id: EntityId::from(2),
            name: "Tucuruí".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: Some(EntityId::from(3)),
            travel_time_hours: Some(360.0),
            entry_stage_id: Some(1),
            exit_stage_id: Some(600),
            min_storage_hm3: 50.0,
            max_storage_hm3: 45_000.0,
            min_outflow_m3s: 1000.0,
            max_outflow_m3s: Some(100_000.0),
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 500.0,
            max_turbined_m3s: 22_500.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 8370.0,
            unit_groups: Vec::new(),
            tailrace: Some(TailraceModel::Polynomial {
                coefficients: vec![5.0, 0.001],
            }),
            hydraulic_losses: Some(HydraulicLossesModel::Factor { value: 0.03 }),
            efficiency: Some(EfficiencyModel::Constant { value: 0.93 }),
            evaporation_coefficients_mm: Some([
                80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0,
            ]),
            evaporation_reference_volumes_hm3: Some([
                12_000.0, 11_500.0, 11_000.0, 10_500.0, 10_000.0, 9_500.0, 10_000.0, 10_500.0,
                11_000.0, 11_500.0, 12_000.0, 12_500.0,
            ]),
            diversion: Some(DiversionChannel {
                downstream_id: EntityId::from(4),
                max_flow_m3s: 200.0,
            }),
            filling: Some(FillingConfig {
                start_stage_id: 48,
                filling_min_rate_m3s: 100.0,
            }),
            penalties: HydroPenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.02,
                turbined_cost: 0.03,
                storage_violation_below_cost: 1.0,
                filling_target_violation_cost: 2.0,
                turbined_violation_below_cost: 3.0,
                outflow_violation_below_cost: 4.0,
                outflow_violation_above_cost: 5.0,
                generation_violation_below_cost: 6.0,
                evaporation_violation_cost: 7.0,
                water_withdrawal_violation_cost: 8.0,
                water_withdrawal_violation_pos_cost: 8.0,
                water_withdrawal_violation_neg_cost: 8.0,
                evaporation_violation_pos_cost: 7.0,
                evaporation_violation_neg_cost: 7.0,
                inflow_nonnegativity_cost: 1000.0,
            },
        };
        hydro.declare_mirror_unit_group(EntityId::from(20));
        let json = serde_json::to_string(&hydro).unwrap();
        let deserialized: Hydro = serde_json::from_str(&json).unwrap();
        assert_eq!(hydro, deserialized);
    }

    #[test]
    fn test_hydro_evaporation_reference_volumes() {
        let volumes: [f64; 12] = [
            12_000.0, 11_500.0, 11_000.0, 10_500.0, 10_000.0, 9_500.0, 10_000.0, 10_500.0,
            11_000.0, 11_500.0, 12_000.0, 12_500.0,
        ];
        let hydro = Hydro {
            evaporation_reference_volumes_hm3: Some(volumes),
            ..minimal_hydro(HydroGenerationModel::ConstantProductivity)
        };

        assert_eq!(hydro.evaporation_reference_volumes_hm3, Some(volumes));
        assert_eq!(
            hydro.evaporation_reference_volumes_hm3.map(|a| a.len()),
            Some(12)
        );
        assert!(
            (hydro.evaporation_reference_volumes_hm3.unwrap()[0] - 12_000.0).abs() < f64::EPSILON
        );
        assert!(
            (hydro.evaporation_reference_volumes_hm3.unwrap()[5] - 9_500.0).abs() < f64::EPSILON
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn specific_productivity_defaults_to_none_in_json() {
        let hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        assert_eq!(hydro.specific_productivity_mw_per_m3s_per_m, None);

        let json = serde_json::to_string(&hydro).expect("serialize");
        let parsed: Hydro = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.specific_productivity_mw_per_m3s_per_m, None);

        // serde(default): a key omitted entirely also deserializes to None.
        let json_without_key = json.replace(",\"specific_productivity_mw_per_m3s_per_m\":null", "");
        let parsed_missing: Hydro =
            serde_json::from_str(&json_without_key).expect("deserialize without key");
        assert_eq!(parsed_missing.specific_productivity_mw_per_m3s_per_m, None);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn specific_productivity_round_trips_when_some() {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.specific_productivity_mw_per_m3s_per_m = Some(0.0085);

        let json = serde_json::to_string(&hydro).expect("serialize");
        assert!(json.contains("specific_productivity_mw_per_m3s_per_m"));

        let parsed: Hydro = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.specific_productivity_mw_per_m3s_per_m, Some(0.0085));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn travel_time_hours_defaults_to_none_in_json() {
        let hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        assert_eq!(hydro.travel_time_hours, None);

        let json = serde_json::to_string(&hydro).expect("serialize");
        let parsed: Hydro = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.travel_time_hours, None);

        // serde(default): a key omitted entirely also deserializes to None.
        let json_without_key = json.replace(",\"travel_time_hours\":null", "");
        let parsed_missing: Hydro =
            serde_json::from_str(&json_without_key).expect("deserialize without key");
        assert_eq!(parsed_missing.travel_time_hours, None);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn travel_time_hours_round_trips_when_some() {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.travel_time_hours = Some(360.0);

        let json = serde_json::to_string(&hydro).expect("serialize");
        assert!(json.contains("travel_time_hours"));

        let parsed: Hydro = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.travel_time_hours, Some(360.0));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn hydro_generation_model_serde_tagged_form() {
        // Serializes as a bare snake_case string, not internally-tagged
        // (#[serde(tag = "model")]): postcard, used for MPI broadcast, does not
        // support internally-tagged enums. The {"model": "..."} input shape lives
        // on the `RawGeneration` mirror in cobre-io that owns the JSON contract.
        let cp_json =
            serde_json::to_string(&HydroGenerationModel::ConstantProductivity).expect("serialize");
        assert_eq!(cp_json, r#""constant_productivity""#);
        let cp_rt: HydroGenerationModel =
            serde_json::from_str(&cp_json).expect("deserialize constant_productivity");
        assert_eq!(cp_rt, HydroGenerationModel::ConstantProductivity);

        let lh_json =
            serde_json::to_string(&HydroGenerationModel::LinearizedHead).expect("serialize");
        assert_eq!(lh_json, r#""linearized_head""#);
        let lh_rt: HydroGenerationModel =
            serde_json::from_str(&lh_json).expect("deserialize linearized_head");
        assert_eq!(lh_rt, HydroGenerationModel::LinearizedHead);

        let fpha_json = serde_json::to_string(&HydroGenerationModel::Fpha).expect("serialize");
        assert_eq!(fpha_json, r#""fpha""#);
        let fpha_rt: HydroGenerationModel =
            serde_json::from_str(&fpha_json).expect("deserialize fpha");
        assert_eq!(fpha_rt, HydroGenerationModel::Fpha);
    }

    #[cfg(feature = "serde")]
    fn hydro_with_two_unit_groups() -> Hydro {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId::from(3),
                name: "Group A".to_string(),
                bus_id: EntityId::from(4),
                min_generation_mw: 10.0,
                max_generation_mw: 20.0,
                min_turbined_m3s: 30.0,
                max_turbined_m3s: 40.0,
            },
            HydroUnitGroup {
                id: EntityId::from(7),
                name: "Group B".to_string(),
                bus_id: EntityId::from(9),
                min_generation_mw: 50.0,
                max_generation_mw: 60.0,
                min_turbined_m3s: 70.0,
                max_turbined_m3s: 80.0,
            },
        ];
        hydro
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hydro_unit_groups_survive_json_roundtrip() {
        let hydro = hydro_with_two_unit_groups();

        let json = serde_json::to_string(&hydro).expect("serialize");
        let deserialized: Hydro = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(hydro, deserialized);
        assert_eq!(deserialized.unit_groups.len(), 2);
        for (original, round_tripped) in hydro.unit_groups.iter().zip(&deserialized.unit_groups) {
            assert_eq!(original.id, round_tripped.id);
            assert_eq!(original.name, round_tripped.name);
            assert_eq!(original.bus_id, round_tripped.bus_id);
            assert_eq!(
                original.min_generation_mw.to_bits(),
                round_tripped.min_generation_mw.to_bits()
            );
            assert_eq!(
                original.max_generation_mw.to_bits(),
                round_tripped.max_generation_mw.to_bits()
            );
            assert_eq!(
                original.min_turbined_m3s.to_bits(),
                round_tripped.min_turbined_m3s.to_bits()
            );
            assert_eq!(
                original.max_turbined_m3s.to_bits(),
                round_tripped.max_turbined_m3s.to_bits()
            );
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_hydro_unit_groups_key_is_required() {
        let hydro = hydro_with_two_unit_groups();
        let json = serde_json::to_string(&hydro).expect("serialize");

        let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse json value");
        value
            .as_object_mut()
            .expect("hydro json must be an object")
            .remove("unit_groups");
        let json_without_key = serde_json::to_string(&value).expect("reserialize json");

        let err = serde_json::from_str::<Hydro>(&json_without_key)
            .expect_err("missing unit_groups key must fail to deserialize");
        assert!(
            err.to_string().contains("unit_groups"),
            "error message must mention unit_groups, got: {err}"
        );
    }

    #[test]
    fn test_declare_mirror_unit_group_copies_plant_bus_name_and_bounds() {
        let mut hydro = Hydro {
            id: EntityId::from(1),
            name: "AlphaPlant".to_string(),
            min_generation_mw: 10.0,
            max_generation_mw: 90.0,
            min_turbined_m3s: 5.0,
            max_turbined_m3s: 200.0,
            unit_groups: Vec::new(),
            ..minimal_hydro(HydroGenerationModel::ConstantProductivity)
        };

        hydro.declare_mirror_unit_group(EntityId::from(10));

        assert_eq!(hydro.unit_groups.len(), 1);
        let group = &hydro.unit_groups[0];
        assert_eq!(group.id, EntityId(0));
        assert_eq!(group.name, "AlphaPlant");
        assert_eq!(group.bus_id, EntityId::from(10));
        assert_eq!(group.min_generation_mw.to_bits(), 10.0_f64.to_bits());
        assert_eq!(group.max_generation_mw.to_bits(), 90.0_f64.to_bits());
        assert_eq!(group.min_turbined_m3s.to_bits(), 5.0_f64.to_bits());
        assert_eq!(group.max_turbined_m3s.to_bits(), 200.0_f64.to_bits());
    }

    #[test]
    fn test_declare_mirror_unit_group_argument_wins_over_plant_bus_field() {
        let mut hydro = Hydro {
            id: EntityId::from(1),
            name: "AlphaPlant".to_string(),
            min_generation_mw: 10.0,
            max_generation_mw: 90.0,
            min_turbined_m3s: 5.0,
            max_turbined_m3s: 200.0,
            unit_groups: Vec::new(),
            ..minimal_hydro(HydroGenerationModel::ConstantProductivity)
        };

        hydro.declare_mirror_unit_group(EntityId::from(77));

        assert_eq!(hydro.unit_groups.len(), 1);
        let group = &hydro.unit_groups[0];
        assert_eq!(group.bus_id, EntityId::from(77));
        assert_ne!(group.bus_id, EntityId::from(10));
    }

    #[test]
    fn test_declare_mirror_unit_group_is_a_noop_when_a_group_exists() {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.unit_groups = vec![HydroUnitGroup {
            id: EntityId::from(7),
            name: "ExistingGroup".to_string(),
            bus_id: EntityId::from(20),
            min_generation_mw: 1.0,
            max_generation_mw: 2.0,
            min_turbined_m3s: 3.0,
            max_turbined_m3s: 4.0,
        }];

        hydro.declare_mirror_unit_group(EntityId::from(10));

        assert_eq!(hydro.unit_groups.len(), 1);
        assert_eq!(hydro.unit_groups[0].id, EntityId::from(7));
    }

    #[test]
    fn test_declare_mirror_unit_group_does_not_sort() {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId::from(5),
                name: "GroupFive".to_string(),
                bus_id: EntityId::from(20),
                min_generation_mw: 1.0,
                max_generation_mw: 2.0,
                min_turbined_m3s: 3.0,
                max_turbined_m3s: 4.0,
            },
            HydroUnitGroup {
                id: EntityId::from(2),
                name: "GroupTwo".to_string(),
                bus_id: EntityId::from(21),
                min_generation_mw: 5.0,
                max_generation_mw: 6.0,
                min_turbined_m3s: 7.0,
                max_turbined_m3s: 8.0,
            },
        ];

        hydro.declare_mirror_unit_group(EntityId::from(10));

        assert_eq!(hydro.unit_groups[0].id, EntityId::from(5));
        assert_eq!(hydro.unit_groups[1].id, EntityId::from(2));
    }

    #[test]
    fn test_sort_unit_groups_leaves_empty_groups_empty() {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.unit_groups = Vec::new();

        hydro.sort_unit_groups();

        assert!(hydro.unit_groups.is_empty());
    }

    #[test]
    fn test_sort_unit_groups_orders_by_id_ascending() {
        let mut hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity);
        hydro.unit_groups = vec![
            HydroUnitGroup {
                id: EntityId::from(5),
                name: "GroupFive".to_string(),
                bus_id: EntityId::from(20),
                min_generation_mw: 1.0,
                max_generation_mw: 2.0,
                min_turbined_m3s: 3.0,
                max_turbined_m3s: 4.0,
            },
            HydroUnitGroup {
                id: EntityId::from(2),
                name: "GroupTwo".to_string(),
                bus_id: EntityId::from(21),
                min_generation_mw: 5.0,
                max_generation_mw: 6.0,
                min_turbined_m3s: 7.0,
                max_turbined_m3s: 8.0,
            },
            HydroUnitGroup {
                id: EntityId::from(9),
                name: "GroupNine".to_string(),
                bus_id: EntityId::from(22),
                min_generation_mw: 9.0,
                max_generation_mw: 10.0,
                min_turbined_m3s: 11.0,
                max_turbined_m3s: 12.0,
            },
        ];

        hydro.sort_unit_groups();

        let ids: Vec<i32> = hydro.unit_groups.iter().map(|g| g.id.0).collect();
        assert_eq!(ids, vec![2, 5, 9]);
    }
}
