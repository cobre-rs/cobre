//! Hydro plant entity — reservoir, turbine, spillage, and cascade topology.
//!
//! A `Hydro` represents a hydroelectric power plant with a reservoir. Hydro plants
//! have a generation model (constant productivity for optimization), reservoir storage
//! bounds, turbine and spillage variables, and may participate in a cascade topology
//! via a downstream reference.

use crate::EntityId;

/// A single point on the piecewise tailrace curve.
///
/// Relates total outflow to downstream water level (tailrace height).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TailracePoint {
    /// Total outflow at this point \[m³/s\].
    pub outflow_m3s: f64,
    /// Downstream water level (tailrace height) at this outflow \[m\].
    pub height_m: f64,
}

/// A diversion channel that routes water from this plant to a downstream plant.
///
/// Diverted flow bypasses turbines and spillways and is routed directly to
/// the downstream reservoir identified by `downstream_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiversionChannel {
    /// Identifier of the downstream hydro plant receiving diverted water.
    pub downstream_id: EntityId,
    /// Maximum diversion flow capacity \[m³/s\].
    pub max_flow_m3s: f64,
}

/// Configuration for reservoir filling operations.
///
/// Filling is an operational mode where a reservoir is intentionally filled
/// from a fixed inflow source (e.g., diversion works) during a defined stage
/// window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillingConfig {
    /// Stage index at which filling begins (inclusive).
    pub start_stage_id: i32,
    /// Constant inflow applied during filling \[m³/s\].
    pub filling_inflow_m3s: f64,
}

/// Resolved penalty costs for a hydro plant.
///
/// All penalties are pre-resolved from the three-tier cascade (global → entity → stage).
/// A `HydroPenalties` instance always contains final, ready-to-use values.
///
/// See DEC-006 for the three-tier penalty resolution design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HydroPenalties {
    /// Penalty per m³/s of water spilled over the spillway \[$/m³/s\].
    pub spillage_cost: f64,
    /// Penalty per m³/s of water diverted beyond diversion channel limits \[$/m³/s\].
    pub diversion_cost: f64,
    /// Penalty per `MWh` of turbined generation used in FPHA approximation \[$/`MWh`\].
    pub fpha_turbined_cost: f64,
    /// Penalty per hm³ of storage below minimum bound \[$/hm³\].
    pub storage_violation_below_cost: f64,
    /// Penalty per hm³ of storage below filling target \[$/hm³\].
    pub filling_target_violation_cost: f64,
    /// Penalty per m³/s of turbined flow below minimum bound \[$/m³/s\].
    pub turbined_violation_below_cost: f64,
    /// Penalty per m³/s of total outflow below minimum bound \[$/m³/s\].
    pub outflow_violation_below_cost: f64,
    /// Penalty per m³/s of total outflow above maximum bound \[$/m³/s\].
    pub outflow_violation_above_cost: f64,
    /// Penalty per MW of generation below minimum bound \[$/MW\].
    pub generation_violation_below_cost: f64,
    /// Penalty per mm of evaporation constraint violation \[$/mm\].
    pub evaporation_violation_cost: f64,
    /// Penalty per m³/s of water withdrawal constraint violation \[$/m³/s\].
    pub water_withdrawal_violation_cost: f64,
}

/// Production function model for a hydro plant.
///
/// Defines how turbine power output is computed from water flow and head.
#[derive(Debug, Clone, PartialEq)]
pub enum HydroGenerationModel {
    /// Constant power per unit flow, independent of reservoir head.
    ///
    /// This is the minimal viable model: `power_mw = productivity_mw_per_m3s * turbined_m3s`.
    /// Applicable to any analysis procedure.
    ConstantProductivity {
        /// Power output per unit of turbined flow \[MW/(m³/s)\].
        productivity_mw_per_m3s: f64,
    },
    /// Head-dependent productivity linearized around an operating point.
    ///
    /// The linearization is computed from the current head at the start
    /// of each time step.
    LinearizedHead {
        /// Nominal power output per unit of turbined flow at reference head \[MW/(m³/s)\].
        productivity_mw_per_m3s: f64,
    },
    /// Full production function with head-area-productivity tables (FPHA model).
    ///
    /// Requires forebay and tailrace elevation tables for high-fidelity head effects.
    Fpha,
}

/// Downstream water level computation model.
///
/// Models the relationship between total outflow and tailrace elevation,
/// which affects net head and therefore turbine productivity.
#[derive(Debug, Clone, PartialEq)]
pub enum TailraceModel {
    /// Polynomial tailrace curve: `height = a₀ + a₁·Q + a₂·Q² + …`
    ///
    /// `coefficients[i]` is the coefficient for `Q^i`. The vector must have
    /// at least one element.
    Polynomial {
        /// Polynomial coefficients in ascending power order \[m, m/(m³/s), …\].
        coefficients: Vec<f64>,
    },
    /// Piecewise-linear tailrace curve defined by (outflow, height) breakpoints.
    ///
    /// The solver interpolates linearly between adjacent [`TailracePoint`] entries.
    /// Points must be sorted by ascending `outflow_m3s`.
    Piecewise {
        /// Breakpoints defining the piecewise-linear curve.
        points: Vec<TailracePoint>,
    },
}

/// Model for hydraulic losses in the penstock and draft tube.
///
/// Hydraulic losses reduce the effective head available at the turbine.
#[derive(Debug, Clone, Copy, PartialEq)]
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
///
/// Efficiency scales the power output from the hydraulic power available.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EfficiencyModel {
    /// Constant efficiency across all operating points.
    Constant {
        /// Turbine efficiency as a fraction in (0, 1\] (e.g., 0.92 = 92%).
        value: f64,
    },
}

/// Hydroelectric power plant with reservoir storage and cascade topology.
///
/// A `Hydro` plant controls a reservoir and operates turbines and spillways.
/// Multiple plants may form a cascade via `downstream_id` references — water
/// released (turbined + spilled) from an upstream plant flows into the
/// downstream plant's reservoir.
///
/// Source: system/hydro.json. See Input System Entities SS3 and
/// Internal Structures §1.9.4.
#[derive(Debug, Clone, PartialEq)]
pub struct Hydro {
    /// Unique hydro plant identifier.
    pub id: EntityId,
    /// Human-readable plant name.
    pub name: String,
    /// Bus to which this plant's generation is injected.
    pub bus_id: EntityId,
    /// Identifier of the downstream hydro plant in the cascade.
    /// None = run-of-river (outflow leaves the system) or final plant.
    pub downstream_id: Option<EntityId>,
    /// Stage index when the plant enters service. None = always exists.
    pub entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. None = never decommissioned.
    pub exit_stage_id: Option<i32>,
    /// Minimum operational storage (dead volume) \[hm³\].
    pub min_storage_hm3: f64,
    /// Maximum operational storage (flood control level) \[hm³\].
    pub max_storage_hm3: f64,
    /// Minimum total outflow (turbined + spilled) required at all times \[m³/s\].
    pub min_outflow_m3s: f64,
    /// Maximum total outflow constraint \[m³/s\]. None = no upper bound.
    pub max_outflow_m3s: Option<f64>,
    /// Production function model for this plant.
    pub generation_model: HydroGenerationModel,
    /// Minimum turbined flow \[m³/s\].
    pub min_turbined_m3s: f64,
    /// Maximum turbined flow (installed turbine capacity) \[m³/s\].
    pub max_turbined_m3s: f64,
    /// Minimum electrical generation \[MW\].
    pub min_generation_mw: f64,
    /// Maximum electrical generation (installed capacity) \[MW\].
    pub max_generation_mw: f64,
    /// Tailrace elevation model. None = constant zero tailrace height.
    pub tailrace: Option<TailraceModel>,
    /// Penstock hydraulic loss model. None = lossless penstock.
    pub hydraulic_losses: Option<HydraulicLossesModel>,
    /// Turbine efficiency model. None = 100% efficiency (lossless turbine).
    pub efficiency: Option<EfficiencyModel>,
    /// Monthly evaporation coefficients, one per calendar month \[mm/month\].
    /// Index 0 = January, index 11 = December. None = no evaporation modelled.
    pub evaporation_coefficients_mm: Option<[f64; 12]>,
    /// Diversion channel configuration. None = no diversion channel.
    pub diversion: Option<DiversionChannel>,
    /// Reservoir filling configuration. None = no filling operation.
    pub filling: Option<FillingConfig>,
    /// Entity-level penalty costs, resolved from the global → entity cascade.
    /// Always populated — falls back to global defaults when no entity override exists.
    pub penalties: HydroPenalties,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn penalties_all(v: f64) -> HydroPenalties {
        HydroPenalties {
            spillage_cost: v,
            diversion_cost: v,
            fpha_turbined_cost: v,
            storage_violation_below_cost: v,
            filling_target_violation_cost: v,
            turbined_violation_below_cost: v,
            outflow_violation_below_cost: v,
            outflow_violation_above_cost: v,
            generation_violation_below_cost: v,
            evaporation_violation_cost: v,
            water_withdrawal_violation_cost: v,
        }
    }
    fn minimal_hydro(model: HydroGenerationModel) -> Hydro {
        Hydro {
            id: EntityId::from(1),
            name: String::from("Itaipu"),
            bus_id: EntityId::from(10),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 100.0,
            max_storage_hm3: 2000.0,
            min_outflow_m3s: 500.0,
            max_outflow_m3s: None,
            generation_model: model,
            min_turbined_m3s: 200.0,
            max_turbined_m3s: 12_600.0,
            min_generation_mw: 0.0,
            max_generation_mw: 14_000.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            diversion: None,
            filling: None,
            penalties: penalties_all(0.0),
        }
    }

    #[test]
    fn test_hydro_constant_productivity() {
        let hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 0.8765,
        });

        let HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s,
        } = hydro.generation_model
        else {
            panic!("expected ConstantProductivity variant");
        };
        assert!((productivity_mw_per_m3s - 0.8765).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hydro_fpha() {
        let hydro = minimal_hydro(HydroGenerationModel::Fpha);
        assert_eq!(hydro.generation_model, HydroGenerationModel::Fpha);
    }

    #[test]
    fn test_hydro_optional_fields_none() {
        let hydro = minimal_hydro(HydroGenerationModel::ConstantProductivity {
            productivity_mw_per_m3s: 1.0,
        });

        assert_eq!(hydro.downstream_id, None);
        assert_eq!(hydro.entry_stage_id, None);
        assert_eq!(hydro.exit_stage_id, None);
        assert_eq!(hydro.max_outflow_m3s, None);
        assert!(hydro.tailrace.is_none());
        assert!(hydro.hydraulic_losses.is_none());
        assert!(hydro.efficiency.is_none());
        assert_eq!(hydro.evaporation_coefficients_mm, None);
        assert!(hydro.diversion.is_none());
        assert!(hydro.filling.is_none());
    }

    #[test]
    fn test_hydro_optional_fields_some() {
        let hydro = Hydro {
            id: EntityId::from(2),
            name: String::from("Tucuruí"),
            bus_id: EntityId::from(20),
            downstream_id: Some(EntityId::from(3)),
            entry_stage_id: Some(1),
            exit_stage_id: Some(600),
            min_storage_hm3: 50.0,
            max_storage_hm3: 45_000.0,
            min_outflow_m3s: 1000.0,
            max_outflow_m3s: Some(100_000.0),
            generation_model: HydroGenerationModel::LinearizedHead {
                productivity_mw_per_m3s: 0.75,
            },
            min_turbined_m3s: 500.0,
            max_turbined_m3s: 22_500.0,
            min_generation_mw: 0.0,
            max_generation_mw: 8370.0,
            tailrace: Some(TailraceModel::Polynomial {
                coefficients: vec![5.0, 0.001],
            }),
            hydraulic_losses: Some(HydraulicLossesModel::Factor { value: 0.03 }),
            efficiency: Some(EfficiencyModel::Constant { value: 0.93 }),
            evaporation_coefficients_mm: Some([
                80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0,
            ]),
            diversion: Some(DiversionChannel {
                downstream_id: EntityId::from(4),
                max_flow_m3s: 200.0,
            }),
            filling: Some(FillingConfig {
                start_stage_id: 48,
                filling_inflow_m3s: 100.0,
            }),
            penalties: penalties_all(1.0),
        };

        assert_eq!(hydro.downstream_id, Some(EntityId::from(3)));
        assert_eq!(hydro.entry_stage_id, Some(1));
        assert_eq!(hydro.exit_stage_id, Some(600));
        assert_eq!(hydro.max_outflow_m3s, Some(100_000.0));
        assert!(hydro.tailrace.is_some());
        assert!(hydro.hydraulic_losses.is_some());
        assert!(hydro.efficiency.is_some());
        assert!(hydro.evaporation_coefficients_mm.is_some());
        // The fixed-size array always has exactly 12 elements.
        assert_eq!(hydro.evaporation_coefficients_mm.map(|a| a.len()), Some(12));
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
            filling_inflow_m3s: 100.0,
        };

        assert_eq!(config.start_stage_id, 48);
        assert!((config.filling_inflow_m3s - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hydro_penalties_all_fields() {
        let p = HydroPenalties {
            spillage_cost: 1.0,
            diversion_cost: 2.0,
            fpha_turbined_cost: 3.0,
            storage_violation_below_cost: 4.0,
            filling_target_violation_cost: 5.0,
            turbined_violation_below_cost: 6.0,
            outflow_violation_below_cost: 7.0,
            outflow_violation_above_cost: 8.0,
            generation_violation_below_cost: 9.0,
            evaporation_violation_cost: 10.0,
            water_withdrawal_violation_cost: 11.0,
        };

        assert!((p.spillage_cost - 1.0).abs() < f64::EPSILON);
        assert!((p.diversion_cost - 2.0).abs() < f64::EPSILON);
        assert!((p.fpha_turbined_cost - 3.0).abs() < f64::EPSILON);
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
}
