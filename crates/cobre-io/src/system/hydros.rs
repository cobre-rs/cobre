//! Parsing for `system/hydros.json` — hydro plant entity registry.
//!
//! [`parse_hydros`] reads `system/hydros.json` from the case directory and
//! returns a fully-validated, sorted `Vec<Hydro>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://cobre.dev/schemas/v2/hydros.schema.json",
//!   "hydros": [{
//!     "id": 0, "name": "FURNAS", "bus_id": 0,
//!     "downstream_id": 2,
//!     "entry_stage_id": null, "exit_stage_id": null,
//!     "filling": null,
//!     "diversion": null,
//!     "reservoir": { "min_storage_hm3": 5733.0, "max_storage_hm3": 22950.0 },
//!     "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
//!     "generation": {
//!       "model": "constant_productivity",
//!       "productivity_mw_per_m3s": 0.8765,
//!       "min_turbined_m3s": 0.0, "max_turbined_m3s": 1692.0,
//!       "min_generation_mw": 0.0, "max_generation_mw": 1312.0
//!     },
//!     "tailrace": { "type": "polynomial", "coefficients": [326.0, 0.0032, -1.2e-7] },
//!     "hydraulic_losses": { "type": "factor", "value": 0.03 },
//!     "efficiency": { "type": "constant", "value": 0.92 },
//!     "evaporation": { "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150] },
//!     "penalties": { "spillage_cost": 0.05 }
//!   }]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked before conversion:
//!
//! 1. No two hydros share the same `id`.
//! 2. `min_storage_hm3 >= 0`, `max_storage_hm3 >= 0`, `min_storage_hm3 <= max_storage_hm3`.
//! 3. `min_outflow_m3s >= 0`.
//! 4. `min_turbined_m3s >= 0`, `max_turbined_m3s >= 0`, `min_turbined_m3s <= max_turbined_m3s`.
//! 5. `min_generation_mw <= max_generation_mw`.
//! 6. Evaporation array, if present, must have exactly 12 elements.
//!
//! Cross-reference validation (`bus_id`, `downstream_id`, `diversion.downstream_id`)
//! is deferred to Layer 3 (Epic 06).

use cobre_core::{
    entities::{
        DiversionChannel, EfficiencyModel, FillingConfig, HydraulicLossesModel, Hydro,
        HydroGenerationModel, TailraceModel, TailracePoint,
    },
    penalty::{resolve_hydro_penalties, GlobalPenaltyDefaults, HydroPenaltyOverrides},
    EntityId,
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use crate::LoadError;

// ── Intermediate serde types ──────────────────────────────────────────────────

/// Top-level intermediate type for `hydros.json`.
///
/// Private — only used during deserialization. Not re-exported.
#[derive(Deserialize)]
struct RawHydroFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,

    /// Array of hydro plant entries.
    hydros: Vec<RawHydro>,
}

/// Intermediate type for a single hydro plant entry.
#[derive(Deserialize)]
struct RawHydro {
    /// Hydro plant identifier. Must be unique within the file.
    id: i32,
    /// Human-readable plant name.
    name: String,
    /// Bus to which this plant's generation is injected.
    bus_id: i32,
    /// Downstream hydro plant in the cascade. `null` = no downstream.
    downstream_id: Option<i32>,
    /// Stage index when the plant enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Reservoir storage bounds.
    reservoir: RawReservoir,
    /// Total outflow bounds.
    outflow: RawOutflow,
    /// Generation model configuration (tagged union on `model` field).
    generation: RawGeneration,
    /// Tailrace elevation model. Absent or null = no tailrace.
    #[serde(default)]
    tailrace: Option<RawTailrace>,
    /// Hydraulic loss model. Absent or null = lossless penstock.
    #[serde(default)]
    hydraulic_losses: Option<RawHydraulicLosses>,
    /// Turbine efficiency model. Absent or null = 100% efficiency.
    #[serde(default)]
    efficiency: Option<RawEfficiency>,
    /// Monthly evaporation coefficients. Absent or null = no evaporation.
    #[serde(default)]
    evaporation: Option<RawEvaporation>,
    /// Diversion channel configuration. Absent or null = no diversion channel.
    #[serde(default)]
    diversion: Option<RawDiversionChannel>,
    /// Reservoir filling configuration. Absent or null = no filling operation.
    #[serde(default)]
    filling: Option<RawFillingConfig>,
    /// Entity-level penalty overrides. Absent = all penalties use global defaults.
    #[serde(default)]
    penalties: Option<RawHydroPenaltyOverrides>,
}

/// Intermediate type for the `reservoir` sub-object.
#[derive(Deserialize)]
struct RawReservoir {
    /// Minimum operational storage (dead volume) [hm³].
    min_storage_hm3: f64,
    /// Maximum operational storage [hm³].
    max_storage_hm3: f64,
}

/// Intermediate type for the `outflow` sub-object.
#[derive(Deserialize)]
struct RawOutflow {
    /// Minimum total outflow [m³/s].
    min_outflow_m3s: f64,
    /// Maximum total outflow [m³/s]. `null` = no upper bound.
    max_outflow_m3s: Option<f64>,
}

/// Tagged-union intermediate type for the `generation` sub-object.
///
/// Uses `#[serde(tag = "model")]` (internally-tagged) to dispatch on the
/// `"model"` field value. Each variant carries only the fields relevant to
/// that model — notably, `Fpha` does NOT have `productivity_mw_per_m3s`.
#[derive(Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
enum RawGeneration {
    /// Constant productivity: `power = productivity * turbined_m3s`.
    ConstantProductivity {
        /// Power output per unit of turbined flow [MW/(m³/s)].
        productivity_mw_per_m3s: f64,
        /// Minimum turbined flow [m³/s].
        min_turbined_m3s: f64,
        /// Maximum turbined flow [m³/s].
        max_turbined_m3s: f64,
        /// Minimum electrical generation [MW].
        min_generation_mw: f64,
        /// Maximum electrical generation [MW].
        max_generation_mw: f64,
    },
    /// Head-dependent productivity linearized around an operating point.
    LinearizedHead {
        /// Nominal power output per unit of turbined flow at reference head [MW/(m³/s)].
        productivity_mw_per_m3s: f64,
        /// Minimum turbined flow [m³/s].
        min_turbined_m3s: f64,
        /// Maximum turbined flow [m³/s].
        max_turbined_m3s: f64,
        /// Minimum electrical generation [MW].
        min_generation_mw: f64,
        /// Maximum electrical generation [MW].
        max_generation_mw: f64,
    },
    /// Full production function with head-area-productivity tables (FPHA model).
    ///
    /// Does NOT have `productivity_mw_per_m3s` — FPHA computes head-dependent
    /// productivity from internal tables.
    Fpha {
        /// Minimum turbined flow [m³/s].
        min_turbined_m3s: f64,
        /// Maximum turbined flow [m³/s].
        max_turbined_m3s: f64,
        /// Minimum electrical generation [MW].
        min_generation_mw: f64,
        /// Maximum electrical generation [MW].
        max_generation_mw: f64,
    },
}

impl RawGeneration {
    /// Extract the turbine and generation bounds shared across all variants.
    fn bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Self::ConstantProductivity {
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
                ..
            }
            | Self::LinearizedHead {
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
                ..
            }
            | Self::Fpha {
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
            } => (
                *min_turbined_m3s,
                *max_turbined_m3s,
                *min_generation_mw,
                *max_generation_mw,
            ),
        }
    }
}

/// Tagged-union intermediate type for the `tailrace` sub-object.
///
/// Uses `#[serde(tag = "type")]` internally-tagged on the `"type"` field.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawTailrace {
    /// Polynomial tailrace curve.
    Polynomial {
        /// Polynomial coefficients in ascending power order.
        coefficients: Vec<f64>,
    },
    /// Piecewise-linear tailrace curve.
    Piecewise {
        /// Breakpoints defining the piecewise-linear curve.
        points: Vec<RawTailracePoint>,
    },
}

/// Intermediate type for a single piecewise tailrace breakpoint.
#[derive(Deserialize)]
struct RawTailracePoint {
    /// Total outflow at this point [m³/s].
    outflow_m3s: f64,
    /// Downstream water level (tailrace height) at this outflow [m].
    height_m: f64,
}

/// Tagged-union intermediate type for the `hydraulic_losses` sub-object.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawHydraulicLosses {
    /// Losses as a fraction of net head.
    Factor {
        /// Dimensionless loss factor.
        value: f64,
    },
    /// Constant head loss independent of flow or head.
    Constant {
        /// Fixed head loss [m].
        value_m: f64,
    },
}

/// Tagged-union intermediate type for the `efficiency` sub-object.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawEfficiency {
    /// Constant efficiency across all operating points.
    Constant {
        /// Turbine efficiency as a fraction in (0, 1].
        value: f64,
    },
}

/// Intermediate type for the `evaporation` sub-object.
#[derive(Deserialize)]
struct RawEvaporation {
    /// Monthly evaporation coefficients [mm/month], one per calendar month.
    /// Index 0 = January, index 11 = December.
    coefficients_mm: Vec<f64>,
}

/// Intermediate type for the `diversion` sub-object.
#[derive(Deserialize)]
struct RawDiversionChannel {
    /// Identifier of the downstream hydro plant receiving diverted water.
    downstream_id: i32,
    /// Maximum diversion flow capacity [m³/s].
    max_flow_m3s: f64,
}

/// Intermediate type for the `filling` sub-object.
#[derive(Deserialize)]
struct RawFillingConfig {
    /// Stage index at which filling begins (inclusive).
    start_stage_id: i32,
    /// Constant inflow applied during filling [m³/s].
    /// Absent = passive filling (no active inflow, defaults to 0.0 per spec).
    #[serde(default)]
    filling_inflow_m3s: f64,
}

/// Intermediate type for entity-level hydro penalty overrides.
///
/// All 11 fields are `Option<f64>`. Absent fields default to `None`,
/// meaning the global default for that penalty is used.
///
/// JSON field names mirror `HydroPenalties` and `HydroPenaltyOverrides` field names.
#[allow(clippy::struct_field_names)]
#[derive(Deserialize, Default)]
struct RawHydroPenaltyOverrides {
    #[serde(default)]
    spillage_cost: Option<f64>,
    #[serde(default)]
    diversion_cost: Option<f64>,
    #[serde(default)]
    fpha_turbined_cost: Option<f64>,
    #[serde(default)]
    storage_violation_below_cost: Option<f64>,
    #[serde(default)]
    filling_target_violation_cost: Option<f64>,
    #[serde(default)]
    turbined_violation_below_cost: Option<f64>,
    #[serde(default)]
    outflow_violation_below_cost: Option<f64>,
    #[serde(default)]
    outflow_violation_above_cost: Option<f64>,
    #[serde(default)]
    generation_violation_below_cost: Option<f64>,
    #[serde(default)]
    evaporation_violation_cost: Option<f64>,
    #[serde(default)]
    water_withdrawal_violation_cost: Option<f64>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Load and validate `system/hydros.json` from `path`.
///
/// Reads the JSON file, deserializes it through intermediate serde types,
/// performs post-deserialization validation, then converts to `Vec<Hydro>` using
/// the three-tier penalty resolution cascade (global → entity). The result is
/// sorted by `id` ascending to satisfy declaration-order invariance.
///
/// Cross-reference validation (`bus_id`, `downstream_id`,
/// `diversion.downstream_id`) is deferred to Layer 3 (Epic 06).
///
/// # Errors
///
/// | Condition                                           | Error variant              |
/// | --------------------------------------------------- | -------------------------- |
/// | File not found / read failure                       | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field       | [`LoadError::ParseError`]  |
/// | Unknown `generation.model` variant                  | [`LoadError::SchemaError`] |
/// | Duplicate `id` within the hydros array              | [`LoadError::SchemaError`] |
/// | `min_storage_hm3 < 0` or `max_storage_hm3 < 0`    | [`LoadError::SchemaError`] |
/// | `min_storage_hm3 > max_storage_hm3`                | [`LoadError::SchemaError`] |
/// | `min_outflow_m3s < 0`                              | [`LoadError::SchemaError`] |
/// | `min_turbined_m3s < 0` or `max_turbined_m3s < 0`  | [`LoadError::SchemaError`] |
/// | `max_turbined_m3s < min_turbined_m3s`              | [`LoadError::SchemaError`] |
/// | `max_generation_mw < min_generation_mw`            | [`LoadError::SchemaError`] |
/// | Evaporation array not exactly 12 elements          | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_hydros;
/// use cobre_core::penalty::GlobalPenaltyDefaults;
/// use std::path::Path;
///
/// # fn make_global() -> GlobalPenaltyDefaults { unimplemented!() }
/// let global = make_global();
/// let hydros = parse_hydros(Path::new("case/system/hydros.json"), &global).unwrap();
/// assert!(!hydros.is_empty());
/// ```
pub fn parse_hydros(
    path: &Path,
    global_penalties: &GlobalPenaltyDefaults,
) -> Result<Vec<Hydro>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawHydroFile = serde_json::from_str(&raw_text).map_err(|e| {
        let msg = e.to_string();
        // Unknown generation model variants and unknown tailrace/efficiency/etc.
        // types are caught by serde's internally-tagged enum deserialization and
        // produce a message containing "unknown variant". Treat these as
        // SchemaError (not ParseError) so callers can distinguish bad JSON syntax
        // from unknown enum discriminants.
        if msg.contains("unknown variant") || msg.contains("missing field") {
            LoadError::SchemaError {
                path: path.to_path_buf(),
                field: extract_field_from_serde_msg(&msg),
                message: msg,
            }
        } else {
            LoadError::parse(path, msg)
        }
    })?;

    validate_raw_hydros(&raw, path)?;

    Ok(convert_hydros(raw, global_penalties))
}

// ── Validation ────────────────────────────────────────────────────────────────

/// Validate all invariants on the raw deserialized hydro data.
fn validate_raw_hydros(raw: &RawHydroFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_hydro_ids(&raw.hydros, path)?;
    for (i, hydro) in raw.hydros.iter().enumerate() {
        validate_reservoir(&hydro.reservoir, i, path)?;
        validate_outflow(&hydro.outflow, i, path)?;
        validate_generation(&hydro.generation, i, path)?;
        if let Some(evap) = &hydro.evaporation {
            validate_evaporation(evap, i, path)?;
        }
    }
    Ok(())
}

/// Check that no two hydros share the same `id`.
fn validate_no_duplicate_hydro_ids(hydros: &[RawHydro], path: &Path) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, hydro) in hydros.iter().enumerate() {
        if !seen.insert(hydro.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("hydros[{i}].id"),
                message: format!("duplicate id {} in hydros array", hydro.id),
            });
        }
    }
    Ok(())
}

/// Validate reservoir storage bounds for hydro at `hydro_index`.
///
/// Checks: `min_storage_hm3 >= 0`, `max_storage_hm3 >= 0`,
/// `min_storage_hm3 <= max_storage_hm3`.
fn validate_reservoir(
    reservoir: &RawReservoir,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if reservoir.min_storage_hm3 < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].reservoir.min_storage_hm3"),
            message: format!(
                "min_storage_hm3 must be >= 0, got {}",
                reservoir.min_storage_hm3
            ),
        });
    }
    if reservoir.max_storage_hm3 < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].reservoir.max_storage_hm3"),
            message: format!(
                "max_storage_hm3 must be >= 0, got {}",
                reservoir.max_storage_hm3
            ),
        });
    }
    if reservoir.min_storage_hm3 > reservoir.max_storage_hm3 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].reservoir"),
            message: format!(
                "min_storage_hm3 ({}) must be <= max_storage_hm3 ({})",
                reservoir.min_storage_hm3, reservoir.max_storage_hm3
            ),
        });
    }
    Ok(())
}

/// Validate outflow bounds for hydro at `hydro_index`.
///
/// Checks: `min_outflow_m3s >= 0`.
fn validate_outflow(
    outflow: &RawOutflow,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if outflow.min_outflow_m3s < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].outflow.min_outflow_m3s"),
            message: format!(
                "min_outflow_m3s must be >= 0, got {}",
                outflow.min_outflow_m3s
            ),
        });
    }
    Ok(())
}

/// Validate generation bounds for hydro at `hydro_index`.
///
/// Checks: `min_turbined_m3s >= 0`, `max_turbined_m3s >= 0`,
/// `max_turbined_m3s >= min_turbined_m3s`, `max_generation_mw >= min_generation_mw`.
fn validate_generation(
    generation: &RawGeneration,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    let (min_turbined, max_turbined, min_gen, max_gen) = generation.bounds();

    if min_turbined < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.min_turbined_m3s"),
            message: format!("min_turbined_m3s must be >= 0, got {min_turbined}"),
        });
    }
    if max_turbined < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.max_turbined_m3s"),
            message: format!("max_turbined_m3s must be >= 0, got {max_turbined}"),
        });
    }
    if max_turbined < min_turbined {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.max_turbined_m3s"),
            message: format!(
                "max_turbined_m3s ({max_turbined}) must be >= min_turbined_m3s ({min_turbined})"
            ),
        });
    }
    if max_gen < min_gen {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].generation.max_generation_mw"),
            message: format!(
                "max_generation_mw ({max_gen}) must be >= min_generation_mw ({min_gen})"
            ),
        });
    }
    Ok(())
}

/// Validate evaporation coefficients array for hydro at `hydro_index`.
///
/// Checks: the array must contain exactly 12 elements (one per calendar month).
fn validate_evaporation(
    evaporation: &RawEvaporation,
    hydro_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    let len = evaporation.coefficients_mm.len();
    if len != 12 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("hydros[{hydro_index}].evaporation.coefficients_mm"),
            message: format!(
                "evaporation coefficients_mm must have exactly 12 elements (one per calendar month), got {len}"
            ),
        });
    }
    Ok(())
}

// ── Conversion ────────────────────────────────────────────────────────────────

/// Convert validated raw hydro data into `Vec<Hydro>`, sorted by `id` ascending.
///
/// Precondition: [`validate_raw_hydros`] has returned `Ok(())` for this data.
fn convert_hydros(raw: RawHydroFile, global: &GlobalPenaltyDefaults) -> Vec<Hydro> {
    let mut hydros: Vec<Hydro> = raw
        .hydros
        .into_iter()
        .map(|raw_hydro| {
            // Extract generation model and turbine/generation bounds.
            let (
                generation_model,
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
            ) = convert_generation(raw_hydro.generation);

            // Convert optional tailrace model.
            let tailrace = raw_hydro.tailrace.map(convert_tailrace);

            // Convert optional hydraulic losses model.
            let hydraulic_losses = raw_hydro.hydraulic_losses.map(convert_hydraulic_losses);

            // Convert optional efficiency model.
            let efficiency = raw_hydro.efficiency.map(convert_efficiency);

            // Convert optional evaporation coefficients to [f64; 12].
            let evaporation_coefficients_mm = raw_hydro.evaporation.map(|evap| {
                let v = evap.coefficients_mm;
                // SAFETY: validated to have exactly 12 elements by validate_evaporation.
                // We use try_into() which succeeds because the length was validated.
                v.try_into()
                    .unwrap_or_else(|_| unreachable!("evaporation length validated to be 12"))
            });

            // Convert optional diversion channel.
            let diversion = raw_hydro.diversion.map(|d| DiversionChannel {
                downstream_id: EntityId(d.downstream_id),
                max_flow_m3s: d.max_flow_m3s,
            });

            // Convert optional filling config.
            let filling = raw_hydro.filling.map(|f| FillingConfig {
                start_stage_id: f.start_stage_id,
                filling_inflow_m3s: f.filling_inflow_m3s,
            });

            // Convert entity-level penalty overrides and resolve against global defaults.
            let entity_overrides: Option<HydroPenaltyOverrides> =
                raw_hydro.penalties.map(convert_penalty_overrides);
            let penalties = resolve_hydro_penalties(&entity_overrides, global);

            Hydro {
                id: EntityId(raw_hydro.id),
                name: raw_hydro.name,
                bus_id: EntityId(raw_hydro.bus_id),
                downstream_id: raw_hydro.downstream_id.map(EntityId),
                entry_stage_id: raw_hydro.entry_stage_id,
                exit_stage_id: raw_hydro.exit_stage_id,
                min_storage_hm3: raw_hydro.reservoir.min_storage_hm3,
                max_storage_hm3: raw_hydro.reservoir.max_storage_hm3,
                min_outflow_m3s: raw_hydro.outflow.min_outflow_m3s,
                max_outflow_m3s: raw_hydro.outflow.max_outflow_m3s,
                generation_model,
                min_turbined_m3s,
                max_turbined_m3s,
                min_generation_mw,
                max_generation_mw,
                tailrace,
                hydraulic_losses,
                efficiency,
                evaporation_coefficients_mm,
                diversion,
                filling,
                penalties,
            }
        })
        .collect();

    // Sort by id ascending to satisfy declaration-order invariance.
    hydros.sort_by_key(|h| h.id.0);
    hydros
}

/// Convert a `RawGeneration` into the core `HydroGenerationModel` and its bounds.
///
/// Returns `(model, min_turbined_m3s, max_turbined_m3s, min_generation_mw, max_generation_mw)`.
// Clippy flags this as needless_pass_by_value, but the function consumes its
// argument by destructuring (moving fields out). Taking &RawGeneration would
// require cloning the copied f64 fields, which is no improvement.
#[allow(clippy::needless_pass_by_value)]
fn convert_generation(raw: RawGeneration) -> (HydroGenerationModel, f64, f64, f64, f64) {
    match raw {
        RawGeneration::ConstantProductivity {
            productivity_mw_per_m3s,
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        } => (
            HydroGenerationModel::ConstantProductivity {
                productivity_mw_per_m3s,
            },
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        ),
        RawGeneration::LinearizedHead {
            productivity_mw_per_m3s,
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        } => (
            HydroGenerationModel::LinearizedHead {
                productivity_mw_per_m3s,
            },
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        ),
        RawGeneration::Fpha {
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        } => (
            HydroGenerationModel::Fpha,
            min_turbined_m3s,
            max_turbined_m3s,
            min_generation_mw,
            max_generation_mw,
        ),
    }
}

/// Convert a `RawTailrace` into the core `TailraceModel`.
fn convert_tailrace(raw: RawTailrace) -> TailraceModel {
    match raw {
        RawTailrace::Polynomial { coefficients } => TailraceModel::Polynomial { coefficients },
        RawTailrace::Piecewise { points } => TailraceModel::Piecewise {
            points: points
                .into_iter()
                .map(|p| TailracePoint {
                    outflow_m3s: p.outflow_m3s,
                    height_m: p.height_m,
                })
                .collect(),
        },
    }
}

/// Convert a `RawHydraulicLosses` into the core `HydraulicLossesModel`.
// Clippy flags this as needless_pass_by_value; the function consumes its argument
// by destructuring (no heap allocation involved). Allow here since the by-value
// API correctly signals ownership transfer.
#[allow(clippy::needless_pass_by_value)]
fn convert_hydraulic_losses(raw: RawHydraulicLosses) -> HydraulicLossesModel {
    match raw {
        RawHydraulicLosses::Factor { value } => HydraulicLossesModel::Factor { value },
        RawHydraulicLosses::Constant { value_m } => HydraulicLossesModel::Constant { value_m },
    }
}

/// Convert a `RawEfficiency` into the core `EfficiencyModel`.
// Clippy flags this as needless_pass_by_value; same rationale as convert_hydraulic_losses.
#[allow(clippy::needless_pass_by_value)]
fn convert_efficiency(raw: RawEfficiency) -> EfficiencyModel {
    match raw {
        RawEfficiency::Constant { value } => EfficiencyModel::Constant { value },
    }
}

/// Convert `RawHydroPenaltyOverrides` into `HydroPenaltyOverrides`.
// Clippy flags this as needless_pass_by_value; the struct fields (Option<f64>) are
// all Copy, but taking by reference would require dereferencing every field.
// By-value is idiomatic for this conversion pattern.
#[allow(clippy::needless_pass_by_value)]
fn convert_penalty_overrides(raw: RawHydroPenaltyOverrides) -> HydroPenaltyOverrides {
    HydroPenaltyOverrides {
        spillage_cost: raw.spillage_cost,
        diversion_cost: raw.diversion_cost,
        fpha_turbined_cost: raw.fpha_turbined_cost,
        storage_violation_below_cost: raw.storage_violation_below_cost,
        filling_target_violation_cost: raw.filling_target_violation_cost,
        turbined_violation_below_cost: raw.turbined_violation_below_cost,
        outflow_violation_below_cost: raw.outflow_violation_below_cost,
        outflow_violation_above_cost: raw.outflow_violation_above_cost,
        generation_violation_below_cost: raw.generation_violation_below_cost,
        evaporation_violation_cost: raw.evaporation_violation_cost,
        water_withdrawal_violation_cost: raw.water_withdrawal_violation_cost,
    }
}

/// Extract a field name hint from a `serde_json` error message.
///
/// Mirrors the implementation in `config.rs`. `serde_json` error messages follow
/// patterns such as:
/// - `"unknown variant 'foo', expected one of …"`
/// - `"missing field 'xyz' at line 1 column 2"`
///
/// This helper extracts the identifier between backticks, returning a best-effort
/// field name or `"<unknown>"` when no match is found.
fn extract_field_from_serde_msg(msg: &str) -> String {
    if let Some(start) = msg.find('`') {
        if let Some(end) = msg[start + 1..].find('`') {
            return msg[start + 1..start + 1 + end].to_string();
        }
    }
    "<unknown>".to_string()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::expect_used
)]
mod tests {
    use super::*;
    use cobre_core::entities::{DeficitSegment, HydroPenalties};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Write a string to a temp file and return the file handle (keeps it alive).
    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    /// Build a canonical `GlobalPenaltyDefaults` for test use.
    fn make_global() -> GlobalPenaltyDefaults {
        GlobalPenaltyDefaults {
            bus_deficit_segments: vec![
                DeficitSegment {
                    depth_mw: Some(500.0),
                    cost_per_mwh: 1000.0,
                },
                DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 5000.0,
                },
            ],
            bus_excess_cost: 100.0,
            line_exchange_cost: 2.0,
            hydro: HydroPenalties {
                spillage_cost: 0.01,
                fpha_turbined_cost: 0.05,
                diversion_cost: 0.1,
                storage_violation_below_cost: 10_000.0,
                filling_target_violation_cost: 50_000.0,
                turbined_violation_below_cost: 500.0,
                outflow_violation_below_cost: 500.0,
                outflow_violation_above_cost: 500.0,
                generation_violation_below_cost: 1_000.0,
                evaporation_violation_cost: 5_000.0,
                water_withdrawal_violation_cost: 1_000.0,
            },
            ncs_curtailment_cost: 0.005,
        }
    }

    /// Minimal hydro entry (`constant_productivity`, no optional fields).
    const MINIMAL_HYDRO_JSON: &str = r#"{
      "id": 1,
      "name": "Minimal",
      "bus_id": 0,
      "downstream_id": null,
      "reservoir": { "min_storage_hm3": 100.0, "max_storage_hm3": 2000.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.75,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1000.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 750.0
      }
    }"#;

    /// Full hydro entry (all optional fields populated, polynomial tailrace).
    const FULL_HYDRO_JSON: &str = r#"{
      "id": 0,
      "name": "FURNAS",
      "bus_id": 0,
      "downstream_id": 2,
      "entry_stage_id": 1,
      "exit_stage_id": 600,
      "reservoir": { "min_storage_hm3": 5733.0, "max_storage_hm3": 22950.0 },
      "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": 4000.0 },
      "generation": {
        "model": "constant_productivity",
        "productivity_mw_per_m3s": 0.8765,
        "min_turbined_m3s": 0.0,
        "max_turbined_m3s": 1692.0,
        "min_generation_mw": 0.0,
        "max_generation_mw": 1312.0
      },
      "tailrace": { "type": "polynomial", "coefficients": [326.0, 0.0032, -1.2e-7] },
      "hydraulic_losses": { "type": "factor", "value": 0.03 },
      "efficiency": { "type": "constant", "value": 0.92 },
      "evaporation": { "coefficients_mm": [150, 130, 120, 90, 60, 40, 30, 40, 70, 100, 130, 150] },
      "diversion": { "downstream_id": 3, "max_flow_m3s": 200.0 },
      "filling": { "start_stage_id": 48, "filling_inflow_m3s": 100.0 },
      "penalties": { "spillage_cost": 0.05 }
    }"#;

    // ── AC: parse valid hydros — full and minimal ──────────────────────────────

    /// Given a valid `hydros.json` with 2 hydros (one full, one minimal), `parse_hydros`
    /// returns `Ok(vec)` sorted by `id`; the full hydro has all optional fields mapped.
    #[test]
    fn test_parse_valid_full_and_minimal() {
        let json = format!(r#"{{ "hydros": [{FULL_HYDRO_JSON}, {MINIMAL_HYDRO_JSON}] }}"#);
        let f = write_json(&json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert_eq!(hydros.len(), 2);

        // Hydro 0 (full) comes first after sorting by id
        let h0 = &hydros[0];
        assert_eq!(h0.id, EntityId(0));
        assert_eq!(h0.name, "FURNAS");
        assert_eq!(h0.bus_id, EntityId(0));
        assert_eq!(h0.downstream_id, Some(EntityId(2)));
        assert_eq!(h0.entry_stage_id, Some(1));
        assert_eq!(h0.exit_stage_id, Some(600));
        assert!((h0.min_storage_hm3 - 5733.0).abs() < f64::EPSILON);
        assert!((h0.max_storage_hm3 - 22950.0).abs() < f64::EPSILON);
        assert!((h0.min_outflow_m3s - 0.0).abs() < f64::EPSILON);
        assert_eq!(h0.max_outflow_m3s, Some(4000.0));
        assert!(
            matches!(
                h0.generation_model,
                HydroGenerationModel::ConstantProductivity {
                    productivity_mw_per_m3s
                } if (productivity_mw_per_m3s - 0.8765).abs() < f64::EPSILON
            ),
            "expected ConstantProductivity with correct productivity"
        );
        assert!((h0.min_turbined_m3s - 0.0).abs() < f64::EPSILON);
        assert!((h0.max_turbined_m3s - 1692.0).abs() < f64::EPSILON);
        assert!((h0.min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((h0.max_generation_mw - 1312.0).abs() < f64::EPSILON);
        // Tailrace: polynomial with 3 coefficients
        assert!(
            matches!(&h0.tailrace, Some(TailraceModel::Polynomial { coefficients }) if coefficients.len() == 3),
            "expected Polynomial tailrace with 3 coefficients"
        );
        // Hydraulic losses: factor
        assert!(matches!(
            h0.hydraulic_losses,
            Some(HydraulicLossesModel::Factor { value }) if (value - 0.03).abs() < f64::EPSILON
        ));
        // Efficiency: constant
        assert!(matches!(
            h0.efficiency,
            Some(EfficiencyModel::Constant { value }) if (value - 0.92).abs() < f64::EPSILON
        ));
        // Evaporation: 12 elements
        assert!(h0.evaporation_coefficients_mm.is_some());
        assert_eq!(h0.evaporation_coefficients_mm.map(|a| a.len()), Some(12));
        // Diversion
        assert!(matches!(
            &h0.diversion,
            Some(DiversionChannel { downstream_id, max_flow_m3s })
            if *downstream_id == EntityId(3) && (max_flow_m3s - 200.0).abs() < f64::EPSILON
        ));
        // Filling
        assert!(matches!(
            &h0.filling,
            Some(FillingConfig { start_stage_id: 48, filling_inflow_m3s })
            if (filling_inflow_m3s - 100.0).abs() < f64::EPSILON
        ));
        // Penalties: spillage_cost overridden to 0.05, rest from global
        assert!((h0.penalties.spillage_cost - 0.05).abs() < f64::EPSILON);
        assert!((h0.penalties.diversion_cost - 0.1).abs() < f64::EPSILON);

        // Hydro 1 (minimal)
        let h1 = &hydros[1];
        assert_eq!(h1.id, EntityId(1));
        assert_eq!(h1.name, "Minimal");
        assert_eq!(h1.downstream_id, None);
        assert_eq!(h1.entry_stage_id, None);
        assert_eq!(h1.exit_stage_id, None);
        assert_eq!(h1.max_outflow_m3s, None);
        assert!(h1.tailrace.is_none());
        assert!(h1.hydraulic_losses.is_none());
        assert!(h1.efficiency.is_none());
        assert!(h1.evaporation_coefficients_mm.is_none());
        assert!(h1.diversion.is_none());
        assert!(h1.filling.is_none());
        // Penalties: all from global (no override block)
        assert!((h1.penalties.spillage_cost - 0.01).abs() < f64::EPSILON);
    }

    // ── AC: FPHA generation model ──────────────────────────────────────────────

    /// Given a hydro with `generation.model: "fpha"`, the resulting `Hydro` has
    /// `generation_model: HydroGenerationModel::Fpha` (no `productivity_mw_per_m3s`).
    #[test]
    fn test_parse_fpha_generation_model() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "FPHA Plant", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 100.0, "max_storage_hm3": 5000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "fpha",
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 2000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 8000.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert_eq!(hydros.len(), 1);
        assert_eq!(hydros[0].generation_model, HydroGenerationModel::Fpha);
        assert!((hydros[0].min_turbined_m3s - 0.0).abs() < f64::EPSILON);
        assert!((hydros[0].max_turbined_m3s - 2000.0).abs() < f64::EPSILON);
    }

    // ── AC: linearized_head generation model ─────────────────────────────────

    /// Given a hydro with `generation.model: "linearized_head"`, it parses correctly.
    #[test]
    fn test_parse_linearized_head_generation_model() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "LH Plant", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "linearized_head",
              "productivity_mw_per_m3s": 0.65,
              "min_turbined_m3s": 100.0,
              "max_turbined_m3s": 3000.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 1950.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert_eq!(hydros.len(), 1);
        assert!(
            matches!(
                &hydros[0].generation_model,
                HydroGenerationModel::LinearizedHead { productivity_mw_per_m3s }
                if (productivity_mw_per_m3s - 0.65).abs() < f64::EPSILON
            ),
            "expected LinearizedHead with productivity 0.65"
        );
    }

    // ── AC: tailrace piecewise variant ────────────────────────────────────────

    /// Tailrace with `"type": "piecewise"` is parsed correctly.
    #[test]
    fn test_parse_tailrace_piecewise() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Piecewise", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "tailrace": {
              "type": "piecewise",
              "points": [
                { "outflow_m3s": 0.0, "height_m": 3.0 },
                { "outflow_m3s": 5000.0, "height_m": 4.5 }
              ]
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(
            matches!(
                &hydros[0].tailrace,
                Some(TailraceModel::Piecewise { points }) if points.len() == 2
            ),
            "expected Piecewise tailrace with 2 points"
        );
    }

    // ── AC: hydraulic losses constant variant ─────────────────────────────────

    /// Hydraulic losses with `"type": "constant"` is parsed correctly.
    #[test]
    fn test_parse_hydraulic_losses_constant() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "ConstantLoss", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "hydraulic_losses": { "type": "constant", "value_m": 2.5 }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(
            matches!(
                hydros[0].hydraulic_losses,
                Some(HydraulicLossesModel::Constant { value_m }) if (value_m - 2.5).abs() < f64::EPSILON
            ),
            "expected Constant hydraulic losses with value_m = 2.5"
        );
    }

    // ── AC: entity-level penalty partial override ─────────────────────────────

    /// Entity-level penalty partial override: overridden fields use entity value,
    /// non-overridden fields use global default.
    #[test]
    fn test_entity_level_penalty_partial_override() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Override", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "penalties": { "spillage_cost": 0.05 }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        // spillage_cost is overridden to 0.05
        assert!(
            (hydros[0].penalties.spillage_cost - 0.05).abs() < f64::EPSILON,
            "spillage_cost should be 0.05 (entity override)"
        );
        // diversion_cost falls back to global default (0.1)
        assert!(
            (hydros[0].penalties.diversion_cost - 0.1).abs() < f64::EPSILON,
            "diversion_cost should be 0.1 (global default)"
        );
        // storage_violation_below_cost falls back to global default (10_000.0)
        assert!(
            (hydros[0].penalties.storage_violation_below_cost - 10_000.0).abs() < f64::EPSILON,
            "storage_violation_below_cost should be 10_000.0 (global default)"
        );
    }

    // ── AC: entity-level penalty all-default (no penalties block) ─────────────

    /// No `penalties` block in JSON → all hydro penalties use global defaults.
    #[test]
    fn test_entity_level_penalty_all_global_defaults() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "NoOverride", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        let g = &global.hydro;
        let p = &hydros[0].penalties;
        assert!((p.spillage_cost - g.spillage_cost).abs() < f64::EPSILON);
        assert!((p.diversion_cost - g.diversion_cost).abs() < f64::EPSILON);
        assert!((p.fpha_turbined_cost - g.fpha_turbined_cost).abs() < f64::EPSILON);
        assert!(
            (p.storage_violation_below_cost - g.storage_violation_below_cost).abs() < f64::EPSILON
        );
        assert!(
            (p.filling_target_violation_cost - g.filling_target_violation_cost).abs()
                < f64::EPSILON
        );
        assert!(
            (p.turbined_violation_below_cost - g.turbined_violation_below_cost).abs()
                < f64::EPSILON
        );
        assert!(
            (p.outflow_violation_below_cost - g.outflow_violation_below_cost).abs() < f64::EPSILON
        );
        assert!(
            (p.outflow_violation_above_cost - g.outflow_violation_above_cost).abs() < f64::EPSILON
        );
        assert!(
            (p.generation_violation_below_cost - g.generation_violation_below_cost).abs()
                < f64::EPSILON
        );
        assert!((p.evaporation_violation_cost - g.evaporation_violation_cost).abs() < f64::EPSILON);
        assert!(
            (p.water_withdrawal_violation_cost - g.water_withdrawal_violation_cost).abs()
                < f64::EPSILON
        );
    }

    // ── AC: filling config with absent filling_inflow_m3s ─────────────────────

    /// Filling config with no `filling_inflow_m3s` field defaults to 0.0 (passive filling).
    #[test]
    fn test_filling_inflow_defaults_to_zero() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Fill", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0,
              "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0,
              "max_generation_mw": 250.0
            },
            "filling": { "start_stage_id": 10 }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();

        assert!(matches!(
            &hydros[0].filling,
            Some(FillingConfig {
                start_stage_id: 10,
                filling_inflow_m3s,
            }) if (*filling_inflow_m3s - 0.0).abs() < f64::EPSILON
        ));
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `hydros.json` with duplicate `id` values, `parse_hydros` returns
    /// `Err(LoadError::SchemaError)` with field containing `"hydros[N].id"` and
    /// message containing `"duplicate"`.
    #[test]
    fn test_duplicate_hydro_id() {
        let entry = r#"{
          "id": 5, "name": "Alpha", "bus_id": 0,
          "downstream_id": null,
          "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
          "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
          "generation": {
            "model": "constant_productivity",
            "productivity_mw_per_m3s": 0.5,
            "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
            "min_generation_mw": 0.0, "max_generation_mw": 250.0
          }
        }"#;
        let json = format!(r#"{{ "hydros": [{entry}, {entry}] }}"#);
        let f = write_json(&json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("hydros[1].id"),
                    "field should contain 'hydros[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: reservoir validation ───────────────────────────────────────────────

    /// `min_storage_hm3 > max_storage_hm3` → `SchemaError` with field containing
    /// `"reservoir"` and message containing `"min_storage_hm3"`.
    #[test]
    fn test_invalid_reservoir_bounds_min_gt_max() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 5000.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("reservoir"),
                    "field should contain 'reservoir', got: {field}"
                );
                assert!(
                    message.contains("min_storage_hm3"),
                    "message should contain 'min_storage_hm3', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `min_storage_hm3` → `SchemaError`.
    #[test]
    fn test_invalid_reservoir_negative_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": -1.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("min_storage_hm3")),
            "expected SchemaError for negative min_storage_hm3, got: {err:?}"
        );
    }

    /// Negative `max_storage_hm3` → `SchemaError`.
    #[test]
    fn test_invalid_reservoir_negative_max() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": -100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("max_storage_hm3")),
            "expected SchemaError for negative max_storage_hm3, got: {err:?}"
        );
    }

    // ── AC: generation bound validation ───────────────────────────────────────

    /// `max_generation_mw < min_generation_mw` → `SchemaError`.
    #[test]
    fn test_invalid_generation_bounds_max_lt_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 500.0, "max_generation_mw": 100.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_generation_mw"),
                    "field should contain 'max_generation_mw', got: {field}"
                );
                assert!(
                    message.contains("min_generation_mw"),
                    "message should reference min_generation_mw, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `max_turbined_m3s < min_turbined_m3s` → `SchemaError`.
    #[test]
    fn test_invalid_turbined_bounds_max_lt_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 600.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("max_turbined_m3s")),
            "expected SchemaError for max_turbined < min_turbined, got: {err:?}"
        );
    }

    /// Negative `min_outflow_m3s` → `SchemaError`.
    #[test]
    fn test_invalid_outflow_negative_min() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": -10.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(&err, LoadError::SchemaError { field, .. } if field.contains("min_outflow_m3s")),
            "expected SchemaError for negative min_outflow_m3s, got: {err:?}"
        );
    }

    // ── AC: evaporation array length validation ────────────────────────────────

    /// Evaporation array with wrong element count → `SchemaError`.
    #[test]
    fn test_invalid_evaporation_wrong_length() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            },
            "evaporation": { "coefficients_mm": [10.0, 20.0] }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("coefficients_mm"),
                    "field should contain 'coefficients_mm', got: {field}"
                );
                assert!(
                    message.contains("12"),
                    "message should mention 12 elements, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: unknown generation model → SchemaError ────────────────────────────

    /// Unknown `generation.model` value → `SchemaError`.
    #[test]
    fn test_unknown_generation_model() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Bad", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "unknown_model_xyz",
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::SchemaError { .. }),
            "unknown generation model should produce SchemaError, got: {err:?}"
        );
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given hydros in reverse ID order in JSON, `parse_hydros` returns a
    /// `Vec<Hydro>` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let entry_a = r#"{
          "id": 0, "name": "Alpha", "bus_id": 0,
          "downstream_id": null,
          "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 500.0 },
          "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
          "generation": {
            "model": "constant_productivity",
            "productivity_mw_per_m3s": 0.5,
            "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
            "min_generation_mw": 0.0, "max_generation_mw": 250.0
          }
        }"#;
        let entry_b = r#"{
          "id": 1, "name": "Beta", "bus_id": 0,
          "downstream_id": null,
          "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 1000.0 },
          "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
          "generation": {
            "model": "constant_productivity",
            "productivity_mw_per_m3s": 0.8,
            "min_turbined_m3s": 0.0, "max_turbined_m3s": 1000.0,
            "min_generation_mw": 0.0, "max_generation_mw": 800.0
          }
        }"#;

        let json_forward = format!(r#"{{ "hydros": [{entry_a}, {entry_b}] }}"#);
        let json_reversed = format!(r#"{{ "hydros": [{entry_b}, {entry_a}] }}"#);
        let global = make_global();

        let f1 = write_json(&json_forward);
        let f2 = write_json(&json_reversed);
        let hydros1 = parse_hydros(f1.path(), &global).unwrap();
        let hydros2 = parse_hydros(f2.path(), &global).unwrap();

        assert_eq!(
            hydros1, hydros2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(hydros1[0].id, EntityId(0));
        assert_eq!(hydros1[1].id, EntityId(1));
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_hydros` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/hydros.json");
        let global = make_global();
        let err = parse_hydros(path, &global).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_hydros` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"hydros": [not valid json}}"#);
        let global = make_global();
        let err = parse_hydros(f.path(), &global).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `hydros` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_hydros_array() {
        let json = r#"{ "hydros": [] }"#;
        let f = write_json(json);
        let global = make_global();
        let hydros = parse_hydros(f.path(), &global).unwrap();
        assert!(hydros.is_empty());
    }

    /// `min_storage_hm3 == max_storage_hm3` (degenerate reservoir) is valid.
    #[test]
    fn test_reservoir_min_equals_max_is_valid() {
        let json = r#"{
          "hydros": [{
            "id": 0, "name": "Deg", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 500.0, "max_storage_hm3": 500.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 500.0,
              "min_generation_mw": 0.0, "max_generation_mw": 250.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let result = parse_hydros(f.path(), &global);
        assert!(
            result.is_ok(),
            "min_storage == max_storage should be valid, got: {result:?}"
        );
    }

    /// `$schema` field is accepted and ignored.
    #[test]
    fn test_schema_field_is_ignored() {
        let json = r#"{
          "$schema": "https://cobre.dev/schemas/v2/hydros.schema.json",
          "hydros": [{
            "id": 0, "name": "H", "bus_id": 0,
            "downstream_id": null,
            "reservoir": { "min_storage_hm3": 0.0, "max_storage_hm3": 100.0 },
            "outflow": { "min_outflow_m3s": 0.0, "max_outflow_m3s": null },
            "generation": {
              "model": "constant_productivity",
              "productivity_mw_per_m3s": 0.5,
              "min_turbined_m3s": 0.0, "max_turbined_m3s": 100.0,
              "min_generation_mw": 0.0, "max_generation_mw": 50.0
            }
          }]
        }"#;
        let f = write_json(json);
        let global = make_global();
        let result = parse_hydros(f.path(), &global);
        assert!(
            result.is_ok(),
            "$schema field should be ignored, got: {result:?}"
        );
    }
}
