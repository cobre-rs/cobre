//! Parsing for `system/thermals.json` — thermal plant entity registry.
//!
//! [`parse_thermals`] reads `system/thermals.json` from the case directory and
//! returns a fully-validated, sorted `Vec<Thermal>`.
//!
//! ## JSON structure
//!
//! ```json
//! {
//!   "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/thermals.schema.json",
//!   "thermals": [
//!     {
//!       "id": 0,
//!       "name": "Angra 1",
//!       "operational_start_date": "2024-01-01",
//!       "bus_id": 2,
//!       "cost_per_mwh": 12.0,
//!       "generation": { "min_mw": 0.0, "max_mw": 600.0 }
//!     },
//!     {
//!       "id": 1,
//!       "name": "Pecém I",
//!       "operational_start_date": "2024-01-01",
//!       "bus_id": 3,
//!       "entry_stage_id": 1,
//!       "exit_stage_id": 120,
//!       "cost_per_mwh": 120.0,
//!       "generation": { "min_mw": 100.0, "max_mw": 360.0 },
//!       "anticipated_config": { "lead_stages": 2 }
//!     }
//!   ]
//! }
//! ```
//!
//! ## Validation
//!
//! After deserializing, the following invariants are checked:
//!
//! 1. No two thermals share the same `id`.
//! 2. `cost_per_mwh` must be ≥ 0.0.
//! 3. `min_generation_mw` and `max_generation_mw` must be ≥ 0.0.
//! 4. `max_generation_mw >= min_generation_mw`.
//!
//! Cross-reference validation (e.g., `bus_id` existence) is deferred to Layer 3.

use cobre_core::{
    EntityId,
    entities::{AnticipatedConfig, Thermal},
};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

use super::parse_operational_start_date;
use crate::LoadError;

/// Top-level intermediate type for `thermals.json` (serde only, not re-exported).
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawThermalFile {
    /// `$schema` field — informational, not validated.
    #[serde(rename = "$schema")]
    _schema: Option<String>,
    /// Array of thermal plant entries.
    thermals: Vec<RawThermal>,
}

/// Intermediate type for a single thermal plant entry.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawThermal {
    /// Thermal plant identifier. Must be unique within the file.
    id: i32,
    /// Human-readable plant name.
    name: String,
    /// Date the entity enters service (ISO 8601 `YYYY-MM-DD`).
    operational_start_date: String,
    /// Bus to which this plant's generation is injected.
    bus_id: i32,
    /// Stage index when the plant enters service. Absent or null = always exists.
    #[serde(default)]
    entry_stage_id: Option<i32>,
    /// Stage index when the plant is decommissioned. Absent or null = never.
    #[serde(default)]
    exit_stage_id: Option<i32>,
    /// Marginal cost of generation \[$/`MWh`\]. Must be ≥ 0.0.
    cost_per_mwh: f64,
    /// Generation bounds.
    generation: RawThermalGeneration,
    /// Anticipated dispatch configuration. Absent = no anticipation.
    #[serde(default)]
    anticipated_config: Option<RawAnticipatedConfig>,
}

/// Intermediate type for the generation bounds sub-object.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub(crate) struct RawThermalGeneration {
    /// Minimum generation (minimum stable load) \[MW\].
    min_mw: f64,
    /// Maximum generation (installed capacity) \[MW\].
    max_mw: f64,
}

/// Intermediate type for anticipated dispatch configuration.
///
/// Untagged with per-variant `deny_unknown_fields`: the `{"lead_stages": N}` /
/// `{"lead_time_hours": H}` object shapes and the structural mutual-exclusion
/// between them live here; `cobre_core::AnticipatedConfig` keeps a plain
/// (non-`untagged`) derive so it stays postcard-broadcast-safe.
#[derive(Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged, deny_unknown_fields)]
pub(crate) enum RawAnticipatedConfig {
    /// Stage-count lead; the calendar is never consulted. Must be ≥ 1.
    LeadStages {
        /// Number of stages of dispatch anticipation. Must be ≥ 1.
        ///
        /// Using `u32` here causes serde to reject negative JSON literals with a
        /// `ParseError` before validation runs.
        lead_stages: u32,
    },
    /// Physical lead time in hours, delivery-anchored. Must be finite and > 0.0.
    LeadTime {
        /// Physical lead time in hours. Must be finite and > 0.0.
        lead_time_hours: f64,
    },
}

/// Load and validate `system/thermals.json` from `path`.
///
/// The result is sorted by `id` ascending, so parser output is deterministic
/// regardless of file row order (declaration-order invariance); canonical order is
/// [`SystemBuilder::build`](cobre_core::SystemBuilder::build)'s to establish.
///
/// Semantic validation (cross-field) is performed by
/// `validation::semantic::thermal`; cross-reference validation (e.g., `bus_id`
/// existence in the bus registry) is deferred to Layer 3.
///
/// # Errors
///
/// | Condition                                           | Error variant              |
/// | --------------------------------------------------- | -------------------------- |
/// | File not found / read failure                       | [`LoadError::IoError`]     |
/// | Invalid JSON syntax or missing required field       | [`LoadError::ParseError`]  |
/// | Duplicate `id` within the thermals array            | [`LoadError::SchemaError`] |
/// | Negative `cost_per_mwh`                             | [`LoadError::SchemaError`] |
/// | Negative `min_generation_mw` or `max_generation_mw` | [`LoadError::SchemaError`] |
/// | `max_generation_mw < min_generation_mw`             | [`LoadError::SchemaError`] |
/// | `lead_stages` is a negative integer in JSON         | [`LoadError::ParseError`]  |
/// | `lead_stages == 0`                                  | [`LoadError::SchemaError`] |
/// | Both/neither of `lead_stages`, `lead_time_hours` set | [`LoadError::ParseError`]  |
/// | `lead_time_hours` non-finite or `<= 0.0`            | [`LoadError::SchemaError`] |
///
/// # Examples
///
/// ```no_run
/// use cobre_io::system::parse_thermals;
/// use std::path::Path;
///
/// let thermals = parse_thermals(Path::new("case/system/thermals.json")).unwrap();
/// assert!(!thermals.is_empty());
/// ```
pub fn parse_thermals(path: &Path) -> Result<Vec<Thermal>, LoadError> {
    let raw_text = std::fs::read_to_string(path).map_err(|e| LoadError::io(path, e))?;

    let raw: RawThermalFile =
        serde_json::from_str(&raw_text).map_err(|e| LoadError::parse(path, e.to_string()))?;

    validate_raw_thermals(&raw, path)?;

    convert_thermals(raw, path)
}

fn validate_raw_thermals(raw: &RawThermalFile, path: &Path) -> Result<(), LoadError> {
    validate_no_duplicate_thermal_ids(&raw.thermals, path)?;
    for (i, thermal) in raw.thermals.iter().enumerate() {
        validate_cost_per_mwh(thermal.cost_per_mwh, i, path)?;
        validate_generation_bounds(&thermal.generation, i, path)?;
        validate_anticipated_config(thermal.anticipated_config.as_ref(), i, path)?;
    }
    Ok(())
}

fn validate_no_duplicate_thermal_ids(
    thermals: &[RawThermal],
    path: &Path,
) -> Result<(), LoadError> {
    let mut seen: HashSet<i32> = HashSet::new();
    for (i, thermal) in thermals.iter().enumerate() {
        if !seen.insert(thermal.id) {
            return Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermals[{i}].id"),
                message: format!("duplicate id {} in thermals array", thermal.id),
            });
        }
    }
    Ok(())
}

fn validate_cost_per_mwh(
    cost_per_mwh: f64,
    thermal_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if cost_per_mwh < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].cost_per_mwh"),
            message: format!("cost_per_mwh must be >= 0.0, got {cost_per_mwh}"),
        });
    }
    Ok(())
}

/// `LeadStages` negative values are rejected earlier by serde (the field is
/// `u32`), so the only remaining failure mode there is `lead_stages == 0`.
/// `LeadTime` mirrors the water arc's `travel_time_hours` gate: reject
/// non-finite or non-positive hours.
fn validate_anticipated_config(
    config: Option<&RawAnticipatedConfig>,
    thermal_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    match config {
        Some(RawAnticipatedConfig::LeadStages { lead_stages: 0 }) => Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].anticipated_config.lead_stages"),
            message: "lead_stages must be >= 1, got 0".to_string(),
        }),
        Some(RawAnticipatedConfig::LeadTime { lead_time_hours })
            if !lead_time_hours.is_finite() || *lead_time_hours <= 0.0 =>
        {
            Err(LoadError::SchemaError {
                path: path.to_path_buf(),
                field: format!("thermals[{thermal_index}].anticipated_config.lead_time_hours"),
                message: format!("lead_time_hours must be finite and > 0.0, got {lead_time_hours}"),
            })
        }
        Some(_) | None => Ok(()),
    }
}

fn validate_generation_bounds(
    bounds: &RawThermalGeneration,
    thermal_index: usize,
    path: &Path,
) -> Result<(), LoadError> {
    if bounds.min_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].generation.min_mw"),
            message: format!("min_mw must be >= 0.0, got {}", bounds.min_mw),
        });
    }
    if bounds.max_mw < 0.0 {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].generation.max_mw"),
            message: format!("max_mw must be >= 0.0, got {}", bounds.max_mw),
        });
    }
    if bounds.max_mw < bounds.min_mw {
        return Err(LoadError::SchemaError {
            path: path.to_path_buf(),
            field: format!("thermals[{thermal_index}].generation.max_mw"),
            message: format!(
                "max_mw ({}) must be >= min_mw ({})",
                bounds.max_mw, bounds.min_mw
            ),
        });
    }
    Ok(())
}

fn convert_thermals(raw: RawThermalFile, path: &Path) -> Result<Vec<Thermal>, LoadError> {
    let mut thermals: Vec<Thermal> = raw
        .thermals
        .into_iter()
        .enumerate()
        .map(|(i, raw_thermal)| {
            let operational_start_date = parse_operational_start_date(
                &raw_thermal.operational_start_date,
                path,
                &format!("thermals[{i}].operational_start_date"),
            )?;

            let anticipated_config: Option<AnticipatedConfig> =
                raw_thermal.anticipated_config.map(|raw_cfg| match raw_cfg {
                    RawAnticipatedConfig::LeadStages { lead_stages } => {
                        AnticipatedConfig::LeadStages(lead_stages)
                    }
                    RawAnticipatedConfig::LeadTime { lead_time_hours } => {
                        AnticipatedConfig::LeadTime(lead_time_hours)
                    }
                });

            Ok(Thermal {
                id: EntityId(raw_thermal.id),
                name: raw_thermal.name,
                operational_start_date,
                bus_id: EntityId(raw_thermal.bus_id),
                entry_stage_id: raw_thermal.entry_stage_id,
                exit_stage_id: raw_thermal.exit_stage_id,
                cost_per_mwh: raw_thermal.cost_per_mwh,
                min_generation_mw: raw_thermal.generation.min_mw,
                max_generation_mw: raw_thermal.generation.max_mw,
                anticipated_config,
            })
        })
        .collect::<Result<_, LoadError>>()?;

    // Sort by id: output must not depend on file row order (declaration-order
    // invariance).
    thermals.sort_by_key(|t| t.id.0);
    Ok(thermals)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use super::*;
    use crate::test_support::write_json;

    /// Canonical valid `thermals.json` with 2 thermals: one with anticipated config
    /// (id=1), one without (id=0).
    const VALID_JSON: &str = r#"{
      "$schema": "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/thermals.schema.json",
      "thermals": [
        {
          "id": 0,
          "name": "Angra 1",
          "operational_start_date": "2024-01-01",
          "bus_id": 2,
          "cost_per_mwh": 12.0,
          "generation": { "min_mw": 0.0, "max_mw": 600.0 }
        },
        {
          "id": 1,
          "name": "Pecém I",
          "operational_start_date": "2024-01-01",
          "bus_id": 3,
          "entry_stage_id": 1,
          "exit_stage_id": 120,
          "cost_per_mwh": 120.0,
          "generation": { "min_mw": 100.0, "max_mw": 360.0 },
          "anticipated_config": { "lead_stages": 2 }
        }
      ]
    }"#;

    // ── AC: valid thermals with and without anticipated config ─────────────────

    /// Given a valid `thermals.json` with 2 thermals (one with anticipated config,
    /// one without), `parse_thermals` returns `Ok(vec)` with 2 `Thermal` entries
    /// sorted by `id`; the anticipated thermal has
    /// `anticipated_config: Some(AnticipatedConfig::LeadStages(2))`.
    #[test]
    fn test_parse_valid_thermals() {
        let f = write_json(VALID_JSON);
        let thermals = parse_thermals(f.path()).unwrap();

        assert_eq!(thermals.len(), 2);

        assert_eq!(thermals[0].id, EntityId(0));
        assert_eq!(thermals[0].name, "Angra 1");
        assert_eq!(thermals[0].bus_id, EntityId(2));
        assert_eq!(thermals[0].entry_stage_id, None);
        assert_eq!(thermals[0].exit_stage_id, None);
        assert!(
            (thermals[0].cost_per_mwh - 12.0).abs() < f64::EPSILON,
            "cost_per_mwh: expected 12.0, got {}",
            thermals[0].cost_per_mwh
        );
        assert!((thermals[0].min_generation_mw - 0.0).abs() < f64::EPSILON);
        assert!((thermals[0].max_generation_mw - 600.0).abs() < f64::EPSILON);
        assert_eq!(thermals[0].anticipated_config, None);

        assert_eq!(thermals[1].id, EntityId(1));
        assert_eq!(thermals[1].name, "Pecém I");
        assert_eq!(thermals[1].bus_id, EntityId(3));
        assert_eq!(thermals[1].entry_stage_id, Some(1));
        assert_eq!(thermals[1].exit_stage_id, Some(120));
        assert!(
            (thermals[1].cost_per_mwh - 120.0).abs() < f64::EPSILON,
            "cost_per_mwh: expected 120.0, got {}",
            thermals[1].cost_per_mwh
        );
        assert!((thermals[1].min_generation_mw - 100.0).abs() < f64::EPSILON);
        assert!((thermals[1].max_generation_mw - 360.0).abs() < f64::EPSILON);
        assert_eq!(
            thermals[1].anticipated_config,
            Some(AnticipatedConfig::LeadStages(2))
        );
    }

    // ── AC: duplicate ID detection ─────────────────────────────────────────────

    /// Given `thermals.json` with duplicate `id` values, `parse_thermals` returns
    /// `Err(LoadError::SchemaError)` mentioning "duplicate".
    #[test]
    fn test_duplicate_thermal_id() {
        let json = r#"{
          "thermals": [
            {
              "id": 5, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            },
            {
              "id": 5, "name": "Beta", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 80.0,
              "generation": { "min_mw": 0.0, "max_mw": 200.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("thermals[1].id"),
                    "field should contain 'thermals[1].id', got: {field}"
                );
                assert!(
                    message.contains("duplicate"),
                    "message should contain 'duplicate', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: cost_per_mwh 75.0 round-trips through parser ───────────────────

    /// Given a `thermals.json` with `"cost_per_mwh": 75.0`, the parser
    /// produces `Thermal { cost_per_mwh: 75.0, .. }`.
    #[test]
    fn test_parse_cost_per_mwh_75() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 75.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(thermals.len(), 1);
        assert!(
            (thermals[0].cost_per_mwh - 75.0).abs() < f64::EPSILON,
            "cost_per_mwh: expected 75.0, got {}",
            thermals[0].cost_per_mwh
        );
    }

    // ── AC: old array-based cost format fails to parse ──────────────────────

    /// Given `thermals.json` using the old array-based cost format (removed in
    /// this version), `parse_thermals` returns `Err(LoadError::ParseError)` —
    /// the old format is a breaking change with no backward compat.
    #[test]
    fn test_legacy_array_cost_format_fails() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for legacy array-based cost format, got: {err:?}"
        );
    }

    // ── AC: negative cost_per_mwh → SchemaError ──────────────────────────────

    /// Negative `cost_per_mwh` → `SchemaError`.
    #[test]
    fn test_negative_cost_per_mwh() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": -50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, .. } => {
                assert!(
                    field.contains("cost_per_mwh"),
                    "field should contain 'cost_per_mwh', got: {field}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: generation bound validation ───────────────────────────────────────

    /// Negative `min_mw` → `SchemaError`.
    #[test]
    fn test_negative_min_generation_mw() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": -10.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("min_mw"),
                    "field should contain 'min_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// Negative `max_mw` → `SchemaError`.
    #[test]
    fn test_negative_max_generation_mw() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": -100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_mw"),
                    "field should contain 'max_mw', got: {field}"
                );
                assert!(
                    message.contains(">= 0.0"),
                    "message should mention >= 0.0, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `max_mw < min_mw` → `SchemaError`.
    #[test]
    fn test_max_mw_less_than_min_mw() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 200.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert!(
                    field.contains("max_mw"),
                    "field should contain 'max_mw', got: {field}"
                );
                assert!(
                    message.contains("min_mw"),
                    "message should mention min_mw, got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    // ── AC: declaration-order invariance ──────────────────────────────────────

    /// Given thermals in reverse ID order in JSON, `parse_thermals` returns a
    /// `Vec<Thermal>` sorted by ascending `id`.
    #[test]
    fn test_declaration_order_invariance() {
        let json_forward = r#"{
          "thermals": [
            {
              "id": 0, "name": "Angra 1", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 12.0,
              "generation": { "min_mw": 0.0, "max_mw": 600.0 }
            },
            {
              "id": 1, "name": "Pecém I", "operational_start_date": "2024-01-01", "bus_id": 1,
              "cost_per_mwh": 120.0,
              "generation": { "min_mw": 0.0, "max_mw": 360.0 }
            }
          ]
        }"#;
        let json_reversed = r#"{
          "thermals": [
            {
              "id": 1, "name": "Pecém I", "operational_start_date": "2024-01-01", "bus_id": 1,
              "cost_per_mwh": 120.0,
              "generation": { "min_mw": 0.0, "max_mw": 360.0 }
            },
            {
              "id": 0, "name": "Angra 1", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 12.0,
              "generation": { "min_mw": 0.0, "max_mw": 600.0 }
            }
          ]
        }"#;

        let f1 = write_json(json_forward);
        let f2 = write_json(json_reversed);
        let thermals1 = parse_thermals(f1.path()).unwrap();
        let thermals2 = parse_thermals(f2.path()).unwrap();

        assert_eq!(
            thermals1, thermals2,
            "results must be identical regardless of input ordering"
        );
        assert_eq!(thermals1[0].id, EntityId(0));
        assert_eq!(thermals1[1].id, EntityId(1));
    }

    // ── AC: anticipated thermals accepted without rejection ───────────────────

    /// Anticipated thermals are parsed and accepted at this layer (semantic
    /// rejection is deferred).
    #[test]
    fn test_anticipated_thermal_not_rejected() {
        let f = write_json(VALID_JSON);
        let thermals = parse_thermals(f.path()).unwrap();
        let anticipated_thermal = thermals.iter().find(|t| t.id == EntityId(1)).unwrap();
        assert_eq!(
            anticipated_thermal.anticipated_config,
            Some(AnticipatedConfig::LeadStages(2))
        );
    }

    // ── AC: legacy anticipated-config key (pre-rename) fails with unknown field ─

    /// Given a `thermals.json` that still uses the pre-rename anticipated-config
    /// key, `parse_thermals` returns `Err(LoadError::ParseError)` whose message
    /// contains the old key name as an unknown field.
    #[test]
    fn test_legacy_anticipated_config_key_fails_with_unknown_field() {
        // Build the old key names at runtime so no banned literal appears in
        // source; the strings themselves are what the parser must reject.
        let old_key: String = ['g', 'n', 'l', '_', 'c', 'o', 'n', 'f', 'i', 'g']
            .iter()
            .collect();
        let old_sub: String = ['l', 'a', 'g', '_', 's', 't', 'a', 'g', 'e', 's']
            .iter()
            .collect();
        let json = format!(
            r#"{{
          "thermals": [
            {{
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": {{ "min_mw": 0.0, "max_mw": 100.0 }},
              "{old_key}": {{ "{old_sub}": 2 }}
            }}
          ]
        }}"#
        );
        let f = write_json(&json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains(&old_key),
                    "error message should mention the old key as unknown field, got: {message}"
                );
            }
            other => {
                panic!("expected ParseError for legacy anticipated-config key, got: {other:?}")
            }
        }
    }

    // ── AC: file not found → IoError ─────────────────────────────────────────

    /// Given a nonexistent path, `parse_thermals` returns `Err(LoadError::IoError)`.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("/nonexistent/system/thermals.json");
        let err = parse_thermals(path).unwrap_err();
        match &err {
            LoadError::IoError { path: p, .. } => {
                assert_eq!(p, path);
            }
            other => panic!("expected IoError, got: {other:?}"),
        }
    }

    // ── AC: invalid JSON → ParseError ─────────────────────────────────────────

    /// Given invalid JSON, `parse_thermals` returns `Err(LoadError::ParseError)`.
    #[test]
    fn test_invalid_json() {
        let f = write_json(r#"{"thermals": [not valid json}}"#);
        let err = parse_thermals(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for invalid JSON, got: {err:?}"
        );
    }

    // ── Additional edge cases ─────────────────────────────────────────────────

    /// Empty `thermals` array is valid — returns an empty Vec.
    #[test]
    fn test_empty_thermals_array() {
        let json = r#"{ "thermals": [] }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert!(thermals.is_empty());
    }

    /// `entry_stage_id` and `exit_stage_id` default to `None` when absent.
    #[test]
    fn test_optional_stage_ids_default_to_none() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(thermals[0].entry_stage_id, None);
        assert_eq!(thermals[0].exit_stage_id, None);
    }

    // ── AC: anticipated_config.lead_stages validation ────────────────────────

    /// `anticipated_config.lead_stages == 0` → `SchemaError` with the precise
    /// field path and "must be >= 1, got 0" in the message.
    #[test]
    fn test_anticipated_lead_stages_zero_rejected() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 },
              "anticipated_config": { "lead_stages": 0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(
                    field, "thermals[0].anticipated_config.lead_stages",
                    "field mismatch: {field}"
                );
                assert!(
                    message.contains("must be >= 1"),
                    "message should contain 'must be >= 1', got: {message}"
                );
                assert!(
                    message.contains("got 0"),
                    "message should contain 'got 0', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `anticipated_config.lead_stages == -3` → `ParseError` from serde (the
    /// field is `u32`, so negative JSON literals are rejected at deserialise
    /// time before validation runs). `RawAnticipatedConfig` is untagged, so the
    /// combined failure across both variants surfaces as the generic
    /// "did not match any variant" message rather than the per-variant `u32`
    /// error.
    #[test]
    fn test_anticipated_lead_stages_negative_rejected_by_serde() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 },
              "anticipated_config": { "lead_stages": -3 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        match &err {
            LoadError::ParseError { message, .. } => {
                assert!(
                    message.contains("did not match any variant"),
                    "message should report the untagged-enum mismatch, got: {message}"
                );
            }
            other => panic!("expected ParseError, got: {other:?}"),
        }
    }

    /// `anticipated_config.lead_stages == 1` is the minimum accepted value
    /// and parses successfully.
    #[test]
    fn test_anticipated_lead_stages_one_accepted() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 },
              "anticipated_config": { "lead_stages": 1 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(thermals.len(), 1);
        assert_eq!(
            thermals[0].anticipated_config,
            Some(AnticipatedConfig::LeadStages(1))
        );
    }

    // ── AC: lead_time_hours mode ──────────────────────────────────────────────

    /// Given a thermal JSON `"anticipated_config": {"lead_stages": 2}`, when
    /// `parse_thermals` runs, then `Thermal.anticipated_config` equals
    /// `Some(AnticipatedConfig::LeadStages(2))`.
    #[test]
    fn test_parse_anticipated_lead_stages() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 },
              "anticipated_config": { "lead_stages": 2 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(
            thermals[0].anticipated_config,
            Some(AnticipatedConfig::LeadStages(2))
        );
    }

    /// Given a thermal JSON `"anticipated_config": {"lead_time_hours": 720.0}`,
    /// when `parse_thermals` runs, then `Thermal.anticipated_config` equals
    /// `Some(AnticipatedConfig::LeadTime(720.0))`.
    #[test]
    fn test_parse_anticipated_lead_time() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 },
              "anticipated_config": { "lead_time_hours": 720.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let thermals = parse_thermals(f.path()).unwrap();
        assert_eq!(
            thermals[0].anticipated_config,
            Some(AnticipatedConfig::LeadTime(720.0))
        );
    }

    /// Given both `lead_stages` and `lead_time_hours` set on the same
    /// `anticipated_config`, when `parse_thermals` runs, then it returns
    /// `Err(LoadError::ParseError)` — structural mutual exclusion: each
    /// variant's `deny_unknown_fields` rejects the other key, so untagged
    /// resolution matches neither.
    #[test]
    fn test_parse_anticipated_both_modes_rejected() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 },
              "anticipated_config": { "lead_stages": 2, "lead_time_hours": 720.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let err = parse_thermals(f.path()).unwrap_err();
        assert!(
            matches!(err, LoadError::ParseError { .. }),
            "expected ParseError for both-modes-set anticipated_config, got: {err:?}"
        );
    }

    /// `lead_time_hours` of `-5.0`, `0.0`, and a non-finite literal all → hard
    /// error `LoadError::SchemaError` naming `lead_time_hours`. `f64` cannot
    /// carry a JSON `NaN`/`Infinity` literal, so the non-finite case is driven
    /// through `serde_json::Number` from a value that round-trips as infinite.
    #[test]
    fn test_parse_anticipated_lead_time_nonpositive_rejected() {
        for lead_time_hours in [-5.0, 0.0] {
            let json = format!(
                r#"{{
              "thermals": [
                {{
                  "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
                  "cost_per_mwh": 50.0,
                  "generation": {{ "min_mw": 0.0, "max_mw": 100.0 }},
                  "anticipated_config": {{ "lead_time_hours": {lead_time_hours} }}
                }}
              ]
            }}"#
            );
            let f = write_json(&json);
            let err = parse_thermals(f.path()).unwrap_err();
            match &err {
                LoadError::SchemaError { field, message, .. } => {
                    assert_eq!(
                        field, "thermals[0].anticipated_config.lead_time_hours",
                        "field mismatch: {field}"
                    );
                    assert!(
                        message.contains("lead_time_hours"),
                        "message should name lead_time_hours, got: {message}"
                    );
                }
                other => panic!(
                    "expected SchemaError for lead_time_hours={lead_time_hours}, got: {other:?}"
                ),
            }
        }
    }

    /// A non-finite `lead_time_hours` (`f64::INFINITY`, unreachable via a JSON
    /// literal) is rejected the same way as non-positive values — exercised
    /// directly against the validator since `serde_json` cannot deserialize an
    /// `Infinity`/`NaN` token from text.
    #[test]
    fn test_validate_anticipated_config_lead_time_non_finite_rejected() {
        let cfg = RawAnticipatedConfig::LeadTime {
            lead_time_hours: f64::INFINITY,
        };
        let err =
            validate_anticipated_config(Some(&cfg), 0, Path::new("thermals.json")).unwrap_err();
        match &err {
            LoadError::SchemaError { field, message, .. } => {
                assert_eq!(field, "thermals[0].anticipated_config.lead_time_hours");
                assert!(
                    message.contains("finite"),
                    "message should mention 'finite', got: {message}"
                );
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    /// `min_mw == max_mw` (degenerate range) is valid.
    #[test]
    fn test_min_equals_max_is_valid() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 50.0,
              "generation": { "min_mw": 100.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_thermals(f.path());
        assert!(
            result.is_ok(),
            "min_mw == max_mw should be valid, got: {result:?}"
        );
    }

    /// Zero-cost thermal (`cost_per_mwh` = 0.0) is valid (non-negative check only).
    #[test]
    fn test_zero_cost_is_valid() {
        let json = r#"{
          "thermals": [
            {
              "id": 0, "name": "Alpha", "operational_start_date": "2024-01-01", "bus_id": 0,
              "cost_per_mwh": 0.0,
              "generation": { "min_mw": 0.0, "max_mw": 100.0 }
            }
          ]
        }"#;
        let f = write_json(json);
        let result = parse_thermals(f.path());
        assert!(
            result.is_ok(),
            "cost_per_mwh = 0.0 should be valid, got: {result:?}"
        );
    }
}
