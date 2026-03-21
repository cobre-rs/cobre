//! Layer 1 — Structural validation.
//!
//! Checks that required files exist in the case directory and records whether
//! optional files are present.  This layer does **not** parse any file content;
//! it only tests for the existence of paths on disk.
//!
//! Call [`validate_structure`] with a path to the case root and a mutable
//! [`ValidationContext`].  It returns a [`FileManifest`] that records which of
//! the 34 input files are present.  Missing required files produce
//! [`ErrorKind::FileNotFound`] entries in the context.  Missing optional files
//! leave the corresponding manifest field `false` without adding any error.
//!
//! # Examples
//!
//! ```no_run
//! use std::path::Path;
//! use cobre_io::validation::{ValidationContext, structural::validate_structure};
//!
//! let mut ctx = ValidationContext::new();
//! let manifest = validate_structure(Path::new("/path/to/case"), &mut ctx);
//! assert!(!ctx.has_errors());
//! assert!(manifest.config_json);
//! ```

use std::path::Path;

use super::{ErrorKind, ValidationContext};

// ── FileManifest ─────────────────────────────────────────────────────────────

/// Records whether each of the 34 input files is present in the case directory.
///
/// All fields default to `false`.  After calling [`validate_structure`], each field
/// is `true` if the corresponding file was found on disk.
///
/// The 36 files are organised by subdirectory following the input directory structure spec.
/// Each bool is an independent "present/absent" flag for a distinct file.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default)]
pub struct FileManifest {
    // ── Root-level files (4) ──────────────────────────────────────────────────
    /// `config.json` — required
    pub config_json: bool,
    /// `penalties.json` — required
    pub penalties_json: bool,
    /// `stages.json` — required
    pub stages_json: bool,
    /// `initial_conditions.json` — required
    pub initial_conditions_json: bool,

    // ── system/ (10 files) ───────────────────────────────────────────────────
    /// `system/buses.json` — required
    pub system_buses_json: bool,
    /// `system/lines.json` — required
    pub system_lines_json: bool,
    /// `system/hydros.json` — required
    pub system_hydros_json: bool,
    /// `system/thermals.json` — required
    pub system_thermals_json: bool,
    /// `system/non_controllable_sources.json` — optional
    pub system_non_controllable_sources_json: bool,
    /// `system/pumping_stations.json` — optional
    pub system_pumping_stations_json: bool,
    /// `system/energy_contracts.json` — optional
    pub system_energy_contracts_json: bool,
    /// `system/hydro_geometry.parquet` — optional
    pub system_hydro_geometry_parquet: bool,
    /// `system/hydro_production_models.json` — optional
    pub system_hydro_production_models_json: bool,
    /// `system/fpha_hyperplanes.parquet` — optional
    pub system_fpha_hyperplanes_parquet: bool,

    // ── scenarios/ (8 files) ─────────────────────────────────────────────────
    /// `scenarios/inflow_history.parquet` — optional
    pub scenarios_inflow_history_parquet: bool,
    /// `scenarios/inflow_seasonal_stats.parquet` — optional
    pub scenarios_inflow_seasonal_stats_parquet: bool,
    /// `scenarios/inflow_ar_coefficients.parquet` — optional
    pub scenarios_inflow_ar_coefficients_parquet: bool,
    /// `scenarios/external_scenarios.parquet` — optional
    pub scenarios_external_scenarios_parquet: bool,
    /// `scenarios/load_seasonal_stats.parquet` — optional
    pub scenarios_load_seasonal_stats_parquet: bool,
    /// `scenarios/load_factors.json` — optional
    pub scenarios_load_factors_json: bool,
    /// `scenarios/correlation.json` — optional
    pub scenarios_correlation_json: bool,
    /// `scenarios/noise_openings.parquet` — optional
    pub scenarios_noise_openings_parquet: bool,
    /// `scenarios/non_controllable_factors.json` — optional
    pub scenarios_non_controllable_factors_json: bool,
    /// `scenarios/non_controllable_stats.parquet` — optional
    pub scenarios_non_controllable_stats_parquet: bool,

    // ── constraints/ (12 files) ──────────────────────────────────────────────
    /// `constraints/thermal_bounds.parquet` — optional
    pub constraints_thermal_bounds_parquet: bool,
    /// `constraints/hydro_bounds.parquet` — optional
    pub constraints_hydro_bounds_parquet: bool,
    /// `constraints/line_bounds.parquet` — optional
    pub constraints_line_bounds_parquet: bool,
    /// `constraints/pumping_bounds.parquet` — optional
    pub constraints_pumping_bounds_parquet: bool,
    /// `constraints/contract_bounds.parquet` — optional
    pub constraints_contract_bounds_parquet: bool,
    /// `constraints/exchange_factors.json` — optional
    pub constraints_exchange_factors_json: bool,
    /// `constraints/generic_constraints.json` — optional
    pub constraints_generic_constraints_json: bool,
    /// `constraints/generic_constraint_bounds.parquet` — optional
    pub constraints_generic_constraint_bounds_parquet: bool,
    /// `constraints/penalty_overrides_bus.parquet` — optional
    pub constraints_penalty_overrides_bus_parquet: bool,
    /// `constraints/penalty_overrides_line.parquet` — optional
    pub constraints_penalty_overrides_line_parquet: bool,
    /// `constraints/penalty_overrides_hydro.parquet` — optional
    pub constraints_penalty_overrides_hydro_parquet: bool,
    /// `constraints/penalty_overrides_ncs.parquet` — optional
    pub constraints_penalty_overrides_ncs_parquet: bool,
    /// `constraints/ncs_bounds.parquet` — optional
    pub constraints_ncs_bounds_parquet: bool,
}

// ── validate_structure ────────────────────────────────────────────────────────

/// Describes a single file entry for the structural check.
struct FileEntry {
    /// Relative path from the case root.
    relative: &'static str,
    /// `true` if this file must be present.
    required: bool,
}

/// All 34 input files in canonical order.
const FILE_ENTRIES: &[FileEntry] = &[
    // Root-level — required
    FileEntry {
        relative: "config.json",
        required: true,
    },
    FileEntry {
        relative: "penalties.json",
        required: true,
    },
    FileEntry {
        relative: "stages.json",
        required: true,
    },
    FileEntry {
        relative: "initial_conditions.json",
        required: true,
    },
    // system/ — required
    FileEntry {
        relative: "system/buses.json",
        required: true,
    },
    FileEntry {
        relative: "system/lines.json",
        required: true,
    },
    FileEntry {
        relative: "system/hydros.json",
        required: true,
    },
    FileEntry {
        relative: "system/thermals.json",
        required: true,
    },
    // system/ — optional
    FileEntry {
        relative: "system/non_controllable_sources.json",
        required: false,
    },
    FileEntry {
        relative: "system/pumping_stations.json",
        required: false,
    },
    FileEntry {
        relative: "system/energy_contracts.json",
        required: false,
    },
    FileEntry {
        relative: "system/hydro_geometry.parquet",
        required: false,
    },
    FileEntry {
        relative: "system/hydro_production_models.json",
        required: false,
    },
    FileEntry {
        relative: "system/fpha_hyperplanes.parquet",
        required: false,
    },
    // scenarios/ — optional
    FileEntry {
        relative: "scenarios/inflow_history.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/inflow_seasonal_stats.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/inflow_ar_coefficients.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/external_scenarios.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/load_seasonal_stats.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/load_factors.json",
        required: false,
    },
    FileEntry {
        relative: "scenarios/correlation.json",
        required: false,
    },
    FileEntry {
        relative: "scenarios/noise_openings.parquet",
        required: false,
    },
    FileEntry {
        relative: "scenarios/non_controllable_factors.json",
        required: false,
    },
    FileEntry {
        relative: "scenarios/non_controllable_stats.parquet",
        required: false,
    },
    // constraints/ — optional
    FileEntry {
        relative: "constraints/thermal_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/hydro_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/line_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/pumping_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/contract_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/exchange_factors.json",
        required: false,
    },
    FileEntry {
        relative: "constraints/generic_constraints.json",
        required: false,
    },
    FileEntry {
        relative: "constraints/generic_constraint_bounds.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_bus.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_line.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_hydro.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/penalty_overrides_ncs.parquet",
        required: false,
    },
    FileEntry {
        relative: "constraints/ncs_bounds.parquet",
        required: false,
    },
];

/// Performs Layer 1 structural validation on the case directory at `case_root`.
///
/// For each of the 34 known input files:
///
/// - If the file is present, the corresponding [`FileManifest`] field is set to `true`.
/// - If the file is absent **and required**, an [`ErrorKind::FileNotFound`] error is
///   added to `ctx`.
/// - If the file is absent **and optional**, the manifest field remains `false` and no
///   error is added.
///
/// This function does **not** read or parse any file content.
///
/// # Arguments
///
/// * `case_root` — path to the case directory root.
/// * `ctx` — mutable validation context that accumulates diagnostics.
///
/// # Returns
///
/// A [`FileManifest`] recording presence/absence of all 36 files.
#[must_use]
pub fn validate_structure(case_root: &Path, ctx: &mut ValidationContext) -> FileManifest {
    let mut manifest = FileManifest::default();

    for (entry, present) in FILE_ENTRIES.iter().zip(manifest_fields_mut(&mut manifest)) {
        let full_path = case_root.join(entry.relative);
        if full_path.exists() {
            *present = true;
        } else if entry.required {
            ctx.add_error(
                ErrorKind::FileNotFound,
                entry.relative,
                None::<&str>,
                format!(
                    "required file '{}' not found in case directory",
                    entry.relative
                ),
            );
        }
    }

    manifest
}

/// Returns mutable references to every `bool` field of [`FileManifest`] in the same
/// order as [`FILE_ENTRIES`].
///
/// This keeps the mapping between entries and manifest fields explicit and avoids
/// fragile positional indexing elsewhere.
fn manifest_fields_mut(m: &mut FileManifest) -> [&mut bool; 37] {
    [
        // Root (4)
        &mut m.config_json,
        &mut m.penalties_json,
        &mut m.stages_json,
        &mut m.initial_conditions_json,
        // system/ required (4)
        &mut m.system_buses_json,
        &mut m.system_lines_json,
        &mut m.system_hydros_json,
        &mut m.system_thermals_json,
        // system/ optional (6)
        &mut m.system_non_controllable_sources_json,
        &mut m.system_pumping_stations_json,
        &mut m.system_energy_contracts_json,
        &mut m.system_hydro_geometry_parquet,
        &mut m.system_hydro_production_models_json,
        &mut m.system_fpha_hyperplanes_parquet,
        // scenarios/ (8)
        &mut m.scenarios_inflow_history_parquet,
        &mut m.scenarios_inflow_seasonal_stats_parquet,
        &mut m.scenarios_inflow_ar_coefficients_parquet,
        &mut m.scenarios_external_scenarios_parquet,
        &mut m.scenarios_load_seasonal_stats_parquet,
        &mut m.scenarios_load_factors_json,
        &mut m.scenarios_correlation_json,
        &mut m.scenarios_noise_openings_parquet,
        &mut m.scenarios_non_controllable_factors_json,
        &mut m.scenarios_non_controllable_stats_parquet,
        // constraints/ (13)
        &mut m.constraints_thermal_bounds_parquet,
        &mut m.constraints_hydro_bounds_parquet,
        &mut m.constraints_line_bounds_parquet,
        &mut m.constraints_pumping_bounds_parquet,
        &mut m.constraints_contract_bounds_parquet,
        &mut m.constraints_exchange_factors_json,
        &mut m.constraints_generic_constraints_json,
        &mut m.constraints_generic_constraint_bounds_parquet,
        &mut m.constraints_penalty_overrides_bus_parquet,
        &mut m.constraints_penalty_overrides_line_parquet,
        &mut m.constraints_penalty_overrides_hydro_parquet,
        &mut m.constraints_penalty_overrides_ncs_parquet,
        &mut m.constraints_ncs_bounds_parquet,
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::validation::ValidationContext;
    use std::fs;
    use tempfile::TempDir;

    /// Create a temporary case directory containing all 8 required files as empty files.
    fn make_case_with_required(dir: &TempDir) {
        let root = dir.path();
        // Create subdirectories
        fs::create_dir_all(root.join("system")).unwrap();
        // Root-level required files
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("penalties.json"), b"{}").unwrap();
        fs::write(root.join("stages.json"), b"{}").unwrap();
        fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
        // system/ required files
        fs::write(root.join("system/buses.json"), b"{}").unwrap();
        fs::write(root.join("system/lines.json"), b"{}").unwrap();
        fs::write(root.join("system/hydros.json"), b"{}").unwrap();
        fs::write(root.join("system/thermals.json"), b"{}").unwrap();
    }

    #[test]
    fn test_structural_all_required_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "should have 0 errors when all required files present, got: {:?}",
            ctx.errors()
        );

        // All 8 required fields must be true
        assert!(manifest.config_json, "config.json should be present");
        assert!(manifest.penalties_json, "penalties.json should be present");
        assert!(manifest.stages_json, "stages.json should be present");
        assert!(
            manifest.initial_conditions_json,
            "initial_conditions.json should be present"
        );
        assert!(
            manifest.system_buses_json,
            "system/buses.json should be present"
        );
        assert!(
            manifest.system_lines_json,
            "system/lines.json should be present"
        );
        assert!(
            manifest.system_hydros_json,
            "system/hydros.json should be present"
        );
        assert!(
            manifest.system_thermals_json,
            "system/thermals.json should be present"
        );
    }

    #[test]
    fn test_structural_missing_required_hydros() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // Remove the required file
        fs::remove_file(dir.path().join("system/hydros.json")).unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            ctx.has_errors(),
            "should have at least 1 error when system/hydros.json is missing"
        );
        assert_eq!(ctx.errors().len(), 1, "should have exactly 1 error");
        let entry = &ctx.errors()[0];
        assert_eq!(
            entry.kind,
            crate::validation::ErrorKind::FileNotFound,
            "error kind should be FileNotFound"
        );
        assert!(
            entry.file.to_string_lossy().contains("hydros.json"),
            "error file should reference hydros.json, got: {}",
            entry.file.display()
        );
        assert!(
            !manifest.system_hydros_json,
            "manifest.system_hydros_json should be false"
        );
    }

    #[test]
    fn test_structural_optional_absent_no_error() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // No optional files are created

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional files should not produce errors"
        );

        // Verify representative optional files are false
        assert!(!manifest.system_non_controllable_sources_json);
        assert!(!manifest.system_hydro_geometry_parquet);
        assert!(!manifest.scenarios_inflow_history_parquet);
        assert!(!manifest.constraints_thermal_bounds_parquet);
        assert!(!manifest.scenarios_correlation_json);
    }

    #[test]
    fn test_structural_optional_present_in_manifest() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // Create an optional file
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("correlation.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(!ctx.has_errors());
        assert!(
            manifest.scenarios_correlation_json,
            "present optional file should be marked true in manifest"
        );
    }

    #[test]
    fn test_structural_multiple_missing_required() {
        let dir = TempDir::new().unwrap();
        // Create only the system/ subdirectory but no files
        fs::create_dir_all(dir.path().join("system")).unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            ctx.has_errors(),
            "should have errors for all 8 missing required files"
        );
        assert_eq!(
            ctx.errors().len(),
            8,
            "should have exactly 8 errors (one per required file), got: {}",
            ctx.errors().len()
        );
        // All errors should be FileNotFound
        for entry in ctx.errors() {
            assert_eq!(
                entry.kind,
                crate::validation::ErrorKind::FileNotFound,
                "all errors should be FileNotFound"
            );
        }
    }

    #[test]
    fn test_structural_manifest_fields_count() {
        // Verify the FILE_ENTRIES array and manifest_fields_mut are consistent (37 entries)
        assert_eq!(
            FILE_ENTRIES.len(),
            37,
            "FILE_ENTRIES should have exactly 37 entries"
        );

        let mut manifest = FileManifest::default();
        let fields = manifest_fields_mut(&mut manifest);
        assert_eq!(
            fields.len(),
            37,
            "manifest_fields_mut should return exactly 37 fields"
        );
    }

    #[test]
    fn test_manifest_noise_openings_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // No scenarios/noise_openings.parquet created

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !manifest.scenarios_noise_openings_parquet,
            "scenarios_noise_openings_parquet should be false when file is absent"
        );
    }

    #[test]
    fn test_manifest_noise_openings_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // Create the optional file
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("noise_openings.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.scenarios_noise_openings_parquet,
            "scenarios_noise_openings_parquet should be true when file is present"
        );
    }
}
