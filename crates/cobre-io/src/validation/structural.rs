//! Layer 1 — Structural validation.
//!
//! Checks that required files exist in the case directory and records whether
//! optional files are present.  This layer does **not** parse any file content;
//! it only tests for the existence of paths on disk. It also rejects any file
//! whose input contract has been withdrawn but is still present
//! (`ErrorKind::BusinessRuleViolation`).
//!
//! Call [`validate_structure`] with a path to the case root and a mutable
//! [`ValidationContext`].  It returns a [`FileManifest`] recording, for each
//! [`InputFile`], whether that file was found on disk.  Missing required files
//! produce `ErrorKind::FileNotFound` entries in the context.  Missing
//! optional files leave [`FileManifest::present`] `false` for that key without
//! adding any error.
//!
//! # Examples
//!
//! ```no_run
//! use std::path::Path;
//! use cobre_io::validation::{ValidationContext, structural::{InputFile, validate_structure}};
//!
//! let mut ctx = ValidationContext::new();
//! let manifest = validate_structure(Path::new("/path/to/case"), &mut ctx);
//! assert!(!ctx.has_errors());
//! assert!(manifest.present(InputFile::ConfigJson));
//! ```

use std::path::Path;

use super::{ValidationContext, rules};

// ── InputFile ────────────────────────────────────────────────────────────────

/// The single registry of every case-directory input file. Each variant names
/// one file; the private `INPUT_FILES` table pairs it with its relative path
/// and required flag, and `FileManifest::present` reports whether
/// [`validate_structure`] found it on disk. A file is declared here and only
/// here — no other list repeats the file set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFile {
    /// `config.json`.
    ConfigJson,
    /// `penalties.json`.
    PenaltiesJson,
    /// `stages.json`.
    StagesJson,
    /// `initial_conditions.json`.
    InitialConditionsJson,
    /// `post_study_stages.json`.
    PostStudyStagesJson,

    /// `system/buses.json`.
    SystemBusesJson,
    /// `system/lines.json`.
    SystemLinesJson,
    /// `system/hydros.json`.
    SystemHydrosJson,
    /// `system/thermals.json`.
    SystemThermalsJson,
    /// `system/non_controllable_sources.json`.
    SystemNonControllableSourcesJson,
    /// `system/pumping_stations.json`.
    SystemPumpingStationsJson,
    /// `system/energy_contracts.json`.
    SystemEnergyContractsJson,
    /// `system/hydro_geometry.parquet`.
    SystemHydroGeometryParquet,
    /// `system/hydro_production_models.json`.
    SystemHydroProductionModelsJson,
    /// `system/fpha_hyperplanes.parquet`.
    SystemFphaHyperplanesParquet,
    /// `system/hydro_energy_productivity.parquet`.
    SystemHydroEnergyProductivityParquet,
    /// `system/tailrace_curves.parquet`.
    SystemTailraceCurvesParquet,

    /// `scenarios/inflow_history.parquet`.
    ScenariosInflowHistoryParquet,
    /// `scenarios/inflow_seasonal_stats.parquet`.
    ScenariosInflowSeasonalStatsParquet,
    /// `scenarios/inflow_ar_coefficients.parquet`.
    ScenariosInflowArCoefficientsParquet,
    /// `scenarios/inflow_annual_component.parquet`.
    ScenariosInflowAnnualComponentParquet,
    /// `scenarios/external_inflow_scenarios.parquet`.
    ScenariosExternalInflowScenariosParquet,
    /// `scenarios/external_load_scenarios.parquet`.
    ScenariosExternalLoadScenariosParquet,
    /// `scenarios/external_ncs_scenarios.parquet`.
    ScenariosExternalNcsScenariosParquet,
    /// `scenarios/load_seasonal_stats.parquet`.
    ScenariosLoadSeasonalStatsParquet,
    /// `scenarios/load_factors.json`.
    ScenariosLoadFactorsJson,
    /// `scenarios/correlation.json`.
    ScenariosCorrelationJson,
    /// `scenarios/non_controllable_factors.json`.
    ScenariosNonControllableFactorsJson,
    /// `scenarios/non_controllable_stats.parquet`.
    ScenariosNonControllableStatsParquet,

    /// `constraints/thermal_bounds.parquet`.
    ConstraintsThermalBoundsParquet,
    /// `constraints/hydro_bounds.parquet`.
    ConstraintsHydroBoundsParquet,
    /// `constraints/line_bounds.parquet`.
    ConstraintsLineBoundsParquet,
    /// `constraints/pumping_bounds.parquet`.
    ConstraintsPumpingBoundsParquet,
    /// `constraints/contract_bounds.parquet`.
    ConstraintsContractBoundsParquet,
    /// `constraints/generic_constraints.json`.
    ConstraintsGenericConstraintsJson,
    /// `constraints/generic_constraint_bounds.parquet`.
    ConstraintsGenericConstraintBoundsParquet,
    /// `constraints/generic_parameters.json`.
    ConstraintsGenericParametersJson,
    /// `constraints/penalty_overrides_bus.parquet`.
    ConstraintsPenaltyOverridesBusParquet,
    /// `constraints/penalty_overrides_line.parquet`.
    ConstraintsPenaltyOverridesLineParquet,
    /// `constraints/penalty_overrides_hydro.parquet`.
    ConstraintsPenaltyOverridesHydroParquet,
    /// `constraints/penalty_overrides_ncs.parquet`.
    ConstraintsPenaltyOverridesNcsParquet,
    /// `constraints/ncs_bounds.parquet`.
    ConstraintsNcsBoundsParquet,
    /// `constraints/hydro_unit_group_bounds.parquet`.
    ConstraintsHydroUnitGroupBoundsParquet,
}

/// One row of the input-file registry: the compiler-checked key, its path
/// relative to the case root, and whether it is required.
struct FileEntry {
    key: InputFile,
    relative: &'static str,
    required: bool,
}

/// Every input file, keyed by [`InputFile`] in declaration order — the single
/// source [`validate_structure`] iterates and [`FileManifest`] indexes by
/// ordinal.
const INPUT_FILES: &[FileEntry] = &[
    // Root-level — required
    FileEntry {
        key: InputFile::ConfigJson,
        relative: "config.json",
        required: true,
    },
    FileEntry {
        key: InputFile::PenaltiesJson,
        relative: "penalties.json",
        required: true,
    },
    FileEntry {
        key: InputFile::StagesJson,
        relative: "stages.json",
        required: true,
    },
    FileEntry {
        key: InputFile::InitialConditionsJson,
        relative: "initial_conditions.json",
        required: true,
    },
    // Root-level — optional
    FileEntry {
        key: InputFile::PostStudyStagesJson,
        relative: "post_study_stages.json",
        required: false,
    },
    // system/ — required
    FileEntry {
        key: InputFile::SystemBusesJson,
        relative: "system/buses.json",
        required: true,
    },
    FileEntry {
        key: InputFile::SystemLinesJson,
        relative: "system/lines.json",
        required: true,
    },
    FileEntry {
        key: InputFile::SystemHydrosJson,
        relative: "system/hydros.json",
        required: true,
    },
    FileEntry {
        key: InputFile::SystemThermalsJson,
        relative: "system/thermals.json",
        required: true,
    },
    // system/ — optional
    FileEntry {
        key: InputFile::SystemNonControllableSourcesJson,
        relative: "system/non_controllable_sources.json",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemPumpingStationsJson,
        relative: "system/pumping_stations.json",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemEnergyContractsJson,
        relative: "system/energy_contracts.json",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemHydroGeometryParquet,
        relative: "system/hydro_geometry.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemHydroProductionModelsJson,
        relative: "system/hydro_production_models.json",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemFphaHyperplanesParquet,
        relative: "system/fpha_hyperplanes.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemHydroEnergyProductivityParquet,
        relative: "system/hydro_energy_productivity.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::SystemTailraceCurvesParquet,
        relative: "system/tailrace_curves.parquet",
        required: false,
    },
    // scenarios/ — optional
    FileEntry {
        key: InputFile::ScenariosInflowHistoryParquet,
        relative: "scenarios/inflow_history.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosInflowSeasonalStatsParquet,
        relative: "scenarios/inflow_seasonal_stats.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosInflowArCoefficientsParquet,
        relative: "scenarios/inflow_ar_coefficients.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosInflowAnnualComponentParquet,
        relative: "scenarios/inflow_annual_component.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosExternalInflowScenariosParquet,
        relative: "scenarios/external_inflow_scenarios.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosExternalLoadScenariosParquet,
        relative: "scenarios/external_load_scenarios.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosExternalNcsScenariosParquet,
        relative: "scenarios/external_ncs_scenarios.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosLoadSeasonalStatsParquet,
        relative: "scenarios/load_seasonal_stats.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosLoadFactorsJson,
        relative: "scenarios/load_factors.json",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosCorrelationJson,
        relative: "scenarios/correlation.json",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosNonControllableFactorsJson,
        relative: "scenarios/non_controllable_factors.json",
        required: false,
    },
    FileEntry {
        key: InputFile::ScenariosNonControllableStatsParquet,
        relative: "scenarios/non_controllable_stats.parquet",
        required: false,
    },
    // constraints/ — optional
    FileEntry {
        key: InputFile::ConstraintsThermalBoundsParquet,
        relative: "constraints/thermal_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsHydroBoundsParquet,
        relative: "constraints/hydro_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsLineBoundsParquet,
        relative: "constraints/line_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsPumpingBoundsParquet,
        relative: "constraints/pumping_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsContractBoundsParquet,
        relative: "constraints/contract_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsGenericConstraintsJson,
        relative: "constraints/generic_constraints.json",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsGenericConstraintBoundsParquet,
        relative: "constraints/generic_constraint_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsGenericParametersJson,
        relative: "constraints/generic_parameters.json",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsPenaltyOverridesBusParquet,
        relative: "constraints/penalty_overrides_bus.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsPenaltyOverridesLineParquet,
        relative: "constraints/penalty_overrides_line.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsPenaltyOverridesHydroParquet,
        relative: "constraints/penalty_overrides_hydro.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsPenaltyOverridesNcsParquet,
        relative: "constraints/penalty_overrides_ncs.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsNcsBoundsParquet,
        relative: "constraints/ncs_bounds.parquet",
        required: false,
    },
    FileEntry {
        key: InputFile::ConstraintsHydroUnitGroupBoundsParquet,
        relative: "constraints/hydro_unit_group_bounds.parquet",
        required: false,
    },
];

const INPUT_FILE_COUNT: usize = INPUT_FILES.len();

// ── FileManifest ─────────────────────────────────────────────────────────────

/// Records which [`InputFile`]s are present in the case directory.
///
/// Flags default to `false`; [`validate_structure`] sets one to `true` for
/// each file found on disk. Read with [`FileManifest::present`].
#[derive(Debug, Clone)]
pub struct FileManifest {
    flags: [bool; INPUT_FILE_COUNT],
}

impl Default for FileManifest {
    fn default() -> Self {
        Self {
            flags: [false; INPUT_FILE_COUNT],
        }
    }
}

impl FileManifest {
    /// Returns whether `file` was found in the case directory.
    #[must_use]
    pub fn present(&self, file: InputFile) -> bool {
        self.flags[file as usize]
    }

    /// Records that `file` was found on disk.
    pub(crate) fn set_present(&mut self, file: InputFile) {
        self.flags[file as usize] = true;
    }
}

// ── validate_structure ────────────────────────────────────────────────────────

/// Describes an input file no longer read; `replacement` names its migration
/// in the rejection message.
struct RemovedFile {
    /// Path relative to the case root.
    relative: &'static str,
    replacement: &'static str,
}

/// An input file no longer read by any loader is rejected when present —
/// never accepted and silently ignored.
const REMOVED_FILES: &[RemovedFile] = &[
    RemovedFile {
        relative: "constraints/exchange_factors.json",
        replacement: "per-block line capacity is now declared as absolute MW in \
            constraints/line_bounds.parquet via direct_mw / reverse_mw rows \
            carrying a block_id (use base_capacity * factor for each block). \
            Remove the file.",
    },
    RemovedFile {
        relative: "system/scalar_parameters.json",
        replacement: "scalar parameters are now read from \
            constraints/generic_parameters.json, beside the constraints that \
            reference them by @name. Move the file (its contents are unchanged).",
    },
];

/// Performs Layer 1 structural validation on the case directory at `case_root`,
/// returning a [`FileManifest`] of which files are present.
///
/// A present file sets its manifest flag to `true`. An absent **required** file
/// adds an `ErrorKind::FileNotFound` error; an absent **optional** file leaves
/// its flag `false` with no error. A present file listed in `REMOVED_FILES`
/// adds an `ErrorKind::BusinessRuleViolation` error naming its replacement.
/// This function does **not** read or parse any file content.
#[must_use]
pub fn validate_structure(case_root: &Path, ctx: &mut ValidationContext) -> FileManifest {
    let mut manifest = FileManifest::default();

    for removed in REMOVED_FILES {
        if case_root.join(removed.relative).exists() {
            ctx.emit(
                &rules::STRUCTURAL_REMOVED_FILE_PRESENT,
                removed.relative,
                None::<&str>,
                format!(
                    "{} is no longer read; {}",
                    removed.relative, removed.replacement
                ),
            );
        }
    }

    for entry in INPUT_FILES {
        if case_root.join(entry.relative).exists() {
            manifest.set_present(entry.key);
        } else if entry.required {
            ctx.emit(
                &rules::STRUCTURAL_REQUIRED_FILE_MISSING,
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

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Every case input path, relative to the case root, in `INPUT_FILES` order.
#[cfg(test)]
pub(crate) fn input_file_relative_paths() -> impl Iterator<Item = &'static str> {
    INPUT_FILES.iter().map(|entry| entry.relative)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::validation::ErrorKind;
    use std::fs;
    use tempfile::TempDir;

    /// Create a temporary case directory containing all 8 required files.
    fn make_case_with_required(dir: &TempDir) {
        let root = dir.path();
        fs::create_dir_all(root.join("system")).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("penalties.json"), b"{}").unwrap();
        fs::write(root.join("stages.json"), b"{}").unwrap();
        fs::write(root.join("initial_conditions.json"), b"{}").unwrap();
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

        assert!(
            manifest.present(InputFile::ConfigJson),
            "config.json should be present"
        );
        assert!(
            manifest.present(InputFile::PenaltiesJson),
            "penalties.json should be present"
        );
        assert!(
            manifest.present(InputFile::StagesJson),
            "stages.json should be present"
        );
        assert!(
            manifest.present(InputFile::InitialConditionsJson),
            "initial_conditions.json should be present"
        );
        assert!(
            manifest.present(InputFile::SystemBusesJson),
            "system/buses.json should be present"
        );
        assert!(
            manifest.present(InputFile::SystemLinesJson),
            "system/lines.json should be present"
        );
        assert!(
            manifest.present(InputFile::SystemHydrosJson),
            "system/hydros.json should be present"
        );
        assert!(
            manifest.present(InputFile::SystemThermalsJson),
            "system/thermals.json should be present"
        );
    }

    #[test]
    fn test_structural_missing_required_hydros() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
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
            ErrorKind::FileNotFound,
            "error kind should be FileNotFound"
        );
        assert!(
            entry.file.to_string_lossy().contains("hydros.json"),
            "error file should reference hydros.json, got: {}",
            entry.file.display()
        );
        assert!(
            !manifest.present(InputFile::SystemHydrosJson),
            "manifest.present(SystemHydrosJson) should be false"
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
        assert!(!manifest.present(InputFile::SystemNonControllableSourcesJson));
        assert!(!manifest.present(InputFile::SystemHydroGeometryParquet));
        assert!(!manifest.present(InputFile::ScenariosInflowHistoryParquet));
        assert!(!manifest.present(InputFile::ConstraintsThermalBoundsParquet));
        assert!(!manifest.present(InputFile::ScenariosCorrelationJson));
    }

    #[test]
    fn test_structural_optional_present_in_manifest() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("correlation.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(!ctx.has_errors());
        assert!(
            manifest.present(InputFile::ScenariosCorrelationJson),
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
        for entry in ctx.errors() {
            assert_eq!(
                entry.kind,
                ErrorKind::FileNotFound,
                "all errors should be FileNotFound"
            );
        }
    }

    /// Registry invariants: every row's relative path is unique, the table
    /// covers every [`InputFile`] variant exactly once in declaration order,
    /// and the required set is exactly the eight files documented above.
    #[test]
    fn test_input_files_registry_invariants() {
        assert_eq!(
            INPUT_FILES.len(),
            INPUT_FILE_COUNT,
            "INPUT_FILES must have one row per InputFile variant"
        );

        let mut seen_paths = std::collections::HashSet::new();
        for entry in INPUT_FILES {
            assert!(
                seen_paths.insert(entry.relative),
                "duplicate relative path in INPUT_FILES: {}",
                entry.relative
            );
        }

        for (index, entry) in INPUT_FILES.iter().enumerate() {
            assert_eq!(
                entry.key as usize, index,
                "INPUT_FILES row {index} ({}) is out of variant-declaration order",
                entry.relative
            );
        }

        let required_paths: std::collections::HashSet<&'static str> = INPUT_FILES
            .iter()
            .filter(|entry| entry.required)
            .map(|entry| entry.relative)
            .collect();
        let expected_required: std::collections::HashSet<&'static str> = [
            "config.json",
            "penalties.json",
            "stages.json",
            "initial_conditions.json",
            "system/buses.json",
            "system/lines.json",
            "system/hydros.json",
            "system/thermals.json",
        ]
        .into_iter()
        .collect();
        assert_eq!(
            required_paths, expected_required,
            "the required set must be exactly the eight files documented today"
        );
    }

    #[test]
    fn test_scalar_parameters_json_and_hydro_energy_productivity_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::create_dir_all(dir.path().join("constraints")).unwrap();
        fs::write(
            dir.path().join("constraints/generic_parameters.json"),
            b"{\"scalar_parameters\":[]}",
        )
        .unwrap();
        fs::write(
            dir.path().join("system/hydro_energy_productivity.parquet"),
            b"",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no errors expected when all required files present"
        );
        assert!(
            manifest.present(InputFile::ConstraintsGenericParametersJson),
            "constraints/generic_parameters.json should be present"
        );
        assert!(
            manifest.present(InputFile::SystemHydroEnergyProductivityParquet),
            "system/hydro_energy_productivity.parquet should be present"
        );
    }

    #[test]
    fn test_scalar_parameters_json_and_hydro_energy_productivity_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // The optional files are deliberately not created

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional files must not produce errors"
        );
        assert!(
            !manifest.present(InputFile::ConstraintsGenericParametersJson),
            "constraints/generic_parameters.json should be false when file is absent"
        );
        assert!(
            !manifest.present(InputFile::SystemHydroEnergyProductivityParquet),
            "system/hydro_energy_productivity.parquet should be false when file is absent"
        );
    }

    /// AC: a case directory containing `constraints/generic_parameters.json` must set
    /// `manifest.present(InputFile::ConstraintsGenericParametersJson) == true`.
    #[test]
    fn manifest_detects_scalar_parameters_json_when_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::create_dir_all(dir.path().join("constraints")).unwrap();
        fs::write(
            dir.path().join("constraints/generic_parameters.json"),
            b"{\"scalar_parameters\":[]}",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "no errors expected when all required files are present"
        );
        assert!(
            manifest.present(InputFile::ConstraintsGenericParametersJson),
            "constraints/generic_parameters.json must be true when constraints/generic_parameters.json exists"
        );
    }

    /// AC: a case directory with no scalar parameter file must report
    /// `manifest.present(InputFile::ConstraintsGenericParametersJson) == false` without producing an error.
    #[test]
    fn manifest_reports_absent_when_no_parameter_file() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // constraints/generic_parameters.json is deliberately absent.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file must not produce errors"
        );
        assert!(
            !manifest.present(InputFile::ConstraintsGenericParametersJson),
            "constraints/generic_parameters.json must be false when constraints/generic_parameters.json is absent"
        );
    }

    #[test]
    fn test_manifest_tailrace_curves_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        fs::write(dir.path().join("system/tailrace_curves.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.present(InputFile::SystemTailraceCurvesParquet),
            "system/tailrace_curves.parquet should be true when file is present"
        );
    }

    #[test]
    fn test_manifest_tailrace_curves_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // system/tailrace_curves.parquet deliberately absent.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !manifest.present(InputFile::SystemTailraceCurvesParquet),
            "system/tailrace_curves.parquet should be false when file is absent"
        );
    }

    /// AC #5: `scenarios/inflow_annual_component.parquet` present → manifest flag `true`.
    #[test]
    fn test_manifest_detects_inflow_annual_component_present() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("inflow_annual_component.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.present(InputFile::ScenariosInflowAnnualComponentParquet),
            "scenarios/inflow_annual_component.parquet should be true when file is present"
        );
    }

    /// AC #6: `scenarios/inflow_annual_component.parquet` absent → manifest flag `false`, no error.
    #[test]
    fn test_manifest_inflow_annual_component_absent() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // Do not create the optional annual component file.

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !manifest.present(InputFile::ScenariosInflowAnnualComponentParquet),
            "scenarios/inflow_annual_component.parquet should be false when file is absent"
        );
    }

    /// AC: `constraints/hydro_unit_group_bounds.parquet` present -> manifest flag
    /// `true`; absent (a separate case directory) -> `false`, no error either way.
    #[test]
    fn test_manifest_hydro_unit_group_bounds() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        fs::write(constraints_dir.join("hydro_unit_group_bounds.parquet"), b"").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "present optional file should not produce errors"
        );
        assert!(
            manifest.present(InputFile::ConstraintsHydroUnitGroupBoundsParquet),
            "constraints/hydro_unit_group_bounds.parquet should be true when file is present"
        );

        let absent_dir = TempDir::new().unwrap();
        make_case_with_required(&absent_dir);

        let mut absent_ctx = ValidationContext::new();
        let absent_manifest = validate_structure(absent_dir.path(), &mut absent_ctx);

        assert!(
            !absent_ctx.has_errors(),
            "absent optional file should not produce errors"
        );
        assert!(
            !absent_manifest.present(InputFile::ConstraintsHydroUnitGroupBoundsParquet),
            "constraints/hydro_unit_group_bounds.parquet should be false when file is absent"
        );
    }

    #[test]
    fn removed_exchange_factors_file_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        // An empty factor list, not garbage bytes: Layer 1 rejects on presence
        // alone and never parses, so even a well-formed empty file is refused.
        fs::write(
            constraints_dir.join("exchange_factors.json"),
            b"{\"exchange_factors\": []}",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert_eq!(
            ctx.errors().len(),
            1,
            "should have exactly 1 error when constraints/exchange_factors.json is present, got: {:?}",
            ctx.errors()
        );
        let entry = ctx.errors()[0];
        assert_eq!(
            entry.kind,
            ErrorKind::BusinessRuleViolation,
            "removed-file rejection should carry BusinessRuleViolation"
        );
        assert!(
            entry
                .file
                .to_string_lossy()
                .contains("constraints/exchange_factors.json"),
            "error file should reference constraints/exchange_factors.json, got: {}",
            entry.file.display()
        );
        assert!(
            entry.message.contains("constraints/exchange_factors.json"),
            "message should name the removed file, got: {}",
            entry.message
        );
        assert!(
            entry.message.contains("constraints/line_bounds.parquet"),
            "message should name the replacement file, got: {}",
            entry.message
        );
    }

    #[test]
    fn removed_scalar_parameters_file_is_rejected() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // The relocated parameters file at its old system/ path: Layer 1 rejects
        // on presence alone and never parses, so a well-formed empty list is
        // still refused rather than silently read.
        fs::write(
            dir.path().join("system/scalar_parameters.json"),
            b"{\"scalar_parameters\": []}",
        )
        .unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert_eq!(
            ctx.errors().len(),
            1,
            "should have exactly 1 error when system/scalar_parameters.json is present, got: {:?}",
            ctx.errors()
        );
        let entry = ctx.errors()[0];
        assert_eq!(
            entry.kind,
            ErrorKind::BusinessRuleViolation,
            "removed-file rejection should carry BusinessRuleViolation"
        );
        assert!(
            entry
                .file
                .to_string_lossy()
                .contains("system/scalar_parameters.json"),
            "error file should reference the withdrawn path, got: {}",
            entry.file.display()
        );
        assert!(
            entry
                .message
                .contains("constraints/generic_parameters.json"),
            "message should name the new path so the old file is rejected loudly, got: {}",
            entry.message
        );
    }

    #[test]
    fn absent_removed_file_produces_no_finding() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        // constraints/exchange_factors.json deliberately absent.

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "absent removed file should not produce any error, got: {:?}",
            ctx.errors()
        );
    }

    #[test]
    fn removed_file_check_does_not_disturb_manifest_flags() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        fs::write(constraints_dir.join("exchange_factors.json"), b"{}").unwrap();
        fs::write(constraints_dir.join("line_bounds.parquet"), b"").unwrap();
        fs::write(constraints_dir.join("hydro_bounds.parquet"), b"").unwrap();
        let scenarios_dir = dir.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).unwrap();
        fs::write(scenarios_dir.join("load_factors.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let manifest = validate_structure(dir.path(), &mut ctx);

        // The three named optional files stay tracked in the manifest; the
        // removed file itself carries no manifest field — REMOVED_FILES is a
        // separate rejection loop, not an INPUT_FILES row.
        assert!(manifest.present(InputFile::ConstraintsLineBoundsParquet));
        assert!(manifest.present(InputFile::ConstraintsHydroBoundsParquet));
        assert!(manifest.present(InputFile::ScenariosLoadFactorsJson));

        // Every other optional flag stays false — pins that the removed-file
        // loop did not shift the INPUT_FILES / FileManifest ordinal keying.
        assert!(!manifest.present(InputFile::SystemNonControllableSourcesJson));
        assert!(!manifest.present(InputFile::SystemPumpingStationsJson));
        assert!(!manifest.present(InputFile::SystemEnergyContractsJson));
        assert!(!manifest.present(InputFile::SystemHydroGeometryParquet));
        assert!(!manifest.present(InputFile::SystemHydroProductionModelsJson));
        assert!(!manifest.present(InputFile::SystemFphaHyperplanesParquet));
        assert!(!manifest.present(InputFile::SystemHydroEnergyProductivityParquet));
        assert!(!manifest.present(InputFile::SystemTailraceCurvesParquet));

        assert!(!manifest.present(InputFile::ScenariosInflowHistoryParquet));
        assert!(!manifest.present(InputFile::ScenariosInflowSeasonalStatsParquet));
        assert!(!manifest.present(InputFile::ScenariosInflowArCoefficientsParquet));
        assert!(!manifest.present(InputFile::ScenariosInflowAnnualComponentParquet));
        assert!(!manifest.present(InputFile::ScenariosExternalInflowScenariosParquet));
        assert!(!manifest.present(InputFile::ScenariosExternalLoadScenariosParquet));
        assert!(!manifest.present(InputFile::ScenariosExternalNcsScenariosParquet));
        assert!(!manifest.present(InputFile::ScenariosLoadSeasonalStatsParquet));
        assert!(!manifest.present(InputFile::ScenariosCorrelationJson));
        assert!(!manifest.present(InputFile::ScenariosNonControllableFactorsJson));
        assert!(!manifest.present(InputFile::ScenariosNonControllableStatsParquet));

        assert!(!manifest.present(InputFile::ConstraintsThermalBoundsParquet));
        assert!(!manifest.present(InputFile::ConstraintsPumpingBoundsParquet));
        assert!(!manifest.present(InputFile::ConstraintsContractBoundsParquet));
        assert!(!manifest.present(InputFile::ConstraintsGenericConstraintsJson));
        assert!(!manifest.present(InputFile::ConstraintsGenericConstraintBoundsParquet));
        assert!(!manifest.present(InputFile::ConstraintsGenericParametersJson));
        assert!(!manifest.present(InputFile::ConstraintsPenaltyOverridesBusParquet));
        assert!(!manifest.present(InputFile::ConstraintsPenaltyOverridesLineParquet));
        assert!(!manifest.present(InputFile::ConstraintsPenaltyOverridesHydroParquet));
        assert!(!manifest.present(InputFile::ConstraintsPenaltyOverridesNcsParquet));
        assert!(!manifest.present(InputFile::ConstraintsNcsBoundsParquet));
        assert!(!manifest.present(InputFile::ConstraintsHydroUnitGroupBoundsParquet));
    }

    #[test]
    fn similarly_named_file_alongside_removed_file_is_not_rejected() {
        let dir = TempDir::new().unwrap();
        make_case_with_required(&dir);
        let constraints_dir = dir.path().join("constraints");
        fs::create_dir_all(&constraints_dir).unwrap();
        // Substring "exchange", not the exact REMOVED_FILES path: pins that the
        // check is exact-path equality, not a substring/prefix match that would
        // also catch this file.
        fs::write(constraints_dir.join("exchange_factors_backup.json"), b"{}").unwrap();

        let mut ctx = ValidationContext::new();
        let _manifest = validate_structure(dir.path(), &mut ctx);

        assert!(
            !ctx.has_errors(),
            "a similarly-named file must not trigger the removed-file rejection, got: {:?}",
            ctx.errors()
        );
    }
}
