//! The owner of output paths: one row per Parquet file a run can write, each
//! naming the [`OUTPUT_SCHEMAS`](super::schemas::OUTPUT_SCHEMAS) row that backs
//! it. One schema may back several files, so paths are keyed here rather than
//! on the schema table.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "registry metadata that only tests read until the registry is exported"
    )
)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileLayout {
    File,
    Hive { partition_by: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FileFormat {
    Parquet,
}

/// The phase whose writer produces the file. A `Training` or `Simulation` file
/// is covered by that phase's `_SUCCESS` marker; a `Setup` file, written before
/// training, by neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WritePhase {
    Setup,
    Training,
    Simulation,
}

#[derive(Debug)]
pub(crate) struct FileRegistryEntry {
    /// POSIX, relative to the output directory; a `Hive` dataset is its
    /// directory, with no trailing `/`.
    pub(crate) path: &'static str,
    pub(crate) layout: FileLayout,
    pub(crate) format: FileFormat,
    pub(crate) phase: WritePhase,
    /// A [`SchemaRegistryEntry::name`](super::schemas::SchemaRegistryEntry::name).
    pub(crate) schema: &'static str,
}

pub(crate) const OUTPUT_FILES: &[FileRegistryEntry] = &[
    FileRegistryEntry {
        path: "simulation/costs",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "costs",
    },
    FileRegistryEntry {
        path: "simulation/hydros",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "hydros",
    },
    FileRegistryEntry {
        path: "simulation/hydro_bus_generation",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "hydro_bus_generation",
    },
    FileRegistryEntry {
        path: "simulation/thermals",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "thermals",
    },
    FileRegistryEntry {
        path: "simulation/exchanges",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "exchanges",
    },
    FileRegistryEntry {
        path: "simulation/buses",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "buses",
    },
    FileRegistryEntry {
        path: "simulation/pumping_stations",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "pumping_stations",
    },
    FileRegistryEntry {
        path: "simulation/contracts",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "contracts",
    },
    FileRegistryEntry {
        path: "simulation/non_controllables",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "non_controllables",
    },
    FileRegistryEntry {
        path: "simulation/inflow_lags",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "inflow_lags",
    },
    FileRegistryEntry {
        path: "simulation/in_transit",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "in_transit",
    },
    FileRegistryEntry {
        path: "simulation/transit_seed",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "transit_seed",
    },
    FileRegistryEntry {
        path: "simulation/violations/generic",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "generic_violations",
    },
    FileRegistryEntry {
        path: "simulation/paths.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "paths",
    },
    FileRegistryEntry {
        path: "simulation/scenario_summary.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "scenario_summary",
    },
    FileRegistryEntry {
        path: "training/convergence.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "convergence",
    },
    FileRegistryEntry {
        path: "training/timing/iterations.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "iteration_timing",
    },
    FileRegistryEntry {
        path: "training/cut_selection/iterations.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "row_selection",
    },
    FileRegistryEntry {
        path: "training/solver/iterations.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "solver_iterations",
    },
    FileRegistryEntry {
        path: "simulation/solver/iterations.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "solver_iterations",
    },
    FileRegistryEntry {
        path: "training/solver/retry_histogram.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "retry_histogram",
    },
    FileRegistryEntry {
        path: "simulation/solver/retry_histogram.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "retry_histogram",
    },
    FileRegistryEntry {
        path: "anticipated/fixed_deliveries.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "fixed_delivery",
    },
    FileRegistryEntry {
        path: "simulation/anticipated_lanes",
        layout: FileLayout::Hive {
            partition_by: "scenario_id",
        },
        format: FileFormat::Parquet,
        phase: WritePhase::Simulation,
        schema: "anticipated_lanes",
    },
    FileRegistryEntry {
        path: "generic_constraints/resolved_echo.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "generic_constraint_echo",
    },
    FileRegistryEntry {
        path: "training/dictionaries/bounds.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "bounds",
    },
    FileRegistryEntry {
        path: "hydro_models/fpha_hyperplanes.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "fpha_hyperplanes",
    },
    FileRegistryEntry {
        path: "hydro_models/evaporation_models.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "evaporation_models",
    },
    FileRegistryEntry {
        path: "hydro_models/fpha_deviation_points.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Training,
        schema: "fpha_deviation_points",
    },
    FileRegistryEntry {
        path: "stochastic/noise_openings.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Setup,
        schema: "noise_openings",
    },
    FileRegistryEntry {
        path: "stochastic/inflow_seasonal_stats.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Setup,
        schema: "inflow_seasonal_stats",
    },
    FileRegistryEntry {
        path: "stochastic/inflow_ar_coefficients.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Setup,
        schema: "inflow_ar_coefficients",
    },
    FileRegistryEntry {
        path: "stochastic/inflow_annual_component.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Setup,
        schema: "inflow_annual_component",
    },
    FileRegistryEntry {
        path: "stochastic/load_seasonal_stats.parquet",
        layout: FileLayout::File,
        format: FileFormat::Parquet,
        phase: WritePhase::Setup,
        schema: "load_seasonal_stats",
    },
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::output::schemas::OUTPUT_SCHEMAS;
    use crate::output::simulation_writer::simulation_family_subpaths;
    use crate::validation::structural::input_file_relative_paths;

    #[track_caller]
    fn assert_no_problems(subject: &str, problems: &[String]) {
        assert!(
            problems.is_empty(),
            "{} {subject} problem(s):\n{}",
            problems.len(),
            problems.join("\n")
        );
    }

    #[test]
    fn output_file_paths_are_unique_relative_posix_paths() {
        let mut problems: Vec<String> = Vec::new();
        for (i, entry) in OUTPUT_FILES.iter().enumerate() {
            let path = entry.path;
            if OUTPUT_FILES[..i].iter().any(|earlier| earlier.path == path) {
                problems.push(format!("{path}: repeated"));
            }
            if path.starts_with('/') || path.ends_with('/') {
                problems.push(format!("{path}: starts or ends with `/`"));
            }
            if path.contains('\\') {
                problems.push(format!("{path}: contains `\\`"));
            }
            if path
                .split('/')
                .any(|segment| matches!(segment, "" | "." | ".."))
            {
                problems.push(format!("{path}: has an empty, `.` or `..` segment"));
            }
            if path.contains(['*', '?', '[', ']', '{', '}']) {
                problems.push(format!("{path}: contains a glob character"));
            }
            let extension = match entry.format {
                FileFormat::Parquet => ".parquet",
            };
            let last_segment = path.rsplit('/').next().unwrap_or(path);
            match entry.layout {
                FileLayout::File if !path.ends_with(extension) => {
                    problems.push(format!("{path}: file entry does not end in {extension}"));
                }
                FileLayout::Hive { .. } if last_segment.contains('.') => {
                    problems.push(format!(
                        "{path}: Hive dataset's last segment {last_segment:?} contains `.`"
                    ));
                }
                FileLayout::File | FileLayout::Hive { .. } => {}
            }
        }
        assert_no_problems("OUTPUT_FILES path", &problems);
    }

    #[test]
    fn every_output_file_names_a_registry_schema_and_every_schema_backs_a_file() {
        const EXPECTED_FILE_COUNT: usize = 34;
        let mut problems: Vec<String> = Vec::new();
        if OUTPUT_FILES.len() != EXPECTED_FILE_COUNT {
            problems.push(format!(
                "OUTPUT_FILES has {} rows, expected {EXPECTED_FILE_COUNT}; a written file \
                 added or removed must add or remove its row and update this count",
                OUTPUT_FILES.len()
            ));
        }
        for entry in OUTPUT_FILES {
            let named = OUTPUT_SCHEMAS
                .iter()
                .filter(|schema| schema.name == entry.schema)
                .count();
            if named != 1 {
                problems.push(format!(
                    "{}: schema {:?} matches {named} OUTPUT_SCHEMAS rows, expected 1",
                    entry.path, entry.schema
                ));
            }
        }
        for schema in OUTPUT_SCHEMAS {
            if !OUTPUT_FILES.iter().any(|entry| entry.schema == schema.name) {
                problems.push(format!(
                    "OUTPUT_SCHEMAS row {:?} backs no OUTPUT_FILES entry",
                    schema.name
                ));
            }
        }
        assert_no_problems("OUTPUT_FILES schema", &problems);
    }

    #[test]
    fn simulation_families_are_the_hive_outputs_partitioned_by_scenario_id() {
        let families: BTreeSet<String> = simulation_family_subpaths()
            .map(|subpath| format!("simulation/{subpath}"))
            .collect();
        let mut problems: Vec<String> = Vec::new();
        let mut hive_paths: BTreeSet<String> = BTreeSet::new();
        for entry in OUTPUT_FILES {
            if let FileLayout::Hive { partition_by } = entry.layout {
                hive_paths.insert(entry.path.to_string());
                if partition_by != "scenario_id" {
                    problems.push(format!(
                        "{}: partitioned by {partition_by:?}, expected \"scenario_id\"",
                        entry.path
                    ));
                }
                if entry.phase != WritePhase::Simulation {
                    problems.push(format!(
                        "{}: Hive dataset in phase {:?}, expected Simulation",
                        entry.path, entry.phase
                    ));
                }
            }
        }
        for missing in families.difference(&hive_paths) {
            problems.push(format!("{missing}: simulation family with no Hive entry"));
        }
        for extra in hive_paths.difference(&families) {
            problems.push(format!("{extra}: Hive entry that is no simulation family"));
        }
        assert_no_problems("Hive entry", &problems);
    }

    fn lies_under_a_case_input_directory(path: &str) -> bool {
        let first_segment = path.split('/').next();
        input_file_relative_paths()
            .filter_map(|input| input.split_once('/').map(|(directory, _)| directory))
            .any(|directory| Some(directory) == first_segment)
    }

    #[test]
    fn no_output_file_lies_under_a_case_input_directory() {
        assert!(
            lies_under_a_case_input_directory("system/hydro_energy_productivity.parquet"),
            "the case-input directory guard must flag a case input path"
        );
        let problems: Vec<String> = OUTPUT_FILES
            .iter()
            .filter(|entry| lies_under_a_case_input_directory(entry.path))
            .map(|entry| format!("{}: lies under a case-input directory", entry.path))
            .collect();
        assert_no_problems("case-input directory", &problems);
    }
}
