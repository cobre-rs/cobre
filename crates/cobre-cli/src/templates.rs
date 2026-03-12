//! Template registry for the `cobre init` subcommand.
//!
//! Provides a compile-time registry of embedded case templates. Each template
//! contains a named collection of files that reproduce a complete, runnable
//! case directory. All file bytes are embedded at compile time via
//! [`include_bytes!`], so the registry is entirely allocation-free.
//!
//! The template source files live at `examples/1dtoy/` in the workspace root.
//! The build script (`build.rs`) copies them into `OUT_DIR/templates/1dtoy/`
//! so that `include_bytes!` works with `cargo publish` (which only packages
//! files within the crate directory).
//!
//! # Example
//!
//! ```rust
//! use cobre::templates;
//!
//! let templates = templates::available_templates();
//! assert!(!templates.is_empty());
//!
//! let toy = templates::find_template("1dtoy").expect("1dtoy template must exist");
//! assert_eq!(toy.name, "1dtoy");
//! ```

/// A single file belonging to a template.
#[derive(Clone, Copy)]
pub(crate) struct TemplateFile {
    pub(crate) relative_path: &'static str,
    pub(crate) content: &'static [u8],
    pub(crate) description: &'static str,
}

/// A named collection of files that together form a runnable case directory.
#[derive(Clone, Copy)]
pub(crate) struct Template {
    pub(crate) name: &'static str,
    pub(crate) description: &'static str,
    pub(crate) files: &'static [TemplateFile],
}

static DTOY1_FILES: &[TemplateFile] = &[
    TemplateFile {
        relative_path: "config.json",
        content: include_bytes!(concat!(env!("OUT_DIR"), "/templates/1dtoy/config.json")),
        description: "Algorithm configuration: training (forward passes, stopping rules) and simulation settings",
    },
    TemplateFile {
        relative_path: "initial_conditions.json",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/initial_conditions.json"
        )),
        description: "Initial reservoir storage volumes for each hydro plant at the start of the planning horizon",
    },
    TemplateFile {
        relative_path: "penalties.json",
        content: include_bytes!(concat!(env!("OUT_DIR"), "/templates/1dtoy/penalties.json")),
        description: "Global penalty costs for constraint violations (deficit, excess, spillage, storage bounds, etc.)",
    },
    TemplateFile {
        relative_path: "stages.json",
        content: include_bytes!(concat!(env!("OUT_DIR"), "/templates/1dtoy/stages.json")),
        description: "Planning horizon definition: policy graph type, discount rate, stage dates, time blocks, and scenario counts",
    },
    TemplateFile {
        relative_path: "system/buses.json",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/system/buses.json"
        )),
        description: "Electrical bus definitions with deficit cost segments",
    },
    TemplateFile {
        relative_path: "system/hydros.json",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/system/hydros.json"
        )),
        description: "Hydro plant definitions: reservoir bounds, outflow limits, turbine model, and generation limits",
    },
    TemplateFile {
        relative_path: "system/lines.json",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/system/lines.json"
        )),
        description: "Transmission line definitions (empty in this single-bus example)",
    },
    TemplateFile {
        relative_path: "system/thermals.json",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/system/thermals.json"
        )),
        description: "Thermal plant definitions with piecewise cost segments and generation bounds",
    },
    TemplateFile {
        relative_path: "scenarios/inflow_seasonal_stats.parquet",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/scenarios/inflow_seasonal_stats.parquet"
        )),
        description: "Seasonal PAR(p) statistics for hydro inflow scenario generation (mean, std, lag correlations)",
    },
    TemplateFile {
        relative_path: "scenarios/load_seasonal_stats.parquet",
        content: include_bytes!(concat!(
            env!("OUT_DIR"),
            "/templates/1dtoy/scenarios/load_seasonal_stats.parquet"
        )),
        description: "Seasonal PAR(p) statistics for electrical load scenario generation (mean, std, lag correlations)",
    },
];

static DTOY1_TEMPLATE: Template = Template {
    name: "1dtoy",
    description: "Single-bus hydrothermal system with one hydro plant and two thermals over a 4-stage finite horizon",
    files: DTOY1_FILES,
};

static ALL_TEMPLATES: &[Template] = &[DTOY1_TEMPLATE];

/// Return all registered case templates.
///
/// The returned slice is ordered by registration order and is never empty.
pub(crate) fn available_templates() -> &'static [Template] {
    ALL_TEMPLATES
}

/// Look up a template by its short name (case-sensitive).
///
/// Returns `None` when no template with the given name is registered.
///
/// # Example
///
/// ```rust
/// use cobre::templates;
///
/// assert!(templates::find_template("1dtoy").is_some());
/// assert!(templates::find_template("nonexistent").is_none());
/// ```
pub(crate) fn find_template(name: &str) -> Option<&'static Template> {
    ALL_TEMPLATES.iter().find(|t| t.name == name)
}

#[cfg(test)]
mod tests {
    use super::{available_templates, find_template};

    #[test]
    fn test_available_templates_contains_1dtoy() {
        let templates = available_templates();
        assert!(templates.iter().any(|t| t.name == "1dtoy"));
    }

    #[test]
    fn test_find_template_1dtoy_returns_some() {
        let template = find_template("1dtoy").unwrap();
        assert_eq!(template.name, "1dtoy");
        assert!(!template.description.is_empty());
    }

    #[test]
    fn test_find_template_unknown_returns_none() {
        assert!(find_template("nonexistent").is_none());
    }

    #[test]
    fn test_1dtoy_template_has_files() {
        let template = find_template("1dtoy").unwrap();
        assert_eq!(template.files.len(), 10);
    }

    #[test]
    fn test_1dtoy_files_have_descriptions() {
        let template = find_template("1dtoy").unwrap();
        for file in template.files {
            assert!(!file.description.is_empty());
        }
    }

    #[test]
    fn test_1dtoy_files_have_relative_paths() {
        let template = find_template("1dtoy").unwrap();
        for file in template.files {
            assert!(!file.relative_path.is_empty());
            assert!(!file.relative_path.starts_with('/'));
        }
    }

    #[test]
    fn test_1dtoy_config_json_content_matches_source() {
        let template = find_template("1dtoy").unwrap();
        let embedded = template
            .files
            .iter()
            .find(|f| f.relative_path == "config.json")
            .expect("1dtoy template must contain config.json");

        let on_disk = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/1dtoy/config.json"
        ))
        .expect("examples/1dtoy/config.json must be readable");

        assert_eq!(
            embedded.content,
            on_disk.as_slice(),
            "embedded config.json content must be byte-identical to the source file"
        );
    }
}
