//! `cobre init --template <NAME> <DIRECTORY>` subcommand.
//!
//! Scaffolds a new case directory from an embedded template. With `--list`,
//! prints all available template names and descriptions to stdout and exits.
//! With `--template <NAME> <DIRECTORY>`, writes template files to the given
//! directory, prints the Cobre banner and a file summary to stderr, and exits.

use std::path::PathBuf;

use clap::Args;
use console::{Term, style};

use crate::error::CliError;
use crate::templates;

/// Arguments for the `cobre init` subcommand.
#[derive(Debug, Args)]
#[command(
    about = "Scaffold a new case directory from an embedded template",
    long_about = "Scaffold a new case directory from an embedded template.\n\n\
        Use --list to see all available templates, or --template <NAME> <DIRECTORY> \
        to create a new case directory from a template."
)]
pub struct InitArgs {
    /// Template name to scaffold (e.g. `1dtoy`). Mutually informative with `--list`.
    #[arg(
        long,
        value_name = "NAME",
        required_unless_present = "list",
        conflicts_with = "list"
    )]
    pub template: Option<String>,

    /// List all available templates and exit.
    #[arg(long, conflicts_with = "template")]
    pub list: bool,

    /// Overwrite existing files in the target directory.
    #[arg(long)]
    pub force: bool,

    /// Target directory where the template files will be written.
    ///
    /// Required when `--template` is provided; ignored with `--list`.
    #[arg(required_unless_present = "list", conflicts_with = "list")]
    pub directory: Option<PathBuf>,
}

/// Execute the `init` subcommand.
///
/// When `--list` is set, prints all template names and descriptions to stdout.
/// When `--template <NAME> <DIRECTORY>` is provided, looks up the template,
/// creates the directory tree, writes all template files, and prints a summary.
///
/// # Errors
///
/// Returns [`CliError::Validation`] when the template name is not found.
/// Returns [`CliError::Io`] when the target directory is non-empty and
/// `--force` is not set, or when a write operation fails.
#[allow(clippy::needless_pass_by_value)]
pub fn execute(args: InitArgs) -> Result<(), CliError> {
    if args.list {
        let stdout = Term::stdout();
        for tmpl in templates::available_templates() {
            let _ = stdout.write_line(&format!("{:<16}  {}", tmpl.name, tmpl.description));
        }
        return Ok(());
    }

    let template_name = args.template.ok_or_else(|| CliError::Internal {
        message: "template argument missing despite clap contract".to_string(),
    })?;
    let directory = args.directory.ok_or_else(|| CliError::Internal {
        message: "directory argument missing despite clap contract".to_string(),
    })?;

    execute_scaffold(&template_name, &directory, args.force)
}

/// Scaffold the named template into the given directory.
fn execute_scaffold(
    template_name: &str,
    directory: &std::path::Path,
    force: bool,
) -> Result<(), CliError> {
    let stderr = Term::stderr();

    // Resolve template — exit code 1 on unknown name.
    let template = templates::find_template(template_name).ok_or_else(|| {
        let available: Vec<&str> = templates::available_templates()
            .iter()
            .map(|t| t.name)
            .collect();
        CliError::Validation {
            report: format!(
                "unknown template '{}'. Available templates: {}",
                template_name,
                available.join(", ")
            ),
        }
    })?;

    if directory.exists() {
        let is_nonempty = directory
            .read_dir()
            .map(|mut d| d.next().is_some())
            .unwrap_or(false);

        if is_nonempty && !force {
            return Err(CliError::Io {
                source: std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "directory '{}' already exists and is not empty",
                        directory.display()
                    ),
                ),
                context: format!(
                    "use --force to overwrite existing files in '{}'",
                    directory.display()
                ),
            });
        }
    }

    std::fs::create_dir_all(directory).map_err(|source| CliError::Io {
        source,
        context: format!("creating directory '{}'", directory.display()),
    })?;

    for file in template.files {
        let dest = directory.join(file.relative_path);

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|source| CliError::Io {
                source,
                context: format!("creating directory '{}'", parent.display()),
            })?;
        }

        std::fs::write(&dest, file.content).map_err(|source| CliError::Io {
            source,
            context: dest.display().to_string(),
        })?;
    }

    crate::banner::print_banner(&stderr);
    print_summary(&stderr, template, directory);

    Ok(())
}

fn print_summary(stderr: &Term, template: &templates::Template, directory: &std::path::Path) {
    let check = style("✔").green().bold();
    let dim_arrow = style("->").yellow();

    let _ = stderr.write_line(&format!(
        "Created {} case directory from template '{}':",
        style(directory.display().to_string()).bold(),
        style(template.name).cyan()
    ));
    let _ = stderr.write_line("");

    for file in template.files {
        let _ = stderr.write_line(&format!(
            "  {} {}  {}",
            check,
            style(file.relative_path).bold(),
            style(file.description).dim()
        ));
    }

    let _ = stderr.write_line("");
    let _ = stderr.write_line("Next steps:");
    let _ = stderr.write_line(&format!(
        "  {} cobre validate {}",
        dim_arrow,
        directory.display()
    ));
    let _ = stderr.write_line(&format!(
        "  {} cobre run {} --output {}/results",
        dim_arrow,
        directory.display(),
        directory.display()
    ));
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn test_init_list_prints_template_names() {
        let templates = templates::available_templates();
        let names: Vec<&str> = templates.iter().map(|t| t.name).collect();
        assert!(names.contains(&"1dtoy"));
    }

    #[test]
    fn test_init_list_execute_returns_ok() {
        let args = InitArgs {
            template: None,
            list: true,
            force: false,
            directory: None,
        };
        assert!(execute(args).is_ok());
    }

    #[test]
    fn test_init_unknown_template_returns_validation_error() {
        let tmp = TempDir::new().unwrap();
        let args = InitArgs {
            template: Some("bogus_template_xyz".to_string()),
            list: false,
            force: false,
            directory: Some(tmp.path().to_path_buf()),
        };
        let Err(CliError::Validation { report }) = execute(args) else {
            panic!("expected CliError::Validation");
        };
        assert!(report.contains("bogus_template_xyz"));
        assert!(report.contains("1dtoy"));
    }

    #[test]
    fn test_init_creates_directory_and_files() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("new_case");
        let args = InitArgs {
            template: Some("1dtoy".to_string()),
            list: false,
            force: false,
            directory: Some(target.clone()),
        };

        assert!(execute(args).is_ok());

        assert!(target.exists());
        assert!(target.is_dir());

        assert!(target.join("config.json").exists());
        assert!(target.join("system").join("hydros.json").exists());
        assert!(
            target
                .join("scenarios")
                .join("inflow_seasonal_stats.parquet")
                .exists()
        );

        let template = templates::find_template("1dtoy").unwrap();
        for file in template.files {
            let dest = target.join(file.relative_path);
            assert!(dest.exists());
        }
    }

    #[test]
    fn test_init_existing_non_empty_dir_without_force_returns_io_error() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("existing.txt"), b"some content").unwrap();

        let args = InitArgs {
            template: Some("1dtoy".to_string()),
            list: false,
            force: false,
            directory: Some(tmp.path().to_path_buf()),
        };

        let Err(CliError::Io { context, .. }) = execute(args) else {
            panic!("expected CliError::Io");
        };
        assert!(context.contains("--force"));
    }

    #[test]
    fn test_init_existing_non_empty_dir_with_force_succeeds() {
        let tmp = TempDir::new().unwrap();
        let pre_existing = tmp.path().join("existing.txt");
        std::fs::write(&pre_existing, b"some content").unwrap();

        let args = InitArgs {
            template: Some("1dtoy".to_string()),
            list: false,
            force: true,
            directory: Some(tmp.path().to_path_buf()),
        };

        assert!(execute(args).is_ok());
        assert!(pre_existing.exists());
        assert!(tmp.path().join("config.json").exists());
    }
}
