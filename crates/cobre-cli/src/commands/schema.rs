//! `cobre schema export [--output-dir DIR]` subcommand.
//!
//! Generates JSON Schema files for all user-facing case directory input types
//! and writes them to the specified output directory. The subcommand uses a
//! nested `export` sub-subcommand to leave room for future additions such as
//! `cobre schema validate` or `cobre schema list`.

use std::path::PathBuf;

use clap::{Args, Subcommand};
use console::Term;

use crate::error::CliError;

/// Arguments for the `cobre schema` subcommand.
#[derive(Debug, Args)]
#[command(about = "Manage JSON Schema files for case directory input types")]
pub struct SchemaArgs {
    /// Schema operation to perform.
    #[command(subcommand)]
    pub command: SchemaCommand,
}

/// Sub-subcommands for `cobre schema`.
#[derive(Debug, Subcommand)]
pub enum SchemaCommand {
    /// Export JSON Schema files for all input types to a directory.
    Export(ExportArgs),
}

/// Arguments for the `cobre schema export` sub-subcommand.
#[derive(Debug, Args)]
#[command(about = "Export JSON Schema files for all input types")]
pub struct ExportArgs {
    /// Directory to write schema files into.
    ///
    /// The directory is created if it does not exist. Existing schema files
    /// are overwritten without prompting (schemas are generated, not hand-edited).
    #[arg(long, default_value = ".")]
    pub output_dir: PathBuf,
}

/// Execute the `schema` subcommand by dispatching to the appropriate operation.
///
/// # Errors
///
/// Returns [`CliError::Internal`] if schema generation fails.
/// Returns [`CliError::Io`] if the output directory cannot be created or a
/// file write fails.
#[allow(clippy::needless_pass_by_value)]
pub fn execute(args: SchemaArgs) -> Result<(), CliError> {
    match args.command {
        SchemaCommand::Export(ref export_args) => execute_export(export_args),
    }
}

/// Write all generated schemas to `args.output_dir`.
fn execute_export(args: &ExportArgs) -> Result<(), CliError> {
    let output_dir = &args.output_dir;

    std::fs::create_dir_all(output_dir).map_err(|source| CliError::Io {
        source,
        context: format!("creating output directory '{}'", output_dir.display()),
    })?;

    let schemas = cobre_io::schema::generate_schemas().map_err(|e| CliError::Internal {
        message: format!("schema generation failed: {e}"),
    })?;

    let count = schemas.len();

    for (filename, value) in schemas {
        let dest = output_dir.join(&filename);
        let content = serde_json::to_string_pretty(&value).map_err(|e| CliError::Internal {
            message: format!("failed to serialize schema '{filename}': {e}"),
        })?;
        std::fs::write(&dest, content).map_err(|source| CliError::Io {
            source,
            context: dest.display().to_string(),
        })?;
    }

    let stderr = Term::stderr();
    let _ = stderr.write_line(&format!(
        "Exported {count} schema files to {}",
        output_dir.display()
    ));

    Ok(())
}
