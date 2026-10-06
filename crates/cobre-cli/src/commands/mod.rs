//! Subcommand implementations for the `cobre` binary.
//!
//! Each module pairs a clap-derived `Args` struct with an `execute` function.

use std::path::{Path, PathBuf};

pub(crate) mod broadcast;
pub mod init;
pub mod run;
pub mod schema;
pub mod validate;
pub mod version;

/// The one resolution of the output directory for `cobre run` (writes) and
/// `cobre validate` (reads policy): `output` verbatim, never canonicalised, else
/// `<case_dir>/output`.
pub(crate) fn resolve_output_dir(case_dir: &Path, output: Option<&Path>) -> PathBuf {
    output.map_or_else(|| case_dir.join("output"), Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_output_dir_is_the_case_output_subdirectory() {
        assert_eq!(
            resolve_output_dir(Path::new("cases/a"), None),
            PathBuf::from("cases/a/output")
        );
    }

    #[test]
    fn explicit_output_dir_is_used_verbatim() {
        assert_eq!(
            resolve_output_dir(Path::new("cases/a"), Some(Path::new("rel_out"))),
            PathBuf::from("rel_out")
        );
    }
}
