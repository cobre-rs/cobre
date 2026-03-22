//! JSON writer for the LP scaling report.
//!
//! The scaling report is a one-time diagnostic artifact produced after
//! template build and column/row scaling. It captures the coefficient
//! ranges before and after scaling for every stage.

use std::io::BufWriter;
use std::path::Path;

use super::error::OutputError;

/// Write a scaling report as pretty-printed JSON.
///
/// Accepts any `Serialize`-implementing value to avoid cross-crate type
/// dependencies (the `ScalingReport` struct lives in `cobre-sddp`).
///
/// Uses atomic write: writes to a `.json.tmp` file first, then renames.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] on filesystem failures, or
/// [`OutputError::SerializationError`] if JSON serialization fails.
pub fn write_scaling_report(
    path: &Path,
    report: &impl serde::Serialize,
) -> Result<(), OutputError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| OutputError::io(parent, e))?;
    }

    let tmp_path = path.with_extension("json.tmp");
    let file = std::fs::File::create(&tmp_path).map_err(|e| OutputError::io(&tmp_path, e))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, report).map_err(|e| {
        OutputError::serialization("scaling_report", format!("JSON serialization: {e}"))
    })?;
    std::fs::rename(&tmp_path, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use tempfile::TempDir;

    #[derive(Serialize)]
    struct MockReport {
        cost_scale_factor: f64,
        num_stages: usize,
    }

    #[test]
    fn write_and_read_back_json() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("training/scaling_report.json");

        let report = MockReport {
            cost_scale_factor: 1000.0,
            num_stages: 3,
        };

        write_scaling_report(&path, &report).expect("write should succeed");

        let content = std::fs::read_to_string(&path).expect("read");
        assert!(content.contains("\"cost_scale_factor\": 1000.0"));
        assert!(content.contains("\"num_stages\": 3"));
    }

    #[test]
    fn tmp_file_is_cleaned_up() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("report.json");

        let report = MockReport {
            cost_scale_factor: 1.0,
            num_stages: 1,
        };

        write_scaling_report(&path, &report).expect("write should succeed");

        let tmp_path = path.with_extension("json.tmp");
        assert!(
            !tmp_path.exists(),
            "tmp file should be removed after rename"
        );
        assert!(path.exists(), "final file should exist");
    }
}
