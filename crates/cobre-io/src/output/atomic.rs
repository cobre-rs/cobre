//! Single owner of the write-side crash-safety contract: write to `{path}.tmp`,
//! flush **explicitly** (never via `Drop`), then `rename` onto `path`.
//!
//! The explicit flush before the rename is load-bearing: `BufWriter` flushes on
//! drop, but `Drop::drop` cannot return an error, so a drop-flush swallows an
//! `ENOSPC`/`EIO` on the buffered tail and the rename then installs a truncated
//! file with no error surfaced. Always flush via [`std::io::Write::flush`] and
//! propagate with `?` — never rely on drop-flush before a rename.
//!
//! Checkpoint commits also `sync_all` every file and, on unix, fsync every
//! directory they change, so a committed copy survives a power loss.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use serde::Serialize;

use super::error::OutputError;
use super::parquet_config::WRITER_PROPERTIES;

/// Temporary sibling path for an atomic write, preserving the original extension
/// as a prefix of `.tmp` (`foo.json` → `foo.json.tmp`) so the temp never
/// collides with a differently-typed sibling.
pub(crate) fn tmp_path(path: &Path) -> PathBuf {
    path.with_extension(path.extension().map_or_else(
        || "tmp".to_string(),
        |ext| format!("{}.tmp", ext.to_string_lossy()),
    ))
}

/// Create `path`'s parent directory, if it has one and it does not already exist.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if directory creation fails.
pub(crate) fn ensure_parent_dir(path: &Path) -> Result<(), OutputError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| OutputError::io(parent, e))?;
    }
    Ok(())
}

/// Write `bytes` to `path` atomically (write to `{path}.tmp`, flush, rename).
///
/// The parent directory must already exist. On any I/O error the target `path`
/// is left untouched; a partial `.tmp` may remain on disk.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if creating, writing, flushing, or renaming
/// the temporary file fails.
pub(crate) fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), OutputError> {
    let tmp = tmp_path(path);

    let file = std::fs::File::create(&tmp).map_err(|e| OutputError::io(&tmp, e))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .map_err(|e| OutputError::io(&tmp, e))?;
    // Explicit flush before rename — see module doc.
    writer.flush().map_err(|e| OutputError::io(&tmp, e))?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Write `bytes` to `path`, then flush and `sync_all` it before returning.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if creating, writing, flushing, or syncing
/// the file fails.
pub(crate) fn write_bytes_synced(path: &Path, bytes: &[u8]) -> Result<(), OutputError> {
    let file = std::fs::File::create(path).map_err(|e| OutputError::io(path, e))?;
    let mut writer = BufWriter::new(file);
    writer
        .write_all(bytes)
        .map_err(|e| OutputError::io(path, e))?;
    writer.flush().map_err(|e| OutputError::io(path, e))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// [`write_bytes_atomic`], with the temporary file synced before the rename.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if creating, writing, flushing, syncing, or
/// renaming the temporary file fails.
pub(crate) fn write_bytes_atomic_synced(path: &Path, bytes: &[u8]) -> Result<(), OutputError> {
    let tmp = tmp_path(path);
    write_bytes_synced(&tmp, bytes)?;
    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Make the entries of directory `dir` durable.
///
/// # Errors
///
/// Returns [`OutputError::IoError`] if opening or syncing `dir` fails.
#[cfg(unix)]
pub(crate) fn sync_dir(dir: &Path) -> Result<(), OutputError> {
    std::fs::File::open(dir)
        .and_then(|handle| handle.sync_all())
        .map_err(|e| OutputError::io(dir, e))
}

/// Does nothing: off unix, a directory cannot be synced through
/// [`std::fs::File::sync_all`].
#[cfg(not(unix))]
pub(crate) fn sync_dir(_dir: &Path) -> Result<(), OutputError> {
    Ok(())
}

/// Serialize `value` to pretty-printed JSON and write it to `path` atomically.
///
/// Byte content is identical to `serde_json::to_vec_pretty`. The parent
/// directory must already exist. `entity` labels any
/// [`OutputError::SerializationError`].
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if JSON serialization fails, or
/// [`OutputError::IoError`] if creating, flushing, or renaming the temporary
/// file fails.
pub(crate) fn write_json_atomic(
    path: &Path,
    value: &impl Serialize,
    entity: &str,
) -> Result<(), OutputError> {
    let tmp = tmp_path(path);

    let file = std::fs::File::create(&tmp).map_err(|e| OutputError::io(&tmp, e))?;
    serialize_json_then_flush(file, value, entity, &tmp)?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Stream `value` as pretty JSON into a flushed `BufWriter` over `sink`.
///
/// Split from [`write_json_atomic`] so the serialize-then-explicit-flush step
/// can be driven with a failing writer in tests; the caller's rename runs only
/// on `Ok`, so a flush error can never install a target file.
fn serialize_json_then_flush<W: Write>(
    sink: W,
    value: &impl Serialize,
    entity: &str,
    tmp: &Path,
) -> Result<(), OutputError> {
    let mut writer = BufWriter::new(sink);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|e| OutputError::serialization(entity, format!("JSON serialization: {e}")))?;
    // Explicit flush before the caller's rename — see module doc.
    writer.flush().map_err(|e| OutputError::io(tmp, e))?;
    Ok(())
}

/// Write a `RecordBatch` to `path` as a Parquet file, atomically.
///
/// Uses the crate-wide frozen encoding — see [`super::parquet_config`] for
/// the parameters. The parent directory must already exist.
///
/// # Errors
///
/// Returns [`OutputError::SerializationError`] if the Parquet writer fails, or
/// [`OutputError::IoError`] if creating, flushing, or renaming the temporary
/// file fails.
pub(crate) fn write_parquet_atomic(path: &Path, batch: &RecordBatch) -> Result<(), OutputError> {
    let tmp = tmp_path(path);

    let file = std::fs::File::create(&tmp).map_err(|e| OutputError::io(&tmp, e))?;
    let buf = BufWriter::new(file);

    let mut writer = ArrowWriter::try_new(buf, batch.schema(), Some(WRITER_PROPERTIES.clone()))
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;
    writer
        .write(batch)
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;

    // Explicit flush to surface I/O errors before rename (into_inner's flush
    // behavior may vary; do not remove without verifying the parquet version).
    let mut buf = writer
        .into_inner()
        .map_err(|e| OutputError::serialization("parquet_writer", e.to_string()))?;
    buf.flush().map_err(|e| OutputError::io(&tmp, e))?;

    std::fs::rename(&tmp, path).map_err(|e| OutputError::io(path, e))?;
    Ok(())
}

/// Create `path`'s parent directory, then write `batch` as Parquet with the
/// crate-default writer configuration, atomically.
///
/// # Errors
///
/// See [`write_parquet_atomic`].
pub(crate) fn write_batch_atomic(path: &Path, batch: &RecordBatch) -> Result<(), OutputError> {
    ensure_parent_dir(path)?;
    write_parquet_atomic(path, batch)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::super::error::OutputError;
    use super::{
        serialize_json_then_flush, tmp_path, write_batch_atomic, write_bytes_atomic,
        write_json_atomic,
    };
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use serde::Serialize;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[derive(Serialize)]
    struct Mock {
        a: i32,
        b: String,
    }

    #[test]
    fn tmp_path_preserves_extension() {
        assert_eq!(
            tmp_path(Path::new("/x/foo.parquet")),
            PathBuf::from("/x/foo.parquet.tmp")
        );
        assert_eq!(
            tmp_path(Path::new("/x/foo.json")),
            PathBuf::from("/x/foo.json.tmp")
        );
        assert_eq!(tmp_path(Path::new("/x/foo")), PathBuf::from("/x/foo.tmp"));
    }

    #[test]
    fn write_json_atomic_produces_exact_bytes_and_removes_tmp() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("out.json");

        let value = Mock {
            a: 7,
            b: "hi".to_string(),
        };
        write_json_atomic(&path, &value, "mock").expect("write should succeed");

        let expected = serde_json::to_vec_pretty(&value).expect("serialize");
        let actual = std::fs::read(&path).expect("read");
        assert_eq!(actual, expected, "written bytes must match pretty JSON");

        assert!(
            !tmp_path(&path).exists(),
            "tmp file must be removed after rename"
        );
    }

    /// A writer that fails on every `flush`, to drive the drop-flush error path.
    struct FlushFails;

    impl io::Write for FlushFails {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failed"))
        }
    }

    #[test]
    fn flush_error_propagates_and_leaves_no_target_file() {
        let dir = TempDir::new().expect("temp dir");
        let target = dir.path().join("never_installed.json");
        let tmp = tmp_path(&target);

        let value = Mock {
            a: 1,
            b: "x".to_string(),
        };

        let result = serialize_json_then_flush(FlushFails, &value, "mock", &tmp);

        assert!(
            matches!(result, Err(OutputError::IoError { .. })),
            "flush failure must surface as IoError, got: {result:?}"
        );
        assert!(
            !target.exists(),
            "no file may be installed at the target path on flush failure"
        );
    }

    #[test]
    fn write_bytes_atomic_round_trips_and_removes_tmp() {
        let dir = TempDir::new().expect("temp dir");
        let path = dir.path().join("blob.bin");

        let bytes = b"some payload bytes";
        write_bytes_atomic(&path, bytes).expect("write should succeed");

        assert_eq!(std::fs::read(&path).expect("read"), bytes);
        assert!(
            !tmp_path(&path).exists(),
            "tmp file must be removed after rename"
        );
    }

    #[test]
    fn write_batch_atomic_creates_missing_parent_and_round_trips() {
        let dir = TempDir::new().expect("temp dir");
        let target = dir.path().join("a").join("b").join("batch.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let record_batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![7, 9]))])
                .expect("batch");

        write_batch_atomic(&target, &record_batch).expect("write should succeed");

        assert!(target.exists(), "target file must exist");
        assert!(
            !tmp_path(&target).exists(),
            "tmp file must be removed after rename"
        );

        let read_back = crate::test_support::output::read_first_batch(&target);
        assert_eq!(read_back.num_rows(), 2, "must round-trip two rows");
        let values = read_back
            .column_by_name("value")
            .expect("value column")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32Array");
        assert_eq!(values.value(0), 7);
        assert_eq!(values.value(1), 9);
    }
}
