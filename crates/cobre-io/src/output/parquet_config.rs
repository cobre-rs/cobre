//! Parquet writer configuration for output files.
//!
//! [`ParquetWriterConfig`] carries the compression, row group size, and
//! dictionary encoding settings used by all Parquet output writers in this
//! crate. The defaults match the recommendations in the binary-formats spec
//! (§5): Zstd level 3, 100 000-row groups, dictionary encoding enabled.

use parquet::basic::{Compression, ZstdLevel};

/// Configuration for Parquet output writers.
///
/// Holds the three Parquet writer knobs relevant to output files: compression
/// codec, row group size, and whether to enable dictionary encoding for
/// categorical columns.
///
/// Construct via [`Default`] or by filling fields directly.
///
/// # Examples
///
/// ```
/// use cobre_io::ParquetWriterConfig;
/// use parquet::basic::Compression;
///
/// let cfg = ParquetWriterConfig::default();
/// assert_eq!(cfg.row_group_size, 100_000);
/// assert!(cfg.dictionary_encoding);
/// // Compression is Zstd level 3 — variant matching shown in tests.
/// ```
#[derive(Debug, Clone)]
pub struct ParquetWriterConfig {
    /// Parquet compression codec applied to all column chunks.
    ///
    /// Default: `Compression::ZSTD(ZstdLevel::try_new(3))` — a good balance
    /// between compression ratio and write-time CPU cost.
    pub compression: Compression,

    /// Maximum number of rows per Parquet row group.
    ///
    /// Default: `100_000`. Larger row groups improve compression and columnar
    /// scan throughput; smaller groups reduce peak write-time memory. 100 000
    /// rows is the recommended value from the binary-formats spec (§5).
    pub row_group_size: usize,

    /// Whether to enable dictionary encoding for categorical columns.
    ///
    /// When `true`, the writer uses dictionary pages for columns that contain
    /// repeated values (entity IDs, stage IDs, operative-state codes). This
    /// reduces file size significantly for such columns.
    ///
    /// Default: `true`.
    pub dictionary_encoding: bool,
}

impl Default for ParquetWriterConfig {
    fn default() -> Self {
        #[allow(clippy::expect_used)]
        let zstd_level = ZstdLevel::try_new(3).expect("ZstdLevel 3 is always valid");
        Self {
            compression: Compression::ZSTD(zstd_level),
            row_group_size: 100_000,
            dictionary_encoding: true,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use parquet::basic::Compression;

    #[test]
    fn parquet_writer_config_default_values() {
        let cfg = ParquetWriterConfig::default();
        assert_eq!(
            cfg.row_group_size, 100_000,
            "default row_group_size must be 100_000"
        );
        assert!(
            cfg.dictionary_encoding,
            "default dictionary_encoding must be true"
        );
        assert!(
            matches!(cfg.compression, Compression::ZSTD(_)),
            "default compression must be ZSTD, got {:?}",
            cfg.compression
        );
    }

    #[test]
    fn parquet_writer_config_zstd_level_is_three() {
        let cfg = ParquetWriterConfig::default();
        if let Compression::ZSTD(level) = cfg.compression {
            // ZstdLevel encodes as "ZSTD(ZstdLevel(<n>))" in Debug
            let debug = format!("{level:?}");
            assert!(debug.contains('3'), "ZSTD level must be 3, got: {debug}");
        } else {
            panic!("expected Compression::ZSTD, got {:?}", cfg.compression);
        }
    }

    #[test]
    fn parquet_writer_config_clone_is_independent() {
        let cfg = ParquetWriterConfig::default();
        let mut cloned = cfg.clone();
        cloned.row_group_size = 50_000;
        assert_eq!(
            cfg.row_group_size, 100_000,
            "original must be unchanged after clone mutation"
        );
        assert_eq!(cloned.row_group_size, 50_000);
    }

    #[test]
    fn parquet_writer_config_debug_non_empty() {
        let cfg = ParquetWriterConfig::default();
        assert!(!format!("{cfg:?}").is_empty());
    }
}
