//! # cobre-io
//!
//! File parsers and serializers for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate provides interoperability with existing power system data formats:
//!
//! - **PSS/E**: `.raw` and `.dyr` files for steady-state and dynamic data.
//! - **CSV/Arrow/Parquet**: modern columnar formats for time series and results.
//! - **JSON/TOML**: human-readable study configuration.
//!
//! All parsers produce `cobre-core` types, enabling round-trip conversion
//! between formats.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the full roadmap.
