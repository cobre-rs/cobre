//! Build script for cobre-cli.
//!
//! Copies the 1dtoy template files from `templates/1dtoy/` into
//! `OUT_DIR/templates/1dtoy/` so that `include_bytes!` in `templates.rs`
//! resolves to paths inside the build directory.
//!
//! The canonical source for these files is `examples/1dtoy/` at the workspace
//! root. The copies in `templates/1dtoy/` must be kept in sync.

use std::env;
use std::fs;
use std::path::Path;

/// Template files to embed, relative to `templates/1dtoy/`.
const TEMPLATE_FILES: &[&str] = &[
    "config.json",
    "initial_conditions.json",
    "penalties.json",
    "stages.json",
    "system/buses.json",
    "system/hydros.json",
    "system/lines.json",
    "system/thermals.json",
    "scenarios/inflow_seasonal_stats.parquet",
    "scenarios/load_seasonal_stats.parquet",
];

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR must be set");

    let src_root = Path::new(&manifest_dir).join("templates/1dtoy");
    let dst_root = Path::new(&out_dir).join("templates/1dtoy");

    for rel in TEMPLATE_FILES {
        let src = src_root.join(rel);
        let dst = dst_root.join(rel);

        println!("cargo:rerun-if-changed={}", src.display());

        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                panic!("failed to create directory {}: {e}", parent.display());
            });
        }

        fs::copy(&src, &dst).unwrap_or_else(|e| {
            panic!("failed to copy {} -> {}: {e}", src.display(), dst.display());
        });
    }
}
