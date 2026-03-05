//! Build script for cobre-solver.
//!
//! This script:
//! 1. Checks that the `HiGHS` git submodule is initialized at `vendor/HiGHS/`
//! 2. Builds `HiGHS` from source using the `cmake` crate (static library)
//! 3. Compiles the thin C wrapper (`csrc/highs_wrapper.c`) via `cc`
//! 4. Links the built `HiGHS` static library and the C++ standard library

// Build scripts routinely use expect/panic for unrecoverable configuration
// errors. Allow these lints here since there is no caller to propagate errors to.
#![allow(clippy::expect_used, clippy::panic, clippy::manual_assert)]

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=csrc/highs_wrapper.c");
    println!("cargo:rerun-if-changed=csrc/highs_wrapper.h");

    if env::var("CARGO_FEATURE_HIGHS").is_err() {
        return;
    }

    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let highs_src = manifest_dir.join("../../vendor/HiGHS");

    if !highs_src.join("CMakeLists.txt").exists() {
        panic!(
            "HiGHS source not found at vendor/HiGHS/. Run: git submodule update --init --recursive"
        );
    }

    eprintln!("cobre-solver: building HiGHS from {}", highs_src.display());

    let highs_dst = cmake::Config::new(&highs_src)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("HIGHS_NO_DEFAULT_THREADS", "ON")
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .build();

    eprintln!(
        "cobre-solver: HiGHS cmake output at {}",
        highs_dst.display()
    );

    println!(
        "cargo:rustc-link-search=native={}",
        highs_dst.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        highs_dst.join("lib64").display()
    );

    println!("cargo:rustc-link-lib=static=highs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // HMpsFF.cpp uses zstr.hpp for compressed MPS file I/O, which requires zlib.
    println!("cargo:rustc-link-lib=z");

    let highs_include = highs_dst.join("include");
    let highs_include_highs = highs_dst.join("include/highs");

    eprintln!(
        "cobre-solver: compiling C wrapper with include paths: {}, {}",
        highs_include.display(),
        highs_include_highs.display()
    );

    cc::Build::new()
        .file("csrc/highs_wrapper.c")
        .include("csrc")
        .include(&highs_include)
        .include(&highs_include_highs)
        .warnings(true)
        .extra_warnings(true)
        .compile("highs_wrapper");
}
