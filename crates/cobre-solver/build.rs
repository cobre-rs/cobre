//! Build script for cobre-solver.
//!
//! This script:
//! 1. Checks that the `HiGHS` git submodule is initialized at `crates/cobre-solver/vendor/HiGHS/`
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
    let highs_src = manifest_dir.join("vendor/HiGHS");

    if !highs_src.join("CMakeLists.txt").exists() {
        panic!(
            "HiGHS source not found at crates/cobre-solver/vendor/HiGHS/. \
             Run: git submodule update --init --recursive"
        );
    }

    eprintln!("cobre-solver: building HiGHS from {}", highs_src.display());

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Always build HiGHS in Release mode regardless of the Rust profile.
    // HiGHS is a solver library — an unoptimized build is ~10x slower and
    // would produce misleading results even during development.
    let mut cmake_config = cmake::Config::new(&highs_src);
    cmake_config
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("HIGHS_NO_DEFAULT_THREADS", "ON")
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        // Disable zlib in HiGHS. HiGHS uses zlib only for reading compressed
        // .mps.gz/.lp.gz files via Highs_readModel(). Cobre constructs all LPs
        // programmatically via the C API and never uses file-based model I/O.
        // Disabling zlib eliminates a system dependency that causes
        // cross-compilation failures in the Python wheel CI.
        .define("CMAKE_DISABLE_FIND_PACKAGE_ZLIB", "ON");

    // On MSVC, use static CRT to avoid requiring vcruntime140.dll in the wheel.
    if target_env == "msvc" {
        cmake_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
        cmake_config.cflag("/MT");
        cmake_config.cxxflag("/MT");
    }

    let highs_dst = cmake_config.build();

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

    // MSVC cmake may place libraries in a configuration subdirectory.
    if target_env == "msvc" {
        println!(
            "cargo:rustc-link-search=native={}",
            highs_dst.join("lib/Release").display()
        );
    }

    println!("cargo:rustc-link-lib=static=highs");

    // Link the C++ standard library.
    // MSVC links it automatically — no explicit directive needed.
    if target_env != "msvc" {
        if target_os == "macos" {
            println!("cargo:rustc-link-lib=c++");
        } else {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }

    let highs_include = highs_dst.join("include");
    let highs_include_highs = highs_dst.join("include/highs");

    eprintln!(
        "cobre-solver: compiling C wrapper with include paths: {}, {}",
        highs_include.display(),
        highs_include_highs.display()
    );

    let mut build = cc::Build::new();
    build
        .file("csrc/highs_wrapper.c")
        .include("csrc")
        .include(&highs_include)
        .include(&highs_include_highs)
        .warnings(true)
        .extra_warnings(true);

    // GCC/Clang-specific warning suppression.
    if target_env != "msvc" {
        build.flag("-Wno-unused-function");
    }

    build.compile("highs_wrapper");
}
