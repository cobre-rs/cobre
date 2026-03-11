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

    // Always build HiGHS in Release mode regardless of the Rust profile.
    // HiGHS is a solver library — an unoptimized build is ~10x slower and
    // would produce misleading results even during development.
    let mut cmake_cfg = cmake::Config::new(&highs_src);
    cmake_cfg
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("HIGHS_NO_DEFAULT_THREADS", "ON")
        .define("BUILD_TESTING", "OFF")
        .define("BUILD_EXAMPLES", "OFF");
    // Disable ZLIB on Windows to avoid needing a separate zlib installation.
    // We don't read compressed MPS files, so this feature is not needed.
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        cmake_cfg.define("ZLIB", "OFF");
    }
    let highs_dst = cmake_cfg.build();

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
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Link the C++ standard library. MSVC links it automatically via the
    // /MT or /MD runtime flags — no explicit library needed.
    match (target_os.as_str(), target_env.as_str()) {
        ("macos", _) => println!("cargo:rustc-link-lib=c++"),
        (_, "msvc") => {} // MSVC links C++ runtime automatically
        _ => println!("cargo:rustc-link-lib=stdc++"),
    }

    // HMpsFF.cpp uses zstr.hpp for compressed MPS file I/O, which requires zlib.
    // Disabled on MSVC (ZLIB=OFF in cmake) to avoid needing a Windows zlib build.
    if target_env != "msvc" {
        println!("cargo:rustc-link-lib=z");
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
    // Suppress warning from HiGHS header: `Highs_compilationDate` is
    // declared static in highs_c_api.h but defined only in the .cpp file.
    // Use MSVC-style flag on Windows, GCC/Clang-style elsewhere.
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        build.flag("/wd4505");
    } else {
        build.flag("-Wno-unused-function");
    }
    build.compile("highs_wrapper");
}
