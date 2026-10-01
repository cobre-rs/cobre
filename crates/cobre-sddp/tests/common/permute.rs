//! Seeded declaration-order permutation primitive for the invariance-shuffle
//! axis (`tests/parity.rs`'s `parity_hash_<case>`/`shuffle_matrix_<case>`
//! tests and `tests/deterministic.rs`'s
//! `nonzero_stage_fpha_override_regression` prototype).
//!
//! [`permute_case`] copies a golden-case directory into a fresh [`TempDir`],
//! shuffling every JSON array classified below as shuffle-safe with a seeded
//! Fisher-Yates permutation and leaving every other byte in the tree
//! untouched. The classification is an explicit `(case-relative path,
//! top-level keys)` map — never auto-discovery — because a case directory
//! legitimately contains order-bearing arrays (an ordinal per-lag vector
//! nested inside a shuffled record) alongside the id-keyed registries this
//! axis permutes; walking every array
//! indiscriminately would silently corrupt the former. Any top-level JSON
//! array that is neither whitelisted for shuffle nor marked keep-ordered
//! panics ("unclassified array") so a future registry cannot silently escape
//! this axis. The gitignored `output/` subtree is skipped entirely.
//!
//! ## Coverage bound
//!
//! This primitive permutes JSON registry arrays only. Input parquet row order
//! (`fpha_hyperplanes.parquet`, seasonal-stats tables) is NOT exercised by
//! this axis.

use std::ffi::OsStr;
use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use tempfile::TempDir;

/// `(case-relative path, top-level keys)` entries whose JSON array order
/// carries no meaning: each resolves through an id-keyed lookup downstream —
/// `cobre_io::stages::convert_stages` folds `pre_study_stages` into `stages`
/// then `sort_by_key(|s| s.id)`; `cobre_io::initial_conditions::convert`
/// sorts every field by `hydro_id`/`thermal_id`; `System::hydros`/`thermals`
/// canonicalize by `(operational_start_date, id)`; `resolve_load_factors`
/// writes into a dense table keyed by `(bus_id, stage_id, block_id)`;
/// `cobre_io::constraints::generic::convert` sorts `constraints` by `gc.id`;
/// `cobre_io::system::pumping_stations::convert_pumping` sorts
/// `pumping_stations` by `s.id`; `cobre_io::post_study_stages::convert_stages`
/// sorts `stages` by `start_date` and `convert_thermal_bounds` sorts
/// `thermal_bounds` by `(thermal_id, post_study_stage_index)`.
const SHUFFLE_WHITELIST: &[(&str, &[&str])] = &[
    ("stages.json", &["stages", "pre_study_stages"]),
    (
        "initial_conditions.json",
        &[
            "storage",
            "filling_storage",
            "past_anticipated_commitments",
            "recent_observations",
            "past_defluences",
        ],
    ),
    ("system/buses.json", &["buses"]),
    ("system/hydros.json", &["hydros"]),
    ("system/thermals.json", &["thermals"]),
    ("system/lines.json", &["lines"]),
    (
        "system/hydro_production_models.json",
        &["production_models"],
    ),
    (
        "system/non_controllable_sources.json",
        &["non_controllable_sources"],
    ),
    ("system/energy_contracts.json", &["contracts"]),
    ("scenarios/load_factors.json", &["load_factors"]),
    ("constraints/generic_constraints.json", &["constraints"]),
    ("system/pumping_stations.json", &["pumping_stations"]),
    ("post_study_stages.json", &["stages", "thermal_bounds"]),
];

/// `(case-relative path, top-level keys)` entries whose JSON array order IS
/// semantically load-bearing and must never be shuffled by this axis. Empty
/// today — every registry this axis has encountered resolves through an
/// id-keyed lookup (see [`SHUFFLE_WHITELIST`]); add an entry here, with a
/// one-line justification, the day one does not.
const KEEP_ORDERED: &[(&str, &[&str])] = &[];

/// Copy `base_dir` into a fresh [`TempDir`], shuffling every
/// [`SHUFFLE_WHITELIST`]-classified array with an RNG seeded from `seed`.
///
/// # Panics
/// Panics on any I/O or JSON-parse failure, or if a `*.json` file contains a
/// top-level array key that is neither in [`SHUFFLE_WHITELIST`] nor
/// [`KEEP_ORDERED`] ("unclassified array").
#[must_use]
pub fn permute_case(base_dir: &Path, seed: u64) -> TempDir {
    let tmp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
    let mut rng = StdRng::seed_from_u64(seed);
    copy_dir_permuted(base_dir, base_dir, tmp.path(), &mut rng);
    tmp
}

/// `path`'s components relative to `case_root`, joined with `/` regardless of
/// host path separator, for matching against [`SHUFFLE_WHITELIST`]/[`KEEP_ORDERED`].
fn relative_key(case_root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(case_root).unwrap_or_else(|e| {
        panic!(
            "{} is not under case root {}: {e}",
            path.display(),
            case_root.display()
        )
    });
    rel.components()
        .map(|c| {
            c.as_os_str()
                .to_str()
                .unwrap_or_else(|| panic!("non-UTF8 path component in {}", path.display()))
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn classified_keys(
    map: &'static [(&'static str, &'static [&'static str])],
    rel_key: &str,
) -> &'static [&'static str] {
    map.iter()
        .find(|(path, _)| *path == rel_key)
        .map_or(&[][..], |&(_, keys)| keys)
}

fn copy_dir_permuted(case_root: &Path, src_dir: &Path, dst_dir: &Path, rng: &mut StdRng) {
    std::fs::create_dir_all(dst_dir)
        .unwrap_or_else(|e| panic!("create_dir_all {}: {e}", dst_dir.display()));

    // `read_dir` order is platform/filesystem-dependent; sorting first makes the
    // permutation a pure function of (seed, inputs), so a shuffle_matrix seed
    // reproduces the same failure on any machine.
    let mut entries: Vec<_> = std::fs::read_dir(src_dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", src_dir.display()))
        .map(|entry| entry.unwrap_or_else(|e| panic!("dir entry under {}: {e}", src_dir.display())))
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in entries {
        let src_path = entry.path();
        let dst_path = dst_dir.join(entry.file_name());
        let file_type = entry
            .file_type()
            .unwrap_or_else(|e| panic!("file_type {}: {e}", src_path.display()));

        if file_type.is_dir() {
            if relative_key(case_root, &src_path) == "output" {
                continue;
            }
            copy_dir_permuted(case_root, &src_path, &dst_path, rng);
            continue;
        }

        if src_path.extension().and_then(OsStr::to_str) == Some("json") {
            permute_json_file(case_root, &src_path, &dst_path, rng);
        } else {
            std::fs::copy(&src_path, &dst_path).unwrap_or_else(|e| {
                panic!("copy {} -> {}: {e}", src_path.display(), dst_path.display())
            });
        }
    }
}

/// Parse `src_path`, shuffle every classified array, and write the result to
/// `dst_path` — a byte-for-byte copy when nothing was shuffled (no classified
/// key present), or a re-serialization otherwise (matching the JSON-rewrite
/// pattern the axis already relies on for its shuffled files).
fn permute_json_file(case_root: &Path, src_path: &Path, dst_path: &Path, rng: &mut StdRng) {
    let rel_key = relative_key(case_root, src_path);

    let text = std::fs::read_to_string(src_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", src_path.display()));
    let mut value: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", src_path.display()));

    let shuffle_keys = classified_keys(SHUFFLE_WHITELIST, &rel_key);
    let keep_ordered_keys = classified_keys(KEEP_ORDERED, &rel_key);

    let obj = value
        .as_object_mut()
        .unwrap_or_else(|| panic!("{} must be a top-level JSON object", src_path.display()));

    let mut shuffled_any = false;
    for (key, val) in obj.iter_mut() {
        let Some(arr) = val.as_array_mut() else {
            continue;
        };
        if shuffle_keys.contains(&key.as_str()) {
            arr.shuffle(rng);
            shuffled_any = true;
        } else if !keep_ordered_keys.contains(&key.as_str()) {
            let file = src_path.display();
            panic!(
                "unclassified array: {file} top-level key {key:?} is a JSON array with no \
                 shuffle/keep-ordered classification in permute.rs's SHUFFLE_WHITELIST or \
                 KEEP_ORDERED — classify it explicitly before running the invariance-shuffle axis"
            );
        }
    }

    if shuffled_any {
        let rendered = serde_json::to_string_pretty(&value)
            .unwrap_or_else(|e| panic!("serialize {}: {e}", dst_path.display()));
        std::fs::write(dst_path, rendered)
            .unwrap_or_else(|e| panic!("write {}: {e}", dst_path.display()));
    } else {
        std::fs::copy(src_path, dst_path).unwrap_or_else(|e| {
            panic!("copy {} -> {}: {e}", src_path.display(), dst_path.display())
        });
    }
}
