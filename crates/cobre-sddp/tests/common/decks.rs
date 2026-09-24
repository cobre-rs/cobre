//! Committed-deck discovery for the template snapshot manifest
//! (`tests/template_snapshot.rs`) and its downstream consumers (the
//! permutation-invariance check, the patch-ownership sweep, the objective
//! agreement baseline).

use std::path::{Path, PathBuf};

/// A committed deck: a directory containing `config.json`, keyed by its path
/// relative to the repository root with `/` separators.
pub struct Deck {
    pub key: String,
    pub dir: PathBuf,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn repo_key(root: &Path, dir: &Path) -> String {
    dir.strip_prefix(root)
        .unwrap_or_else(|e| {
            panic!(
                "{} is not under repo root {}: {e}",
                dir.display(),
                root.display()
            )
        })
        .components()
        .map(|c| {
            c.as_os_str()
                .to_str()
                .unwrap_or_else(|| panic!("non-UTF8 path component in {}", dir.display()))
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn decks_with_config_under(root: &Path, scan_dir: &Path) -> Vec<Deck> {
    std::fs::read_dir(scan_dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", scan_dir.display()))
        .map(|entry| {
            entry.unwrap_or_else(|e| panic!("dir entry under {}: {e}", scan_dir.display()))
        })
        .map(|entry| entry.path())
        .filter(|path| path.join("config.json").is_file())
        .map(|dir| Deck {
            key: repo_key(root, &dir),
            dir,
        })
        .collect()
}

/// Every committed deck (a directory is a deck when it contains
/// `config.json`): directories directly under `examples/deterministic/` and
/// `crates/cobre-sddp/tests/fixtures/`, plus `examples/1dtoy` and
/// `examples/4ree`, sorted by `key`.
#[must_use]
pub fn committed_decks() -> Vec<Deck> {
    let root = repo_root();

    let mut decks = decks_with_config_under(&root, &root.join("examples/deterministic"));
    decks.extend(decks_with_config_under(
        &root,
        &root.join("crates/cobre-sddp/tests/fixtures"),
    ));
    for extra in ["examples/1dtoy", "examples/4ree"] {
        decks.push(Deck {
            key: extra.to_string(),
            dir: root.join(extra),
        });
    }

    decks.sort_by(|a, b| a.key.cmp(&b.key));
    decks
}

/// Deck keys whose `template_snapshot_matches_manifest` comparison runs only
/// under the `slow-tests` feature: a deck whose `fresh_setup_with` build
/// exceeds 5 seconds in the debug test profile. `examples/4ree` measures well
/// under that threshold, so no deck is currently gated.
pub const SLOW_DECKS: &[&str] = &[];
