//! Committed-deck discovery shared by every committed-deck sweep across the
//! crate's integration tests and cobre-cli's non-root setup-rebuild parity
//! test.

use std::path::{Path, PathBuf};

/// A committed deck: a directory containing `config.json`, keyed by its path
/// relative to the repository root with `/` separators.
pub struct Deck {
    /// The deck's path relative to the repository root, `/`-separated.
    pub key: String,
    /// The deck's directory.
    pub dir: PathBuf,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[expect(
    clippy::panic,
    reason = "a deck path outside the repo root or a non-UTF8 path component is a broken checkout, not a runtime condition to recover from"
)]
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

#[expect(
    clippy::panic,
    reason = "an unreadable examples/deterministic or fixtures directory is a broken checkout, not a runtime condition to recover from"
)]
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
///
/// # Panics
///
/// Panics on a broken checkout: an unreadable scan directory, a deck path
/// outside the repository root, or a non-UTF8 path component.
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

/// Deck keys gated out of every committed-deck sweep (the permutation and
/// patch-ownership checks, the cut oracles, and the non-root setup-rebuild
/// parity test) unless the `slow-tests` feature is
/// enabled: a deck whose `fresh_setup_with` build exceeds 5 seconds in the
/// debug test profile. `examples/4ree` measures well under that threshold, so
/// no deck is currently gated.
pub const SLOW_DECKS: &[&str] = &[];
