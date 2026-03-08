#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::path::Path;
use std::process::Command;

#[test]
fn infrastructure_genericity_no_sddp_references() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).join("..").join("..");
    let src_path = Path::new(manifest_dir).join("src");

    let output = Command::new("grep")
        .args(["-riE", "sddp"])
        .arg(&src_path)
        .current_dir(&workspace_root)
        .output()
        .expect("infrastructure_genericity: failed to execute grep");

    assert_eq!(
        output.status.code(),
        Some(1),
        "grep found algorithm-specific references in cobre-io/src/:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
}
