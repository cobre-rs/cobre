#!/usr/bin/env bash
#
# check-infra-genericity.sh — Infrastructure crate genericity gate.
#
# Scans the five infrastructure crates for algorithm-specific vocabulary that
# must not appear in their production source code:
#
#   cobre-core, cobre-io, cobre-solver, cobre-stochastic, cobre-comm
#
# Patterns flagged (word-boundary anchored to avoid false positives on
# identifiers like "execution", "cuts_active", "cutting_edge"):
#
#   \bSddp\b | \bsddp\b | \bSDDP\b — SDDP algorithm references
#   \bBenders\b             — Benders decomposition references
#   \bCut\b | \bcut\b      — standalone "cut" / "Cut" (word boundary)
#   cut pool                — phrase: cut pool
#   cutting-plane           — phrase: cutting-plane
#   cutting\.plane          — phrase: cutting.plane
#   outer approximation     — phrase: outer approximation
#   CutSync\w*             — CutSync type names
#   CutSelection\w*        — CutSelection type names
#   MetadataCuts\b         — MetadataCuts type name
#
# Exclusions:
#
#   1. Files whose name contains "test" (e.g., integration test files, test
#      helper files named *_test.rs or test_*.rs).
#   2. Inline #[cfg(test)] blocks within production files: once a line matching
#      `#[cfg(test)]` is seen, all subsequent lines until end-of-file are
#      considered test scope and skipped. This handles the common Rust pattern
#      where the test module sits at the bottom of a production file.
#
# Known limitation: the #[cfg(test)] exclusion assumes the test module is a
# tail block. Rust files with mid-file test modules separated by production
# code after them would incorrectly skip that trailing production code. In
# practice, all cobre infra files follow the tail-block convention.
#
# Explicitly excluded files:
#
#   cobre-io/src/output/policy.rs — This file serialises the policy checkpoint
#     format (FlatBuffers schema SS3.1). The schema field names (cut_id,
#     cut_intercept, PolicyCutRecord, StageCutsPayload, etc.) are part of the
#     persisted binary format and cannot be renamed without a format-version
#     bump. Renaming is tracked as tech debt in docs/ROADMAP.md under
#     "Policy format genericity". Until that work is done, policy.rs is
#     excluded from this gate.
#
# Exit codes:
#   0 — No violations found.
#   1 — One or more violations found (details printed to stdout).
#
# Implementation note on \b word boundaries:
#   This script uses `grep -E` for all pattern matching. POSIX/GNU awk silently
#   treats `\b` as a backspace character when the pattern is supplied via
#   `-v pat=`, so awk-based matching was unreliable. grep -E correctly supports
#   \b word-boundary anchors. awk is retained only for structural work
#   (truncating files at the #[cfg(test)] boundary and emitting FILE:LINE:
#   prefixes); all regex matching goes through grep -E.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCAN_DIRS=(
    "${REPO_ROOT}/crates/cobre-core/src"
    "${REPO_ROOT}/crates/cobre-io/src"
    "${REPO_ROOT}/crates/cobre-solver/src"
    "${REPO_ROOT}/crates/cobre-stochastic/src"
    "${REPO_ROOT}/crates/cobre-comm/src"
)

# Files to explicitly exclude from the scan (relative to REPO_ROOT).
EXCLUDED_FILES=(
    "crates/cobre-io/src/output/policy.rs"
)

# Build the grep ERE pattern.
# \b is a word boundary — supported by grep -E but NOT by awk's ERE when
# supplied via -v (awk treats \b as backspace in that context).
PATTERN='\bSddp\b|\bsddp\b|\bSDDP\b|\bBenders\b|\bCut\b|\bcut\b|cut pool|cutting-plane|cutting[.]plane|outer approximation|CutSync[[:alnum:]_]*|CutSelection[[:alnum:]_]*|\bMetadataCuts\b'

violations=0
violation_lines=()

for dir in "${SCAN_DIRS[@]}"; do
    # Find all .rs files, excluding files whose path contains "test".
    while IFS= read -r -d '' file; do
        # Skip explicitly excluded files.
        rel="${file#"${REPO_ROOT}/"}"
        skip=0
        for excl in "${EXCLUDED_FILES[@]}"; do
            if [[ "$rel" == "$excl" ]]; then
                skip=1
                break
            fi
        done
        [[ $skip -eq 1 ]] && continue

        # Skip files whose basename contains "test".
        basename="${file##*/}"
        if [[ "$basename" == *test* ]]; then
            continue
        fi

        # Two-stage pipeline:
        #
        # Stage 1 (awk): Structural pre-filter — stop emitting lines once a
        #   bare `#[cfg(test)]` line is encountered (tail test-module exclusion).
        #   For each production line, emit "FILE:LINENO:CONTENT" so that grep
        #   output retains location information.
        #
        # Stage 2 (grep -E): Pattern matching — apply the ERE pattern with
        #   correct \b word-boundary support. grep -E reads from stdin and
        #   prints only lines that match.
        #
        # NOTE: awk does NOT perform any regex matching here. It only handles
        # the structural concern (line truncation + prefix). This avoids the
        # awk \b backspace bug entirely.
        matches=$(awk -v f="$file" '
            /^#\[cfg\(test\)\]/ { exit }
            { printf "%s:%d:%s\n", f, NR, $0 }
        ' "$file" | grep -E "$PATTERN") || true

        if [[ -n "$matches" ]]; then
            violation_lines+=("$matches")
            violations=$(( violations + 1 ))
        fi
    done < <(find "$dir" -name "*.rs" -print0)
done

if [[ ${#violation_lines[@]} -gt 0 ]]; then
    echo "FAIL: infra-genericity gate found algorithm-specific vocabulary in production code."
    echo ""
    for block in "${violation_lines[@]}"; do
        echo "$block"
    done
    echo ""
    echo "Rename or move the flagged identifiers to cobre-sddp (or another"
    echo "algorithm-specific crate) before committing."
    exit 1
fi

echo "OK: no algorithm-specific vocabulary found in infrastructure crates."
exit 0
