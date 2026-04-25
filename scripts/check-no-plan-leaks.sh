#!/usr/bin/env bash
#
# check-no-plan-leaks.sh — Plan-structure leak gate.
#
# Scans shipped artefacts (production source, book, CHANGELOG) for
# plan-structure tokens that must not appear in user-facing
# content per CLAUDE.md hard rule:
#
#   "No plan-structure references in user-facing artifacts"
#
# Forbidden patterns:
#   Epic [0-9]+      — "Epic 06", "Epic 12", etc.
#   ticket-[0-9]+    — "ticket-001", "ticket-042", etc.
#   T0[0-9][0-9]     — "T002", "T007", "T015", etc.
#   \bsprint\b       — sprint planning vocabulary
#
# Scope: production source under crates/*/src/, book/, CHANGELOG.md,
#   and README.md.
#
# Excluded: plans/ (gitignored), .github/, target/, .git/, Cargo.lock,
#   and the script itself.
#
# Exit codes:
#   0 — No leaks found.
#   1 — Leaks found (details printed to stdout).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

readonly PATTERN='Epic [0-9]+|ticket-[0-9]+|T0[0-9][0-9]|\bsprint\b'

readonly SCAN_PATHS=(
    "${REPO_ROOT}/crates/cobre-core/src"
    "${REPO_ROOT}/crates/cobre-io/src"
    "${REPO_ROOT}/crates/cobre-solver/src"
    "${REPO_ROOT}/crates/cobre-comm/src"
    "${REPO_ROOT}/crates/cobre-stochastic/src"
    "${REPO_ROOT}/crates/cobre-sddp/src"
    "${REPO_ROOT}/crates/cobre-cli/src"
    "${REPO_ROOT}/crates/cobre-python/src"
    "${REPO_ROOT}/crates/cobre-mcp/src"
    "${REPO_ROOT}/crates/cobre-tui/src"
    "${REPO_ROOT}/book"
    "${REPO_ROOT}/CHANGELOG.md"
    "${REPO_ROOT}/README.md"
)

violations=$(grep -rnE "$PATTERN" "${SCAN_PATHS[@]}" 2>/dev/null \
    || true)

if [[ -n "$violations" ]]; then
    echo "FAIL: plan-structure leaks found in shipped artefacts."
    echo ""
    echo "$violations"
    echo ""
    echo "Per CLAUDE.md, plan-structure references must not appear"
    echo "in shipped source, the book, or the CHANGELOG. Rewrite"
    echo "in behavioural terms or move to plans/ (gitignored)."
    exit 1
fi

echo "OK: no plan-structure leaks found in shipped artefacts."
exit 0
