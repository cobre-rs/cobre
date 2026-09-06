#!/usr/bin/env bash
# Pin the evaluation baseline and scaffold the register sections in BACKLOG.md.
# Idempotent: a second run over an already scaffolded register exits 0 with no diff.
#
# usage: tools/pin-baseline.sh <full-40-hex-sha> [pin-date ISO-8601, default today]
# exit 2: <sha> is not HEAD, nor an ancestor of HEAD with the evaluated surfaces
#         (crates docs scripts .github schemas Cargo.* examples tests) diff-free up to HEAD;
# exit 3: tracked modifications present outside plans/architecture-debt-audit; exit 4: register pinned to another baseline.
set -euo pipefail

SHA="${1:?usage: pin-baseline.sh <full-sha> [pin-date]}"
PIN_DATE="${2:-$(date -I)}"
BACKLOG="$(git rev-parse --show-toplevel)/plans/architecture-debt-audit/BACKLOG.md"
SHORT="${SHA:0:8}"

[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "not a full 40-hex sha: $SHA" >&2; exit 1; }
HEAD_SHA=$(git rev-parse HEAD)
EVALUATED=(crates docs scripts .github schemas Cargo.toml Cargo.lock examples tests)
if [[ "$HEAD_SHA" != "$SHA" ]]; then
  git merge-base --is-ancestor "$SHA" HEAD \
    || { echo "baseline $SHA is not an ancestor of HEAD $HEAD_SHA" >&2; exit 2; }
  if git diff --quiet "$SHA" HEAD -- "${EVALUATED[@]}"; then :; else
    echo "evaluated surfaces differ between baseline $SHA and HEAD $HEAD_SHA" >&2; exit 2
  fi
fi
if git status --porcelain --untracked-files=no -- . ':!plans/architecture-debt-audit' | grep -q .; then
  echo 'tracked modifications present outside plans/architecture-debt-audit' >&2; exit 3
fi
git merge-base --is-ancestor v0.15.0 "$SHA" || { echo 'baseline predates v0.15.0' >&2; exit 1; }

header_block() {
  cat <<HDR

Baseline: ${SHA} (pinned ${PIN_DATE})
Ledger: this file is the sole home of finding IDs (\`CD-\` / \`PD-\` / \`OD-\` / \`TD-\`).
Workers return JSON only; the main session is the sole writer. The 2026-09 quality
evaluation writes exactly two tracked surfaces: the ID-free mirror
\`docs/design/reserved-seams-and-deferred-debt.md\` and this directory
(\`plans/architecture-debt-audit/\`, tracked on the evaluation branch so its work
sessions carry git evidence); no crate, docs, script, CI or schema file is touched
(amends the Status bullet above).
Perf calibration bound: TBD s (median of 3 timed runs after one warm-up, layout \`4t\`,
deck \`~/git/cobre-bridge/example/cobre_reduzido_2\`; filled by the calibration run).
Only that deck at 4 workers is sanctioned: \`--threads 4\` (\`4t\`) or
\`mpiexec -n 2 … --threads 2\` (\`2x2\`). A run past 3x this bound is killed and its
claim tagged \`UNMEASURED\`; reasons are \`timeout-3x\`, \`unexercised-path\`,
\`mpi-unavailable\`. Perf fix-shapes stay byte-neutral.

## Milestones

Strict precedence: \`0a\` < \`0b\` < \`1\`. \`neutral\` advances no phase and is ordered by
dependency only. Sequencing principle (pull, don't push; a second consumer proves a seam
before the data model breaks): plans/generalizing/beyond-sddp-generalization.md V.0.

| Milestone | Meaning | Precedes | Source |
| --- | --- | --- | --- |
| \`0a\` | Engine seam; \`study\` config block + admission gate; shared output orchestration in cobre-io; rank-0-executes MPI; byte-identical for SDDP | \`0b\` | beyond-sddp-generalization.md V.1 (split resolved by D12) |
| \`0b\` | Carve \`cobre-model\` from the engine-neutral part of cobre-sddp \`lp/\` | \`1\` | beyond-sddp-generalization.md V.1 / IV.2 (split resolved by D12) |
| \`1\` | Purify the data model: stochastic off System/Stage, training_event out of cobre-core, StageTemplate shed, case v2 + bit-for-bit shim | — | beyond-sddp-generalization.md V.2 |
| \`neutral\` | Advances no phase | — | — |

Triggers: \`gnl-import\` — the GNL anticipated-coupling import. Wave 3 (boundary frame)
MUST land before it (Execution-waves table, Wave 3 row, 2026-08-22 section). SATISFIED
2026-08-22 by the Wave-3 execution (\`44e72b76\`, continuation \`9db2cbdf\`); the roadmap
must not re-open this gate.
HDR
}

if ! grep -qE "^Baseline: ${SHA} " "$BACKLOG"; then
  if grep -qE '^Baseline: [0-9a-f]{40} ' "$BACKLOG"; then
    echo "BACKLOG.md is pinned to a different baseline" >&2; exit 4
  fi
  STATUS_LINE=$(grep -nE '^- \*\*Status\*\*: read-only investigation' "$BACKLOG" | head -1 | cut -d: -f1)
  [[ -n "$STATUS_LINE" ]] || { echo 'Status bullet not found' >&2; exit 1; }
  { head -n "$STATUS_LINE" "$BACKLOG"; header_block; tail -n +"$((STATUS_LINE + 1))" "$BACKLOG"; } > "$BACKLOG.tmp"
  mv "$BACKLOG.tmp" "$BACKLOG"
fi

SECTIONS=(core-io stochastic solver-comm sddp cli-python build-ci test-corpus
  generalization-alignment performance-sweep reconciliation unified-roadmap)
for name in "${SECTIONS[@]}"; do
  heading="## ★ QUALITY EVALUATION (2026-09, baseline ${SHORT}) — ${name}"
  grep -qF -- "$heading" "$BACKLOG" && continue
  printf '\n%s\n\n_(no entries yet)_\n' "$heading" >> "$BACKLOG"
done
