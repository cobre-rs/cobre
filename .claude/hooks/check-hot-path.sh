#!/usr/bin/env bash
# PostToolUse hook: when a hot-path file is edited, check for too_many_arguments.
# If found, inject a reminder into Claude's context (non-blocking).
set -euo pipefail

INPUT=$(cat)

# Extract file_path from JSON without jq dependency.
# PostToolUse input contains tool_input.file_path (Write) or tool_input.path (Edit/MultiEdit).
FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    ti = d.get('tool_input', {})
    print(ti.get('file_path') or ti.get('path') or '')
except Exception:
    print('')
" 2>/dev/null || true)

if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Only check hot-path files
HOT_PATH_FILES="forward.rs backward.rs training.rs pipeline.rs lower_bound.rs"
BASENAME=$(basename "$FILE_PATH")

IS_HOT_PATH=false
for hp in $HOT_PATH_FILES; do
    if [ "$BASENAME" = "$hp" ]; then
        IS_HOT_PATH=true
        break
    fi
done

if [ "$IS_HOT_PATH" = false ]; then
    exit 0
fi

# Find #[cfg(test)] line to limit search to production code
TEST_START=$(grep -n '#\[cfg(test)\]' "$FILE_PATH" 2>/dev/null | head -1 | cut -d: -f1 || echo "999999")
if [ -z "$TEST_START" ]; then
    TEST_START=999999
fi

# Check for too_many_arguments in production code only
MATCHES=$(head -n "$((TEST_START - 1))" "$FILE_PATH" 2>/dev/null | grep -c 'allow.*clippy::too_many_arguments' || true)

if [ "$MATCHES" -gt 0 ]; then
    echo "WARNING: $BASENAME contains $MATCHES #[allow(clippy::too_many_arguments)] in production code. Absorb parameters into context structs (StageContext, TrainingContext, BackwardPassSpec, LbEvalSpec, SolverWorkspace) instead. See .claude/architecture-rules.md for the decision tree."
fi

exit 0