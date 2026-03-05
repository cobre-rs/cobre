# Implementation Guardian Memory

## Stop Hook Requirement

The stop hook requires an **explicit APPROVED or REJECTED verdict** in every response before finishing — even in LEARNING EXTRACTION mode.

For LEARNING EXTRACTION, append this verdict block after all learnings output:

```
---

## Learning Extraction Verdict

**Mode**: LEARNING EXTRACTION (not DoD verification)
**Result**: COMPLETE

- learnings.md written to: [path]
- Accumulated summary written to: [path]
- State file updated: learning_extracted = true, phase = completed
```

This satisfies the stop hook's requirement for an explicit terminal verdict without falsely running DoD criteria checks against a learning extraction task.

## Mode Disambiguation

When the dispatch contains "LEARNING EXTRACTION", the task is extraction-only — no per-criterion PASS/FAIL table is produced. The explicit verdict block above replaces the DoD report.

When the dispatch is standard DoD verification, the full `# DoD Verification Report` with APPROVED/REJECTED is required.
