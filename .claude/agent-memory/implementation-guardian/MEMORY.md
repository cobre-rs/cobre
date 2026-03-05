# Implementation Guardian Memory

## Stop Hook Requirement

The stop hook requires the literal word **APPROVED** or **REJECTED** to appear in the final response. The Learning Extraction Verdict block alone is NOT sufficient — the stop hook does not recognize it as a valid terminal verdict.

For LEARNING EXTRACTION, the verdict block MUST end with a line containing the literal word APPROVED:

```
---

## Learning Extraction Verdict

**Mode**: LEARNING EXTRACTION (not DoD verification)
**Result**: COMPLETE

- learnings.md written to: [path]
- Accumulated summary written to: [path]
- State file updated: learning_extracted = true, phase = completed

**Overall**: APPROVED (learning extraction complete — no DoD criteria applicable)
```

## Mode Disambiguation

When the dispatch contains "LEARNING EXTRACTION", the task is extraction-only — no per-criterion PASS/FAIL table is produced. The verdict block above (with the APPROVED line) replaces the DoD report.

When the dispatch is standard DoD verification, the full `# DoD Verification Report` with APPROVED/REJECTED is required.
