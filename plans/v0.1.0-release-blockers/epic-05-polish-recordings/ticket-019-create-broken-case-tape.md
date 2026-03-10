# ticket-019: Create Broken-Case Validation Tape

## Context

### Background

The `recordings/` directory already contains two VHS tape files (`quickstart.tape` and `validation.tape`) that demonstrate `cobre init`, `cobre run`, and `cobre validate` on a valid case. However, there is no recording that demonstrates the validation system detecting actual errors -- the current `validation.tape` only validates a correct case and shows a success message. A recording that shows `cobre validate` catching real errors with helpful diagnostics would be a compelling demonstration of the validation pipeline and error reporting.

### Relation to Epic

Epic 05 is about polish and recordings for the v0.1.0 release. This ticket creates one of the key demo artifacts: a VHS tape that produces a GIF showing the CLI detecting and reporting validation errors.

### Current State

- `recordings/quickstart.tape` exists and demos `init -> run -> report` on a valid 1dtoy case
- `recordings/validation.tape` exists and demos `init -> validate` on a valid 1dtoy case (shows success)
- `recordings/README.md` documents how to run VHS tapes and generate GIF output
- VHS tapes use consistent settings: FontSize 16, Width 100, Height 30, Theme "Catppuccin Mocha"
- The `cobre validate` command calls `cobre_io::load_case` and prints structured errors with red `error:` labels (see `crates/cobre-cli/src/commands/validate.rs`)
- The validation pipeline catches missing files (IoError), schema errors, cross-reference errors, dimensional errors, and semantic constraint violations

## Specification

### Requirements

1. Create a VHS tape file at `recordings/validation-error.tape` that demonstrates `cobre validate` detecting errors in a broken case directory
2. The tape must use `jq` to corrupt a valid 1dtoy case on the fly (no pre-built broken directory needed -- this keeps the demo self-contained and avoids committing broken test data)
3. Show at least two distinct error categories in the recording: one structural/schema error (e.g., remove a required field from hydros.json) and one semantic error (e.g., set negative max storage capacity)
4. The tape must use the same VHS settings as existing tapes (FontSize 16, Width 100, Height 30, Theme "Catppuccin Mocha")

### Inputs/Props

- The `examples/1dtoy/` directory provides the source case files to copy and corrupt
- `jq` is listed as a prerequisite in `recordings/README.md` and is available on the developer machine

### Outputs/Behavior

- `recordings/validation-error.tape` produces `recordings/validation-error.gif` when run with `vhs recordings/validation-error.tape`
- The GIF shows: (a) `cobre init --template 1dtoy demo/`, (b) corruption of one or more JSON files via `jq`, (c) `cobre validate demo/` printing error diagnostics with exit code 1

### Error Handling

- If `jq` is not installed, VHS will fail at the `Type` step that runs jq -- this is acceptable since jq is a documented prerequisite in `recordings/README.md`

## Acceptance Criteria

- [ ] Given the file `recordings/validation-error.tape` exists, when `cat recordings/validation-error.tape` is run, then the file contains `Set FontSize 16`, `Set Width 100`, `Set Height 30`, and `Set Theme "Catppuccin Mocha"`
- [ ] Given the file `recordings/validation-error.tape` exists, when `cat recordings/validation-error.tape` is run, then the file contains at least one `jq` command that modifies a JSON file from the 1dtoy case
- [ ] Given the file `recordings/validation-error.tape` exists, when `cat recordings/validation-error.tape` is run, then the file contains `cobre validate` invocation
- [ ] Given `recordings/README.md` exists, when reading the file, then it includes a section documenting how to generate `validation-error.gif` from the new tape

## Implementation Guide

### Suggested Approach

1. Create `recordings/validation-error.tape` following the pattern of the existing tapes
2. The tape should:
   - `cobre init --template 1dtoy demo/` to scaffold a valid case
   - Use `jq 'del(.hydros[0].max_storage)' demo/system/hydros.json > tmp && mv tmp demo/system/hydros.json` (or equivalent) to remove a required field
   - Run `cobre validate demo/` to show the error output
3. Calibrate `Sleep` durations: 2s after init, 1s after jq (fast), 5s after validate (to let user read the output), 2s final
4. Update `recordings/README.md` to add the new tape to the "VHS Recordings" section

### Key Files to Modify

- `recordings/validation-error.tape` (new file)
- `recordings/README.md` (add documentation for the new tape)

### Patterns to Follow

- Match the exact VHS settings from `recordings/quickstart.tape`: `Set Shell "bash"`, `Require cobre`, `Set FontSize 16`, `Set Width 100`, `Set Height 30`, `Set Theme "Catppuccin Mocha"`
- Follow the `Output <name>.gif` pattern

### Pitfalls to Avoid

- Do not create a committed broken case directory -- use jq to corrupt on the fly within the tape
- Do not modify the existing `validation.tape` -- it demos the success path and should remain as-is
- The exact jq expression depends on the actual JSON schema of `hydros.json` in `examples/1dtoy/` -- read the file to verify field names before writing the jq command

## Testing Requirements

### Unit Tests

Not applicable -- this is a VHS tape file, not Rust code.

### Integration Tests

Not applicable -- VHS tapes are run manually by the developer.

### E2E Tests (if applicable)

Manual verification: run `vhs recordings/validation-error.tape` and confirm the GIF shows the error output. This is a visual artifact, not an automated test.

## Dependencies

- **Blocked By**: None
- **Blocks**: ticket-023-final-review-pass.md

## Effort Estimate

**Points**: 1
**Confidence**: High
