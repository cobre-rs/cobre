# ticket-018: Generate and Calibrate VHS Recordings

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Build the cobre binary, run existing VHS tape files with calibrated Sleep durations against actual run times, and generate GIF/cast output files for the quickstart and training demos.

## Anticipated Scope

- **Files likely to be modified**:
  - `recordings/quickstart.tape` -- calibrate Sleep durations
  - `recordings/training.sh` -- verify works with rayon threads
  - `recordings/report.sh` -- verify output format
  - `recordings/output/` (new) -- generated GIF and cast files

- **Key decisions needed**:
  - Target GIF file size (affects quality settings and duration)
  - Whether to use VHS or asciinema for the primary README recording

- **Open questions**:
  - What is the actual wall-clock time for `cobre run examples/1dtoy/` with --threads 4?
  - Should recordings show the `--threads` flag in action?

## Dependencies

- **Blocked By**: Epic 03 (4ree example for multi-bus demo)
- **Blocks**: ticket-020-embed-recordings

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
