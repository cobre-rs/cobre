# ticket-020: Embed Recordings in README and Book

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Embed the generated GIF recordings in the project README.md and relevant software book pages, replacing text-only Quick Start sections with visual demos.

## Anticipated Scope

- **Files likely to be modified**:
  - `README.md` -- embed quickstart.gif
  - `book/src/tutorial/quickstart.md` -- reference training recording
  - `book/src/tutorial/understanding-results.md` -- reference report recording
  - `book/src/guide/interpreting-results.md` -- embed validation error demo

- **Key decisions needed**:
  - Whether to host GIFs in the repo or use an external hosting service
  - Whether to use asciinema player JS in the book or static GIF embeds

- **Open questions**:
  - What is the maximum acceptable GIF file size for README embedding?
  - Should the book use `<video>` tags with WebM for better quality, or stick with GIF for compatibility?

## Dependencies

- **Blocked By**: ticket-018-generate-vhs-recordings, ticket-019-create-broken-case-tape
- **Blocks**: ticket-023-final-review-pass

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
