# ADR-001: Documentation Split

**Status:** Accepted
**Date:** 2026-03-02

## Context

The Cobre documentation was developed in a separate `cobre-docs` repository during the
specification phase, before any implementation code existed. As implementation begins,
software documentation (installation, guides, API overviews) needs to live next to the
code for atomic updates, while the specification corpus and mathematical reference
material have a different editorial lifecycle.

## Decision

Split documentation into three locations:

- **`cobre/book/`** -- Software book (mdBook): installation, user guides, crate
  overviews, contributing guide. Updated during/after implementation.
- **`cobre-docs/`** -- Specification corpus and methodology reference: 74 spec files,
  algorithm theory, math formulations, format specifications. Updated before
  implementation (specs define contracts).
- **`cobre/docs/adr/`** -- Architecture Decision Records: implementation-level decisions
  not covered by the spec corpus.

## Consequences

- Code and software docs update in the same commit
- Specs can be reviewed and cited independently of code changes
- Two CI pipelines for documentation deployment (GitHub Pages for each repo)
- Cross-linking between sites required (absolute URLs)
- The spec corpus in cobre-docs remains the source of truth for design; the code in
  cobre is the source of truth for what has been built
