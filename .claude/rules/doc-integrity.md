---
paths:
  - "*.md"
  - "CONTRIBUTING.md"
  - "CHANGELOG.md"
  - ".claude/*.md"
  - ".claude/rules/*.md"
---

# Cobre Prose Documentation Integrity Rules

Governs every Markdown file that serves as a user-facing or agent-facing artifact:
`CLAUDE.md`, `.claude/rules/*`, `CONTRIBUTING.md`, `CHANGELOG.md`,
and root-level `*.md` files. The rule auto-loads on the matching globs.

For the code-comment counterpart (Four Voices, Earned-Comment Test, directives
D1–D5 / N1–N6), see `.claude/rules/comments.md`. That file governs `.rs` files;
this one governs prose.

---

## 1. Scope matrix — how far each concern transfers, per doc

Transfer strength tracks **(machine reader) × (low teaching mandate)**.

| Doc                                          | Reader                                                 | Transfer profile                                                                                                                                                 |
| -------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CLAUDE.md**                                | an agent that acts on assertions, no human in the loop | **near-total**: Durability/anti-drift dominates; Plan-leakage (it defines the rule); Provenance (clean). Voice machinery N/A.                                    |
| **`.claude/rules/*`, architecture-rules.md** | agent + maintainer                                     | **near-total**: machine-consumed contracts; same as code-adjacent docs.                                                                                          |
| **CHANGELOG / THIRD_PARTY_NOTICES / CoC**    | release / license / community                          | **strong** for Plan-leakage; Provenance **inverted** (CHANGELOG _is_ history — the carve-out); Durability adapted (NOTICES is a maintained single-owner mirror). |
| **CONTRIBUTING.md**                          | external contributors                                  | **adapted**: every path/command/flag must resolve; Self-executable-instruction applies; teaching repetition is signal, not bloat.                                |
| **README.md**                                | newcomers / evaluators                                 | **adapted**: Plan-leakage full; Durability in spirit; narrative/teaching voice is the job, **promotional/hype voice is not (§5)**; Length N/A.                   |

### Which of the six concerns transfer

- **Transfer with force (and bite harder than in code):**
  - **Durability / anti-drift** — re-expressed for prose (see §2, the key adaptation).
  - **Plan-leakage** — already a CLAUDE.md hard rule for user-facing artifacts;
    `comments.md` N4 extends the pattern.
  - **Contract-mirroring** — "a drifted mirror is a lie"; prose mirrors many
    code-owned facts and is the densest drift source.
- **Transfers adapted:**
  - **Provenance** — present-tense rule holds, **except** the CHANGELOG, which is
    history by design (the standing carve-out). Forward-looking "planned" status
    prose is a legitimate register but a future stale-snapshot vector.
- **Does NOT transfer (out of scope for prose):**
  - **Voice machinery** (the Four Voices, Intra-Comment Surgery, the contract/
    rationale-seam distinction) — category error; the prose reader cannot see a
    code seam. The teaching/narrative register (explaining, second-person
    instructions, worked examples) is the _correct_ register for README/CoC/
    CONTRIBUTING. These structural concerns are governed by
    `.claude/rules/comments.md` for `.rs` files and do not apply to Markdown prose.
  - **Caveat — this does NOT exempt voice.** "Narrative is the correct register"
    is not a licence for **promotional / marketing voice**. Hype is a distinct,
    in-scope concern governed by §5 below; it transfers with force to every
    user-facing doc, README included.
  - **Strict Truth-density / Length** — superseded by the teaching mandate. Prose
    docs legitimately repeat, explain, and elaborate; the Length heuristic from
    `comments.md` does not transfer here.

---

## 2. The one adaptation that does the heavy lifting

> The code rule "**reference by stable SYMBOL, never `file.rs:NNN`**" relies on
> the compiler validating the symbol. Prose cannot link a symbol. So it becomes:
>
> **Name the external contract (filename / flag / config field / path), but
> never freeze a COUNT, VERSION, ENUMERATION, or run-snapshot NUMBER. Pin any
> literal that must appear with a _guard_, not by hand — otherwise state the
> invariant instead of the number.**

Corollaries:

- `env!(CARGO_PKG_VERSION)` is **not** the prose answer — rendered Markdown is not
  compiled. The answer is a guard script (e.g. `check_schemas.sh`).
- "State the invariant" beats "pin a source" when no generator exists: write
  "every CLI output is mirrored in Python" (a rule a gate enforces), not
  "currently none are missing" (a snapshot that rots); write "one analytic case
  per modeled feature; indices are sparse where a planned case was retired", not
  "27 cases; d12, d17, d18 unoccupied" (false — 29 ship, only d18 is missing).

---

## 3. Six prose-only failure modes

A code comment is co-located with one code site, so these cannot arise there:

1. **Single-source fan-out (cache coherency).** One authoritative fact cached
   into N audience docs that drift apart. _Rule:_ one owner per fact; every other
   doc is a shape-only pointer or a guard-pinned literal.
   _Example (historical):_ MSRV 1.86 (book) vs 1.88 (Cargo.toml) — the guard
   that caught this (`check_book_version.py`) was retired with book/ (mdBook
   decommission); the failure mode remains live for any multi-doc fact without
   a comparable guard.

2. **Stale current-state snapshots presented as standing truth.** A census or
   completeness claim that rots silently (no adjacent diff forces an edit).
   _Rule:_ state the durable invariant, not the transient count.
   _Example:_ crate-partition prose in `CLAUDE.md`; deterministic-suite case count.

3. **Audience-bleed.** Detail aimed at the wrong reader (internal struct names on
   an end-user page; finding-IDs in a release ledger). _Rule:_ match content to
   the doc's declared reader; relocate or signpost.

4. **Executable/resolvable claim with no compiler backstop (scoped).** Paths,
   commands, flags, and copy-paste code blocks a reader runs. _Rule:_ every cited
   **repo-relative** path/command/flag must resolve against the live tree or
   binary. **Scope bound (critical):** repo-relative prefixes only — **never**
   "every cited filename resolves", because README/CONTRIBUTING legitimately
   cite the external `cobre-docs` spec/theory pages absent from this repo. The
   path/link checker (`check-doc-paths.sh`) enforces this bound; it is the spec
   the checker must honor.

5. **Self-executable agent instruction.** A command an agent-facing doc tells the
   reader to run must itself _succeed_ — verified by running it, not by
   cross-checking sibling prose. Distinct from cross-doc contradiction: it can be
   false even if every other doc were deleted.
   _Example:_ a non-compiling `--all-features` invocation in CLAUDE.md.

6. **Misplaced artifact + false-confidence guards.** _Where_ a Markdown file sits
   is itself a truth claim — a dated working artifact committed at the repo root
   reads as canonical. _Corollary:_ a partial guard lies; a passing guard is
   evidence only for the fact-class it parses. Audit guard **coverage**, not just
   pass/fail.
   _Examples:_ `allow-attribute-inventory.md` tracked at root; a version-drift
   guard green while the underlying prose has since been deleted along with
   its scan target; a schema-count assert that passes while the actual count
   has grown.

---

## 4. Applying the rules

### When writing or reviewing prose docs

- For every **number, version, or enumeration** in a doc: ask whether a guard pins
  it. If not, replace the count/snapshot with the invariant form. Do not hand-edit
  a number that a guard script could own.
- For every **path/command/flag** cited in a repo-relative context: confirm it
  resolves in the live tree. Do not cite paths to external repositories (e.g.,
  `cobre-docs` spec pages) as if they were repo-relative — they are legitimately
  absent.
- For every **fact that appears in more than one doc**: identify the single owner.
  Secondary docs may carry a shape-only pointer ("see CONTRIBUTING.md for the full
  list") or a guard-pinned literal; they must not silently diverge.
- For **CHANGELOG entries**: Provenance is inverted — historical narrative is
  correct here. Plan-leakage still applies: no `Epic`/`ticket`/`workstream` tokens
  in released CHANGELOG entries. Admission and entry shape are governed by §6.

### Directives

- **Do** name the external contract (filename, flag, config field, path).
- **Do** default to deleting a number and stating the invariant. Pin a literal
  with a generator/guard ONLY where one must appear (schema export, version,
  MSRV). Invariant-first is the strict default, not a preference.
- **Do** flag audience-bleed: if a reader of this doc would not recognize a term,
  it belongs in a different doc or needs a signpost.
- **Do not** freeze a COUNT, VERSION, ENUMERATION, or run-snapshot NUMBER without
  a guard — this includes a "typical"/"common in practice"/"usually ~N" value, a
  single-run benchmark (speedup, timing, memory), and a structural census (crate /
  test / field / option / case counts), **even when correct today**. State the
  mechanism or rule; cite a regenerable artifact for measurements.
- **Do not** ship promotional/marketing voice in user-facing prose (§5).
- **Do not** add prose that only makes sense to a reader familiar with internal
  plan structure (`Epic N`, `ticket-NNN`, `workstream F-NNN`).
- **Do not** conflate the scope bound: the path/link checker (when wired) checks
  repo-relative prefixes only — treat external `cobre-docs` citations as
  intentionally unresolvable from this repo.

---

## 5. Voice register — sober reference, not marketing

`README.md` and the other narrative docs **teach**; they do not **sell**.
The teaching/narrative register (§1) is welcome — explain, address the reader
directly, walk through worked examples. **Promotional / marketing register is
not.** A reader should never feel pitched to. Capabilities and mechanisms are
stated as facts; the specifics carry the weight, not the adjectives.

### Banned constructs (delete, or rewrite to a plain factual statement)

- **Hype adjectives / superlatives** — "powerful", "blazing-fast", "seamless",
  "cutting-edge", "world-class", "production-grade", "robust", "high-fidelity",
  "Fastest", "smallest possible", and "zero-cost" / "zero-overhead" used as a
  _claim_. State the concrete property (e.g. "compiles to a no-op after inlining",
  with the basis) or say nothing.
- **Contrasting-affirmative / inflation** — "not just X, it's Y", "more than a
  Z", "isn't merely…", "X reimagined", "delivers a full …".
- **Unsubstantiated quality claims** — any adjective asserting goodness with no
  checkable basis _on the same page_. If the basis exists, state it; if it lives
  elsewhere, link it; otherwise drop the adjective.
- **Excitement / emotion framing** — "The performance benefit is meaningful:",
  "the frustrating … cycle", exclamation marks, and branded slogans as headings
  ("The honest representation principle").
- **Reader-minimizers** — "simply", "just", "easily", "obviously", "of course",
  "trivial". They hide difficulty and condescend.
- **Vague value words** — "modern infrastructure", "best practices",
  "industry-standard", "leverages" with no specific referent.

### Allowed (these are NOT hype — keep)

- Accurate, checkable technical descriptors: "deterministic", "type-safe",
  "zero-copy" (when literally true and load-bearing), "single-process",
  "RAII-managed", "ahead-of-time".
- Teaching scaffolding and second-person instructions ("run", "you will need").
- Comparative facts stated **as facts**, with the mechanism or a reference —
  "warm-starting reuses the prior basis, so subsequent solves do fewer pivots" —
  not as selling points ("dramatically faster").

### The deletion test for voice

Delete the adjective or clause. If the sentence still states a true, checkable
fact about _what the software does_, the deletion was correct. If it only lost
enthusiasm, it was hype. Keep a quality word **only** when its basis sits on the
page or one link away.

---

## 6. CHANGELOG entries — the CLI user is the only reader

The reader of `CHANGELOG.md` runs the `cobre` CLI and works with its input
files and outputs. Crate consumers, Python-only users and maintainers are not
its readers; the PR body carries what they need.

### Admission — does the change get an entry?

An entry exists only for a change a CLI user can observe:

- different results or outputs (a value, a file, a column);
- an input that is now refused, or newly accepted;
- a new or changed error or warning;
- a changed default, command, flag, or install requirement.

Leave out: crate API changes (breaking ones included — they go in the PR
body), changes seen only through the Python bindings (raise them as a
judgment call), refactors, solver-retry and other internal mechanics, limits
no authored input reaches, and cosmetic fixes such as a duplicated message.

### Shape — short and action-first

- One or two lines per entry: what changed, then what the user must do.
  Lead with the config key, file or command, with its path
  (`training.cut_selection.selection.lookback_iterations`,
  `constraints/hydro_bounds.parquet`, `cobre validate --json`).
- Keep the new default, the refusal condition, the remedy ("set it to
  `active` for the earlier behaviour", "retrain `level1` policies", "update
  readers of the file") and a "results change" note when outputs move.
- Drop the mechanism, the rationale, tie-break rules, per-column null
  semantics and edge-case sub-conditions. The user docs own that detail.
- Mark `**BREAKING:**` a change that needs the user to act: edit an input,
  retrain or re-run, update a reader of an output, or change platform.
- A fix names the symptom the user saw ("made the stage infeasible",
  "aborted the run", "was always 0"), not its cause in the code.
- Merge related changes into one entry — every checkpoint/resume change of a
  release, several refusals of one file, a fix and the warning added with it.
  Before appending to `[Unreleased]`, check whether an entry for the same
  feature should absorb the new change instead.
- Sections follow Keep a Changelog: Added, Changed, Removed, Fixed. A removal
  goes under Removed even when it is breaking.
- Released sections stay as published; trim `[Unreleased]` before a release,
  not history after it.
