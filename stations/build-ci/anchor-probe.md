## INGEST ANCHOR PROBE — build-ci (2026-09, baseline)

Register-shaped stub: one block per candidate of the four merged lens files, every anchor rendered so
`check-anchors.py "INGEST ANCHOR PROBE — build-ci (2026-09, baseline)" --register <this file> --baseline 077dbe2c` resolves it with the register's own parser.
Ingest ref `<lens>-<nn>` (nn = zero-based index among the lens file's candidates). Anchor form at this station:
`path:line` for every YAML / shell / Python / TOML / JSON / Markdown surface; `path::symbol` ONLY on the three build.rs
(the checker's symbol resolver is a Rust-item regex). The paired drift evidence (`docClaim` / `treeFact`) and the
`twoSite` sites are rendered as extra anchors of the same block so they resolve under the same parser.

**CD-900 · probe · architecture-00**

The infra-genericity gate's crate list is a hand-kept literal in two unpaired places, so the rule's coverage cannot follow the target layering without a synchronized script-and-prose edit that nothing enforces

- **Anchors:** `scripts/ci/check-infra-genericity.sh:62` `.github/workflows/ci.yml:261` `CLAUDE.md:39` `scripts/ci/check-infra-genericity.sh:62` `.github/workflows/ci.yml:262`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-901 · probe · architecture-01**

The infra-genericity gate matches one engine's vocabulary rather than the engine-concept rule it enforces, so a sibling engine's nouns and an underscore-joined identifier both pass unflagged

- **Anchors:** `scripts/ci/check-infra-genericity.sh:79` `scripts/ci/check-infra-genericity.sh:10` `CLAUDE.md:41` `scripts/ci/check-infra-genericity.sh:79` `.github/workflows/ci.yml:262`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Re-raise-of:** CD-061 — a SHARPEN of the live core-io entry's gate half (prior-register.md: CD-061 may be sharpened only through that station's owner)

**CD-902 · probe · architecture-02**

No CI workflow runs on a pull request into develop, so every blocking gate first sees a develop change only after it has already merged

- **Anchors:** `.github/workflows/ci.yml:6` `.github/workflows/ci.yml:4` `scripts/ci/check-allow-rationale.sh:422`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-903 · probe · architecture-03**

The invariance-shuffle workflow cannot compile the test target it dispatches, and the automatic suite skips the same tests, so the declaration-order-invariance shuffle matrix has no executable path in CI

- **Anchors:** `.github/workflows/invariance-shuffle.yml:34` `.github/workflows/invariance-shuffle.yml:55` `crates/cobre-sddp/Cargo.toml:55` `.github/workflows/ci.yml:114` `CLAUDE.md:26` `.github/workflows/invariance-shuffle.yml:34` `.github/workflows/ci.yml:114`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-904 · probe · architecture-04**

The CLI template mirror is guarded in one direction only: a file added to the canonical example deck but not to the embedded list leaves both guards green while cobre init ships an incomplete case

- **Anchors:** `crates/cobre-cli/build.rs::TEMPLATE_FILES` `scripts/ci/check-docs-examples.sh:32` `scripts/ci/check-docs-examples.sh:113`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-905 · probe · architecture-05**

No clippy invocation in CI reaches the published Python bindings crate, so the lint bar the crate declares for itself, including the deny-level unwrap rule, is never evaluated

- **Anchors:** `crates/cobre-python/Cargo.toml:46` `crates/cobre-python/Cargo.toml:50` `.github/workflows/ci.yml:154` `.github/workflows/ci.yml:221` `CLAUDE.md:20` `.github/workflows/ci.yml:154` `.github/workflows/ci.yml:116`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-906 · probe · over-engineering-00**

The MPICH Cache/Build/Set step triple is copied byte-for-byte eight times inside ci.yml, with two near-variants in mpi-slurm.yml and release-mpi.yml and no composite action anywhere under .github/

- **Anchors:** `.github/workflows/ci.yml:43` `.github/workflows/ci.yml:49` `.github/workflows/ci.yml:87` `.github/workflows/ci.yml:130` `.github/workflows/ci.yml:180` `.github/workflows/ci.yml:316` `.github/workflows/ci.yml:377` `.github/workflows/ci.yml:468` `.github/workflows/ci.yml:522` `.github/workflows/mpi-slurm.yml:63` `.github/workflows/release-mpi.yml:75`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-907 · probe · over-engineering-01**

The MPICH version literal 4.2.3 has three independent owners in the SLURM image path (workflow env, Dockerfile ARG default, pushed image tag) and no build-arg links them, so the tag can name a version the image was not built from

- **Anchors:** `tests/slurm/Dockerfile.base:42` `tests/slurm/Dockerfile.base:8` `tests/slurm/Dockerfile:15`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-908 · probe · over-engineering-02**

Five comment and doc gates each hand-maintain their own copy of the crate-source scan-directory list, in two divergent shapes, and four of them do so while already sourcing the shared scan library that exists to hold exactly this kind of shared fact

- **Anchors:** `scripts/ci/check-comment-refs.sh:56` `scripts/ci/check-comment-refs.sh:57` `scripts/ci/check-comment-line-refs.sh:48` `scripts/ci/check-doc-placeholders.sh:57` `scripts/ci/check-comment-banners.sh:49` `scripts/ci/check-no-plan-leaks.sh:126` `scripts/ci/lib/comment_scan.sh:5` `CLAUDE.md:57` `scripts/ci/check-comment-refs.sh:57` `.github/workflows/ci.yml:276`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-909 · probe · over-engineering-03**

cobre-cli declares an empty slow-tests feature with no cfg consumer anywhere in the crate, justified by a workspace-consistency claim that three other members with feature tables falsify

- **Anchors:** `crates/cobre-cli/Cargo.toml:55` `crates/cobre-cli/Cargo.toml:57`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-910 · probe · over-engineering-04**

cobre-io declares an empty slow-tests feature with no cfg consumer anywhere in the crate, justified by the same workspace-consistency claim the workspace falsifies

- **Anchors:** `crates/cobre-io/Cargo.toml:17` `crates/cobre-io/Cargo.toml:19`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-911 · probe · over-engineering-05**

The step named Publish all crates publishes eight of the thirteen workspace members from a hand-written ordered list, while the five members it omits carry crates.io-facing descriptions and no publish = false, and nothing reconciles the list with the manifest

- **Anchors:** `.github/workflows/publish.yml:36` `.github/workflows/publish.yml:80` `.github/workflows/publish.yml:105` `Cargo.toml:12` `crates/cobre-mcp/Cargo.toml:3`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-912 · probe · drift-00**

docs/design/README.md declares a closed five-value status vocabulary and a delete-on-ship convention that three of its eight rows and two still-present docs contradict

- **Anchors:** `docs/design/README.md:4` `docs/design/README.md:24` `docs/design/README.md:26` `docs/design/README.md:27` `docs/design/README.md:31` `docs/design/README.md:4` `docs/design/anticipated-fixed-post-horizon-commitments.md:3`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-913 · probe · drift-01**

docs/design/README.md calls itself the map of the directory while post-horizon-input-unification.md has no row in the table

- **Anchors:** `docs/design/README.md:4` `docs/design/README.md:18` `docs/design/README.md:27` `docs/design/README.md:4` `docs/design/post-horizon-input-unification.md:3`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-914 · probe · drift-02**

Two index rows disagree with the status the doc itself carries: one doc declares a status outside the vocabulary, the other carries no status line at all

- **Anchors:** `docs/design/README.md:4` `docs/design/README.md:23` `docs/design/README.md:25` `docs/design/README.md:4` `docs/design/backward-warm-start-channels.md:10`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-915 · probe · drift-03**

ARCHITECTURE.md's crate map counts five dependencies for cobre-python while naming six on the same line, and its cobre-cli entry omits cobre-core

- **Anchors:** `ARCHITECTURE.md:95` `ARCHITECTURE.md:88` `ARCHITECTURE.md:95` `crates/cobre-cli/Cargo.toml:21`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-916 · probe · drift-04**

ci.yml's per-crate feature table calls itself the deduplicated union across crates but credits test-support to cobre-solver alone, while four other crates declare it

- **Anchors:** `.github/workflows/ci.yml:18` `.github/workflows/ci.yml:22` `.github/workflows/ci.yml:18` `crates/cobre-core/Cargo.toml:30`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-917 · probe · drift-05**

The genericity hard rule names three forbidden tokens while the gate it is enforced by rejects eleven alternatives, including standalone cut and Cut

- **Anchors:** `CLAUDE.md:39` `scripts/ci/check-infra-genericity.sh:79` `CLAUDE.md:39` `scripts/ci/check-infra-genericity.sh:79` `CLAUDE.md:39` `scripts/ci/check-infra-genericity.sh:79` `.github/workflows/ci.yml:262`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`
- **Re-raise-of:** CD-061 — a SHARPEN of the live core-io entry's gate half (prior-register.md: CD-061 may be sharpened only through that station's owner)

**CD-918 · probe · drift-06**

A shipped conformance test states it verifies the contracts of a backend-testing.md that has never existed in this repository, and two gate blind spots keep the reference invisible

- **Anchors:** `docs/design/README.md:18` `scripts/ci/check-comment-refs.sh:57` `scripts/ci/check-comment-refs.sh:84` `.github/workflows/ci.yml:276` `crates/cobre-comm/tests/local_conformance.rs:4` `docs/design/README.md:18` `CLAUDE.md:57` `scripts/ci/check-comment-refs.sh:57` `.github/workflows/ci.yml:276`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-919 · probe · drift-07**

Four gate headers cite rule-file sections that do not resolve, and a sibling gate sanctions the filename-plus-section form as durable

- **Anchors:** `scripts/ci/check-doc-paths.sh:5` `scripts/ci/check-doc-placeholders.sh:6` `scripts/ci/check-comment-refs.sh:5` `scripts/ci/check-comment-line-refs.sh:7` `scripts/ci/check-no-plan-leaks.sh:48` `scripts/ci/check-doc-paths.sh:5` `.claude/rules/doc-integrity.md:173`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-920 · probe · drift-08**

The roadmap's Part-I item 6 records a genericity-gate exemption for the four cobre-io policy files that the gate retired before this baseline

- **Anchors:** `scripts/ci/check-infra-genericity.sh:74` `scripts/ci/check-infra-genericity.sh:70` `plans/architecture-debt-audit/tools/target-layering-brief.md:91` `scripts/ci/check-infra-genericity.sh:74`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-921 · probe · drift-09**

The unsafe-code hard rule enumerates one unsafe island in cobre-sddp while the crate's own manifest states there are two

- **Anchors:** `CLAUDE.md:19` `crates/cobre-sddp/Cargo.toml:112` `CLAUDE.md:19` `crates/cobre-sddp/src/hull/ffi.rs:27` `CLAUDE.md:19` `Cargo.toml:36` `.github/workflows/ci.yml:154`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-922 · probe · performance-00**

The schemas job carries the MPICH Cache/Build/Set triple although its only compile is the default-feature release CLI binary, and the sibling docs-examples job runs that identical build with no triple and says in writing why none is needed

- **Anchors:** `.github/workflows/ci.yml:310` `.github/workflows/ci.yml:316` `.github/workflows/ci.yml:339` `.github/workflows/ci.yml:344` `.github/workflows/ci.yml:360` `scripts/ci/check_schemas.sh:23` `crates/cobre-cli/Cargo.toml:42` `crates/cobre-comm/Cargo.toml:20`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-923 · probe · performance-01**

The python job carries the MPICH Cache/Build/Set triple on every instance of its version matrix although none of its three compiles enables an mpi feature

- **Anchors:** `.github/workflows/ci.yml:511` `.github/workflows/ci.yml:516` `.github/workflows/ci.yml:522` `.github/workflows/ci.yml:559` `.github/workflows/ci.yml:561` `.github/workflows/ci.yml:568` `crates/cobre-python/Cargo.toml:38` `crates/cobre-comm/Cargo.toml:20`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-924 · probe · performance-02**

The eight MPICH consumers in ci.yml share one immutable cache key with no dependency edge anywhere in the workflow, so a key change is missed concurrently by every job instance rather than once by a populator

- **Anchors:** `.github/workflows/ci.yml:12` `.github/workflows/ci.yml:45` `.github/workflows/ci.yml:47` `.github/workflows/ci.yml:315` `.github/workflows/ci.yml:506` `.github/workflows/ci.yml:521` `.github/workflows/mpi-slurm.yml:61` `.github/workflows/release-mpi.yml:73`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-925 · probe · performance-03**

Three job definitions each compile the release cobre binary independently, with no artifact handoff and no cache-sharing directive anywhere in the workflow, while three cache-separating directives are present

- **Anchors:** `.github/workflows/ci.yml:337` `.github/workflows/ci.yml:339` `.github/workflows/ci.yml:358` `.github/workflows/ci.yml:360` `.github/workflows/ci.yml:545` `.github/workflows/ci.yml:561` `scripts/ci/check_schemas.sh:23` `scripts/ci/check-docs-examples.sh:86`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-926 · probe · performance-04**

Ten of the fourteen quality-scripts steps each enumerate the crates tree themselves, and the shared scan library they already source exposes only a per-file predicate and no enumeration helper

- **Anchors:** `scripts/ci/lib/comment_scan.sh:21` `scripts/ci/lib/comment_scan.sh:32` `scripts/ci/check-infra-genericity.sh:127` `scripts/ci/check-comment-refs.sh:155` `scripts/ci/check-comment-line-refs.sh:118` `scripts/ci/check-doc-placeholders.sh:104` `scripts/ci/check-no-plan-leaks.sh:180` `scripts/ci/check-allow-rationale.sh:680` `scripts/ci/quality-report.sh:91` `.github/workflows/ci.yml:262`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-927 · probe · performance-05**

Three jobs install their cargo tool from source with cargo install while the same workflow already uses a prebuilt-binary install action for its one other tool

- **Anchors:** `.github/workflows/ci.yml:215` `.github/workflows/ci.yml:217` `.github/workflows/ci.yml:413` `.github/workflows/ci.yml:443` `.github/workflows/ci.yml:496` `.github/workflows/invariance-shuffle.yml:29` `.github/workflows/invariance-shuffle.yml:50`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

**CD-928 · probe · performance-06**

Nine of the fourteen job definitions check out every vendored submodule recursively, and on each of them at least one vendored superbuild tree is fetched that the job's feature set never compiles

- **Anchors:** `.github/workflows/ci.yml:42` `.github/workflows/ci.yml:173` `.github/workflows/ci.yml:219` `.github/workflows/ci.yml:309` `.github/workflows/ci.yml:356` `.github/workflows/ci.yml:515` `crates/cobre-solver/build.rs::main`
- **Baseline:** `077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`

