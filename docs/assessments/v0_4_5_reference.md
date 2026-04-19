# Reference artifact: v0.4.5 pre-epic-03 baseline

Frozen byte-level reference for the 26 deterministic d-cases captured at the
`develop`-tip commit immediately before any Epic 03 production-code changes.
Downstream regression tickets (epic-03 ticket-009, epic-04, epic-05) compare
their outputs against these hashes to detect silent behavior drift.

## Capture procedure

The hashes below were produced by running:

```sh
bash scripts/capture_v045_reference.sh
```

which executes the following commands for each case (27 cases = d01-d30
minus d12, d17, d18 which are not present on disk):

```sh
./target/release/cobre run examples/deterministic/<case> \
    --output target/v045-reference/<case> \
    --quiet
```

And for convertido (single-rank, absent on this machine — see Notes):

```sh
./target/release/cobre run ~/git/cobre-bridge/example/convertido \
    --output target/v045-reference/convertido \
    --threads 5 --quiet
```

After all cases complete, `sha256sum` is applied to every **stable** parquet
file and the result is sorted by relative path:

```sh
find target/v045-reference -name "*.parquet" -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sed "s|target/v045-reference/||g" \
    | sort \
    > target/v045-reference/sha256.txt
```

Timing-bearing files are excluded before hashing; see
[Excluded files](#excluded-files-timing-bearing-parquets).

### Configuration at capture time

| Setting                    | Value                                                        |
| -------------------------- | ------------------------------------------------------------ |
| `warm_start_basis_mode`    | `NonAlienFirst` (default — no override in any d-case config) |
| `canonical_state_strategy` | `Disabled` (default — no override in any d-case config)      |
| Simulation arm             | Baked templates (default)                                    |
| MPI ranks                  | 1 (single-rank only)                                         |

## SHA256 map

90 stable entries, sorted by relative path.
Stability verified by two back-to-back captures on the same commit: zero
divergence between run 3 and run 4 of the script.

Hashes cover: `training/dictionaries/bounds.parquet`,
`training/solver/retry_histogram.parquet`,
`simulation/solver/retry_histogram.parquet`,
and all `simulation/<entity>/scenario_id=*/data.parquet` files.

```
0220d23b3cc9c0ff2bbbfca02b87065c09ec83f90eff8923638d316f73c8c0a0  d28-decomp-weekly-monthly/simulation/buses/scenario_id=0000/data.parquet
0220d23b3cc9c0ff2bbbfca02b87065c09ec83f90eff8923638d316f73c8c0a0  d28-decomp-weekly-monthly/simulation/buses/scenario_id=0001/data.parquet
0220d23b3cc9c0ff2bbbfca02b87065c09ec83f90eff8923638d316f73c8c0a0  d28-decomp-weekly-monthly/simulation/buses/scenario_id=0002/data.parquet
0220d23b3cc9c0ff2bbbfca02b87065c09ec83f90eff8923638d316f73c8c0a0  d28-decomp-weekly-monthly/simulation/buses/scenario_id=0003/data.parquet
0220d23b3cc9c0ff2bbbfca02b87065c09ec83f90eff8923638d316f73c8c0a0  d28-decomp-weekly-monthly/simulation/buses/scenario_id=0004/data.parquet
0dea26c6aa442c5dc0e1bbbb75f0dc806f9692124e77c7cb6a1a0a52932ddb45  d21-min-outflow-regression/training/dictionaries/bounds.parquet
13bd580392e9362ee4d6fd53751fde8011269bc9b2a5209c36ef2bb4669f7969  d28-decomp-weekly-monthly/simulation/hydros/scenario_id=0004/data.parquet
1487aeb8fcc910da0e2f33c5da3a854d3cac2d13959a9a437c4357291721441a  d25-discount-rate/simulation/hydros/scenario_id=0000/data.parquet
16e04d3bccb5fdf736e19b4a99e1345bf7fe87c5291e0273b407ac17901dda22  d27-per-stage-thermal-cost/training/dictionaries/bounds.parquet
260bbebc8d729d2a626728ec4bee0803bedd35807ed7f7d126f0f33c2ea35df1  d30-pattern-d-monthly-quarterly/simulation/inflow_lags/scenario_id=0000/data.parquet
2743ad1b251ab212b53ed8883f7b04fd9c436aa147f63da911a8d36bcde46baf  d09-multi-deficit/training/dictionaries/bounds.parquet
2743ad1b251ab212b53ed8883f7b04fd9c436aa147f63da911a8d36bcde46baf  d13-generic-constraint/training/dictionaries/bounds.parquet
2fcfaab9f4f432f85dbd9447c81f90f685d1d7a5c43856e8dac3870ef30f50ce  d30-pattern-d-monthly-quarterly/simulation/costs/scenario_id=0000/data.parquet
305311a324dfc5d315ea2c003804e7e9955eb39393972f5b456f30c0ecac604c  d25-discount-rate/simulation/thermals/scenario_id=0000/data.parquet
3a99c0bacad60ba69076d19cb13328b59e62b192f86e6614afaf60c544b2c1fc  d29-pattern-c-weekly-par/simulation/inflow_lags/scenario_id=0000/data.parquet
3fdf080ea48929d5774f5da99e63f02677dd745877e90b10754b398195a49d71  d23-bidirectional-withdrawal/training/dictionaries/bounds.parquet
4ebe6c7eaa5ec053043a12c2e52fcb17486010e5c7cec887c7dd67e53c2395b0  d28-decomp-weekly-monthly/simulation/thermals/scenario_id=0000/data.parquet
4ebe6c7eaa5ec053043a12c2e52fcb17486010e5c7cec887c7dd67e53c2395b0  d28-decomp-weekly-monthly/simulation/thermals/scenario_id=0001/data.parquet
4ebe6c7eaa5ec053043a12c2e52fcb17486010e5c7cec887c7dd67e53c2395b0  d28-decomp-weekly-monthly/simulation/thermals/scenario_id=0002/data.parquet
4ebe6c7eaa5ec053043a12c2e52fcb17486010e5c7cec887c7dd67e53c2395b0  d28-decomp-weekly-monthly/simulation/thermals/scenario_id=0003/data.parquet
4ebe6c7eaa5ec053043a12c2e52fcb17486010e5c7cec887c7dd67e53c2395b0  d28-decomp-weekly-monthly/simulation/thermals/scenario_id=0004/data.parquet
507aa779516a498ba45942b6ec751c96d7f063629d74ad0576972a24cd83a393  d28-decomp-weekly-monthly/simulation/costs/scenario_id=0002/data.parquet
509cc61f976685a102f09bf69939b5dc9bee412f690376f6e089b4803a4f430b  d30-pattern-d-monthly-quarterly/simulation/buses/scenario_id=0000/data.parquet
64d91d5152242fe7e437f4ce5002e0581ff917cc9c483335b088d02a5a50a29a  d04-transmission/training/dictionaries/bounds.parquet
65778c2f6963a7b538ff6a2499042e53a4ec56a61623dd2bbb6d0b3dbdf97abd  d22-per-block-min-outflow/training/dictionaries/bounds.parquet
6a578d343e1ec2e32c8b088b3bc193fca61482a2587ba978466578b230cae351  d16-par1-lag-shift/training/dictionaries/bounds.parquet
868c3036a49983326675563128251b41f7e480759a24f6d6774dd178df6bef5d  d30-pattern-d-monthly-quarterly/simulation/hydros/scenario_id=0000/data.parquet
86f84a85dadbb735c082423abe44d0f86bfca6c79e777de6f971e370cd6dba83  d03-two-hydro-cascade/training/dictionaries/bounds.parquet
9013da169e3299ab66045b8dead51714f517226ab73f4a58f611ab8119244b10  d30-pattern-d-monthly-quarterly/training/dictionaries/bounds.parquet
948b1269be9487dde47d638b32736fe19e8d2902eab9baafe83e6142434da918  d28-decomp-weekly-monthly/simulation/hydros/scenario_id=0002/data.parquet
974ab53ec9d4e323e92aa96e2865039b2a51a178940fb1bbbb0f2f14aafd9130  d28-decomp-weekly-monthly/simulation/costs/scenario_id=0000/data.parquet
984b7f68f2e0b69f752434a653cea5599ace3d567b53b77608d739d4f0ff80dd  d29-pattern-c-weekly-par/simulation/buses/scenario_id=0000/data.parquet
9d10166d4f39757bef73707608f890c1a2e3f66894a820def85a6b7e1dfb81ea  d28-decomp-weekly-monthly/simulation/costs/scenario_id=0004/data.parquet
9d4c0f445a4ab03bd46688eef1d0911e7c72faed2f6fd4cf66cbcc144c8211d1  d29-pattern-c-weekly-par/simulation/costs/scenario_id=0000/data.parquet
9f21cd18b654ced8f6bb9695905b5fa858bd9aa56bcc80460aff5e7ae60831aa  d28-decomp-weekly-monthly/simulation/costs/scenario_id=0001/data.parquet
9f5cfb8f390a1b778f86c15b80fe06850482c1b3db54c7ec370ea2e0800b765e  d28-decomp-weekly-monthly/simulation/hydros/scenario_id=0003/data.parquet
a1d7ceca8bcd2ffdf8866888519f1d637e6bf569fcaf74cf74d5ae9b76e049ad  d26-estimated-par2/training/dictionaries/bounds.parquet
a2cefa26ea5194ad10b85c5ad64ebd59774cb5bfc8b4eddca171c113e1e7f59c  d25-discount-rate/simulation/buses/scenario_id=0000/data.parquet
a6a893c7e67bfb5ca24b62495d6b8b3bee9f341048800a49ab89061993a3fc05  d28-decomp-weekly-monthly/simulation/hydros/scenario_id=0001/data.parquet
b2720468fb2126a7fe2219f020fd7dd59738966ce05c31c1d0281f95d9b69e3c  d25-discount-rate/simulation/costs/scenario_id=0000/data.parquet
c89534ee7fee7573fd958e30dcde72c49e487edb3ef431cfb90e105703b8bcc2  d19-multi-hydro-par/training/dictionaries/bounds.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d01-thermal-dispatch/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d02-single-hydro/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d03-two-hydro-cascade/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d04-transmission/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d05-fpha-constant-head/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d06-fpha-variable-head/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d07-fpha-computed/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d08-evaporation/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d09-multi-deficit/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d10-inflow-nonnegativity/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d11-water-withdrawal/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d13-generic-constraint/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d14-block-factors/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d15-non-controllable-source/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d16-par1-lag-shift/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d19-multi-hydro-par/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d20-operational-violations/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d21-min-outflow-regression/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d22-per-block-min-outflow/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d23-bidirectional-withdrawal/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d24-productivity-override/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d25-discount-rate/simulation/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d25-discount-rate/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d26-estimated-par2/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d27-per-stage-thermal-cost/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d28-decomp-weekly-monthly/simulation/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d28-decomp-weekly-monthly/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d29-pattern-c-weekly-par/simulation/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d29-pattern-c-weekly-par/training/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d30-pattern-d-monthly-quarterly/simulation/solver/retry_histogram.parquet
d009bf4ddebcdbc8f2e221f690f01addfcdefcd0e90cf99b6baea107181612f3  d30-pattern-d-monthly-quarterly/training/solver/retry_histogram.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d02-single-hydro/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d05-fpha-constant-head/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d06-fpha-variable-head/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d07-fpha-computed/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d08-evaporation/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d10-inflow-nonnegativity/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d11-water-withdrawal/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d24-productivity-override/training/dictionaries/bounds.parquet
d576b0bac37b6e191b1f7bb3ef85dc352cdf6ebc36bb34f17e4f6373e192cc27  d25-discount-rate/training/dictionaries/bounds.parquet
e1ddbccadf0052b006caec132e5d32117085edb02676265becf1b5f8ef8a0692  d29-pattern-c-weekly-par/simulation/hydros/scenario_id=0000/data.parquet
ebd495bcff98d7b063113078eb9b65fbf47cf764c6738bbbed8256e1b53e9178  d29-pattern-c-weekly-par/training/dictionaries/bounds.parquet
f120c067d3e352bc261d6df4955dbdec01436ab02536983398a76ac7d34cf429  d20-operational-violations/training/dictionaries/bounds.parquet
f2c8336b7c76ce712f814fc646fc2753f840fccac5ef8a314734c48c917ed588  d15-non-controllable-source/training/dictionaries/bounds.parquet
f33f11198b8ba2edbdd66779f8a5fd93d76f642b2dc391b298a847ad0a00d744  d28-decomp-weekly-monthly/simulation/hydros/scenario_id=0000/data.parquet
f5243a4c219f5bffe6ac3da6928feb8bf3fa489ecdc2d36eea260a456122aa32  d01-thermal-dispatch/training/dictionaries/bounds.parquet
f5243a4c219f5bffe6ac3da6928feb8bf3fa489ecdc2d36eea260a456122aa32  d14-block-factors/training/dictionaries/bounds.parquet
f93344f91b865bc62ac7a16bb404220181aa857234204af0a7e3a85489c5427e  d28-decomp-weekly-monthly/training/dictionaries/bounds.parquet
f9ec2eb64775873729a720e8177311c921e9c4cd6404c23aa4eec1ddf24e3f9e  d28-decomp-weekly-monthly/simulation/costs/scenario_id=0003/data.parquet
```

## Excluded files: timing-bearing parquets

The following four path patterns were discovered to be **unstable** (hashes
differ on back-to-back runs on the same commit) because they embed actual
wall-clock durations. They are excluded from `sha256.txt` and from the map
above.

| Path pattern                             | Unstable columns                                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `*/training/convergence.parquet`         | `time_forward_ms`, `time_backward_ms`, `time_total_ms`                                               |
| `*/training/solver/iterations.parquet`   | `solve_time_ms`, `load_model_time_ms`, `add_rows_time_ms`, `set_bounds_time_ms`, `basis_set_time_ms` |
| `*/training/timing/iterations.parquet`   | All columns (`forward_wall_ms`, `backward_wall_ms`, `cut_selection_ms`, `mpi_allreduce_ms`, etc.)    |
| `*/simulation/solver/iterations.parquet` | `solve_time_ms`, `load_model_time_ms`, `add_rows_time_ms`, `set_bounds_time_ms`, `basis_set_time_ms` |

Regression tickets (starting with epic-03 ticket-009) **inherit this exclusion
list**: they must compare only stable parquet files. The timing files are
orthogonal to algorithmic correctness.

## Captured at

| Field                     | Value                                                        |
| ------------------------- | ------------------------------------------------------------ |
| Commit SHA                | `c67f3d8fedd544b784c97fb3b1d5b67048d2f5a2`                   |
| `cargo pkgid cobre-cli`   | `path+file:///home/rogerio/git/cobre/crates/cobre-cli#0.4.4` |
| Capture date              | 2026-04-18                                                   |
| Machine                   | `Linux 6.19.12-200.fc43.x86_64`                              |
| Stable entries            | 90                                                           |
| Excluded (timing) entries | 85                                                           |
| Convertido                | Absent (see below)                                           |

The commit SHA above is the value of `git rev-parse HEAD` at the time the
script was run. After this document is committed, the new commit SHA is the
one recorded by `git log -1 --format=%H docs/assessments/v0_4_5_reference.md`.

## Notes on convertido

The convertido benchmark case (`~/git/cobre-bridge/example/convertido`) was
**not present** on this machine at capture time. The script emitted a WARNING
and continued without error. The 90 hashes above are from the 27 deterministic
d-cases only.

To add convertido hashes to a future revision of this document:

1. Ensure `~/git/cobre-bridge/example/convertido` exists on the reference
   machine.
2. Run `bash scripts/capture_v045_reference.sh` (no extra flags needed; the
   script auto-detects the default path).
3. Paste the additional rows from `target/v045-reference/sha256.txt` into the
   SHA256 map section above, keeping the alphabetical sort order.

The acceptance gate for epic-03/04/05 regression tickets does not require
convertido hashes; the 26 d-case hashes are sufficient for the epics as
currently scoped.

## Post-epic-03 verification

### Summary

All 90 stable parquet entries from the v0.4.5 reference map are **byte-identical**
between the pre-epic-03 baseline and the post-epic-03 build. Zero unexpected drifts.
Epic-03 R1 risk (latent bug surfacing) is cleared.

### Capture details

| Field                   | Value                                      |
| ----------------------- | ------------------------------------------ |
| Post-epic-03 commit SHA | `def90bad5f40855c5571e74d9399f298a7989115` |
| Capture date            | 2026-04-18                                 |
| Machine                 | `Linux 6.19.12-200.fc43.x86_64`            |
| Stable entries compared | 90                                         |
| Byte-identical          | 90                                         |
| Allowlisted drifts      | 0 (see below)                              |
| Unexpected drifts       | 0                                          |
| Convertido              | Absent (see Notes on convertido)           |

**Note on commit SHA**: The value above is `git rev-parse HEAD` at the time the
capture script ran. After this document is committed, the HEAD SHA advances by
one more commit (this edit itself). This intentional mismatch mirrors the
same pattern noted in the "Captured at" section above.

### Comparison command

```sh
bash scripts/capture_v045_reference.sh
cp target/v045-reference/sha256.txt target/v045-reference-post-epic03/sha256.txt
python3 scripts/compare_v045_reference.py \
    --reference-sha256 docs/assessments/v0_4_5_reference.md \
    --actual-sha256 target/v045-reference-post-epic03/sha256.txt
```

Output:

```
all mismatches are in the expected-drifts allowlist (90/90 files byte-identical, 0 allowlisted drift(s))
```

Exit code: 0.

### Allowed-drifts list

The `scripts/compare_v045_reference.py` script contains the following
`EXPECTED_DRIFTS` allowlist. In this run, none of these paths appeared in the
stable hash map (the timing-bearing files were already excluded by the capture
script, and the `iterations.parquet` schema-rename files are also
timing-bearing and therefore also excluded). The allowlist is recorded here
for completeness:

| Path pattern                              | Justification                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| `**/training/solver/iterations.parquet`   | ticket-007: `basis_consistency_failures` schema rename causes hash drift |
| `**/simulation/solver/iterations.parquet` | ticket-007: `basis_consistency_failures` schema rename causes hash drift |
| `**/training/convergence.parquet`         | ticket-001: timing columns (`time_*_ms`) are wall-clock unstable         |
| `**/training/timing/iterations.parquet`   | ticket-001: pure wall-clock timing file, excluded from stable map        |
| `**/metadata.json`                        | embeds `completed_at` timestamp and hostname, changes on every run       |

### Convertido wall-clock delta

The convertido benchmark case (`~/git/cobre-bridge/example/convertido`) was
**not present** on this machine at re-capture time. The ±5% wall-clock
acceptance criterion (ticket-009 R2 risk gate) cannot be evaluated here.
No performance regression is flagged; the measurement is deferred to the
reference machine where convertido is available. See the "Notes on
convertido" section above for instructions on adding convertido hashes when
the reference machine is available.
