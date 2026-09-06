# Harness verification

## Run 2026-09-06T17:54:05Z

- Baseline: `d5c6c39a0a889e501c2898b5b83ea3adf9759f52` (register pin `a136840d4f2ea137f685f0af6dac04254b983b60`)
- Deck: `/home/rogerio/git/cobre-bridge/example/cobre_reduzido_2` (sha256 e46d944548e816803a36a1ea05e36ddc4235c8c4d25fd2837e44e3347ce6bbc6, byte-identical to the calibration stamp)
- Calibration bound: `234.781 s` (from `measurements/CAL/median.txt`)
- Unit suite: Ran 10 tests in 1.119s OK 
- Status: **PASS**

### Checks

| check | expected exit | actual exit |
| --- | --- | --- |
| `check-anchors.py --self-test` | 0 | 0 |
| `check-reraise.py --self-test` | 0 | 0 |
| `check-roadmap-dag.py --self-test` | 0 | 0 |
| `fields-check.py --self-test` | 0 | 0 |
| `check-anchors.py good-section.md` | 0 | 0 |
| `check-anchors.py bad-anchor.md` | 1 | 1 |
| `check-anchors.py NO SUCH SECTION` | 3 | 3 |
| `check-reraise.py good-section.md` | 0 | 0 |
| `check-reraise.py reraise-seeded.md` | 1 | 1 |
| `fields-check.py reraise-seeded.md` | 0 | 0 |
| `fields-check.py missing-alignment.md` | 1 | 1 |
| `fields-check.py duplicate-id (derived)` | 4 | 4 |
| `check-roadmap-dag.py good-roadmap.md` | 0 | 0 |
| `check-roadmap-dag.py cyclic-roadmap.md` | 1 | 1 |
| `perf-run.sh --dry-run CAL 4t` | 0 | 0 |
| `perf-run.sh deck-missing (HOME=/nonexistent)` | 3 | 3 |
| `python3 -m unittest discover tools/tests` | 0 | 0 |

### Tracked-tree assertion

`git status --porcelain -- crates docs scripts .github schemas Cargo.toml` -> empty

### Not exercised

- `perf-run.sh` exit 4 (`timeout-3x`): needs a runaway solve; exercised ad hoc during the calibration ticket with `--bound 1`, not by this gate.
- `perf-run.sh` exit 5 (`mpi-unavailable`): layout `2x2` is outside the harness epic; exercised ad hoc during the calibration ticket with `mpiexec` off PATH.
- `perf-run.sh` exit 6 (deck mutated): would require corrupting the reference deck.
