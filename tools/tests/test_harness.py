"""Executable proof of the evaluation harness (E01): checkers, dry run, baseline pin, clean tree.

Run from the repository root:
    python3 -m unittest discover -s plans/architecture-debt-audit/tools/tests -p 'test_*.py'
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
)
AUDIT = ROOT / "plans" / "architecture-debt-audit"
TOOLS = AUDIT / "tools"
FIXTURES = TOOLS / "fixtures"
BACKLOG = AUDIT / "BACKLOG.md"
DECK = pathlib.Path(
    os.environ.get(
        "COBRE_PERF_DECK",
        str(pathlib.Path.home() / "git/cobre-bridge/example/cobre_reduzido"),
    )
)
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
EVALUATED = [
    "crates",
    "docs",
    "scripts",
    ".github",
    "schemas",
    "Cargo.toml",
    "Cargo.lock",
    "examples",
    "tests",
    f":(exclude){MIRROR}",
]
# The two register pins seen so far: a136840d minted the core-io and stochastic stations,
# 077dbe2c superseded it once the Tier fix waves had removed several anchored symbols.
FIRST_PIN = "a136840d4f2ea137f685f0af6dac04254b983b60"
SECOND_PIN = "077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c"

sys_path_tools = str(TOOLS)
if sys_path_tools not in sys.path:
    sys.path.insert(0, sys_path_tools)
from lib import backlog_parse as bp  # noqa: E402

# checker, argv for the good fixture, argv for the seeded-bad fixture, expected bad exit, expected bad message
MATRIX = [
    (
        "check-anchors.py",
        ["--register", FIXTURES / "good-section.md", "good-section"],
        ["--register", FIXTURES / "bad-anchor.md", "bad-anchor"],
        1,
        "anchor-missing",
    ),
    (
        "check-reraise.py",
        [FIXTURES / "good-section.md"],
        [FIXTURES / "reraise-seeded.md"],
        1,
        "unjustified re-raise",
    ),
    (
        "fields-check.py",
        [FIXTURES / "reraise-seeded.md"],
        [FIXTURES / "missing-alignment.md"],
        1,
        "alignment-invalid",
    ),
    (
        "check-roadmap-dag.py",
        ["--backlog", FIXTURES / "good-roadmap.md"],
        ["--backlog", FIXTURES / "cyclic-roadmap.md"],
        1,
        "cycle",
    ),
]


def run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(a) for a in args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def checker(tool: str, *args) -> subprocess.CompletedProcess[str]:
    return run("python3", TOOLS / tool, *args)


def deck_digest(path: pathlib.Path) -> str:
    """Byte-for-byte the digest perf-run.sh stamps: sha256 over `sha256sum` lines of the sorted absolute paths."""
    h = hashlib.sha256()
    for f in sorted(
        (str(p) for p in path.rglob("*") if p.is_file()), key=lambda x: x.encode()
    ):
        h.update(
            f"{hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()}  {f}\n".encode()
        )
    return h.hexdigest()


def tree_snapshot(path: pathlib.Path) -> list[tuple[str, int]]:
    return sorted(
        (str(p.relative_to(path)), p.stat().st_size)
        for p in path.rglob("*")
        if p.is_file()
    )


class CheckerSelfTests(unittest.TestCase):
    def test_every_checker_passes_its_self_test(self) -> None:
        for tool, *_ in MATRIX:
            with self.subTest(tool=tool):
                proc = run("python3", TOOLS / tool, "--self-test")
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class CheckerFixtureTests(unittest.TestCase):
    def test_good_passes_and_bad_fails_with_documented_code(self) -> None:
        for tool, good, bad, code, message in MATRIX:
            with self.subTest(tool=tool, fixture=str(good[-1])):
                proc = checker(tool, *good)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            with self.subTest(tool=tool, fixture=str(bad[-1])):
                proc = checker(tool, *bad)
                self.assertEqual(proc.returncode, code, proc.stdout + proc.stderr)
                self.assertIn(message, proc.stdout + proc.stderr)

    def test_section_not_found_is_exit_3(self) -> None:
        proc = checker(
            "check-anchors.py",
            "--register",
            FIXTURES / "good-section.md",
            "NO SUCH SECTION",
        )
        self.assertEqual(proc.returncode, 3)
        self.assertIn("section-not-found", proc.stderr)

    def test_duplicate_id_is_exit_4(self) -> None:
        text = (
            (FIXTURES / "missing-alignment.md")
            .read_text(encoding="utf-8")
            .replace("CD-903", "CD-902", 1)
        )
        dup = AUDIT / "tools" / "tests" / ".dup-fixture.md"
        dup.write_text(text, encoding="utf-8")
        try:
            proc = checker("fields-check.py", "--register", dup)
        finally:
            dup.unlink()
        self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)
        self.assertIn("duplicate-id", proc.stderr)


class DryRunTests(unittest.TestCase):
    def test_dry_run_prints_invocation_and_creates_nothing(self) -> None:
        self.assertTrue(DECK.is_dir(), f"reference deck missing at {DECK}")
        measurements = AUDIT / "measurements"
        before_tree = tree_snapshot(measurements)
        before_deck = deck_digest(DECK)
        proc = run("bash", TOOLS / "perf-run.sh", "--dry-run", "CAL", "4t")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertRegex(proc.stdout, r"cobre run .*--threads 4( |$)")
        self.assertIn("--quiet", proc.stdout)
        self.assertIn("measurements/CAL", proc.stdout)
        self.assertEqual(tree_snapshot(measurements), before_tree)
        self.assertEqual(deck_digest(DECK), before_deck)
        env = measurements / "CAL" / "env.txt"
        if env.exists() and "deck_sha256" in env.read_text(encoding="utf-8"):
            self.assertIn(
                before_deck,
                env.read_text(encoding="utf-8"),
                "deck mutated since calibration",
            )

    def test_deck_missing_is_exit_3_without_a_solve(self) -> None:
        env = dict(os.environ, HOME="/nonexistent")
        proc = run("bash", TOOLS / "perf-run.sh", "--dry-run", "CAL", "4t", env=env)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("missing", proc.stderr)


class BaselineLineTests(unittest.TestCase):
    def baseline(self) -> str:
        m = re.search(
            r"^Baseline: ([0-9a-f]{8,40}) ", BACKLOG.read_text(encoding="utf-8"), re.M
        )
        self.assertIsNotNone(m, "no column-0 Baseline line in BACKLOG.md")
        return m.group(1)

    def test_baseline_is_head_or_an_evaluated_surface_equal_ancestor(self) -> None:
        pinned = self.baseline()
        head = run("git", "rev-parse", "HEAD").stdout.strip()
        if head.startswith(pinned):
            return
        self.assertEqual(
            run("git", "merge-base", "--is-ancestor", pinned, "HEAD").returncode,
            0,
            f"baseline-drift pinned={pinned} head={head}: pin is not an ancestor of HEAD",
        )
        diff = run("git", "diff", "--stat", pinned, "HEAD", "--", *EVALUATED)
        self.assertEqual(
            diff.stdout.strip(),
            "",
            f"baseline-drift pinned={pinned} head={head}:\n{diff.stdout}",
        )

    def test_milestones_block_and_bound(self) -> None:
        text = BACKLOG.read_text(encoding="utf-8")
        self.assertRegex(
            text, re.compile(r"^#+ .*Milestones", re.M), "Milestones block missing"
        )
        self.assertRegex(text, r"`0a` *< *`0b` *< *`1`", "0a < 0b < 1 not declared")
        self.assertIn("gnl-import", text)
        self.assertRegex(
            text,
            re.compile(r"^Protocol bound: [0-9]+(\.[0-9]+)? s", re.M),
            "numeric wall-time bound missing",
        )

    def test_second_pin_run_is_byte_identical(self) -> None:
        pinned = self.baseline()
        full = run("git", "rev-parse", pinned).stdout.strip()
        before = BACKLOG.read_bytes()
        proc = run("bash", TOOLS / "pin-baseline.sh", full)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(BACKLOG.read_bytes(), before)

    def test_moving_the_pin_without_repin_is_exit_4_and_writes_nothing(self) -> None:
        pinned = self.baseline()
        other = FIRST_PIN if not FIRST_PIN.startswith(pinned) else SECOND_PIN
        before = BACKLOG.read_bytes()
        proc = run("bash", TOOLS / "pin-baseline.sh", other)
        self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)
        self.assertIn("--repin", proc.stderr)
        self.assertEqual(BACKLOG.read_bytes(), before)

    def test_superseded_pin_is_recorded_and_each_entry_keeps_its_own(self) -> None:
        text = BACKLOG.read_text(encoding="utf-8")
        if self.baseline().startswith(FIRST_PIN[:8]):
            self.skipTest("register still on its first pin")
        self.assertRegex(
            text,
            re.compile(
                rf"^Previous baselines: {FIRST_PIN[:8]} \(pinned 2026-09-05, superseded ",
                re.M,
            ),
        )
        lines = text.splitlines()
        for slug in ("core-io", "stochastic"):
            section = bp.find_section(lines, slug)
            pins = bp.entry_baselines(section.lines)
            self.assertTrue(pins, f"{slug}: no entry carries a Baseline field")
            self.assertEqual(
                {p[:8] for p in pins.values()},
                {FIRST_PIN[:8]},
                f"{slug}: a ratified entry moved off the pin it was evaluated at",
            )


class AnchorBaselineTests(unittest.TestCase):
    """An anchor resolves at its entry's Baseline; provenance bullets resolve at the register pin."""

    SECTION = [
        "**CD-901 · Sev C · duplication · effort S · confidence high**",
        "A type the Tier-4/5 wave later deleted.",
        f"- **Baseline:** `{FIRST_PIN[:8]}`",
        "- **Anchors:** `crates/cobre-core/src/topology/network.rs::NetworkTopology`",
        "- **Evidence:** `crates/cobre-core/src/topology/network.rs:91`",
        "- **Alignment:** neutral",
        "- **Status:** fixed (2026-09-15) — hoisted into `crates/cobre-core/src/test_support.rs`;",
        "  the `crates/cobre-io/src/test_support.rs` copy followed.",
        "- **Correction (2026-09-15):** see `crates/cobre-io/tests/helpers/mod.rs`.",
        "",
        "**CD-902 · Sev C · duplication · effort S · confidence high**",
        "An entry without a Baseline field resolves at the register pin: `Cargo.toml`.",
        "- **Anchors:** `crates/cobre-core/src/lib.rs`",
    ]

    def anchors(self) -> list[bp.Anchor]:
        return bp.parse_anchors(self.SECTION)

    def test_entry_baselines_reads_the_first_sha_of_the_field(self) -> None:
        self.assertEqual(bp.entry_baselines(self.SECTION), {"CD-901": FIRST_PIN[:8]})

    def test_anchors_and_evidence_carry_the_entry_pin(self) -> None:
        pins = {a.raw: a.baseline for a in self.anchors()}
        self.assertEqual(
            pins["crates/cobre-core/src/topology/network.rs::NetworkTopology"],
            FIRST_PIN[:8],
        )
        self.assertEqual(
            pins["crates/cobre-core/src/topology/network.rs:91"], FIRST_PIN[:8]
        )

    def test_provenance_bullets_and_their_continuations_carry_no_pin(self) -> None:
        pins = {a.raw: a.baseline for a in self.anchors()}
        self.assertIsNone(pins["crates/cobre-core/src/test_support.rs"])
        self.assertIsNone(pins["crates/cobre-io/src/test_support.rs"])
        self.assertIsNone(pins["crates/cobre-io/tests/helpers/mod.rs"])
        self.assertIsNone(pins["crates/cobre-core/src/lib.rs"])

    def test_checker_resolves_a_deleted_symbol_at_the_entry_pin_only(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        fixture = pathlib.Path(scratch.name) / "entry-baseline.md"
        fixture.write_text(
            "## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — entry-baseline\n\n"
            + "\n".join(self.SECTION)
            + "\n",
            encoding="utf-8",
        )
        own_pin = checker(
            "check-anchors.py",
            "--allow-drift",
            "--baseline",
            SECOND_PIN,
            "--register",
            fixture,
            "entry-baseline",
        )
        self.assertEqual(own_pin.returncode, 0, own_pin.stdout + own_pin.stderr)
        self.assertIn("checked 6 anchors, 0 failing", own_pin.stdout)
        flattened = [
            line for line in self.SECTION if not line.startswith("- **Baseline:**")
        ]
        fixture.write_text(
            "## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — entry-baseline\n\n"
            + "\n".join(flattened)
            + "\n",
            encoding="utf-8",
        )
        register_pin = checker(
            "check-anchors.py",
            "--allow-drift",
            "--baseline",
            SECOND_PIN,
            "--register",
            fixture,
            "entry-baseline",
        )
        self.assertEqual(register_pin.returncode, 1)
        self.assertIn("NetworkTopology (entry CD-901", register_pin.stdout)
        self.assertIn(f"at {SECOND_PIN[:8]}", register_pin.stdout)


class PythonAnchorTests(unittest.TestCase):
    """`.py` anchors resolve def / class / column-0 constant declarations and nothing else."""

    GOOD = [
        "**CD-903 · Sev C · duplication · effort S · confidence high**",
        "A pytest fixture and a module constant of the cobre-python test corpus.",
        f"- **Baseline:** `{SECOND_PIN[:8]}`",
        "- **Anchors:** `crates/cobre-python/tests/conftest.py::cli_binary` "
        "`crates/cobre-python/tests/test_contract_output_parity.py::CONTRACT_SCHEMA_FIELDS`",
        "- **Alignment:** neutral",
    ]
    BAD = [
        "**CD-904 · Sev C · duplication · effort S · confidence high**",
        "An imported module name and a name that does not exist are not declarations.",
        f"- **Baseline:** `{SECOND_PIN[:8]}`",
        "- **Anchors:** `crates/cobre-python/tests/conftest.py::pytest` "
        "`crates/cobre-python/tests/conftest.py::no_such_def`",
        "- **Alignment:** neutral",
    ]

    def run_fixture(self, section: list[str]) -> subprocess.CompletedProcess[str]:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        fixture = pathlib.Path(scratch.name) / "py-anchor.md"
        fixture.write_text(
            "## ★ QUALITY EVALUATION (2026-09, baseline FIXTURE) — py-anchor\n\n"
            + "\n".join(section)
            + "\n",
            encoding="utf-8",
        )
        return checker(
            "check-anchors.py",
            "--allow-drift",
            "--baseline",
            SECOND_PIN,
            "--register",
            fixture,
            "py-anchor",
        )

    def test_def_and_module_constant_resolve(self) -> None:
        good = self.run_fixture(self.GOOD)
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        self.assertIn("checked 2 anchors, 0 failing", good.stdout)

    def test_non_declarations_are_symbol_missing(self) -> None:
        bad = self.run_fixture(self.BAD)
        self.assertEqual(bad.returncode, 1, bad.stdout + bad.stderr)
        self.assertIn("checked 2 anchors, 2 failing", bad.stdout)
        self.assertIn("pytest", bad.stdout)
        self.assertIn("no_such_def", bad.stdout)

    def test_station_checks_resolver_agrees_with_the_register_checker(self) -> None:
        from lib import station_checks as sc

        tree = sc.Tree(SECOND_PIN)
        self.assertTrue(
            sc.anchor_exists(
                "`crates/cobre-python/tests/conftest.py::cli_binary`", tree
            )
        )
        self.assertTrue(
            sc.anchor_exists(
                "`crates/cobre-python/tests/test_contract_output_parity.py::CONTRACT_SCHEMA_FIELDS`",
                tree,
            )
        )
        self.assertFalse(
            sc.anchor_exists("`crates/cobre-python/tests/conftest.py::pytest`", tree)
        )
        self.assertFalse(
            sc.anchor_exists(
                "`crates/cobre-python/tests/conftest.py::no_such_def`", tree
            )
        )


class ReadOnlyMirrorTests(unittest.TestCase):
    def test_the_mirror_is_not_a_read_only_offender(self) -> None:
        sys.path.insert(0, str(TOOLS))
        import station_verify  # noqa: PLC0415

        worktree = [f" M {MIRROR}", " M .gitignore", " M crates/cobre-core/src/lib.rs"]
        committed = [MIRROR, "crates/cobre-io/src/lib.rs"]
        wt, cm = station_verify.readonly_offenders(worktree, committed)
        self.assertEqual(wt, ["M crates/cobre-core/src/lib.rs"])
        self.assertEqual(cm, ["crates/cobre-io/src/lib.rs"])

    def test_drift_test_ignores_the_mirror(self) -> None:
        self.assertIn(f":(exclude){MIRROR}", bp.SURFACE_PATHSPEC)


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_evaluation_surface_is_dirty(self) -> None:
        proc = run(
            "git",
            "status",
            "--porcelain",
            "--",
            "crates",
            "docs",
            "scripts",
            ".github",
            "schemas",
            "Cargo.toml",
        )
        self.assertEqual(proc.stdout.strip(), "", f"tracked-path-dirty:\n{proc.stdout}")


if __name__ == "__main__":
    unittest.main()
