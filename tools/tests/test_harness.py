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
import unittest

ROOT = pathlib.Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                                   text=True, check=True).stdout.strip())
AUDIT = ROOT / "plans" / "architecture-debt-audit"
TOOLS = AUDIT / "tools"
FIXTURES = TOOLS / "fixtures"
BACKLOG = AUDIT / "BACKLOG.md"
DECK = pathlib.Path(os.environ.get("COBRE_PERF_DECK", str(pathlib.Path.home() / "git/cobre-bridge/example/cobre_reduzido_2")))
EVALUATED = ["crates", "docs", "scripts", ".github", "schemas", "Cargo.toml", "Cargo.lock", "examples", "tests"]

# checker, argv for the good fixture, argv for the seeded-bad fixture, expected bad exit, expected bad message
MATRIX = [
    ("check-anchors.py", ["--register", FIXTURES / "good-section.md", "good-section"],
     ["--register", FIXTURES / "bad-anchor.md", "bad-anchor"], 1, "anchor-missing"),
    ("check-reraise.py", [FIXTURES / "good-section.md"], [FIXTURES / "reraise-seeded.md"], 1, "unjustified re-raise"),
    ("fields-check.py", [FIXTURES / "reraise-seeded.md"], [FIXTURES / "missing-alignment.md"], 1, "alignment-invalid"),
    ("check-roadmap-dag.py", ["--backlog", FIXTURES / "good-roadmap.md"],
     ["--backlog", FIXTURES / "cyclic-roadmap.md"], 1, "cycle"),
]


def run(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(a) for a in args], cwd=ROOT, capture_output=True, text=True,
                          check=False, env=env)


def checker(tool: str, *args) -> subprocess.CompletedProcess[str]:
    return run("python3", TOOLS / tool, *args)


def deck_digest(path: pathlib.Path) -> str:
    """Byte-for-byte the digest perf-run.sh stamps: sha256 over `sha256sum` lines of the sorted absolute paths."""
    h = hashlib.sha256()
    for f in sorted((str(p) for p in path.rglob("*") if p.is_file()), key=lambda x: x.encode()):
        h.update(f"{hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()}  {f}\n".encode())
    return h.hexdigest()


def tree_snapshot(path: pathlib.Path) -> list[tuple[str, int]]:
    return sorted((str(p.relative_to(path)), p.stat().st_size) for p in path.rglob("*") if p.is_file())


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
        proc = checker("check-anchors.py", "--register", FIXTURES / "good-section.md", "NO SUCH SECTION")
        self.assertEqual(proc.returncode, 3)
        self.assertIn("section-not-found", proc.stderr)

    def test_duplicate_id_is_exit_4(self) -> None:
        text = (FIXTURES / "missing-alignment.md").read_text(encoding="utf-8").replace("CD-903", "CD-902", 1)
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
            self.assertIn(before_deck, env.read_text(encoding="utf-8"), "deck mutated since calibration")

    def test_deck_missing_is_exit_3_without_a_solve(self) -> None:
        env = dict(os.environ, HOME="/nonexistent")
        proc = run("bash", TOOLS / "perf-run.sh", "--dry-run", "CAL", "4t", env=env)
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        self.assertIn("missing", proc.stderr)


class BaselineLineTests(unittest.TestCase):
    def baseline(self) -> str:
        m = re.search(r"^Baseline: ([0-9a-f]{8,40}) ", BACKLOG.read_text(encoding="utf-8"), re.M)
        self.assertIsNotNone(m, "no column-0 Baseline line in BACKLOG.md")
        return m.group(1)

    def test_baseline_is_head_or_an_evaluated_surface_equal_ancestor(self) -> None:
        pinned = self.baseline()
        head = run("git", "rev-parse", "HEAD").stdout.strip()
        if head.startswith(pinned):
            return
        self.assertEqual(run("git", "merge-base", "--is-ancestor", pinned, "HEAD").returncode, 0,
                         f"baseline-drift pinned={pinned} head={head}: pin is not an ancestor of HEAD")
        diff = run("git", "diff", "--stat", pinned, "HEAD", "--", *EVALUATED)
        self.assertEqual(diff.stdout.strip(), "", f"baseline-drift pinned={pinned} head={head}:\n{diff.stdout}")

    def test_milestones_block_and_bound(self) -> None:
        text = BACKLOG.read_text(encoding="utf-8")
        self.assertRegex(text, re.compile(r"^#+ .*Milestones", re.M), "Milestones block missing")
        self.assertRegex(text, r"`0a` *< *`0b` *< *`1`", "0a < 0b < 1 not declared")
        self.assertIn("gnl-import", text)
        self.assertRegex(text, re.compile(r"^Protocol bound: [0-9]+(\.[0-9]+)? s", re.M), "numeric wall-time bound missing")

    def test_second_pin_run_is_byte_identical(self) -> None:
        pinned = self.baseline()
        full = run("git", "rev-parse", pinned).stdout.strip()
        before = BACKLOG.read_bytes()
        proc = run("bash", TOOLS / "pin-baseline.sh", full, "2026-09-05")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertEqual(BACKLOG.read_bytes(), before)


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_evaluation_surface_is_dirty(self) -> None:
        proc = run("git", "status", "--porcelain", "--", "crates", "docs", "scripts", ".github", "schemas", "Cargo.toml")
        self.assertEqual(proc.stdout.strip(), "", f"tracked-path-dirty:\n{proc.stdout}")


if __name__ == "__main__":
    unittest.main()
