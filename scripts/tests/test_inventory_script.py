#!/usr/bin/env python3
"""Unit tests for scripts/test_inventory.py.

Run via:
    python3 -m unittest scripts/tests/test_inventory_script.py
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path

# Make the scripts package importable when run from repo root
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import test_inventory as inv


def _run_on_fixture(rs_source: str, filename: str = "fixture.rs") -> list[dict[str, str]]:
    """Write source to temp, run inventory, return CSV rows."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # Minimal Cargo.toml so --repo-root validation passes
        (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")

        # Place the fixture inside crates/cobre-fixture/src/
        crate_src = tmp / "crates" / "cobre-fixture" / "src"
        crate_src.mkdir(parents=True)
        (crate_src / filename).write_text(rs_source, encoding="utf-8")

        # Capture stdout CSV
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        try:
            inv.main(
                [
                    "--repo-root", str(tmp),
                    "--crates", "cobre-fixture",
                ]
            )
        except SystemExit:
            pass
        finally:
            sys.stdout = old_stdout

        buf.seek(0)
        reader = csv.DictReader(buf)
        return list(reader)


def _run_on_fixture_tests_dir(rs_source: str, filename: str = "foo.rs") -> list[dict[str, str]]:
    """Write source to tests/, run inventory, return CSV rows."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")

        crate_tests = tmp / "crates" / "cobre-fixture" / "tests"
        crate_tests.mkdir(parents=True)
        (crate_tests / filename).write_text(rs_source, encoding="utf-8")

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        try:
            inv.main(
                [
                    "--repo-root", str(tmp),
                    "--crates", "cobre-fixture",
                ]
            )
        except SystemExit:
            pass
        finally:
            sys.stdout = old_stdout

        buf.seek(0)
        reader = csv.DictReader(buf)
        return list(reader)


class TestSingleTopLevelTest(unittest.TestCase):
    """Single #[test] in tests/foo.rs (no enclosing mod)."""

    _RS = """\
#[test]
fn foo() {
    assert!(true);
}
"""

    def test_single_row_emitted(self) -> None:
        rows = _run_on_fixture_tests_dir(self._RS, filename="foo.rs")
        self.assertEqual(len(rows), 1, f"Expected 1 row, got {len(rows)}: {rows}")

    def test_function_name_correct(self) -> None:
        rows = _run_on_fixture_tests_dir(self._RS, filename="foo.rs")
        self.assertEqual(rows[0]["function"], "foo")

    def test_body_loc_correct(self) -> None:
        rows = _run_on_fixture_tests_dir(self._RS, filename="foo.rs")
        self.assertEqual(int(rows[0]["body_loc"]), 3)

    def test_test_module_empty(self) -> None:
        rows = _run_on_fixture_tests_dir(self._RS, filename="foo.rs")
        self.assertEqual(rows[0]["test_module"], "")

    def test_file_field_ends_with_foo_rs(self) -> None:
        rows = _run_on_fixture_tests_dir(self._RS, filename="foo.rs")
        self.assertTrue(
            rows[0]["file"].endswith("foo.rs"),
            f"file field should end with foo.rs, got: {rows[0]['file']}",
        )


class TestTwoTestsInsideModTests(unittest.TestCase):
    """Two #[test] functions inside mod tests."""

    _RS = """\
fn production_code() {}

mod tests {
    use super::*;

    #[test]
    fn alpha() {
        assert_eq!(1, 1);
    }

    #[test]
    fn beta() {
        let x = 1;
        let y = 2;
        assert_eq!(x + y, 3);
        assert!(true);
        let _z = x + y;
        assert_ne!(x, y);
        assert!(x < y);
        assert!(y > x);
        assert_eq!(x * y, 2);
        // end
    }
}
"""

    def test_two_rows_emitted(self) -> None:
        rows = _run_on_fixture(self._RS)
        self.assertEqual(len(rows), 2, f"Expected 2 rows, got {len(rows)}: {rows}")

    def test_function_names(self) -> None:
        rows = _run_on_fixture(self._RS)
        names = {r["function"] for r in rows}
        self.assertIn("alpha", names)
        self.assertIn("beta", names)

    def test_test_module_is_tests_for_both(self) -> None:
        rows = _run_on_fixture(self._RS)
        for row in rows:
            self.assertEqual(
                row["test_module"],
                "tests",
                f"Expected test_module='tests' for {row['function']}, got '{row['test_module']}'",
            )

    def test_body_loc_alpha(self) -> None:
        rows = _run_on_fixture(self._RS)
        alpha = next(r for r in rows if r["function"] == "alpha")
        self.assertEqual(int(alpha["body_loc"]), 3)

    def test_body_loc_beta(self) -> None:
        rows = _run_on_fixture(self._RS)
        beta = next(r for r in rows if r["function"] == "beta")
        self.assertEqual(int(beta["body_loc"]), 12)

    def test_csv_header_correct(self) -> None:
        """CSV header must match the required column order."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")
            crate_src = tmp / "crates" / "cobre-fixture" / "src"
            crate_src.mkdir(parents=True)
            (crate_src / "lib.rs").write_text(self._RS, encoding="utf-8")

            out_file = tmp / "out.csv"
            try:
                inv.main(
                    [
                        "--repo-root", str(tmp),
                        "--crates", "cobre-fixture",
                        "--output", str(out_file),
                    ]
                )
            except SystemExit:
                pass

            with out_file.open(encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader)

            self.assertEqual(
                header,
                ["crate", "file", "line", "function", "body_loc",
                 "test_module", "category", "guards", "notes"],
            )


class TestStringLiteralBraceSkip(unittest.TestCase):
    """#[test] body with "}" in string (state machine exercise)."""

    _RS = '''\
#[test]
fn tricky() {
    let s = "}";
    assert_eq!(s, "}");
}
'''

    def test_single_row_emitted(self) -> None:
        rows = _run_on_fixture(self._RS)
        self.assertEqual(len(rows), 1, f"Expected 1 row, got {len(rows)}: {rows}")

    def test_function_name(self) -> None:
        rows = _run_on_fixture(self._RS)
        self.assertEqual(rows[0]["function"], "tricky")

    def test_body_loc_correct_despite_string_braces(self) -> None:
        rows = _run_on_fixture(self._RS)
        self.assertEqual(int(rows[0]["body_loc"]), 4)

    def test_raw_string_brace_skip(self) -> None:
        rs = '''\
#[test]
fn raw_str_test() {
    let s = r#"closing brace: }"#;
    assert!(!s.is_empty());
}
'''
        rows = _run_on_fixture(rs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["function"], "raw_str_test")
        self.assertEqual(int(rows[0]["body_loc"]), 4)


class TestSlowMarkerFlag(unittest.TestCase):
    """--include-slow-marker flag adds slow_marker column."""

    _RS = """\
#[ignore]
#[test]
fn slow_test() {
    assert!(true);
}

#[test]
fn fast_test() {
    assert!(true);
}
"""

    def _rows_with_slow(self) -> list[dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")
            crate_src = tmp / "crates" / "cobre-fixture" / "src"
            crate_src.mkdir(parents=True)
            (crate_src / "lib.rs").write_text(self._RS, encoding="utf-8")

            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf

            try:
                inv.main(
                    [
                        "--repo-root", str(tmp),
                        "--crates", "cobre-fixture",
                        "--include-slow-marker",
                    ]
                )
            except SystemExit:
                pass
            finally:
                sys.stdout = old_stdout

            buf.seek(0)
            return list(csv.DictReader(buf))

    def test_slow_marker_column_present(self) -> None:
        rows = self._rows_with_slow()
        self.assertTrue(
            all("slow_marker" in r for r in rows),
            "slow_marker column missing from rows",
        )

    def test_ignore_attr_captured(self) -> None:
        rows = self._rows_with_slow()
        slow_row = next((r for r in rows if r["function"] == "slow_test"), None)
        self.assertIsNotNone(slow_row)
        assert slow_row is not None
        self.assertIn("#[ignore]", slow_row["slow_marker"])

    def test_fast_test_no_marker(self) -> None:
        rows = self._rows_with_slow()
        fast_row = next((r for r in rows if r["function"] == "fast_test"), None)
        self.assertIsNotNone(fast_row)
        assert fast_row is not None
        self.assertEqual(fast_row["slow_marker"], "")


class TestCratesFilter(unittest.TestCase):
    """--crates flag limits output to named crates."""

    def test_only_named_crate_in_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")

            rs_src = "#[test]\nfn t() { assert!(true); }\n"

            for crate in ("cobre-alpha", "cobre-beta"):
                d = tmp / "crates" / crate / "src"
                d.mkdir(parents=True)
                (d / "lib.rs").write_text(rs_src, encoding="utf-8")

            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf

            try:
                inv.main(
                    [
                        "--repo-root", str(tmp),
                        "--crates", "alpha",  # no cobre- prefix
                    ]
                )
            except SystemExit:
                pass
            finally:
                sys.stdout = old_stdout

            buf.seek(0)
            rows = list(csv.DictReader(buf))

        self.assertTrue(all(r["crate"] == "cobre-alpha" for r in rows))
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
