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


def _run_on_fixture_dir(
    rs_source: str, filename: str = "fixture.rs", subdir: str = "src"
) -> list[dict[str, str]]:
    """Write source to temp crate, run inventory, return CSV rows."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")
        crate_dir = tmp / "crates" / "cobre-fixture" / subdir
        crate_dir.mkdir(parents=True)
        (crate_dir / filename).write_text(rs_source, encoding="utf-8")
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            inv.main(["--repo-root", str(tmp), "--crates", "cobre-fixture"])
        except SystemExit:
            pass
        finally:
            sys.stdout = old_stdout
        buf.seek(0)
        return list(csv.DictReader(buf))


def _run_on_fixture(rs_source: str, filename: str = "fixture.rs") -> list[dict[str, str]]:
    """Write source to src/, run inventory, return CSV rows."""
    return _run_on_fixture_dir(rs_source, filename, "src")


def _run_on_fixture_tests_dir(rs_source: str, filename: str = "foo.rs") -> list[dict[str, str]]:
    """Write source to tests/, run inventory, return CSV rows."""
    return _run_on_fixture_dir(rs_source, filename, "tests")


class TestSingleTopLevelTest(unittest.TestCase):
    """Single #[test] in tests/foo.rs (no enclosing mod)."""

    _RS = """\
#[test]
fn foo() {
    assert!(true);
}
"""

    def setUp(self) -> None:
        self.rows = _run_on_fixture_tests_dir(self._RS, filename="foo.rs")

    def test_single_row_emitted(self) -> None:
        self.assertEqual(len(self.rows), 1)

    def test_function_name_correct(self) -> None:
        self.assertEqual(self.rows[0]["function"], "foo")

    def test_body_loc_correct(self) -> None:
        self.assertEqual(int(self.rows[0]["body_loc"]), 3)

    def test_test_module_empty(self) -> None:
        self.assertEqual(self.rows[0]["test_module"], "")

    def test_file_field_ends_with_foo_rs(self) -> None:
        self.assertTrue(self.rows[0]["file"].endswith("foo.rs"))


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

    def setUp(self) -> None:
        self.rows = _run_on_fixture(self._RS)

    def test_two_rows_emitted(self) -> None:
        self.assertEqual(len(self.rows), 2)

    def test_function_names(self) -> None:
        names = {r["function"] for r in self.rows}
        self.assertIn("alpha", names)
        self.assertIn("beta", names)

    def test_test_module_is_tests_for_both(self) -> None:
        for row in self.rows:
            self.assertEqual(row["test_module"], "tests")

    def test_body_loc_alpha(self) -> None:
        alpha = next(r for r in self.rows if r["function"] == "alpha")
        self.assertEqual(int(alpha["body_loc"]), 3)

    def test_body_loc_beta(self) -> None:
        beta = next(r for r in self.rows if r["function"] == "beta")
        self.assertEqual(int(beta["body_loc"]), 12)

    def test_csv_header_correct(self) -> None:
        """CSV header matches expected column order."""
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            (tmp / "Cargo.toml").write_text('[workspace]\nmembers = []\n', encoding="utf-8")
            crate_src = tmp / "crates" / "cobre-fixture" / "src"
            crate_src.mkdir(parents=True)
            (crate_src / "lib.rs").write_text(self._RS, encoding="utf-8")
            out_file = tmp / "out.csv"
            try:
                inv.main([
                    "--repo-root", str(tmp),
                    "--crates", "cobre-fixture",
                    "--output", str(out_file),
                ])
            except SystemExit:
                pass
            with out_file.open(encoding="utf-8") as fh:
                header = next(csv.reader(fh))
            self.assertEqual(header, [
                "crate", "file", "line", "function", "body_loc",
                "test_module", "category", "guards", "notes",
            ])


class TestStringLiteralBraceSkip(unittest.TestCase):
    """#[test] body with "}" in string (state machine exercise)."""

    _RS = '''\
#[test]
fn tricky() {
    let s = "}";
    assert_eq!(s, "}");
}
'''

    def setUp(self) -> None:
        self.rows = _run_on_fixture(self._RS)

    def test_single_row_emitted(self) -> None:
        self.assertEqual(len(self.rows), 1)

    def test_function_name(self) -> None:
        self.assertEqual(self.rows[0]["function"], "tricky")

    def test_body_loc_correct_despite_string_braces(self) -> None:
        self.assertEqual(int(self.rows[0]["body_loc"]), 4)

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

    def setUp(self) -> None:
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
                inv.main([
                    "--repo-root", str(tmp),
                    "--crates", "cobre-fixture",
                    "--include-slow-marker",
                ])
            except SystemExit:
                pass
            finally:
                sys.stdout = old_stdout
            buf.seek(0)
            self.rows = list(csv.DictReader(buf))

    def test_slow_marker_column_present(self) -> None:
        self.assertTrue(all("slow_marker" in r for r in self.rows))

    def test_ignore_attr_captured(self) -> None:
        slow_row = next((r for r in self.rows if r["function"] == "slow_test"), None)
        self.assertIsNotNone(slow_row)
        self.assertIn("#[ignore]", slow_row["slow_marker"])

    def test_fast_test_no_marker(self) -> None:
        fast_row = next((r for r in self.rows if r["function"] == "fast_test"), None)
        self.assertIsNotNone(fast_row)
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
                inv.main(["--repo-root", str(tmp), "--crates", "alpha"])
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
