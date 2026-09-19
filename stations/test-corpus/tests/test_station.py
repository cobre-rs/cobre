"""Station tests for the test-corpus station (stations/test-corpus/).

InventoryTests binds the E08-1 re-measurement and the E08-2 priors to the pinned
baseline: every figure equals a fresh run of its recorded command, every prior-register
anchor and cited register id resolves at the pin, and every claim class carries its
yardstick section, a verbatim quote found at the stated lines, and a resolving anchor
into the test corpus. Later test-corpus tickets append their stage classes here.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))
from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

ID_RE = re.compile(r"\b((?:CD|PD|OD|TD)-\d{3})\b")
SWEEP_ANCHOR_RE = re.compile(
    r"`(?P<path>[A-Za-z0-9_./-]+\.(?:rs|toml|yml|md|py|json|sh|lock))"
    r"(?:::(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)|:(?P<line>\d+)(?:-(?P<line_to>\d+))?)?`"
)
BLOCK_RE = re.compile(
    r"^=== figure=(?P<id>\S+) source=(?P<source>\S+) definition=(?P<definition>.+)$"
)
NEXTEST_LINE_RE = re.compile(r"^\S+ \S+$")
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
YARDSTICK = "docs/design/testing-architecture.md"
TOOLCHAIN = {"nextest-list", "doctests", "pytest-collect"}


def load_check_anchors():
    spec = importlib.util.spec_from_file_location(
        "check_anchors", sc.TOOLS / "check-anchors.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("tools/check-anchors.py not importable")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_blocks(log: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    lines = log.splitlines()
    i = 0
    while i < len(lines):
        m = BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fid = m.group("id")
        block = {"command": "", "exit": "", "status": "", "stdout": ""}
        i += 1
        while i < len(lines) and not lines[i].startswith("stdout:"):
            for key in ("command", "exit", "status"):
                if lines[i].startswith(f"{key}: "):
                    block[key] = lines[i][len(key) + 2 :]
            i += 1
        i += 1
        out: list[str] = []
        while i < len(lines) and lines[i] != f"=== end {fid}":
            out.append(lines[i])
            i += 1
        block["stdout"] = "\n".join(out).rstrip("\n")
        blocks[fid] = block
        i += 1
    return blocks


def run(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", cmd], cwd=sc.REPO, capture_output=True, text=True, check=False
    )


class TestCorpusCase(sc.StationCase):
    SLUG = "test-corpus"
    SECTION_TITLE = "test-corpus"

    @classmethod
    def baseline(cls) -> str:
        pinned = sc.load_json(cls.station_dir() / "inventory.json")["baseline"]
        return pinned["sha"] if isinstance(pinned, dict) else str(pinned)


class InventoryTests(TestCorpusCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.claims = sc.load_json(cls.station_dir() / "claim-classes.json")
        cls.prior = (cls.station_dir() / "prior-register.md").read_text(
            encoding="utf-8"
        )
        cls.lens = (cls.station_dir() / "lens-rules.md").read_text(encoding="utf-8")
        cls.log = (cls.station_dir() / "remeasure.log").read_text(encoding="utf-8")
        cls.blocks = parse_blocks(cls.log)
        cls.sha = cls.baseline()
        cls.t = cls.tree()
        cls.figures = {f["id"]: f for f in cls.inv["figures"]}
        cls.register = backlog_parse.read_register(sc.BACKLOG)
        cls.entry_lines: dict[str, int] = {}
        for i, ln in enumerate(cls.register, 1):
            m = backlog_parse.ENTRY_RE.match(ln)
            if m:
                cls.entry_lines.setdefault(m.group("id"), i)

    def lines(self, path: str) -> list[str]:
        return self.t.read_text(path).splitlines()

    # ---- baseline ----

    def test_baseline_is_the_register_pin_everywhere(self) -> None:
        self.assertRegex(self.sha, r"^[0-9a-f]{40}$")
        self.assertTrue(sc.is_register_pin(self.sha))
        self.assertEqual(self.claims["baseline"], self.sha)
        self.assertIn(self.sha, self.prior)
        self.assertIn(self.sha, self.lens)
        blob = run(f"git rev-parse {self.sha}:{YARDSTICK}").stdout.strip()
        self.assertEqual(self.claims["sourceBlob"], blob)

    # ---- E08-1 figures ----

    def test_every_figure_is_measured_and_logged(self) -> None:
        self.assertEqual(
            self.inv["figureStatus"], {"measured": len(self.inv["figures"])}
        )
        for fid, fig in self.figures.items():
            self.assertEqual(fig["status"], "measured", fid)
            parts = [
                b for b in self.blocks.values() if b["command"] in fig["command"]
            ]
            self.assertTrue(parts, fid)
            for block in parts:
                self.assertEqual(block["exit"], "0", fid)
        self.assertEqual(self.log.count("/target/"), 0)

    def test_git_census_blocks_equal_a_fresh_run(self) -> None:
        reran = 0
        for fid, block in self.blocks.items():
            if fid in TOOLCHAIN or self.sha not in block["command"]:
                continue
            proc = run(block["command"])
            self.assertEqual(proc.returncode, int(block["exit"]), fid)
            self.assertEqual(proc.stdout.rstrip("\n"), block["stdout"], fid)
            reran += 1
        self.assertGreaterEqual(reran, 20)

    def test_toolchain_figures_equal_a_fresh_run(self) -> None:
        nextest = run(self.blocks["nextest-list"]["command"])
        self.assertEqual(nextest.returncode, 0, nextest.stderr[-2000:])
        listed = [ln for ln in nextest.stdout.splitlines() if NEXTEST_LINE_RE.match(ln)]
        self.assertEqual(len(listed), self.figures["nextest-list"]["value"])
        binaries = {ln.split()[0] for ln in listed}
        self.assertEqual(len(binaries), self.figures["nextest-list"]["listedBinaries"])

        doc = run(self.blocks["doctests"]["command"])
        self.assertEqual(doc.returncode, 0, doc.stderr[-2000:])
        running = sum(
            int(m) for m in re.findall(r"^running (\d+) tests?$", doc.stdout, re.M)
        )
        self.assertEqual(running, self.figures["doctests"]["value"])

        pytest = run(self.blocks["pytest-collect"]["command"])
        self.assertEqual(pytest.returncode, 0, pytest.stderr[-2000:])
        m = re.search(r"^(\d+) tests? collected", pytest.stdout, re.M)
        self.assertIsNotNone(m, pytest.stdout[-500:])
        self.assertEqual(int(m.group(1)), self.figures["pytest-collect"]["value"])  # type: ignore[union-attr]

    # ---- E08-2 prior register ----

    def entries(self) -> dict[str, str]:
        body = self.prior.split("## Ticket-figure deviations", 1)[0]
        parts = re.split(r"^### ", body, flags=re.M)[1:]
        return {p.splitlines()[0].strip(): p for p in parts}

    def test_prior_register_has_exactly_three_mirror_items(self) -> None:
        entries = self.entries()
        self.assertEqual(
            set(entries),
            {
                "Oracle test-harness duplication",
                "Python-binding Rust tests invisible to CI",
                "Mega-file / inline-test-giant asymmetry",
            },
        )
        mirror = self.lines(MIRROR)
        for title, body in entries.items():
            anchor_line = next(
                ln for ln in body.splitlines() if ln.startswith("**Mirror anchor.**")
            )
            self.assertIn(f"`{MIRROR}:", anchor_line, title)
            self.assertIn("## Deferred-debt register", anchor_line, title)
            self.assertTrue("### " in anchor_line or "bullet" in anchor_line, title)
            self.assertRegex(body, r"`[A-Za-z0-9_./-]+\.(?:rs|toml|yml|md):\d+`", title)
            self.assertIn("dup-of-likely", body, title)
            self.assertIn("**Owner (from mirror).**", body, title)
            self.assertIn("**Trigger (from mirror).**", body, title)
            for m in re.finditer(rf"`{re.escape(MIRROR)}:(\d+)`", anchor_line):
                line = mirror[int(m.group(1)) - 1]
                self.assertTrue(
                    line.startswith("##") or line.startswith("- **"), f"{title}: {line}"
                )
        self.assertIn("### Oracle test-harness duplication", mirror[307])
        self.assertIn("### Python-binding Rust tests invisible to CI", mirror[346])
        self.assertIn("Mega-file / inline-test-giant asymmetry", mirror[710])
        self.assertIn("## Deferred-debt register", mirror[239])
        self.assertIn("whole-lifecycle audit findings", mirror[552])

    def test_oracle_entry_names_all_four_carriers_at_pin_lines(self) -> None:
        body = self.entries()["Oracle test-harness duplication"]
        sharpening = next(
            ln
            for ln in body.splitlines()
            if ln.startswith("**Sharpening at baseline.**")
        )
        census = run(
            f"git grep -n 'fn close' {self.sha} -- crates/cobre-sddp/tests"
        ).stdout
        hits = {
            (m.group(1), int(m.group(2)))
            for m in re.finditer(r"^[0-9a-f]{40}:(crates/[^:]+):(\d+):", census, re.M)
        }
        self.assertEqual(
            hits,
            {
                ("crates/cobre-sddp/tests/extensive_form_oracle.rs", 77),
                ("crates/cobre-sddp/tests/branching_value_oracle.rs", 93),
                ("crates/cobre-sddp/tests/node_native_backward_gate.rs", 76),
                ("crates/cobre-sddp/tests/mpi_wire.rs", 2816),
            },
        )
        for path, line in hits:
            self.assertIn(f"{pathlib.PurePosixPath(path).name}:{line}", sharpening)
            self.assertIn(f"`{path}:{line}`", body)
        self.assertIn("**Disposition.** `dup-of-likely`", body)
        self.assertIsNone(re.search(r"\bTD-\d{3}\b.*\bowns this item", body))
        self.assertIn("mirror-only", body)

    def test_python_entry_records_the_superseded_premise(self) -> None:
        body = self.entries()["Python-binding Rust tests invisible to CI"]
        ci = self.lines(".github/workflows/ci.yml")
        self.assertIn(
            "cargo test --manifest-path crates/cobre-python/Cargo.toml", ci[566]
        )
        self.assertIn("Run Rust tests for the bindings crate", ci[561])
        self.assertIn("`.github/workflows/ci.yml:567`", body)
        self.assertIn("SN-06", body)
        self.assertIn("does NOT hold at the pin", body)
        manifest = self.lines("crates/cobre-python/Cargo.toml")
        self.assertIn('crate-type = ["cdylib"]', manifest[16])
        self.assertIn("[dev-dependencies]", manifest[33])
        root = self.lines("Cargo.toml")
        self.assertIn("exclude", root[20])
        self.assertIn("crates/cobre-python", root[21])

    def test_inline_giant_entry_line_counts_match_the_pin(self) -> None:
        body = self.entries()["Mega-file / inline-test-giant asymmetry"]
        lp = "crates/cobre-sddp/src/lp/builder"
        counts = {
            f: len(self.lines(f"{lp}/{f}"))
            for f in (
                "entries.rs",
                "columns.rs",
                "template/tests.rs",
                "layout/tests.rs",
                "template.rs",
                "layout.rs",
            )
        }
        self.assertEqual(counts["entries.rs"], 10093)
        self.assertEqual(counts["columns.rs"], 9319)
        self.assertEqual(counts["template/tests.rs"], 5356)
        self.assertEqual(counts["layout/tests.rs"], 3565)
        for n in counts.values():
            self.assertIn(f"{n:,}", body)
        entries = self.lines(f"{lp}/entries.rs")
        columns = self.lines(f"{lp}/columns.rs")
        self.assertEqual(
            [i + 1 for i, ln in enumerate(entries) if ln == "#[cfg(test)]"],
            [1680, 1716, 2242, 3510],
        )
        self.assertEqual(len([ln for ln in columns if ln == "#[cfg(test)]"]), 12)
        self.assertEqual(columns[1304], "#[cfg(test)]")
        self.assertIn("8,413", body)
        self.assertIn("8,014", body)
        self.assertEqual(self.lines(f"{lp}/layout.rs")[1964], "#[cfg(test)]")
        self.assertIn("CD-021", body)
        self.assertIn("CD-007", body)
        self.assertIn("TD-047", body)

    def test_prior_register_ids_exist_and_their_anchors_resolve(self) -> None:
        ids = set(ID_RE.findall(self.prior))
        self.assertTrue(
            {"CD-007", "CD-021", "TD-047", "TD-030", "TD-043", "CD-103"} <= ids
        )
        for id_ in ids:
            self.assertIn(id_, self.entry_lines, id_)
        for m in re.finditer(r"\b((?:CD|PD|OD|TD)-\d{3})\b \(`([^`]+)`", self.prior):
            self.assertTrue(sc.anchor_exists(f"`{m.group(2)}`", self.t), m.group(0))
        for m in re.finditer(r"\bL(\d{3,4})\b", self.prior):
            self.assertLessEqual(int(m.group(1)), len(self.register))

    def test_anchor_sweep_recorded_and_independently_confirmed(self) -> None:
        self.assertIn("## Anchor verification", self.prior)
        head, sweep = self.prior.split("## Anchor verification", 1)
        self.assertNotIn("anchor-missing `", sweep)
        self.assertIn("stays in the dup-of set", sweep)
        self.assertEqual(sweep.count("stays in the dup-of set"), 3)
        ca = load_check_anchors()
        swept = {m.group(0) for m in SWEEP_ANCHOR_RE.finditer(sweep)}
        for m in SWEEP_ANCHOR_RE.finditer(head):
            raw = m.group(0)
            path = m.group("path")
            if path.startswith(("plans/", "stations/")):
                continue
            self.assertIn(raw, swept, raw)
            self.assertTrue(self.t.is_file(path), raw)
            text = self.t.read_text(path)
            if m.group("symbol"):
                self.assertTrue(
                    ca.decl_pattern(path, m.group("symbol")).search(text), raw
                )
            elif m.group("line"):
                hi = int(m.group("line_to") or m.group("line"))
                self.assertLessEqual(hi, len(text.splitlines()), raw)

    def test_dup_of_set_lists_the_three_items(self) -> None:
        section = self.prior.split("## dup-of set handed to the attacker ticket", 1)[1]
        section = section.split("## Anchor verification", 1)[0]
        for title in self.entries():
            self.assertIn(f"`{title}`", section)

    # ---- E08-2 claim classes ----

    def test_claim_classes_schema_and_section_rule(self) -> None:
        rows = self.claims["rows"]
        self.assertEqual(len(rows), self.claims["rowCount"])
        self.assertGreaterEqual(len(rows), 60)
        ids = [r["id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for r in rows:
            for key in (
                "id",
                "section",
                "lines",
                "quote",
                "sentence",
                "class",
                "driftIsFinding",
                "adjudicate",
                "counterAnchors",
                "anchors",
                "note",
            ):
                self.assertIn(key, r, r["id"])
            self.assertIn(r["class"], {"current-state", "target"}, r["id"])
            self.assertEqual(
                r["driftIsFinding"], r["class"] == "current-state", r["id"]
            )
            if r["section"].startswith("2"):
                self.assertEqual(r["class"], "current-state", r["id"])
            if r["section"].startswith("5"):
                self.assertEqual(r["class"], "target", r["id"])
            if r["section"] in {"6", "7"}:
                self.assertEqual(r["class"], "target", r["id"])
            if r["adjudicate"]:
                self.assertTrue(r["counterAnchors"], r["id"])
                self.assertIn("provisional", r["note"], r["id"])
        sections = {r["section"] for r in rows}
        self.assertTrue({"2.1", "2.2", "2.3", "3.2", "5.2", "5.8", "7"} <= sections)

    def test_claim_rows_quote_the_yardstick_at_the_stated_lines(self) -> None:
        doc = self.lines(YARDSTICK)
        for r in self.claims["rows"]:
            lo, _, hi = r["lines"].partition("-")
            lo_i, hi_i = int(lo), int(hi or lo)
            self.assertLessEqual(hi_i, len(doc), r["id"])
            window = "\n".join(doc[lo_i - 1 : hi_i])
            self.assertIn(r["quote"], window, r["id"])

    def test_claim_rows_carry_a_resolving_anchor_into_the_corpus(self) -> None:
        for r in self.claims["rows"]:
            self.assertTrue(r["anchors"], r["id"])
            for a in r["anchors"]:
                self.assertTrue(sc.anchor_exists(f"`{a}`", self.t), f"{r['id']}: {a}")
            if r.get("figure"):
                self.assertIn(r["figure"], self.figures, r["id"])

    def test_adjudicate_rows_are_the_stubcomm_and_test_support_sentences(self) -> None:
        rows = {r["id"]: r for r in self.claims["rows"]}
        adjudicated = {rid for rid, r in rows.items() if r["adjudicate"]}
        self.assertEqual(
            adjudicated,
            {
                "ta-5.8-stubcomm-home",
                "ta-5.2-stubcomm-in-comm",
                "ta-5.2-keep-feature-three-crates",
                "ta-5.2-adoption-uniform",
            },
        )
        mod = self.lines("crates/cobre-sddp/tests/common/mod.rs")
        self.assertEqual(mod[31], "pub struct StubComm;")
        self.assertEqual(mod[85], "pub struct Rank0Of2;")
        self.assertNotIn(
            "test-support", self.t.read_text("crates/cobre-comm/Cargo.toml")
        )
        for rid in (
            "ta-5.8-stubcomm-home",
            "ta-5.2-stubcomm-in-comm",
            "ta-5.2-adoption-uniform",
        ):
            counters = "\n".join(rows[rid]["counterAnchors"])
            self.assertIn("crates/cobre-sddp/tests/common/mod.rs:32", counters, rid)
            self.assertIn("crates/cobre-sddp/tests/common/mod.rs:86", counters, rid)
            self.assertIn(
                "crates/cobre-comm/Cargo.toml declares no test-support", counters, rid
            )
            self.assertEqual(rows[rid]["class"], "target", rid)
            self.assertFalse(rows[rid]["driftIsFinding"], rid)
        doc = self.lines(YARDSTICK)
        self.assertIn("in `cobre-comm`'s `test-support` surface", doc[523])
        self.assertIn("Keep the `test-support` cargo-feature mechanism", doc[397])
        declarers = self.figures["test-support-declarers-vs-consumers"]["value"][
            "declarers"
        ]
        self.assertEqual(len(declarers), 5)
        self.assertNotIn("cobre-comm", declarers)
        self.assertNotIn("cobre-cli", declarers)

    # ---- E08-2 lens rules ----

    def test_lens_rules_state_both_rules_with_sources_and_candidate_tests(self) -> None:
        testing = self.lines(".claude/rules/testing.md")
        self.assertTrue(testing[39].startswith("## Contracts"))
        self.assertTrue(testing[112].startswith("## Cost discipline"))
        self.assertIn("Declaration-order invariance is a sort contract", testing[41])
        doc = self.lines(YARDSTICK)
        self.assertIn("Reducing coverage to shrink the suite", doc[601])
        self.assertIn("No test is deleted, skipped, renamed, or weakened", doc[352])
        for needle in (
            "## Rule 1",
            "## Rule 2",
            "`.claude/rules/testing.md:40`",
            "`.claude/rules/testing.md:113`",
            "`docs/design/testing-architecture.md:602-604`",
            "**Candidate test a lens applies.**",
            "NEVER a fix-shape",
            "recorded under `positives`",
            "per-**binary** and per-**feature-combo**, never per-**test**",
        ):
            self.assertIn(needle, self.lens, needle)
        self.assertEqual(self.lens.count("**Candidate test a lens applies.**"), 2)
        self.assertIn("62 of the 87", self.lens)
        self.assertEqual(self.figures["int-binaries-solver-linking"]["value"], 62)
        self.assertEqual(self.figures["int-binaries"]["value"], 87)
        self.assertIn("6,371", self.lens)
        self.assertEqual(self.figures["nextest-list"]["value"], 6371)

    def test_lens_rule_one_probe_reproduces_at_the_pin(self) -> None:
        claimers = run(
            f"git grep -il 'invarian' {self.sha} -- 'crates/*/tests/*.rs'"
        ).stdout.split()
        files = [c.split(":", 1)[1] for c in claimers]
        with_permute = {f for f in files if "permute" in self.t.read_text(f)}
        self.assertIn(f"# {len(files)} files claim some invariance", self.lens)
        self.assertIn(f"holds for {len(with_permute)}", self.lens)
        self.assertIn(
            f"The {len(files) - len(with_permute)} files without a permute call", self.lens
        )
        self.assertIn(f"The {len(with_permute)} permute-carrying files", self.lens)
        for f in with_permute:
            self.assertIn(f"`{f}`", self.lens, f)
        self.assertIn(
            "fn permute_case",
            self.lines("crates/cobre-sddp/tests/common/permute.rs")[82],
        )


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_is_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])
        out = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "crates",
                ".github",
                "docs",
                "schemas",
                "scripts",
                "Cargo.toml",
                "Cargo.lock",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
