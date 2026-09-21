"""cobre-sddp station tests: inventory census + partition, prior register.

Run from the repository root:
    python3 -m unittest plans/architecture-debt-audit/stations/sddp/tests/test_station.py
Every figure is re-derived from the tree the station evaluated (inventory.json's
`baseline`, read through `station_checks.Tree`), never from the worktree. Later sddp
tickets append their own stage classes here; the station verification runs the module.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

import station_verify  # noqa: E402
from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

SRC = "crates/cobre-sddp/src"
TESTS = "crates/cobre-sddp/tests"
SUBSTATIONS = {
    "5a": {
        "setup",
        "policy",
        "stochastic",
        "config.rs",
        "validate_phases.rs",
        "horizon_mode.rs",
    },
    "5b": {"lp"},
    "5c": {
        "cut",
        "training",
        "solve",
        "workspace",
        "convergence",
        "gemm.rs",
        "claim_scatter.rs",
        "solver_stats.rs",
    },
    "5d": {
        "simulation",
        "production",
        "hull",
        "lead_time",
        "generic_constraint_echo.rs",
        "fixed_delivery_echo.rs",
        "error.rs",
        "lib.rs",
        "test_support.rs",
    },
}
DECL = re.compile(
    r"^\s*(?P<vis>pub(\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?"
    r"(?P<kind>fn|struct|enum|trait|type|const|static|mod|impl)\s+(?P<name>[A-Za-z_]\w*)\b"
)
HEADER_BASELINE = re.compile(r"^Baseline:\s+([0-9a-f]{40})\b", re.M)
OUTLIERS_NAMED = {
    f"{SRC}/lp/builder/entries.rs",
    f"{SRC}/lp/builder/columns.rs",
}


def header_baseline() -> str:
    head = "\n".join(backlog_parse.read_register(sc.BACKLOG)[:40])
    hit = HEADER_BASELINE.search(head)
    assert hit, "BACKLOG.md header carries no `Baseline: <sha40>` line"
    return hit.group(1)


class InventoryTests(sc.StationCase):
    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.files = cls.inv["src"]["files"]
        cls.by_path = {f["path"]: f for f in cls.files}

    def test_envelope_and_baseline_come_from_the_register_header(self) -> None:
        for key in (
            "station",
            "crate",
            "baseline",
            "generatedAt",
            "src",
            "substation_rollup",
            "tests",
            "coverage",
            "commands",
            "ticketFigureDeviations",
        ):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "sddp")
        self.assertEqual(self.inv["crate"], "cobre-sddp")
        base = self.inv["baseline"]
        self.assertEqual(base, header_baseline())
        self.assertTrue(sc.is_register_pin(base), sc.pin_history())

    def test_src_set_equals_the_tree_at_the_baseline(self) -> None:
        tree = self.tree()
        actual = sorted(tree.rs_files(SRC))
        listed = [f["path"] for f in self.files]
        self.assertEqual(len(listed), len(set(listed)), "a path assigned twice")
        self.assertEqual(sorted(listed), actual)
        self.assertEqual(len(actual), 163)
        cov = self.inv["coverage"]["src"]
        self.assertEqual(cov["find_count"], len(actual))
        self.assertEqual(cov["assigned_count"], len(listed))
        self.assertEqual(cov["unassigned"], [])
        self.assertEqual(cov["double_assigned"], [])
        self.assertEqual(self.inv["src"]["file_count"], len(listed))
        self.assertEqual(
            self.inv["src"]["total_lines"], sum(f["total_lines"] for f in self.files)
        )
        self.assertEqual(
            self.inv["src"]["non_test_lines"],
            sum(f["non_test_lines"] for f in self.files),
        )

    def test_every_file_reproduces_its_lines_kind_and_symbols_at_the_baseline(
        self,
    ) -> None:
        tree = self.tree()
        for f in self.files:
            with self.subTest(path=f["path"]):
                text = tree.read_text(f["path"])
                self.assertEqual(
                    f["total_lines"], tree.read_bytes(f["path"]).count(b"\n")
                )
                phys, _ = sc.classify_lines(f["path"], text)
                self.assertEqual(f["non_test_lines"], phys)
                self.assertEqual(
                    f["inline_test_lines"], f["total_lines"] - f["non_test_lines"]
                )
                base = f["path"].rsplit("/", 1)[-1]
                if base == "tests.rs":
                    self.assertEqual(f["kind"], "sibling-test-module")
                    self.assertEqual(f["non_test_lines"], 0)
                elif base == "test_support.rs":
                    self.assertEqual(f["kind"], "test-support")
                    self.assertEqual(f["non_test_lines"], 0)
                else:
                    self.assertEqual(f["kind"], "source")
                    self.assertTrue(f["top_symbols"], "a source file with no item")
                    self.assertLess(f["non_test_lines"], f["total_lines"] + 1)
                lines = text.splitlines()
                for s in f["top_symbols"][:8]:
                    m = DECL.match(lines[s["line"] - 1])
                    self.assertIsNotNone(m, s)
                    assert m is not None
                    self.assertEqual(m.group("name"), s["name"])
                    self.assertEqual(m.group("kind"), s["kind"])
                    self.assertEqual(
                        s["visibility"], "pub" if m.group("vis") else "private"
                    )
                    self.assertTrue(
                        sc.symbol_resolves(f["path"], s["name"], tree),
                        f"{f['path']}::{s['name']} would not resolve for check-anchors",
                    )
                self.assertLessEqual(len(f["top_symbols"]), 8)

    def test_inline_test_dominated_outliers_are_recorded_as_cd_007_evidence(
        self,
    ) -> None:
        for p in OUTLIERS_NAMED:
            f = self.by_path[p]
            self.assertLess(f["non_test_lines"], f["total_lines"])
            self.assertGreater(f["inline_test_lines"], f["non_test_lines"])
        recorded = {o["path"]: o for o in self.inv["inlineTestOutliers"]}
        self.assertTrue(OUTLIERS_NAMED <= set(recorded), OUTLIERS_NAMED - set(recorded))
        for o in recorded.values():
            f = self.by_path[o["path"]]
            self.assertEqual(o["total_lines"], f["total_lines"])
            self.assertEqual(o["non_test_lines"], f["non_test_lines"])
            self.assertGreater(o["inline_test_lines"], o["non_test_lines"])
            self.assertEqual(o["evidenceFor"], "CD-007")
        self.assertEqual(
            set(recorded),
            {
                f["path"]
                for f in self.files
                if f["kind"] == "source"
                and f["inline_test_lines"] > f["non_test_lines"]
                and f["total_lines"] >= 2000
            },
        )

    def test_the_four_sub_stations_partition_the_modules(self) -> None:
        for f in self.files:
            component = f["path"][len(SRC) + 1 :].split("/", 1)[0]
            self.assertEqual(f["module"], component)
            self.assertIn(f["substation"], SUBSTATIONS)
            self.assertIn(component, SUBSTATIONS[f["substation"]], f["path"])
        rollup = self.inv["substation_rollup"]
        self.assertEqual(set(rollup), set(SUBSTATIONS))
        for sub, r in rollup.items():
            rows = [f for f in self.files if f["substation"] == sub]
            self.assertEqual(set(r["modules"]), SUBSTATIONS[sub])
            self.assertEqual(r["modules_present"], sorted({f["module"] for f in rows}))
            self.assertEqual(set(r["modules_present"]), SUBSTATIONS[sub])
            self.assertEqual(r["file_count"], len(rows))
            self.assertEqual(r["total_lines"], sum(f["total_lines"] for f in rows))
            self.assertEqual(
                r["non_test_lines"], sum(f["non_test_lines"] for f in rows)
            )
        self.assertEqual(sum(r["file_count"] for r in rollup.values()), 163)
        self.assertEqual(
            sum(r["total_lines"] for r in rollup.values()),
            self.inv["src"]["total_lines"],
        )
        census = self.inv["module_census"]
        self.assertEqual(set(census), {m for ms in SUBSTATIONS.values() for m in ms})
        for module, c in census.items():
            rows = [f for f in self.files if f["module"] == module]
            self.assertEqual(c["files"], len(rows))
            self.assertEqual(c["lines"], sum(f["total_lines"] for f in rows))

    def test_sibling_test_files_stay_in_the_partition_with_zero_non_test_lines(
        self,
    ) -> None:
        sibling = self.inv["siblingTestFiles"]
        self.assertEqual(
            sorted(sibling),
            sorted(f["path"] for f in self.files if f["path"].endswith("/tests.rs")),
        )
        self.assertIn(f"{SRC}/setup/tests.rs", sibling)
        self.assertIn(f"{SRC}/lead_time/tests.rs", sibling)
        for p in sibling:
            self.assertEqual(self.by_path[p]["non_test_lines"], 0)
            self.assertIn(self.by_path[p]["substation"], SUBSTATIONS)

    def test_tests_corpus_reproduces_its_three_way_layout(self) -> None:
        tree = self.tree()
        corpus = self.inv["tests"]
        actual = sorted(tree.rs_files(TESTS))
        self.assertEqual(sorted(t["path"] for t in corpus["files"]), actual)
        self.assertEqual(corpus["file_count"], len(actual))
        self.assertEqual(
            corpus["total_lines"], sum(t["total_lines"] for t in corpus["files"])
        )
        groups: dict[str, list[dict[str, Any]]] = {}
        for t in corpus["files"]:
            with self.subTest(path=t["path"]):
                self.assertEqual(
                    t["total_lines"], tree.read_bytes(t["path"]).count(b"\n")
                )
                parts = t["path"][len(TESTS) + 1 :].split("/")
                self.assertEqual(t["group"], "binary" if len(parts) == 1 else parts[0])
            groups.setdefault(t["group"], []).append(t)
        layout = corpus["layout"]
        self.assertEqual(layout["top_level_binaries"], len(groups["binary"]))
        self.assertEqual(
            layout["top_level_binary_lines"],
            sum(t["total_lines"] for t in groups["binary"]),
        )
        self.assertEqual(layout["common_harness"], len(groups["common"]))
        self.assertEqual(
            layout["template_integration"], len(groups["template_integration"])
        )
        self.assertEqual(layout["other_groups"], [])
        self.assertEqual(
            {p.rsplit("/", 1)[-1] for p in (t["path"] for t in groups["common"])},
            {
                "anticipated_structural_assertions.rs",
                "builders.rs",
                "mod.rs",
                "parity_hash.rs",
                "permute.rs",
            },
        )
        self.assertEqual(layout["fixtures_rs_files"], 0)
        self.assertFalse(
            [p for p in tree.ls_files(f"{TESTS}/fixtures") if p.endswith(".rs")]
        )
        self.assertNotEqual(
            corpus["file_count"],
            layout["top_level_binaries"],
            "53-as-binary-count trap: the corpus count and the binary count differ",
        )
        cov = self.inv["coverage"]["tests"]
        self.assertEqual(cov["find_count"], len(actual))
        self.assertEqual(cov["recorded_count"], corpus["file_count"])
        self.assertEqual(cov["unassigned"], [])

    def test_determinism_gates_are_informational_and_reproduce(self) -> None:
        tree = self.tree()
        gates = self.inv["tests"]["informationalGates"]
        parity = sorted(
            p for p in tree.rs_files(TESTS) if "parity_hash" in tree.read_text(p)
        )
        self.assertEqual(gates["parityHashGolden"]["files"], parity)
        self.assertIn(f"{TESTS}/common/parity_hash.rs", parity)
        self.assertIn("informational", gates["parityHashGolden"]["disposition"])
        sites = gates["slowTestsGating"]["attributeSites"]
        expected = {
            p: tree.read_text(p).count("slow-tests")
            for p in [*tree.rs_files(SRC), *tree.rs_files(TESTS)]
            if "slow-tests" in tree.read_text(p)
        }
        self.assertEqual(sites, expected)
        self.assertIn("informational", gates["slowTestsGating"]["disposition"])

    def test_loc_stats_row_reconciles_with_the_non_test_total(self) -> None:
        tree = self.tree()
        loc = self.inv["locStats"]
        self.assertEqual(loc["crate_row"], sc.loc_stats("cobre-sddp", tree))
        extras = loc["reconciliation"]["extraProductionFiles"]
        for p, n in extras.items():
            self.assertEqual(n, tree.read_bytes(p).count(b"\n"))
            self.assertFalse(p.startswith((f"{SRC}/", f"{TESTS}/")))
        self.assertEqual(
            loc["crate_row"]["prod_all"],
            self.inv["src"]["non_test_lines"] + sum(extras.values()),
        )
        self.assertEqual(
            loc["reconciliation"]["benchesBucketedAsTest"],
            sorted(tree.rs_files("crates/cobre-sddp/benches")),
        )

    def test_ticket_figure_deviations_are_real_and_the_sets_hold(self) -> None:
        dev = self.inv["ticketFigureDeviations"]
        figures = {d["figure"] for d in dev["changed"]}
        for d in dev["changed"]:
            self.assertNotEqual(d["ticket"], d["measured"], d)
        self.assertNotIn("src.file_count", figures, "the 163-file set holds")
        self.assertNotIn("substation_rollup.5b.file_count", figures)
        self.assertIn("src.total_lines", figures)
        self.assertIn("tests.file_count", figures)

    def test_lp_inventory_handoff_is_regenerated_at_the_same_baseline(self) -> None:
        lp = sc.load_json(sc.AUDIT / "measurements" / "lp-inventory.json")
        self.assertEqual(lp["baseline"], self.inv["baseline"])
        lp_paths = sorted(f["path"] for f in lp["files"])
        self.assertEqual(
            lp_paths,
            sorted(f["path"] for f in self.files if f["substation"] == "5b"),
        )
        self.assertEqual(lp["totals"]["file_count"], 30)
        self.assertEqual(
            lp["totals"]["total_lines"],
            self.inv["substation_rollup"]["5b"]["total_lines"],
        )
        self.assertEqual(
            lp["totals"]["non_test_lines"],
            self.inv["substation_rollup"]["5b"]["non_test_lines"],
        )


ROSTER_IDS = {
    4: ("CD-004", "CD-005", "CD-024-successor"),
    6: ("CD-015", "CD-022", "CD-018", "CD-035", "CD-034", "CD-023"),
    7: (
        "CD-007",
        "CD-012",
        "CD-014-remnant",
        "CD-016",
        "CD-021",
        "CD-028",
        "CD-030",
        "CD-037",
        "CD-038",
        "OD-009",
    ),
}
PRIOR_SECTIONS = (
    "## Wave 4/6/7 roster — disposition pending",
    "## Retire-with-commit — closed, cite, never re-raise",
    "## Do-not-touch — retracted, refuted, deferred, sanctioned",
    "## Reserved seams — Cleared-with-citation, never a live OD finding",
    "## Contract-first rule — `.claude/rules/sddp.md`",
    "## Byte-neutrality bar — every proposal states it",
    "## Do not re-raise",
    "## Re-derive",
)
MIRROR = "docs/design/reserved-seams-and-deferred-debt.md"
SDDP_MD = ".claude/rules/sddp.md"
ROW_RE = re.compile(r"^\| (?P<id>(?:CD|PD|OD)-\d{3}(?:-\w+)?) \| (?P<wave>\d) \| ")
ID_RE = re.compile(r"\b((?:CD|PD|OD|TD)-\d{3})\b")


def section(text: str, heading: str) -> str:
    assert heading in text, heading
    body = text.split(heading, 1)[1]
    nxt = re.search(r"^## ", body, re.M)
    return body[: nxt.start()] if nxt else body


class PriorRegisterTests(sc.StationCase):
    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (cls.station_dir() / "prior-register.md").read_text(encoding="utf-8")
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))

    def test_sections_and_baseline(self) -> None:
        self.assertIn(f"baseline `{self.baseline()[:8]}`", self.text.splitlines()[0])
        for heading in PRIOR_SECTIONS:
            self.assertIn(heading, self.text)

    def test_roster_has_the_nineteen_rows_pending_with_symbol_anchors(self) -> None:
        body = section(self.text, PRIOR_SECTIONS[0])
        table = [line for line in body.splitlines() if ROW_RE.match(line)]
        rows = {ROW_RE.match(line).group("id"): line for line in table}  # type: ignore[union-attr]
        self.assertEqual(len(table), len(rows), "duplicate roster row")
        expected = {i for ids in ROSTER_IDS.values() for i in ids}
        self.assertEqual(set(rows), expected)
        for wave, ids in ROSTER_IDS.items():
            for i in ids:
                self.assertEqual(int(ROW_RE.match(rows[i]).group("wave")), wave, i)  # type: ignore[union-attr]
        tree = self.tree()
        for i, line in rows.items():
            with self.subTest(id=i):
                self.assertTrue(line.rstrip().endswith("| _pending_ |"), line)
                anchors = sc.anchors_in(line)
                self.assertTrue(anchors, "a roster row with no anchor")
                for a in anchors:
                    self.assertNotRegex(
                        a, r":\d+`$", f"path:line anchor in the roster table: {a}"
                    )
                    self.assertTrue(sc.anchor_exists(a, tree), a)
                if "unresolved" in line:
                    self.assertRegex(line, r"unresolved — \S")
        self.assertIn("workspace/{mod,context,workspace}.rs", rows["CD-021"])
        self.assertIn("`crates/cobre-sddp/src/workspace/workspace.rs::", rows["CD-021"])
        self.assertIn("solve/", rows["CD-023"])
        self.assertIn("`crates/cobre-sddp/src/lp/builder/entries.rs::", rows["CD-007"])
        self.assertIn("10,093", rows["CD-007"])

    def test_owned_elsewhere_and_roster_count_are_stated(self) -> None:
        body = section(self.text, PRIOR_SECTIONS[0])
        self.assertRegex(
            body,
            r"\*\*Owned elsewhere — do not raise here:\*\* CD-002, CD-009 \(cobre-cli station \(E06\)\) · CD-011 \(cobre-io station",
        )
        self.assertIn("19 rows", body)
        self.assertIn("22 owned dispositions", body)
        self.assertIn("**Rule:** anchors are path + symbol, never path + line.", body)

    def test_retire_with_commit_names_the_three_items_and_their_commits_exist(
        self,
    ) -> None:
        body = section(self.text, PRIOR_SECTIONS[1])
        self.assertIn("### CD-001 — closed by `b051c410`", body)
        self.assertIn("### CD-003 (Construction hop) — closed by `4075c4e8`", body)
        self.assertIn("### CD-006 — resolved in-tree, no closing sha registered", body)
        for sha in ("b051c410", "4075c4e8"):
            self.assertEqual(
                subprocess.run(
                    ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                    cwd=sc.REPO,
                    check=False,
                ).returncode,
                0,
                sha,
            )
        tree = self.tree()
        text_all = "\n".join(tree.read_text(p) for p in tree.rs_files("crates"))
        self.assertNotRegex(text_all, r"\bfn rebuild_historical_library_non_root\b")
        self.assertNotIn("ConstructionConfig", text_all)
        self.assertNotIn("into_construction_config", text_all)
        self.assertTrue(
            sc.symbol_resolves(
                "crates/cobre-sddp/src/setup/stochastic_pipeline.rs",
                "build_stochastic_context_for_study",
                tree,
            )
        )
        node_graph = tree.read_text("crates/cobre-sddp/src/setup/node_graph.rs")
        self.assertGreaterEqual(
            len(re.findall(r"^impl NodeGraph\b", node_graph, re.M)), 1
        )
        for fn in (
            "frontier_node",
            "node_parent",
            "backward_cut_levels",
            "stage_frontier",
        ):
            self.assertIsNotNone(
                re.search(rf"^\s+pub(?:\(crate\))? fn {fn}\b", node_graph, re.M),
                f"{fn} is not an impl method",
            )
            self.assertIsNone(
                re.search(rf"^pub(?:\(crate\))? fn {fn}\b", node_graph, re.M),
                f"{fn} still has a free-fn definition",
            )
        self.assertIn("Confirmed at the baseline**: yes", body)
        self.assertNotIn("Confirmed at the baseline**: NO", body)

    def test_do_not_touch_block(self) -> None:
        body = section(self.text, PRIOR_SECTIONS[2])
        self.assertIn("### CD-008 — RETRACTED", body)
        self.assertIn("### PD-001 — REFUTED", body)
        self.assertIn("### PD-004 — DEFERRED pending a profile", body)
        self.assertIn("### sanctioned #[allow(...)] census", body)
        self.assertIn(
            "`crates/cobre-sddp/src/training/backward_pass_state.rs::run_enumerated_backward`",
            body,
        )
        self.assertTrue(
            sc.symbol_resolves(
                "crates/cobre-sddp/src/training/backward_pass_state.rs",
                "run_enumerated_backward",
                self.tree(),
            )
        )
        self.assertIn("perf epic", body)

    def test_reserved_seams_are_cleared_with_citation(self) -> None:
        body = section(self.text, PRIOR_SECTIONS[3])
        for name in (
            "LipschitzConfig.mode",
            "second-family reserved slot body",
            "Anticipated post-study-commitment channel",
            "delivery_date",
            "Legacy (`None`) cost-scale branch in `rescale_cut_records_for_load`",
        ):
            self.assertIn(name, body)
        blocks = [b for b in body.split("\n### ")[1:]]
        self.assertEqual(len(blocks), 4)
        tree = self.tree()
        mirror_lines = tree.read_text(MIRROR).splitlines()
        for b in blocks:
            with self.subTest(seam=b.splitlines()[0]):
                cites = re.findall(rf"`{re.escape(MIRROR)}:(\d+)`", b)
                self.assertTrue(cites, "seam without a mirror citation")
                self.assertLessEqual(max(int(x) for x in cites), len(mirror_lines))
        self.assertIn("**GAP: no entry for this seam at the baseline.**", body)
        self.assertIn(f'`{MIRROR}:30` "## Reserved-seam register"', body)
        self.assertEqual(mirror_lines[29].strip(), "## Reserved-seam register")
        self.assertEqual(
            mirror_lines[53].strip(),
            "### `LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`",
        )
        self.assertIn("closed as **Cleared (sanctioned)**", body)
        self.assertIn("assigns no OD id", body)
        self.assertIn(
            "`crates/cobre-sddp/src/policy/policy_load.rs::rescale_cut_records_for_load`",
            body,
        )
        self.assertIn("reference_date", body)

    def test_contract_table_headings_and_symbols_resolve_at_the_baseline(self) -> None:
        body = section(self.text, PRIOR_SECTIONS[4])
        tree = self.tree()
        sddp = tree.read_text(SDDP_MD).splitlines()
        rows = [
            line
            for line in body.splitlines()
            if line.startswith("| ") and "`.claude/rules/sddp.md:" in line
        ]
        self.assertEqual(len(rows), 8)
        for line in rows:
            with self.subTest(row=line[:60]):
                m = re.search(rf"`{re.escape(SDDP_MD)}:(\d+)` \"([^\"]+)\"", line)
                assert m is not None
                self.assertEqual(sddp[int(m.group(1)) - 1].strip(), f"## {m.group(2)}")
        for heading_line in re.findall(
            rf"^- `{re.escape(SDDP_MD)}:(\d+)` (.+)$", body, re.M
        ):
            self.assertEqual(
                sddp[int(heading_line[0]) - 1].strip(), f"## {heading_line[1]}"
            )
        self.assertEqual(
            len(re.findall(rf"^- `{re.escape(SDDP_MD)}:\d+` ", body, re.M)),
            sum(1 for line in sddp if line.startswith("## ")),
        )
        self.assertIn(
            "`crates/cobre-sddp/src/cut/basis_reconstruct.rs::reconstruct_basis`", body
        )
        self.assertIn("Policy-load compatibility validation is mandatory", body)

    def test_every_anchor_token_in_the_register_resolves_at_the_baseline(self) -> None:
        tree = self.tree()
        anchors = sc.anchors_in(self.text)
        self.assertGreater(len(anchors), 100)
        for a in anchors:
            with self.subTest(anchor=a):
                self.assertTrue(sc.anchor_exists(a, tree), a)

    def test_every_cited_id_exists_in_the_register(self) -> None:
        for i in sorted(set(ID_RE.findall(self.text))):
            with self.subTest(id=i):
                self.assertIsNotNone(
                    re.search(rf"^\*\*{i} (?:·|→|—)", self.register, re.M),
                    f"{i} has no register entry heading",
                )

    def test_byte_neutrality_and_no_fix_rule(self) -> None:
        body = section(self.text, PRIOR_SECTIONS[5])
        for item in ("parity goldens", "rank-invariance", "mpiexec -n 1/2"):
            self.assertIn(item, body)
        self.assertIn("This station executes no measurement and no fix", body)


OWNED = {
    "CD-001",
    "CD-003-construction-hop",
    "CD-004",
    "CD-005",
    "CD-006",
    "CD-007",
    "CD-012",
    "CD-014-remnant",
    "CD-015",
    "CD-016",
    "CD-018",
    "CD-021",
    "CD-022",
    "CD-023",
    "CD-024-successor",
    "CD-028",
    "CD-030",
    "CD-034",
    "CD-035",
    "CD-037",
    "CD-038",
    "OD-009",
}
NOT_OURS = {"CD-002", "CD-009", "CD-011"}
ALIGN = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
RETIRE_COMMITS = {
    "CD-001": "b051c410",
    "CD-003-construction-hop": "4075c4e8",
    "CD-006": "3f4c3db3",
}
PHASE0A_SHAPES = {
    "study-block admission-gate carrier",
    "engine-tagged setup stages",
    "rank-0-executes MPI shape",
}
NINE_WORKSPACE_STRUCTS = (
    "CapturedBasis",
    "WorkspaceSizing",
    "BackwardAccumulators",
    "ByNodeScratch",
    "ScratchBuffers",
    "SolverWorkspace",
    "WorkspacePool",
    "BasisStore",
    "BasisStoreSliceMut",
)
NODE_GRAPH_QUERIES = (
    "frontier_node",
    "stage_frontier",
    "node_parent",
    "node_opening_range",
    "node_pinned_scenario",
    "any_stage_node",
    "build_parent_map",
    "max_successor_outcome_count",
    "backward_cut_levels",
    "pool_cut_stride",
    "forward_solve_counts",
)


def render_probe(title: str, rows: list[dict[str, Any]], baseline: str) -> str:
    out = [f"## {title}", ""]
    for d in rows:
        anchors = [
            d["baselineAnchor"],
            *[a for a in d.get("anchors", []) if a != d["baselineAnchor"]],
        ]
        rid = (
            d["id"]
            .split("-construction-hop")[0]
            .split("-successor")[0]
            .split("-remnant")[0]
        )
        out += [f"**{rid} · probe · {d['id']}**", ""]
        claim = d.get("survivingClaim") or d.get("priorTitle") or d["id"]
        out += [" ".join(str(claim).split()), ""]
        out.append(
            "- **Anchors:** "
            + " ".join(f"`{a['path']}::{a['symbol']}`" for a in anchors)
        )
        out.append(f"- **Baseline:** `{baseline}`")
        overlaps = d.get("retiredOverlaps") or []
        if overlaps:
            refs = "; ".join(o["ref"].split(":", 1)[-1] for o in overlaps)
            why = " / ".join(o["justification"] for o in overlaps)
            out.append(f"- **Re-raise-of:** {refs}; NOT a re-raise — {why}")
        out.append("")
    return "\n".join(out) + "\n"


class WaveReverifyTests(sc.StationCase):
    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = sc.load_json(cls.station_dir() / "wave-dispositions.json")
        cls.handoff = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.by_id = {d["id"]: d for d in cls.env["dispositions"]}

    def test_roster_is_exactly_the_owned_set_and_disowns_the_three(self) -> None:
        ids = [d["id"] for d in self.env["dispositions"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), OWNED)
        self.assertFalse(NOT_OURS & set(ids))
        self.assertEqual({n["id"] for n in self.env["notOwned"]}, NOT_OURS)
        for n in self.env["notOwned"]:
            self.assertTrue(n["owningStation"])
        self.assertEqual(self.env["baseline"], self.baseline())
        self.assertEqual({d["wave"] for d in self.env["dispositions"]}, {4, 6, 7})

    def test_every_anchor_is_symbol_only_and_resolves_at_the_baseline(self) -> None:
        tree = self.tree()
        for d in [*self.env["dispositions"], *self.env["existenceChecks"]]:
            with self.subTest(id=d["id"]):
                for a in [d["baselineAnchor"], *d.get("anchors", [])]:
                    self.assertEqual(set(a), {"path", "symbol"}, a)
                    self.assertNotIn("line", a)
                    self.assertTrue(
                        sc.anchor_exists(f"`{a['path']}::{a['symbol']}`", tree), a
                    )
                if d in self.env["dispositions"]:
                    self.assertTrue(
                        d["priorAnchor"]["raw"], "provenance anchors dropped"
                    )

    def test_anchor_probe_and_reraise_probe_pass_through_the_harness_checkers(
        self,
    ) -> None:
        base = self.baseline()
        rows = [*self.env["dispositions"], *self.env["existenceChecks"]]
        with tempfile.TemporaryDirectory(prefix="sddp-wave-probe.") as tmp:
            stub = pathlib.Path(tmp) / "anchor-probe.md"
            title = "WAVE ANCHOR PROBE — sddp (test)"
            stub.write_text(render_probe(title, rows, base), encoding="utf-8")
            self.assertEqual(
                sc.run_checker(
                    "check-anchors.py",
                    title,
                    "--register",
                    str(stub),
                    "--baseline",
                    base,
                ),
                0,
            )
            stub_r = pathlib.Path(tmp) / "reraise-probe.md"
            title_r = "WAVE RERAISE PROBE — sddp (test)"
            live = [d for d in self.env["dispositions"] if d["disposition"] != "retire"]
            self.assertEqual(len(live), 19)
            stub_r.write_text(render_probe(title_r, live, base), encoding="utf-8")
            self.assertEqual(
                sc.run_checker(
                    "check-reraise.py",
                    title_r,
                    "--register",
                    str(stub_r),
                    "--baseline",
                    base,
                ),
                0,
            )
        self.assertEqual(self.env["probes"]["anchorProbe"]["exit"], 0)
        self.assertEqual(self.env["probes"]["reraiseProbe"]["exit"], 0)

    def test_dispositions_are_well_formed(self) -> None:
        for d in self.env["dispositions"]:
            with self.subTest(id=d["id"]):
                self.assertIn(d["disposition"], ("keep", "retire", "sharpen"))
                if d["disposition"] == "retire":
                    self.assertEqual(d["resolvingCommit"], RETIRE_COMMITS[d["id"]])
                    self.assertEqual(
                        subprocess.run(
                            [
                                "git",
                                "cat-file",
                                "-e",
                                f"{d['resolvingCommit']}^{{commit}}",
                            ],
                            cwd=sc.REPO,
                            check=False,
                        ).returncode,
                        0,
                    )
                else:
                    self.assertIsNone(d["resolvingCommit"])
                    self.assertTrue(d["survivingClaim"])
                if d["disposition"] == "sharpen":
                    self.assertTrue(d["supersededClaim"] and d["survivingClaim"])
                h = d["alignmentHint"]
                self.assertIn(h["value"], ALIGN)
                self.assertRegex(h["citation"], r"Part (IV|V)\.\d")
                self.assertTrue(h["argument"])
                if h["value"] == "conflicts":
                    self.assertTrue(h["guardrailBreach"])
                    self.assertTrue(h["roadmapConsistentAlternative"])
        self.assertEqual(set(RETIRE_COMMITS), set(self.env["byDisposition"]["retire"]))

    def test_wave_4_restatements_name_their_phase_0a_shape(self) -> None:
        restated = {
            d["id"]: d["phase0aRestatement"]
            for d in self.env["dispositions"]
            if d["wave"] == 4 and d["phase0aRestatement"]
        }
        self.assertEqual(
            set(restated),
            {"CD-004", "CD-005", "CD-003-construction-hop", "CD-024-successor"},
        )
        for rid, r in restated.items():
            with self.subTest(id=rid):
                self.assertIn(r["artifact"], PHASE0A_SHAPES)
                for key in ("shape", "guardrail", "byteNeutralityBar", "anchorOwner"):
                    self.assertTrue(r[key], key)
                self.assertIn("Engine enum stays at L4", r["guardrail"])
                self.assertIn("no engine and no paradigm", r["guardrail"])
        self.assertEqual(
            restated["CD-004"]["artifact"], "study-block admission-gate carrier"
        )
        self.assertEqual(restated["CD-005"]["artifact"], "engine-tagged setup stages")
        self.assertEqual(
            restated["CD-003-construction-hop"]["artifact"], "rank-0-executes MPI shape"
        )
        cd004 = self.by_id["CD-004"]
        self.assertEqual(cd004["disposition"], "sharpen")
        self.assertIn("ConstructionConfig", cd004["supersededClaim"])
        for name in ("BroadcastConfig", "StudyParams"):
            self.assertIn(name, cd004["survivingClaim"])
        self.assertEqual(cd004["alignmentHint"]["value"], "advances-0a")
        self.assertRegex(cd004["alignmentHint"]["citation"], r"Part V\.1")
        self.assertRegex(cd004["alignmentHint"]["citation"], r"Part IV\.4")
        self.assertEqual(
            self.by_id["CD-024-successor"]["alignmentHint"]["value"], "conflicts"
        )

    def test_retire_evidence_is_declaration_level(self) -> None:
        tree = self.tree()
        crates = "\n".join(tree.read_text(p) for p in tree.rs_files("crates"))
        self.assertNotRegex(crates, r"\bfn rebuild_historical_library_non_root\b")
        bare = [
            line
            for p in tree.rs_files("crates/cobre-sddp/src/setup")
            for line in tree.read_text(p).splitlines()
            if "rebuild_historical_library_non_root" in line
        ]
        self.assertEqual(len(bare), 3)
        self.assertTrue(all("///" in line for line in bare), bare)
        cd001 = self.by_id["CD-001"]
        self.assertEqual(len(cd001["evidence"]["docCommentResidue"]), 3)
        self.assertIn(
            "doc-comment references remain",
            cd001["evidence"]["docCommentResidueDisposition"],
        )
        self.assertTrue(
            any(
                "build_stochastic_context_for_study" in a["symbol"]
                for a in [cd001["baselineAnchor"], *cd001["anchors"]]
            )
        )
        self.assertNotIn("ConstructionConfig", crates)
        self.assertNotIn("into_construction_config", crates)
        node_graph = tree.read_text("crates/cobre-sddp/src/setup/node_graph.rs")
        for name in NODE_GRAPH_QUERIES:
            self.assertIn(f"fn {name}(&self", node_graph, name)
            self.assertIsNone(
                re.search(rf"^pub(?:\(crate\))? fn {name}\b", node_graph, re.M), name
            )
        self.assertTrue(all(self.by_id["CD-006"]["evidence"]["implMethods"].values()))

    def test_cd_021_and_cd_023_are_sharpened_not_retired(self) -> None:
        tree = self.tree()
        ws = tree.read_text("crates/cobre-sddp/src/workspace/workspace.rs")
        for t in NINE_WORKSPACE_STRUCTS:
            self.assertIsNotNone(
                re.search(rf"^(?:pub |pub\(crate\) )?struct {t}\b", ws, re.M), t
            )
        self.assertTrue(tree.is_file("crates/cobre-sddp/src/workspace/mod.rs"))
        self.assertTrue(tree.is_file("crates/cobre-sddp/src/workspace/context.rs"))
        cd021 = self.by_id["CD-021"]
        self.assertEqual(cd021["disposition"], "sharpen")
        self.assertEqual(
            cd021["baselineAnchor"]["path"],
            "crates/cobre-sddp/src/workspace/workspace.rs",
        )
        self.assertTrue(all(cd021["evidence"]["structCensus"].values()))
        self.assertEqual(cd021["evidence"]["structCount"], 9)
        for f in ("mod.rs", "partition.rs", "solver_phase.rs", "stage_solve.rs"):
            self.assertTrue(tree.is_file(f"crates/cobre-sddp/src/solve/{f}"), f)
        self.assertTrue(
            tree.is_file("crates/cobre-sddp/src/training/stage_solve_prep.rs")
        )
        self.assertTrue(
            tree.is_file("crates/cobre-sddp/src/training/stage_solve_prep/tests.rs")
        )
        self.assertIn(
            "training::stage_solve_prep",
            tree.read_text("crates/cobre-sddp/src/simulation/pipeline.rs"),
        )
        cd023 = self.by_id["CD-023"]
        self.assertEqual(cd023["disposition"], "sharpen")
        self.assertIn("solve/", cd023["survivingClaim"])
        self.assertIn("training/stage_solve_prep.rs", cd023["survivingClaim"])

    def test_pd_004_is_an_existence_check_only(self) -> None:
        checks = self.env["existenceChecks"]
        self.assertEqual([c["id"] for c in checks], ["PD-004"])
        pd = checks[0]
        self.assertTrue(pd["exists"])
        self.assertIsNone(pd["disposition"])
        self.assertNotIn("fixShape", pd)
        self.assertEqual(
            pd["baselineAnchor"],
            {
                "path": "crates/cobre-sddp/src/training/backward_pass_state.rs",
                "symbol": "run_enumerated_backward",
            },
        )
        self.assertIn("perf", pd["queuedTo"])
        self.assertNotIn("PD-004", self.by_id)

    def test_part_i_handoff_is_exactly_item_7(self) -> None:
        entries = self.handoff["entries"]
        self.assertEqual([e["partIRef"] for e in entries], ["I.3-7"])
        e = entries[0]
        self.assertEqual(
            e["baselineAnchor"],
            {"path": "crates/cobre-sddp/src/setup/params.rs", "symbol": "from_config"},
        )
        self.assertEqual(e["owningStation"], "sddp")
        self.assertEqual(e["disposition"], "sharpen")
        self.assertEqual(e["proposedPhase"], "0a")
        self.assertIn("CD-004", e["crossRef"])
        self.assertIn("ConstructionConfig", e["changedSinceV012"])
        self.assertIn("export_states", e["changedSinceV012"])
        self.assertEqual(self.handoff["baseline"], self.baseline())
        tree = self.tree()
        for a in [e["baselineAnchor"], *e["anchors"]]:
            self.assertTrue(sc.anchor_exists(f"`{a['path']}::{a['symbol']}`", tree), a)
        params = tree.read_text("crates/cobre-sddp/src/setup/params.rs")
        self.assertIn("export_states: config.exports.states", params)


LENSES = ("architecture", "performance", "over-engineering", "test-bloat")
PERF_TARGETS = {
    "5c": ("select_for_stage", "SuccessorOutcomes", "reconstruct_basis"),
    "5b": ("PatchBuffer",),
}
SEAM_SYMBOLS = (
    "LipschitzConfig",
    "UpperBoundEvaluationConfig",
    "splice_reserved_state_block",
    "reserve_boundary_inflow_lag_slots",
    "rescale_cut_records_for_load",
    "LEGACY_COST_SCALE_FACTOR",
)
SETTLED_SYMBOLS = (
    "rebuild_historical_library_non_root",
    "ConstructionConfig",
    "into_construction_config",
)
TIMING_NUMBER = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\b|×|%|(?:ms|µs|us|ns|s|sec|secs|seconds|minutes|min|speedup|faster|slower)\b)",
    re.I,
)


class CandidateEnvelopeTests(sc.StationCase):
    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.cells = {
            (lens, sub): sc.load_json(
                cls.station_dir() / f"candidates.{lens}.{sub}.json"
            )
            for lens in LENSES
            for sub in SUBSTATIONS
        }
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.manifest = {f["path"]: f["substation"] for f in cls.inv["src"]["files"]}
        cls.log = (cls.station_dir() / "attacker-log.md").read_text(encoding="utf-8")

    def test_sixteen_cells_validate_and_agree_with_their_filenames(self) -> None:
        for (lens, sub), env in self.cells.items():
            with self.subTest(cell=f"{lens}.{sub}"):
                path = self.station_dir() / f"candidates.{lens}.{sub}.json"
                self.assertEqual(
                    subprocess.run(
                        [
                            "python3",
                            str(sc.TOOLS / "validate-envelope.py"),
                            "--role",
                            "attacker",
                            "--station",
                            "sddp",
                            str(path),
                        ],
                        cwd=sc.REPO,
                        capture_output=True,
                        text=True,
                        check=False,
                    ).returncode,
                    0,
                )
                self.assertEqual(env["station"], "sddp")
                self.assertEqual(env["subStation"], sub)
                self.assertEqual(env["cell"], f"sddp/{sub}")
                self.assertEqual(env["lens"], lens)
                self.assertEqual(env["baseline"], self.baseline())
                self.assertEqual(env["gate"]["validator"], "pass")
                self.assertTrue(
                    env["candidates"] or env["positives"],
                    "a blank cell (no candidates and no positives)",
                )

    def test_every_anchor_is_symbol_only_inside_the_station_and_resolves(self) -> None:
        tree = self.tree()
        for (lens, sub), env in self.cells.items():
            for c in env["candidates"]:
                with self.subTest(cell=f"{lens}.{sub}", title=c["title"][:60]):
                    self.assertTrue(c["anchors"])
                    for a in c["anchors"]:
                        self.assertEqual(set(a), {"path", "symbol"}, a)
                        self.assertTrue(a["path"].startswith("crates/cobre-sddp/"), a)
                        if a["path"].startswith("crates/cobre-sddp/src/"):
                            self.assertIn(a["path"], self.manifest, a)
                        else:
                            self.assertEqual(lens, "test-bloat", a)
                        self.assertTrue(
                            sc.anchor_exists(f"`{a['path']}::{a['symbol']}`", tree), a
                        )
                    self.assertIn(c["alignmentHint"], ALIGN)
                    self.assertIn(c["proposedSeverity"], ("A", "B", "C"))

    def test_performance_cells_carry_layouts_and_no_numbers(self) -> None:
        for sub in SUBSTATIONS:
            env = self.cells[("performance", sub)]
            for c in env["candidates"]:
                with self.subTest(sub=sub, title=c["title"][:60]):
                    self.assertIn(c["claimType"], ("single-process", "collective"))
                    self.assertIn(c["layout"], ("4t", "2x2"))
                    self.assertEqual(c["status"], "UNMEASURED")
                    self.assertIsNone(c["measured"])
                    self.assertEqual(c["queuedTo"], "perf-sweep")
                    blob = (
                        " ".join(
                            str(c.get(k, ""))
                            for k in ("title", "mechanism", "fixShape")
                        )
                        + " "
                        + str(c["evidence"].get("reading", ""))
                    )
                    self.assertIsNone(TIMING_NUMBER.search(blob), blob[:120])
        for sub, symbols in PERF_TARGETS.items():
            env = self.cells[("performance", sub)]
            mentioned = json.dumps(env["candidates"]) + json.dumps(env["positives"])
            for s in symbols:
                self.assertIn(
                    s,
                    mentioned,
                    f"perf target {s} has neither candidate nor positive in {sub}",
                )

    def test_reserved_seams_and_settled_items_never_surface_as_candidates(self) -> None:
        for (lens, sub), env in self.cells.items():
            for c in env["candidates"]:
                blob = c["title"] + " " + json.dumps(c["anchors"])
                for s in (*SEAM_SYMBOLS, *SETTLED_SYMBOLS):
                    self.assertNotRegex(
                        blob, rf"\b{s}\b", f"{lens}.{sub}: {c['title'][:60]}"
                    )
            for p in env["positives"]:
                text = p.get("subject", "") + " " + p.get("why", "")
                if any(re.search(rf"\b{s}\b", text) for s in SEAM_SYMBOLS):
                    self.assertTrue(p.get("sanctionedBy"), p)

    def test_gate_recorded_drops_with_the_six_actions_and_the_log_has_no_blank_cell(
        self,
    ) -> None:
        actions = {
            "settled",
            "sanctioned",
            "dup-of",
            "merged",
            "anchor-missing",
            "needs-human",
        }
        for (lens, sub), env in self.cells.items():
            for d in env["dropped"]:
                self.assertIn(d["action"], actions, f"{lens}.{sub}: {d['action']}")
                self.assertTrue(d["reason"])
            self.assertEqual(env["gate"]["kept"], len(env["candidates"]))
            self.assertEqual(env["gate"]["dropped"], len(env["dropped"]))
        matrix = section(self.log, "## Coverage matrix")
        rows = [
            line
            for line in matrix.splitlines()
            if line.startswith("| ") and not line.startswith("| -")
        ]
        self.assertEqual(len(rows), 5, "header + four lens rows")
        for line in rows[1:]:
            self.assertNotIn("pending", line)
            self.assertNotIn("0/0/0", line)
            self.assertNotIn("FAIL", line)
        lens_rows = [line.split("|")[1].strip() for line in rows[1:]]
        self.assertEqual(sorted(lens_rows), sorted(LENSES))
        for lens in LENSES:
            for sub in SUBSTATIONS:
                self.assertIn(
                    f"{lens}.{sub}", self.log, "a cell without a dispatch record"
                )
        self.assertIn("## Prior-register screen", self.log)

    def test_enumerated_duplication_is_kept_once_across_5c_and_5d(self) -> None:
        enum_files = {
            "crates/cobre-sddp/src/training/forward/enumerated.rs",
            "crates/cobre-sddp/src/simulation/enumerated.rs",
        }
        dup = re.compile(
            r"duplicat|twice|mirror|parallel|skeleton|cop(y|ies)|shared shape", re.I
        )
        carriers = [
            (sub, c)
            for sub in ("5c", "5d")
            for c in self.cells[("architecture", sub)]["candidates"]
            if ({a["path"] for a in c["anchors"]} & enum_files)
            and (
                len({a["path"] for a in c["anchors"]} & enum_files) == 2
                or (re.search(r"enumerat", c["title"], re.I) and dup.search(c["title"]))
            )
        ]
        self.assertLessEqual(len(carriers), 1, [s for s, _ in carriers])
        if carriers:
            _, c = carriers[0]
            self.assertEqual({a["path"] for a in c["anchors"]} & enum_files, enum_files)
            self.assertEqual(c.get("reRaiseOf"), "CD-028")


INGEST_STATES = {
    "accepted",
    "rejected-anchor",
    "rejected-reserved-seam",
    "rejected-defender",
    "merged",
    "needs-human",
}
SANCTIONED_BY_CLOSED_SET = {
    "`LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`",
    "Boundary state-family coupling channels are per-family bespoke",
    "Legacy (`None`) cost-scale branch of `rescale_cut_records_for_load`",
    "`#[allow(...)]` census — Reserved-seam (Voice 4) class",
    "Superseded cut-sync public methods",
}
BYTE_NEUTRAL = {"asserted", "needs-rebaseline", "n/a"}
PROBE_TITLE = "INGEST ANCHOR PROBE — sddp (2026-09, baseline)"


def ingest_state(entry: dict[str, Any]) -> str:
    if entry["disposition"] == "dup-of":
        return "merged"
    if entry["disposition"] == "anchor-missing":
        return "rejected-anchor"
    if entry["disposition"] == "sanctioned":
        return "rejected-reserved-seam"
    if entry["verdict"] == "confirmed":
        return "accepted"
    if entry["verdict"] == "dismissed" or entry["disposition"] == "contract-dismissed":
        return "rejected-defender"
    return "needs-human"


def normalised_tokens(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9 ]", "", text.lower().replace("`", "")).split())


class IngestTests(sc.StationCase):
    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = sc.load_json(cls.station_dir() / "verdicts.json")
        cls.verdicts = cls.doc["verdicts"]
        cls.log = (cls.station_dir() / "ingest-log.md").read_text(encoding="utf-8")
        cls.brief = (cls.station_dir() / "defender-prompt.md").read_text(
            encoding="utf-8"
        )
        cls.expected_refs = {
            f"{sub}-{lens}-{i:02d}"
            for lens in LENSES
            for sub in SUBSTATIONS
            for i in range(
                len(
                    sc.load_json(cls.station_dir() / f"candidates.{lens}.{sub}.json")[
                        "candidates"
                    ]
                )
            )
        }
        cls.headings = {
            line.strip("# \n").replace("`", "")
            for line in (sc.REPO / SDDP_MD).read_text(encoding="utf-8").splitlines()
            if re.match(r"^#{2,3} ", line)
        }

    def test_exactly_one_verdict_per_candidate_across_the_sixteen_files(self) -> None:
        self.assertEqual(set(self.verdicts), self.expected_refs)
        for ref, entry in self.verdicts.items():
            self.assertEqual(entry["candidateRef"], ref)
        self.assertEqual(self.doc["counts"]["received"], len(self.expected_refs))
        self.assertEqual(self.doc["baseline"], header_baseline())

    def test_every_verdict_has_a_known_state_and_the_counts_sum(self) -> None:
        counts = self.doc["counts"]
        states = [ingest_state(e) for e in self.verdicts.values()]
        self.assertTrue(set(states) <= INGEST_STATES, set(states) - INGEST_STATES)
        self.assertEqual(
            counts["received"],
            counts["anchorRejected"]
            + counts["sanctionedCleared"]
            + counts["contractDismissed"]
            + counts["dupOf"]
            + counts["reRaiseRejected"]
            + counts["handedOff"]
            + counts["defended"],
        )
        self.assertEqual(
            counts["defended"],
            counts["confirmed"] + counts["dismissed"] + counts["verdictNull"],
        )
        self.assertEqual(states.count("merged"), counts["dupOf"])
        self.assertEqual(states.count("accepted"), counts["confirmed"])
        self.assertEqual(states.count("needs-human"), counts["verdictNull"])
        for entry in self.verdicts.values():
            if entry["disposition"] == "defended" and entry["verdict"] is None:
                self.assertTrue(entry["_needsHuman"], entry["candidateRef"])

    def test_accepted_anchors_resolve_at_the_baseline(self) -> None:
        tree = self.tree()
        for entry in self.verdicts.values():
            if ingest_state(entry) != "accepted":
                continue
            for anchor in entry["anchors"]:
                with self.subTest(ref=entry["candidateRef"], anchor=anchor):
                    self.assertTrue(anchor["path"].startswith("crates/cobre-sddp/"))
                    self.assertTrue(
                        sc.symbol_resolves(anchor["path"], anchor["symbol"], tree)
                    )

    def test_anchor_probe_resolves_with_the_harness_checker(self) -> None:
        report = json.loads(
            subprocess.run(
                [
                    "python3",
                    str(sc.TOOLS / "check-anchors.py"),
                    PROBE_TITLE,
                    "--register",
                    str(self.station_dir() / "anchor-probe.md"),
                    "--baseline",
                    header_baseline(),
                    "--json",
                ],
                cwd=sc.REPO,
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        )
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            report["checked"],
            sum(len(e["anchors"]) for e in self.verdicts.values()),
        )

    def test_reserved_seam_rejections_cite_the_closed_set(self) -> None:
        for entry in self.verdicts.values():
            if ingest_state(entry) == "rejected-reserved-seam" or entry.get(
                "sanctionedBy"
            ):
                self.assertIn(entry["sanctionedBy"], SANCTIONED_BY_CLOSED_SET)
        for text in SANCTIONED_BY_CLOSED_SET:
            self.assertIn(text, self.brief)

    def test_dup_of_carries_the_prior_id_instead_of_a_fresh_verdict(self) -> None:
        merged = [e for e in self.verdicts.values() if ingest_state(e) == "merged"]
        self.assertTrue(merged)
        for entry in merged:
            self.assertIsNone(entry["verdict"])
            self.assertIn(entry["priorRelation"], {"restates", "intra-station"})
            if entry["priorRelation"] == "restates":
                self.assertRegex(entry["priorId"], r"^(CD|OD|PD)-\d{3}$")
                self.assertIn(entry["priorId"], OWNED)
            else:
                self.assertIn(entry["priorId"], self.verdicts)
                self.assertEqual(
                    self.verdicts[entry["priorId"]]["disposition"], "defended"
                )
                self.assertIn(
                    entry["candidateRef"], self.verdicts[entry["priorId"]]["mergedFrom"]
                )
        for entry in self.verdicts.values():
            if entry.get("priorRelation") == "sharpens":
                self.assertEqual(entry["disposition"], "defended")
                self.assertIn(entry["priorId"], OWNED | {"CD-074"})

    def test_confirmed_verdicts_narrow_and_dismissals_carry_no_claim(self) -> None:
        for entry in self.verdicts.values():
            if entry["disposition"] != "defended":
                continue
            with self.subTest(ref=entry["candidateRef"]):
                self.assertIn(entry["byteNeutral"], BYTE_NEUTRAL | {None})
                self.assertIn(entry["alignmentHint"], ALIGN | {None})
                if entry["verdict"] == "confirmed":
                    claim = entry["survivingClaim"]
                    self.assertTrue(claim and len(claim) >= 40)
                    title_tokens = normalised_tokens(entry["title"])
                    claim_tokens = normalised_tokens(claim)
                    jaccard = len(title_tokens & claim_tokens) / len(
                        title_tokens | claim_tokens
                    )
                    self.assertLess(jaccard, 0.85)
                if entry["verdict"] == "dismissed":
                    self.assertIsNone(entry["survivingClaim"])
                if entry.get("contractCited"):
                    self.assertIn(
                        entry["contractCited"].replace("`", ""), self.headings
                    )
                if entry["lens"] == "performance" and entry["argument"]:
                    self.assertIsNone(
                        TIMING_NUMBER.search(
                            entry["argument"] + " " + (entry["survivingClaim"] or "")
                        )
                    )
                for field in ("argument", "survivingClaim"):
                    self.assertNotIn("```", entry[field] or "")

    def test_ingest_log_has_one_roster_row_per_candidate(self) -> None:
        roster = section(self.log, "## Per-candidate roster")
        refs = re.findall(
            r"^\| (5[a-d]-(?:architecture|performance|over-engineering|test-bloat)-\d{2}) \|",
            roster,
            re.M,
        )
        self.assertEqual(sorted(refs), sorted(self.expected_refs))
        self.assertEqual(len(refs), len(set(refs)))
        header = re.search(r"^Baseline: `([0-9a-f]{40})`", self.log, re.M)
        assert header is not None
        self.assertEqual(header.group(1), header_baseline())
        for heading in (
            "## Candidate census",
            "## Anchor rejections",
            "## Cleared (sanctioned)",
            "## Dup-of merges",
            "## Re-raise rejections",
            "## Contract dismissals",
            "## Out-of-station hand-offs",
            "## Defender pass",
        ):
            self.assertIn(heading, self.log)


CAL_ID_RE = re.compile(r"^(CD|PD|OD|TD)-(\d{3})$")
ID_FLOORS = {"CD": 40, "PD": 6, "OD": 10, "TD": 1}
SEVERITIES = {"A", "B", "B (A-risk)", "C"}
LAYOUTS = {"4t", "2x2"}
REQUIRES = {"none", "enumerated", "external-library"}
ENTRY_HEADING = re.compile(r"^\*\*((?:CD|PD|OD|TD)-\d{3}) · Sev ", re.M)
SCAFFOLD = "## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp"
STATION_TITLE = "STATION 5 — cobre-sddp (2026-09)"
SUBSTATION_HEADINGS = (
    "#### 5a — setup + policy + stochastic + config",
    "#### 5b — lp/ (indexer, builder, template, generic constraints)",
    "#### 5c — cut + training + solve + workspace",
    "#### 5d — simulation + production + support",
)
LENS_CLASS = {
    "architecture": "CD",
    "performance": "PD",
    "over-engineering": "OD",
    "test-bloat": "TD",
}
RANK = {"A": 3, "B (A-risk)": 2.5, "B": 2, "C": 1}
CONTRACT_PATHS = (
    "/cut/",
    "policy/policy_load.rs",
    "lp/builder/columns.rs",
    "lp/builder/patch.rs",
    "lp/builder/entries.rs",
)


_STATION_ORDER = [
    "core-io",
    "stochastic",
    "solver-comm",
    "sddp",
    "cli-python",
    "build-ci",
    "test-corpus",
]


_EPIC_SECTIONS = ("generalization-alignment", "performance-sweep", "unified-roadmap")


def register_before_station(register: str, own_section: str, slug: str) -> str:
    """The register as it stood when this station minted: its own section, every LATER crate-station section and the epic blocks written after all stations (the alignment ledger names every id) removed."""
    prior = register.replace(own_section, "")
    for later in _STATION_ORDER[_STATION_ORDER.index(slug) + 1 :] + list(
        _EPIC_SECTIONS
    ):
        marker = f"## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — {later}"
        if marker in prior:
            block = prior.split(marker, 1)[1].split("\n## ", 1)[0]
            prior = prior.replace(marker + block, "")
    return prior


class CalibrationTests(sc.StationCase):
    """E05-6: id assignment, house calibration, dispositions, alignment, byte-neutrality, queues, section."""

    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.assigned = cls.cal["assigned"]
        cls.by_ref = {r["candidateRef"]: r for r in cls.assigned}
        cls.by_id = {r["id"]: r for r in cls.assigned}
        cls.reused = {r["id"]: r for r in cls.cal["reusedIds"]}
        cls.verdicts = sc.load_json(cls.station_dir() / "verdicts.json")["verdicts"]
        cls.perf = sc.load_json(cls.station_dir() / "perf-queue.json")
        cls.td = sc.load_json(cls.station_dir() / "td-queue.json")
        cls.register = (sc.REPO / "plans/architecture-debt-audit/BACKLOG.md").read_text(
            encoding="utf-8"
        )
        cls.section = cls.register.split(SCAFFOLD, 1)[1].split("\n## ", 1)[0]
        cls.prior = register_before_station(cls.register, cls.section, "sddp")

    def block(self, entry_id: str) -> str:
        return self.section.split(f"**{entry_id} · ", 1)[1].split("\n**", 1)[0]

    def test_envelope_baseline_and_heading_levels(self) -> None:
        self.assertEqual(self.cal["station"], "sddp")
        self.assertEqual(self.cal["baseline"], header_baseline())
        self.assertEqual(self.cal["sectionTitle"], STATION_TITLE)
        self.assertEqual(self.section.count(f"\n### {STATION_TITLE}\n"), 1)
        level3 = re.findall(r"^### .*$", self.section, re.M)
        self.assertEqual(level3, [f"### {STATION_TITLE}"])
        for heading in SUBSTATION_HEADINGS + (
            "#### Wave 4 / 6 / 7 dispositions",
            "#### Part-I cross-references — item 7",
            "#### Positives",
            "#### ↩︎ Cleared",
            "#### Queued out",
            "#### Owner gate — decisions",
        ):
            self.assertIn(heading, self.section, heading)
        gate = self.section.split("#### Owner gate — decisions", 1)[1].split(
            "\n#### ", 1
        )[0]
        if (self.station_dir() / "decisions.json").exists():
            self.assertIn("**Gate: RETURNED ", gate)
        else:
            self.assertIn("_(pending", gate)

    def test_one_assigned_row_per_confirmed_new_candidate_and_none_otherwise(
        self,
    ) -> None:
        confirmed_new = {
            ref
            for ref, e in self.verdicts.items()
            if e["disposition"] == "defended"
            and e["verdict"] == "confirmed"
            and e.get("priorRelation") != "sharpens"
        }
        self.assertEqual(set(self.by_ref), confirmed_new)
        sharpens = {
            e["priorId"]: ref
            for ref, e in self.verdicts.items()
            if e.get("priorRelation") == "sharpens"
        }
        for prior_id, ref in sharpens.items():
            if prior_id in self.reused:
                self.assertEqual(
                    self.reused[prior_id]["e05_5Delta"]["candidateRef"], ref
                )
            self.assertNotIn(ref, self.by_ref, "a sharpening must never mint an id")
        self.assertEqual(
            {c["candidateRef"] for c in self.cal["cleared"]},
            {
                ref
                for ref, e in self.verdicts.items()
                if e["disposition"] == "defended" and e["verdict"] == "dismissed"
            },
        )
        self.assertEqual(
            {d["candidateRef"] for d in self.cal["dupOf"]},
            {ref for ref, e in self.verdicts.items() if e["disposition"] == "dup-of"},
        )
        self.assertTrue(all(d["assignedId"] is None for d in self.cal["dupOf"]))
        for row in self.assigned:
            claim, title = (
                normalised_tokens(row["survivingClaim"]),
                normalised_tokens(row["title"]),
            )
            self.assertTrue(claim and claim != title, row["id"])

    def test_ids_well_formed_in_range_and_class_matches_lens(self) -> None:
        for row in self.assigned:
            m = CAL_ID_RE.match(row["id"])
            self.assertIsNotNone(m, row["id"])
            assert m is not None
            cls, num = m.group(1), int(m.group(2))
            self.assertEqual(cls, row["class"])
            self.assertEqual(cls, LENS_CLASS[row["lens"]])
            self.assertGreaterEqual(num, ID_FLOORS[cls])

    def test_ids_contiguous_from_the_runtime_floor_per_class(self) -> None:
        floors = self.cal["idFloorsSeen"]
        for cls in ID_FLOORS:
            nums = sorted(
                int(r["id"].split("-")[1]) for r in self.assigned if r["class"] == cls
            )
            self.assertEqual(
                nums, list(range(floors[cls], floors[cls] + len(nums))), cls
            )
            prior_max = max(
                [int(n) for n in re.findall(rf"\b{cls}-(\d{{3}})\b", self.prior)]
                + [ID_FLOORS[cls] - 1]
            )
            self.assertEqual(
                floors[cls],
                prior_max + 1,
                f"{cls}: floor is not the register's next free",
            )

    def test_ids_unique_across_the_whole_register_and_reused_ids_never_re_minted(
        self,
    ) -> None:
        headings = ENTRY_HEADING.findall(self.register)
        dup = {h for h in headings if headings.count(h) > 1}
        self.assertEqual(dup, set(), f"duplicate entry headings: {sorted(dup)}")
        for row in self.assigned:
            self.assertIsNone(
                re.search(rf"\b{row['id']}\b", self.prior),
                f"{row['id']} pre-exists in the register",
            )
            self.assertEqual(
                self.section.count(f"**{row['id']} · Sev {row['severity']} · "),
                1,
                row["id"],
            )
        section_headings = ENTRY_HEADING.findall(self.section)
        self.assertEqual(sorted(section_headings), sorted(self.by_id))
        for rid in self.reused:
            self.assertNotIn(rid.split("-construction")[0], section_headings)
        self.assertTrue(all(r["priorId"] is None for r in self.assigned))

    def test_severity_alignment_and_the_house_precedents(self) -> None:
        for row in self.assigned:
            self.assertIn(row["severity"], SEVERITIES, row["id"])
            self.assertIn(row["alignmentHint"], ALIGN, row["id"])
            self.assertIn(row["effort"], ("S", "M", "L"))
            self.assertIn(row["confidence"], ("high", "med", "low"))
            self.assertTrue(row["category"])
        for ref in ("5c-architecture-00", "5c-performance-06"):
            row = self.by_ref[ref]
            self.assertTrue(row["severity"].startswith("B"), ref)
            self.assertIn("retrofitted-variant", row["calibrationBasis"])
            self.assertIn("CD-013", row["calibrationBasis"])
            self.assertIn("retrofitted-variant", self.block(row["id"]))
        ring = self.by_ref["5b-architecture-00"]
        self.assertEqual(ring["severity"], "B (A-risk)")
        self.assertEqual(ring["reviewerRating"], "A")
        self.assertTrue(ring["downgradeReason"])
        cd004 = self.reused["CD-004"]
        self.assertEqual(cd004["severity"], "B (A-risk)")
        self.assertEqual(cd004["alignmentHint"], "advances-0a")
        self.assertRegex(cd004["alignmentCites"] or "", r"Part (IV|V)")
        self.assertIn("silen", cd004["calibrationBasis"])
        self.assertEqual(self.reused["CD-015"]["severity"], "B")
        table = self.section.split("#### Wave 4 / 6 / 7 dispositions", 1)[1].split(
            "\n#### ", 1
        )[0]
        # the owner gate appends a Decision column, so the row is not anchored at `$`
        self.assertIsNotNone(
            re.search(
                r"^\| CD-004 \| 4 \| sharpen \|.*\| B \(A-risk\) \| advances-0a \|",
                table,
                re.M,
            )
        )
        self.assertIn("SILENCE of the divergence", table)

    def test_downgrades_record_the_reviewer_rating(self) -> None:
        for row in self.assigned:
            downgraded = RANK[row["severity"]] < RANK[row["reviewerRating"]]
            block = self.block(row["id"])
            if downgraded:
                self.assertTrue(row["downgradeReason"], row["id"])
                self.assertIn(
                    f"- **Reviewer rating:** {row['reviewerRating']} — recalibrated to {row['severity']}",
                    block,
                )
            else:
                self.assertIsNone(row["downgradeReason"], row["id"])
        self.assertEqual(
            self.cal["counts"]["downgrades"],
            sum(1 for r in self.assigned if r["downgradeReason"]),
        )

    def test_conflicts_row_is_the_cd024_successor_held_with_an_alternative(
        self,
    ) -> None:
        self.assertEqual([c["id"] for c in self.cal["conflicts"]], ["CD-024-successor"])
        held = self.cal["conflicts"][0]
        self.assertTrue(held["alternative"])
        self.assertIn("HELD", held["status"])
        self.assertIn("one-consumer", held["condition"])
        self.assertFalse(self.reused["CD-024-successor"]["actionable"])
        self.assertFalse(
            [r for r in self.assigned if r["alignmentHint"] == "conflicts"]
        )
        self.assertTrue(all(r["actionable"] for r in self.assigned))
        table = self.section.split("#### Wave 4 / 6 / 7 dispositions", 1)[1].split(
            "\n#### ", 1
        )[0]
        self.assertIsNotNone(
            re.search(
                r"^\| CD-024-successor \|.*conflicts — HELD.*activate-or-die.*\|$",
                table,
                re.M,
            )
        )

    def test_byte_neutrality_and_contract_lines_on_every_entry(self) -> None:
        for row in self.assigned:
            block = self.block(row["id"])
            self.assertIn("- **Byte-neutrality:**", block, row["id"])
            self.assertIn(
                row["byteNeutral"], ("asserted", "n/a", "needs-rebaseline"), row["id"]
            )
            paths = [a["path"] for a in row["anchors"]]
            if row["class"] != "TD" and any(
                any(cp in p for cp in CONTRACT_PATHS) for p in paths
            ):
                self.assertTrue(row["contractCited"], row["id"])
                self.assertIn("- **Contract:** .claude/rules/sddp.md", block, row["id"])
            if row["byteNeutral"] == "asserted":
                self.assertIn("parity_hash_highs", block)
                self.assertIn("mpiexec -n 1/2", block)
        self.assertIn("re-baseline", self.reused["CD-005"]["byteNeutrality"])
        table = self.section.split("#### Wave 4 / 6 / 7 dispositions", 1)[1].split(
            "\n#### ", 1
        )[0]
        self.assertIn("re-baseline decision at the owner gate", table)
        headings = {
            line.strip("# \n").replace("`", "")
            for line in (sc.REPO / SDDP_MD).read_text(encoding="utf-8").splitlines()
            if re.match(r"^#{2,3} ", line)
        }
        for row in self.assigned:
            for cited in row["contractCited"]:
                self.assertIn(cited.replace("`", ""), headings, row["id"])

    def test_checkers_exit_zero_with_the_exact_title_and_the_slug(self) -> None:
        for arg in (STATION_TITLE, "sddp"):
            for tool in ("fields-check.py", "check-anchors.py", "check-reraise.py"):
                result = subprocess.run(
                    [sys.executable, str(sc.TOOLS / tool), arg],
                    capture_output=True,
                    text=True,
                    cwd=sc.REPO,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{tool} {arg}: {result.stdout}{result.stderr}",
                )

    def test_every_entry_carries_alignment_with_roadmap_citation_and_the_baseline(
        self,
    ) -> None:
        for row in self.assigned:
            block = self.block(row["id"])
            m = re.search(
                r"^- \*\*Alignment:\*\* (\S+) \((?:provisional; Epic 9 adjudicates .*beyond-sddp-generalization\.md"
                r"|.*; station hint: (\S+), retagged \d{4}-\d{2}-\d{2} by alignment/alignment-ledger\.json\))",
                block,
                re.M,
            )
            self.assertIsNotNone(m, row["id"])
            assert m is not None
            self.assertEqual(m.group(2) or m.group(1), row["alignmentHint"])
            self.assertIn(f"- **Baseline:** `{header_baseline()}`", block)
            self.assertRegex(row["alignmentCites"], r"Part (IV|V)")

    def test_perf_queue_only_sev_ab_pd_with_layout_requires_call_sites_and_no_number(
        self,
    ) -> None:
        ab = {
            r["id"]
            for r in self.assigned
            if r["class"] == "PD" and r["severity"][0] in "AB"
        }
        queue = {q["id"]: q for q in self.perf["queue"]}
        self.assertEqual(set(queue) - {"PD-004"}, ab)
        self.assertTrue(ab)
        for qid, q in queue.items():
            self.assertIn(q["requires"], REQUIRES, qid)
            self.assertEqual(q["status"], "UNMEASURED", qid)
            if qid == "PD-004":
                self.assertTrue(q["existsAtBaseline"])
                self.assertIsNone(q["fixShape"])
                continue
            self.assertIn(q["layout"], LAYOUTS, qid)
            self.assertIn(q["claimType"], ("single-process", "collective"))
            self.assertTrue(q["exercisingCallSites"], qid)
            self.assertTrue(q["byteNeutralFixShape"], qid)
            self.assertIsNone(q.get("measured"))
        blob = json.dumps(self.perf["queue"])
        self.assertIsNone(
            TIMING_NUMBER.search(blob), "perf-queue carries a timing figure"
        )
        for row in self.assigned:
            if row["class"] == "PD":
                body = self.block(row["id"]).split("\n", 1)[1]
                self.assertIsNone(TIMING_NUMBER.search(body), row["id"])
        self.assertIn("perf-queue.json", self.section)

    def test_td_queue_every_td_with_yardstick_and_no_new_fixture_crate(self) -> None:
        tds = {r["id"] for r in self.assigned if r["class"] == "TD"}
        self.assertEqual({q["id"] for q in self.td["queue"]}, tds)
        for q in self.td["queue"]:
            self.assertTrue(q["yardstick"].startswith("§"), q["id"])
            self.assertTrue(q["claim"])
            self.assertIn("determinismGates", self.td["informationalGates"])
        self.assertNotRegex(
            self.section,
            r"(?i)\b(introduce|create|add|extract)\w*\s+(?:a |an )?new (?:fixture|test-support) crate\b(?![^.]*(?:out of bounds|rules that out))",
        )
        self.assertIn("td-queue.json", self.section)

    def test_disposition_table_has_one_row_per_owned_id_with_the_record_contract(
        self,
    ) -> None:
        table = self.section.split("#### Wave 4 / 6 / 7 dispositions", 1)[1].split(
            "\n#### ", 1
        )[0]
        rows = {
            m.group(1): m.group(0)
            for m in re.finditer(
                r"^\| ((?:CD|OD|PD)-\d{3}(?:-[\w-]+)?) \|.*$", table, re.M
            )
        }
        expected = (OWNED - {"CD-003-construction-hop"}) | {
            "CD-003-construction-hop",
            "PD-004",
        }
        self.assertEqual(set(rows), expected)
        self.assertEqual(len(self.cal["reusedIds"]), 22)
        for r in self.cal["reusedIds"]:
            row = rows[r["id"]]
            self.assertRegex(
                row, r"`crates/cobre-sddp/[\w/.-]+\.rs::[A-Za-z_][A-Za-z0-9_]*`"
            )
            if r["disposition"] == "retire":
                self.assertTrue(r["resolvingCommit"])
                self.assertIn(f"resolved by `{r['resolvingCommit']}`", row)
            if r["disposition"] == "sharpen":
                self.assertTrue(r["supersededClaim"] and r["survivingClaim"], r["id"])
                self.assertIn("superseded:", row)
                self.assertIn("surviving:", row)
        self.assertIn("workspace/workspace.rs::", rows["CD-021"])
        self.assertIn("existence-only", rows["PD-004"])

    def test_section_skeleton_lenses_positives_cleared_and_queue_markers(self) -> None:
        for heading in SUBSTATION_HEADINGS:
            body = self.section.split(heading, 1)[1].split("\n#### ", 1)[0]
            parts = re.split(
                r"^\*\*(Architecture|Performance|Over-engineering|Test bloat)\*\*",
                body,
                flags=re.M,
            )
            labels, blocks = parts[1::2], parts[2::2]
            self.assertEqual(
                labels,
                ["Architecture", "Performance", "Over-engineering", "Test bloat"],
                heading,
            )
            for label, lens_block in zip(labels, blocks):
                self.assertTrue(
                    re.search(r"^\*\*(?:CD|PD|OD|TD)-\d{3} · Sev ", lens_block, re.M)
                    or "_No confirmed NEW finding for the" in lens_block,
                    f"{heading} / {label}",
                )
        cleared = self.section.split("#### ↩︎ Cleared", 1)[1].split("\n#### ", 1)[0]
        dismissed = [
            ref
            for ref, e in self.verdicts.items()
            if e["disposition"] == "defended" and e["verdict"] == "dismissed"
        ]
        for ref in dismissed:
            self.assertIsNotNone(
                re.search(rf"^- \*\*{re.escape(ref)} — .*\*\* — .+", cleared, re.M), ref
            )
        self.assertIn("Superseded cut-sync public methods", cleared)
        self.assertIn("LipschitzConfig.mode", cleared)
        positives = self.section.split("#### Positives", 1)[1].split("\n#### ", 1)[0]
        self.assertGreater(positives.count("\n- "), 50)
        queued = self.section.split("#### Queued out", 1)[1].split("\n#### ", 1)[0]
        self.assertIn("perf-queue.json", queued)
        self.assertIn("td-queue.json", queued)


GENERIC_CHECKS = (
    "check-anchors",
    "check-reraise",
    "fields-check",
    "register",
    "inventory-set-equality",
    "infra-genericity",
    "read-only-workspace",
)
STATION_CHECKS = (
    "partition",
    "dispositions",
    "lp-reconciliation",
    "no-timing",
    "read-only-snapshot",
)
SCAFFOLD_TITLE = SCAFFOLD[3:]
LP_SRC = "crates/cobre-sddp/src/lp"
RETIRED_ROWS = ("CD-001", "CD-003-construction-hop", "CD-006")
TIMING_DECOYS = (
    "163 module paths",
    "30 records",
    "L1727",
    "`077dbe2c287b92c2d0c6a12d5f67c2c0cb83c39c`",
    "resolved by `b051c410`",
    "layout `2x2`",
    "--threads 4",
    "mpiexec -n 2 x --threads 2",
    "PD-035",
    "19 × B",
    "exists at the baseline (721)",
    "2026-09-18",
    "-n 1/2",
    "5a-performance-00",
    "entries.rs 10,093 lines",
    "2-opt",
)
TIMING_FIGURES = (
    "12 ms",
    "0.8 s",
    "3.2x faster",
    "40 LP/s",
    "1.5 GiB/s",
    "a speed-up of 2",
    "12% faster",
    "45 min",
    "2 iterations per second",
)


def load_verifier() -> Any:
    path = sc.STATIONS / "sddp" / "verify-sddp.py"
    spec = importlib.util.spec_from_file_location("verify_sddp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SectionVerifyTests(sc.StationCase):
    """Executable proof of the station verification (E05-7).

    verify-station.sh runs this module, so nothing here may invoke it (recursion); the shared
    verifier's checks are re-asserted directly and its rendered verification.md is read when it
    exists. The station driver is invoked with --no-shared for the same reason.
    """

    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = load_verifier()
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.lp = sc.load_json(sc.AUDIT / "measurements" / "lp-inventory.json")
        cls.lines = backlog_parse.read_register(sc.BACKLOG)
        cls.section = backlog_parse.find_section(cls.lines, SCAFFOLD_TITLE)
        cls.entries = backlog_parse.iter_entries(cls.section)

    def run_driver(self, *argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.artifact("verify-sddp.py")), *argv],
            capture_output=True,
            text=True,
            cwd=sc.REPO,
            check=False,
        )

    def test_three_harness_checkers_exit_zero_over_the_section_title(self) -> None:
        for tool in ("check-anchors.py", "check-reraise.py", "fields-check.py"):
            self.assertEqual(sc.run_checker(tool, SCAFFOLD_TITLE), 0, tool)

    def test_no_tracked_file_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])

    def test_station_verify_subcommands_exit_zero_over_the_nested_census(self) -> None:
        base = backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG))
        cmds = {
            "register": [str(sc.AUDIT), self.SLUG],
            "inventory": [str(self.artifact("inventory.json")), str(sc.REPO)],
            "genericity": [str(sc.REPO), str(self.artifact("partI-handoff.json"))],
            "readonly": [str(sc.REPO), base],
        }
        for sub, args in cmds.items():
            proc = subprocess.run(
                [sys.executable, str(sc.TOOLS / "station_verify.py"), sub, *args],
                cwd=sc.REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                proc.returncode, 0, f"station_verify {sub}:\n{proc.stdout}{proc.stderr}"
            )

    def test_shared_tool_reconstructs_the_nested_census_and_its_root(self) -> None:
        listed = station_verify.reconstruct_listed(self.inv)
        roots = station_verify.crate_src_roots(self.inv)
        self.assertEqual(roots, [SRC])
        self.assertEqual(
            sorted(listed), sorted(f["path"] for f in self.inv["src"]["files"])
        )
        tree = sorted(
            station_verify.baseline_rs_files(sc.REPO, self.inv["baseline"], roots)
        )
        self.assertEqual(station_verify.inventory_diff(listed, tree), ([], []))
        self.assertEqual(len(tree), self.inv["src"]["file_count"])

    def test_inventory_module_set_equals_the_tree_at_the_baseline(self) -> None:
        listed = [f["path"] for f in self.inv["src"]["files"]]
        self.assertEqual(len(listed), len(set(listed)))
        actual = set(self.tree().rs_files(SRC))
        self.assertEqual(
            sorted(set(listed) - actual), [], "listed but absent at the baseline"
        )
        self.assertEqual(
            sorted(actual - set(listed)), [], "present at the baseline but unlisted"
        )
        self.assertEqual(len(actual), 163)

    def test_lp_inventory_lists_every_lp_module_once_with_wc_l_line_counts(
        self,
    ) -> None:
        tree = self.tree()
        listed = [f["path"] for f in self.lp["files"]]
        self.assertEqual(sorted(listed), sorted(set(listed)))
        self.assertEqual(sorted(listed), sorted(tree.rs_files(LP_SRC)))
        self.assertEqual(len(listed), 30)
        for f in self.lp["files"]:
            self.assertEqual(f["total_lines"], sc.raw_lines(f["path"], tree), f["path"])
            self.assertTrue(f["top_symbols"], f["path"])
            self.assertLessEqual(f["non_test_lines"], f["total_lines"], f["path"])

    def test_verify_sddp_exits_zero_with_every_station_check_passing(self) -> None:
        result = self.run_driver(SCAFFOLD_TITLE, "--no-shared")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "PASS (5/5 station-specific checks; shared verifier skipped)", result.stdout
        )
        for check in STATION_CHECKS:
            self.assertIsNotNone(
                re.search(rf"^\| {check} \| .* \| 0 \| PASS \|$", result.stdout, re.M),
                check,
            )
        self.assertIsNotNone(
            re.search(
                r"^\| shared-verifier \| .* \| - \| SKIP \|$", result.stdout, re.M
            )
        )

    def test_section_title_must_resolve_to_exactly_one_heading(self) -> None:
        for name in (SCAFFOLD_TITLE, "sddp"):
            self.assertEqual(
                len(self.verifier.matching_headings(self.lines, name)), 1, name
            )
        twice = [
            "## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — sddp",
            "prose",
            "## ★ QUALITY EVALUATION (2026-10, baseline deadbeef) — sddp",
        ]
        self.assertEqual(len(self.verifier.matching_headings(twice, "sddp")), 2)
        self.assertEqual(self.verifier.matching_headings(twice, "core-io"), [])
        result = self.run_driver("no-such-station", "--no-shared")
        self.assertEqual(result.returncode, 2)
        self.assertIn("matches 0 headings in BACKLOG.md", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_partition_names_a_doubly_assigned_path_even_when_the_union_holds(
        self,
    ) -> None:
        tree = sorted(f["path"] for f in self.inv["src"]["files"])
        self.assertEqual(self.verifier.partition_lists(self.inv, tree), ([], [], []))
        doubled = json.loads(json.dumps(self.inv))
        victim = next(f for f in doubled["src"]["files"] if f["substation"] == "5c")
        doubled["src"]["files"].append({**victim, "substation": "5b"})
        doubly, unassigned, phantom = self.verifier.partition_lists(doubled, tree)
        self.assertEqual((unassigned, phantom), ([], []), "the union still holds")
        self.assertEqual(doubly, [f"{victim['path']} <- 5c x2, 5b"])
        rolled = json.loads(json.dumps(self.inv))
        rolled["substation_rollup"]["5d"]["modules"].append("lp")
        doubly, _, _ = self.verifier.partition_lists(rolled, tree)
        self.assertEqual(len(doubly), 30)
        self.assertTrue(all(d.endswith("<- 5b, 5d") for d in doubly))
        thinned = json.loads(json.dumps(self.inv))
        dropped = thinned["src"]["files"].pop()
        thinned["src"]["files"].append({**dropped, "path": f"{SRC}/phantom.rs"})
        self.assertEqual(
            self.verifier.partition_lists(thinned, tree),
            ([], [dropped["path"]], [f"{SRC}/phantom.rs"]),
        )

    def test_timing_regexes_ignore_the_known_decoys_and_catch_figures(self) -> None:
        for decoy in TIMING_DECOYS:
            self.assertEqual(self.verifier.timing_hits(decoy), [], decoy)
        for figure in TIMING_FIGURES:
            self.assertNotEqual(self.verifier.timing_hits(figure), [], figure)
        self.assertEqual(self.verifier.timing_hits("\n".join(self.section.lines)), [])

    def test_retired_items_and_pd_004_never_surface_as_live_entries(self) -> None:
        live = {e.id for e in self.entries}
        self.assertTrue(live.isdisjoint({"CD-001", "CD-003", "CD-006", "PD-004"}))
        table = next(
            t
            for t in backlog_parse.parse_tables(self.section)
            if t and "ID" in t[0] and "Disposition" in t[0]
        )
        by_id = {row["ID"]: row for row in table}
        self.assertEqual(len(table), len(by_id), "one disposition-table row per id")
        for rid in RETIRED_ROWS:
            self.assertEqual(by_id[rid]["Disposition"], "retire", rid)
        resolved = {
            rid: next(v for k, v in by_id[rid].items() if k.startswith("Superseded"))
            for rid in RETIRED_ROWS
        }
        self.assertIn("`b051c410`", resolved["CD-001"])
        self.assertIn("`4075c4e8`", resolved["CD-003-construction-hop"])
        self.assertTrue(by_id["PD-004"]["Disposition"].startswith("existence-only"))
        self.assertTrue(
            (
                sc.REPO / "crates/cobre-sddp/src/training/backward_pass_state.rs"
            ).is_file()
        )

    def test_verification_report_carries_both_tables_all_passing(self) -> None:
        report_path = self.artifact("verification.md")
        if not report_path.exists():
            # verify-station.sh runs this module BEFORE rendering the report, so the very
            # first run has nothing to read; the next run (on the committed report) asserts it.
            self.skipTest("verification.md not rendered yet (bootstrap run)")
        report = report_path.read_text(encoding="utf-8")
        self.assertIn(f"Station baseline: `{self.baseline()}`", report)
        for i, check in enumerate(GENERIC_CHECKS, 1):
            self.assertIsNotNone(
                re.search(
                    rf"^\| {i} \| {check} \| `[^`]+` \| 0 \| PASS \|$", report, re.M
                ),
                check,
            )
        self.assertIn("## Station-specific checks — sddp", report)
        self.assertIsNotNone(
            re.search(r"^\| shared-verifier \| .* \| 0 \| PASS \|$", report, re.M)
        )
        for check in STATION_CHECKS:
            self.assertIsNotNone(
                re.search(rf"^\| {check} \| .* \| 0 \| PASS \|$", report, re.M), check
            )
        self.assertNotIn("| FAIL |", report)
        self.assertNotIn("Failures:", report)
        # The "Test suite: …" line is written AFTER this module runs and reflects this very
        # run, so asserting it here would be circular; only the check rows are asserted.


GATE_DATE = "2026-09-18"
DECISION_VERBS = {
    "ratify-as-presented",
    "re-dispose",
    "accept",
    "downgrade",
    "reject",
    "defer",
    "override-with-rationale",
    "take-the-alternative",
    "accept-with-rebaseline",
    "restate-byte-neutral",
}
HOLD_VERBS = {"take-the-alternative", "override-with-rationale", "reject", "defer"}
REBASELINE_VERBS = {"accept-with-rebaseline", "restate-byte-neutral", "defer", "reject"}
GATE_HEADINGS = (
    "### 1.1 Prior Wave 4/6/7 dispositions",
    "### 1.2 New entries by house severity",
    "### 1.3 Holds",
    "### 1.4 Re-baseline flags",
    "### 1.5 Cleared and dup-of",
    "### 1.6 Queues handed on",
    "### 1.7 Handoffs the alignment epic consumes",
    "### 1.8 Worker needs-human items",
    "## 2. Round plan",
    "## 3. Decision record",
    "### 3.1 Prior Wave 4/6/7 dispositions",
    "### 3.2 New entries",
    "### 3.3 Holds and re-baseline decisions",
    "### 3.4 Worker needs-human answers",
    "## 4. Handoffs after the gate",
)


class GateTests(sc.StationCase):
    """The owner gate (E05-8): gate.md, decisions.json, the BACKLOG owner-gate block and the marker.

    The gate ticket runs this class after recording the owner decision; the class asserts the
    recorded state and never asks a question or writes a file.
    """

    SLUG = "sddp"
    SECTION_TITLE = "sddp"

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = cls.station_dir().joinpath("gate.md").read_text(encoding="utf-8")
        cls.rec = sc.load_json(cls.station_dir() / "decisions.json")
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.env = sc.load_json(cls.station_dir() / "wave-dispositions.json")
        cls.perf = sc.load_json(cls.station_dir() / "perf-queue.json")
        cls.td = sc.load_json(cls.station_dir() / "td-queue.json")
        cls.align = sc.load_json(cls.station_dir() / "alignment-queue.json")
        register = "\n".join(backlog_parse.read_register(sc.BACKLOG))
        cls.section = register.split(SCAFFOLD, 1)[1].split("\n## ", 1)[0]
        cls.owner_gate = cls.section.split("#### Owner gate — decisions", 1)[1].split(
            "\n#### ", 1
        )[0]
        cls.verifier = load_verifier()

    def block(self, entry_id: str) -> str:
        return self.section.split(f"**{entry_id} · ", 1)[1].split("\n**", 1)[0]

    def test_gate_md_carries_one_decision_line_and_presents_the_priors_first(
        self,
    ) -> None:
        lines = [ln for ln in self.gate.splitlines() if ln.startswith("**Decision: ")]
        self.assertEqual(len(lines), 1)
        self.assertIsNotNone(
            re.match(r"^\*\*Decision: (ratified|returned)\*\*", lines[0])
        )
        positions = [self.gate.index(h) for h in GATE_HEADINGS]
        self.assertEqual(positions, sorted(positions), "gate.md sections out of order")
        prior = self.gate.split(GATE_HEADINGS[0], 1)[1].split("\n### ", 1)[0]
        rows = [ln for ln in prior.splitlines() if re.match(r"^\| \d+ \| ", ln)]
        self.assertEqual(len(rows), 22)
        groups = collections.Counter(ln.split("|")[4].strip() for ln in rows)
        self.assertEqual(groups, {"retire-with-commit": 3, "keep": 11, "sharpen": 8})
        for ln in rows:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if cells[3] == "retire-with-commit":
                self.assertRegex(cells[5], r"^[0-9a-f]{8}$", ln)
            if cells[3] == "sharpen":
                self.assertIn("→", cells[6], ln)
        self.assertIn("**PD-004 (existence-and-queue only, no fix option):**", prior)
        self.assertLess(
            self.gate.index(GATE_HEADINGS[0]), self.gate.index(GATE_HEADINGS[1])
        )

    def test_round_plan_keeps_holds_and_rebaseline_out_of_batches(self) -> None:
        plan = self.gate.split("## 2. Round plan", 1)[1].split("\n## ", 1)[0]
        rows = [
            [c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in plan.splitlines()
            if re.match(r"^\| R\d", ln)
        ]
        kinds = [r[1] for r in rows]
        self.assertEqual(kinds[:3], ["prior-disposition"] * 3)
        first_batch = kinds.index("severity-batch")
        self.assertGreater(first_batch, 2)
        for r in rows:
            if r[1] == "severity-batch":
                self.assertLessEqual(len(r[3].split("·")), 4, r[0])
                self.assertNotIn("CD-024-successor", r[2])
                self.assertNotRegex(r[2], r"\bCD-005\b")
            if r[1] == "conflicts-hold":
                self.assertIn("CD-024-successor", r[2])
                self.assertIn("override-with-rationale", r[3])
                self.assertIn("take-the-alternative", r[3])
            if r[1] == "rebaseline":
                self.assertIn("CD-005", r[2])
                self.assertIn("accept-with-rebaseline", r[3])
        self.assertEqual(sum(1 for k in kinds if k == "conflicts-hold"), 1)
        self.assertEqual(sum(1 for k in kinds if k == "rebaseline"), 1)
        self.assertEqual(sum(1 for k in kinds if k == "needs-human"), 43)
        self.assertNotIn("PD-004", " ".join(r[2] for r in rows))

    def test_decisions_cover_every_prior_id_and_calibrated_entry(self) -> None:
        wave_ids = {d["id"] for d in self.env["dispositions"]}
        new_ids = {r["id"] for r in self.cal["assigned"]}
        prior = [d for d in self.rec["decisions"] if d["kind"] == "prior-disposition"]
        new = [d for d in self.rec["decisions"] if d["kind"] == "new-entry"]
        self.assertEqual({d["id"] for d in prior}, wave_ids)
        self.assertEqual({d["id"] for d in new}, new_ids)
        self.assertEqual(len(prior), 22)
        for d in self.rec["decisions"]:
            self.assertIn(d["decision"], DECISION_VERBS, d["id"])
            self.assertTrue(d["rationale"].strip(), d["id"])
            if d.get("ownerDisposition") == "retire":
                self.assertRegex(
                    d["resolvingCommit"] or "", r"^[0-9a-f]{8,40}$", d["id"]
                )
                self.assertTrue(d["clearedMoved"], d["id"])
            if d["decision"] == "defer":
                self.assertTrue(
                    d["deferTrigger"], f"{d['id']}: defer without a trigger"
                )
            if d["decision"] == "override-with-rationale":
                self.assertTrue(d["overrideRationale"], d["id"])
            if d["decision"] == "reject":
                self.assertTrue(d["clearedMoved"], d["id"])
            if d["decision"] == "downgrade":
                self.assertNotEqual(d["newSeverity"], d["reviewerSeverity"])
        by_id = {d["id"]: d for d in prior}
        self.assertEqual(by_id["CD-001"]["resolvingCommit"], "b051c410")
        self.assertEqual(
            by_id["CD-003-construction-hop"]["resolvingCommit"], "4075c4e8"
        )
        for h in self.rec["holds"]:
            self.assertIn(h["decision"], HOLD_VERBS, h["id"])
        self.assertEqual([h["id"] for h in self.rec["holds"]], ["CD-024-successor"])
        for r in self.rec["rebaseline"]:
            self.assertIn(r["decision"], REBASELINE_VERBS, r["id"])
            if r["decision"] == "accept-with-rebaseline":
                self.assertEqual(len(r["rebaselineBar"]), 3)
                for bar in ("parity goldens", "rank-invariance", "mpiexec -n 1/2"):
                    self.assertTrue(any(bar in b for b in r["rebaselineBar"]), bar)
        self.assertEqual([r["id"] for r in self.rec["rebaseline"]], ["CD-005"])
        answered = [n for n in self.rec["needsHuman"] if n.get("answer", "").strip()]
        self.assertEqual(len(answered), len(self.rec["needsHuman"]))
        self.assertEqual(len(answered), 43)
        counts = self.rec["counts"]
        self.assertEqual(counts["needsHumanAnswered"], 43)
        verbs = [d["decision"] for d in new]
        self.assertEqual(counts["accepted"], verbs.count("accept"))
        self.assertEqual(counts["downgraded"], verbs.count("downgrade"))
        self.assertEqual(counts["rejected"], verbs.count("reject"))
        self.assertEqual(counts["deferred"], verbs.count("defer"))
        self.assertEqual(counts["presented"], len(self.rec["decisions"]))
        self.assertEqual(
            counts["priorRetired"],
            sum(1 for d in prior if d["ownerDisposition"] == "retire"),
        )
        self.assertTrue(self.rec["returned"])
        self.assertEqual(self.rec["decision"], "ratified")

    def test_backlog_carries_the_ratified_line_the_decision_column_and_the_marker(
        self,
    ) -> None:
        counts = self.rec["counts"]
        head = self.section.split("\n**Station.**", 1)[0]
        if self.rec["decision"] == "ratified":
            self.assertIn(
                f"**Ratified {GATE_DATE}** — owner gate; baseline `{self.baseline()[:8]}`",
                head,
            )
        else:
            self.assertNotIn("**Ratified ", head)
        table = self.section.split("#### Wave 4 / 6 / 7 dispositions", 1)[1].split(
            "\n#### ", 1
        )[0]
        header = next(ln for ln in table.splitlines() if ln.startswith("| ID | Wave |"))
        self.assertTrue(header.rstrip().endswith("| Decision |"))
        rows = [
            [c.strip() for c in ln.strip().strip("|").split("|")]
            for ln in table.splitlines()
            if ln.startswith("| ") and not ln.startswith("| ID ")
        ]
        self.assertEqual(len(rows), 23)
        for cells in rows:
            self.assertTrue(cells[-1], f"{cells[0]}: empty Decision cell")
            if cells[2] == "retire":
                self.assertIn("ratified → Cleared", cells[-1], cells[0])
            if cells[0] == "PD-004":
                self.assertIn("existence-and-queue only", cells[-1])
        marker = re.search(
            rf"^\*\*Gate: RETURNED {GATE_DATE}\*\* — baseline `{self.baseline()[:8]}`; "
            r"accepted (\d+), amended (\d+), downgraded (\d+), rejected (\d+), deferred (\d+), overridden (\d+)",
            self.owner_gate,
            re.M,
        )
        self.assertIsNotNone(marker, "Gate: RETURNED marker missing or malformed")
        assert marker is not None
        self.assertEqual(
            [int(marker.group(i)) for i in range(1, 7)],
            [
                counts[k]
                for k in (
                    "accepted",
                    "amended",
                    "downgraded",
                    "rejected",
                    "deferred",
                    "overridden",
                )
            ],
        )
        for d in self.rec["decisions"]:
            if d["kind"] != "new-entry":
                continue
            self.assertEqual(
                self.owner_gate.count(f"| {d['id']} | {d['decision']} |"), 1, d["id"]
            )
            if RANK[d["newSeverity"]] < RANK[d["reviewerSeverity"]]:
                self.assertIn(
                    f"| {d['id']} | {d['decision']} | {d['newSeverity']} (reviewer: {d['reviewerSeverity']}) |",
                    self.owner_gate,
                )
            else:
                self.assertNotIn(
                    f"| {d['id']} | {d['decision']} | {d['newSeverity']} (reviewer:",
                    self.owner_gate,
                )
        self.assertNotRegex(self.owner_gate, r"\| defer \|[^\n]*\| *- *\|\s*$")
        self.assertIn("**Cleared by this gate (do not re-raise):**", self.owner_gate)
        cleared = self.section.split("#### ↩︎ Cleared", 1)[1].split("\n#### ", 1)[0]
        for rid, sha in (
            ("CD-001", "b051c410"),
            ("CD-003-construction-hop", "4075c4e8"),
            ("CD-006", "3f4c3db3"),
        ):
            self.assertIsNotNone(
                re.search(
                    rf"^- \*\*{rid} — .*\*\* — retired, resolved by `{sha}`",
                    cleared,
                    re.M,
                ),
                rid,
            )

    def test_owner_decisions_sit_beneath_the_alignment_field(self) -> None:
        with_directions = [
            d
            for d in self.rec["decisions"]
            if d["kind"] == "new-entry" and d["directions"]
        ]
        self.assertGreaterEqual(len(with_directions), 30)
        for d in with_directions:
            block = self.block(d["id"])
            align_at = block.index("- **Alignment:**")
            for direction in d["directions"]:
                line = f"- **Owner decision ({GATE_DATE}, {direction['round']}):** {direction['choice']}"
                self.assertIn(line, block, d["id"])
                self.assertGreater(block.index(line), align_at, d["id"])

    def test_held_entries_have_an_owner_disposition_and_nothing_was_overridden(
        self,
    ) -> None:
        held = {
            r["id"]
            for r in self.cal["reusedIds"]
            if r.get("alignmentHint") == "conflicts"
        }
        held |= {
            r["id"]
            for r in self.cal["assigned"]
            if r.get("alignmentHint") == "conflicts"
        }
        self.assertEqual(held, {h["id"] for h in self.rec["holds"]})
        self.assertEqual(self.rec["holds"][0]["decision"], "take-the-alternative")
        self.assertFalse(self.rec["holds"][0]["idPromoted"])
        self.assertEqual(
            [d for d in self.rec["decisions"] if d.get("overrideRationale")], []
        )
        self.assertIn("no id promoted", self.owner_gate)
        self.assertTrue(self.align["holds"][0]["status"].startswith("CLOSED"))

    def test_queues_and_handoffs_reflect_the_ratified_state(self) -> None:
        kept = {
            d["id"]
            for d in self.rec["decisions"]
            if (d.get("perfQueue") or {}).get("effect") == "kept"
        }
        claim_rows = {q["id"] for q in self.perf["queue"] if q.get("severity")}
        self.assertEqual(claim_rows, kept)
        self.assertEqual(self.perf["gate"]["rowsStruck"], [])
        self.assertEqual(sorted(self.perf["gate"]["rowsKept"]), sorted(kept))
        for q in self.perf["queue"]:
            self.assertEqual(q["status"], "UNMEASURED")
            if q.get("severity"):
                self.assertIn(q["layout"], LAYOUTS)
        td_kept = {
            d["id"]
            for d in self.rec["decisions"]
            if (d.get("tdQueue") or {}).get("effect") == "kept"
        }
        self.assertEqual({q["id"] for q in self.td["queue"]}, td_kept)
        self.assertEqual(self.td["gate"]["rowsStruck"], [])
        decided = {d["id"] for d in self.rec["decisions"]}
        self.assertTrue({r["id"] for r in self.align["rows"]} <= decided)
        for rid in ("CD-082", "TD-045", "CD-004", "CD-085"):
            self.assertIn(rid, {r["id"] for r in self.align["rows"]})
        self.assertIn("released", self.align["lpInventory"])
        self.assertEqual(self.rec["handoffConfirmation"]["decision"], "confirm")
        self.assertEqual(self.verifier.timing_hits(self.gate), [])


class CleanTreeTests(unittest.TestCase):
    def test_no_tracked_file_under_an_evaluated_surface_is_modified(self) -> None:
        out = subprocess.run(
            [
                "git",
                "diff",
                "--stat",
                "HEAD",
                "--",
                "crates",
                "docs",
                "schemas",
                "scripts",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
