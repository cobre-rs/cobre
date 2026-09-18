"""cobre-sddp station tests: inventory census + partition, prior register.

Run from the repository root:
    python3 -m unittest plans/architecture-debt-audit/stations/sddp/tests/test_station.py
Every figure is re-derived from the tree the station evaluated (inventory.json's
`baseline`, read through `station_checks.Tree`), never from the worktree. Later sddp
tickets append their own stage classes here; the station verification runs the module.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

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
