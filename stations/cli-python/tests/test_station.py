"""Executable proof of the cli-python station artifacts.

Every figure in inventory.json is re-measured here on the tree at the station baseline
(`git show <pin>:<path>` blobs through station_checks.Tree), never read back from the JSON;
prior-register.md is checked for the four owned ids, resolving anchors and the supersession block.
Later cli/python tickets append their stage classes to this module.
"""

from __future__ import annotations

import collections
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

CLI = "crates/cobre-cli/src"
PY = "crates/cobre-python/src"
FACADE = "crates/cobre/src"
ROOTS = (CLI, PY, FACADE)
HEADER_BASELINE = re.compile(r"^Baseline:\s+([0-9a-f]{40})\b", re.M)
WRITE_CALL = re.compile(
    r"(?<![\w:])((?:cobre_io::|cobre_sddp::orchestration::)?write_[a-z_0-9]+)\s*\("
)
LOCAL_CLI = {
    "write_line",
    "write_training_outputs",
    "write_simulation_outputs",
    "write_sim_outputs_on_root",
}
LOCAL_PY = {"write_training_artifacts"} | {
    f"write_{s}_if_any"
    for s in (
        "fpha_hyperplanes",
        "evaporation_models",
        "generic_constraint_echo",
        "fixed_delivery",
        "fpha_deviation_points",
    )
}
PRIOR_IDS = ("CD-025", "CD-029", "CD-002", "CD-009")
ID_RE = re.compile(r"\b((?:CD|PD|OD|TD)-\d{3})\b")


def header_baseline() -> str:
    head = "\n".join(backlog_parse.read_register(sc.BACKLOG)[:40])
    hit = HEADER_BASELINE.search(head)
    assert hit, "BACKLOG.md header carries no `Baseline: <sha40>` line"
    return hit.group(1)


def substation_of(rel: str) -> str:
    if (
        rel.startswith(f"{CLI}/commands/run/")
        or rel == f"{CLI}/commands/broadcast.rs"
        or rel == f"{PY}/run.rs"
    ):
        return "S6a"
    if rel.startswith(f"{CLI}/"):
        return "S6b"
    return "S6c"


def writer_scan(tree: sc.Tree, paths: list[str]) -> dict[str, collections.Counter]:
    per_file: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for p in paths:
        for line in tree.read_text(p).splitlines():
            s = line.strip()
            if s.startswith("//") or s.startswith("use "):
                continue
            for m in WRITE_CALL.finditer(line):
                per_file[p][m.group(1).split("::")[-1]] += 1
    return per_file


def decl_line(text: str, pattern: str) -> int | None:
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return i
    return None


def all_lines(text: str, pattern: str) -> list[int]:
    return [
        i for i, line in enumerate(text.splitlines(), 1) if re.search(pattern, line)
    ]


class InventoryTests(sc.StationCase):
    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.files = cls.inv["files"]
        cls.by_path = {f["path"]: f for f in cls.files}

    def test_envelope_and_baseline_come_from_the_register_header(self) -> None:
        for key in (
            "station",
            "baseline",
            "generatedAt",
            "crates",
            "totals",
            "files",
            "substationRollup",
            "coverage",
            "commandSurface",
            "sddpCoupling",
            "writerBoundary",
            "parityEnforcement",
            "testSurface",
            "priorAnchors",
            "supersessions",
            "commands",
            "ticketFigureDeviations",
        ):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "cli-python")
        self.assertEqual(self.inv["baseline"], header_baseline())
        self.assertTrue(sc.is_register_pin(self.inv["baseline"]), sc.pin_history())
        self.assertEqual(
            [c["name"] for c in self.inv["crates"]],
            ["cobre-cli", "cobre-python", "cobre"],
        )

    def test_src_set_equals_the_tree_at_the_baseline(self) -> None:
        tree = self.tree()
        found = sorted(p for r in ROOTS for p in tree.rs_files(r))
        listed = [f["path"] for f in self.files]
        self.assertEqual(len(listed), len(set(listed)), "a path assigned twice")
        self.assertEqual(sorted(listed), found)
        self.assertEqual(len(found), 30)
        cov = self.inv["coverage"]
        self.assertEqual(cov["findCount"], len(found))
        self.assertEqual(cov["assignedCount"], len(listed))
        self.assertEqual(
            (cov["unassigned"], cov["doubleAssigned"], cov["phantom"]), ([], [], [])
        )
        for c in self.inv["crates"]:
            rows = [f for f in self.files if f["crate"] == c["name"]]
            self.assertEqual(c["srcFiles"], len(rows), c["name"])
            self.assertEqual(c["lines"], sum(f["lines"] for f in rows), c["name"])
            self.assertEqual(
                c["nonTestLines"], sum(f["nonTestLines"] for f in rows), c["name"]
            )
        self.assertEqual(self.inv["totals"]["files"], len(listed))
        self.assertEqual(
            self.inv["totals"]["lines"], sum(f["lines"] for f in self.files)
        )

    def test_every_file_reproduces_lines_non_test_substation_and_symbols(self) -> None:
        tree = self.tree()
        for f in self.files:
            text = tree.read_text(f["path"])
            self.assertEqual(
                f["lines"], tree.read_bytes(f["path"]).count(b"\n"), f["path"]
            )
            non_test, _ = sc.classify_lines(f["path"], text)
            self.assertEqual(f["nonTestLines"], non_test, f["path"])
            self.assertEqual(f["inlineTestLines"], f["lines"] - non_test, f["path"])
            self.assertEqual(f["inlineTestFns"], text.count("#[test]"), f["path"])
            self.assertEqual(f["substation"], substation_of(f["path"]), f["path"])
            self.assertEqual(f["crate"], f["path"].split("/")[1])
            if f["path"] != f"{FACADE}/lib.rs":
                self.assertTrue(f["topSymbols"], f["path"])
                first = f["topSymbols"][0]
                self.assertTrue(
                    sc.anchor_exists(f"`{f['path']}::{first['name']}`", tree),
                    f"{f['path']}::{first['name']}",
                )
        self.assertEqual(self.by_path[f"{FACADE}/lib.rs"]["lines"], 2)
        self.assertEqual(self.by_path[f"{FACADE}/lib.rs"]["topSymbols"], [])

    def test_partition_is_the_declared_three_sub_surfaces(self) -> None:
        roll = {r["id"]: r for r in self.inv["substationRollup"]}
        self.assertEqual(sorted(roll), ["S6a", "S6b", "S6c"])
        for r in roll.values():
            rows = [f for f in self.files if f["substation"] == r["id"]]
            self.assertEqual(r["files"], len(rows), r["id"])
            self.assertEqual(r["lines"], sum(f["lines"] for f in rows), r["id"])
            self.assertEqual(
                sorted(r["members"]), sorted(f["path"] for f in rows), r["id"]
            )
        s6a = set(roll["S6a"]["members"])
        self.assertEqual(len(s6a), 8)
        self.assertIn(f"{CLI}/commands/broadcast.rs", s6a)
        self.assertIn(f"{PY}/run.rs", s6a)
        self.assertEqual(
            len([p for p in s6a if p.startswith(f"{CLI}/commands/run/")]), 6
        )
        self.assertTrue(all(p.startswith(f"{CLI}/") for p in roll["S6b"]["members"]))
        self.assertTrue(
            all(p.startswith((f"{PY}/", f"{FACADE}/")) for p in roll["S6c"]["members"])
        )
        self.assertIn(f"{FACADE}/lib.rs", roll["S6c"]["members"])
        self.assertEqual(sum(r["files"] for r in roll.values()), 30)
        self.assertEqual(
            sum(r["lines"] for r in roll.values()), self.inv["totals"]["lines"]
        )

    def test_facade_and_workspace_facts_hold_at_the_baseline(self) -> None:
        tree = self.tree()
        cli, py, facade = self.inv["crates"]
        self.assertEqual(
            tree.read_text(f"{FACADE}/lib.rs").splitlines(), facade["libRs"]
        )
        self.assertIn("re-exports nothing", facade["libRs"][0])
        self.assertFalse(facade["dependenciesSection"])
        self.assertNotIn("[dependencies]", tree.read_text("crates/cobre/Cargo.toml"))
        self.assertEqual(facade["dependencies"], [])
        lo, hi = (int(x) for x in py["workspaceExcluded"]["lines"].split("-"))
        ws = tree.read_text("Cargo.toml").splitlines()
        self.assertTrue(ws[lo - 1].strip().startswith("exclude"))
        self.assertTrue(
            any("crates/cobre-python" in ws[i - 1] for i in range(lo, hi + 1))
        )
        toml = tree.read_text("crates/cobre-python/Cargo.toml").splitlines()
        self.assertIn("cdylib", toml[py["crateType"]["line"] - 1])

    def test_command_surface_reproduces(self) -> None:
        tree = self.tree()
        cs = self.inv["commandSurface"]
        main_rs = tree.read_text(f"{CLI}/main.rs")
        enum_line = decl_line(main_rs, r"^\s*enum Command\b")
        self.assertEqual(cs["enum"]["line"], enum_line)
        variants = []
        for line in main_rs.splitlines()[enum_line:]:
            if line.strip().startswith("}"):
                break
            m = re.match(r"^\s{4}([A-Z]\w*)\s*(\(|,|$)", line)
            if m:
                variants.append(m.group(1))
        self.assertEqual(cs["enum"]["variants"], variants)
        self.assertEqual(cs["enum"]["count"], len(variants))
        self.assertEqual(variants, ["Init", "Run", "Validate", "Schema", "Version"])
        self.assertEqual(
            cs["dispatch"]["line"],
            decl_line(main_rs, r"^\s*let result = match cli\.command"),
        )
        for v in variants:
            self.assertIn(f"Command::{v}", main_rs)
        self.assertFalse(tree.exists(f"{CLI}/commands/report.rs"))
        self.assertFalse(tree.exists(f"{CLI}/commands/summary.rs"))
        self.assertFalse(cs["broadcastModule"]["isSubcommand"])
        self.assertNotIn("Broadcast", variants)
        commands_mod = tree.read_text(f"{CLI}/commands/mod.rs").splitlines()
        self.assertEqual(
            commands_mod[cs["broadcastModule"]["declaredAt"]["line"] - 1].strip(),
            "pub(crate) mod broadcast;",
        )
        self.assertEqual(
            cs["broadcastModule"]["lines"],
            tree.read_bytes(f"{CLI}/commands/broadcast.rs").count(b"\n"),
        )
        sha = cs["removedSinceTicket"]["commit"].split()[0]
        rp = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", f"{sha}^{{commit}}"],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rp.returncode, 0, sha)

    def test_sddp_coupling_reproduces(self) -> None:
        tree = self.tree()
        sc_ = self.inv["sddpCoupling"]
        per_file: dict[str, int] = {}
        use_lines = 0
        for p in tree.rs_files(CLI):
            t = tree.read_text(p)
            if "cobre_sddp::" in t:
                per_file[p] = t.count("cobre_sddp::")
            use_lines += len(all_lines(t, r"^use cobre_sddp"))
        self.assertEqual(sc_["occurrences"], sum(per_file.values()))
        self.assertEqual(sc_["filesWithOccurrences"], len(per_file))
        self.assertEqual(sc_["useLines"], use_lines)
        ranking = [(r["file"], r["occurrences"]) for r in sc_["perFileRanking"]]
        self.assertEqual(
            ranking, sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        self.assertEqual(ranking[0][0], f"{CLI}/commands/run/policy.rs")
        summary = tree.read_text(f"{CLI}/summary.rs").splitlines()
        self.assertTrue(
            summary[sc_["reExport"]["line"] - 1].startswith("pub use cobre_sddp::")
        )
        self.assertEqual(sc_["reExport"]["count"], len(sc_["reExport"]["items"]))
        self.assertGreaterEqual(sc_["reExport"]["count"], 6)
        self.assertFalse(sc_["reportRs"]["exists"])

    def test_writer_boundary_reproduces_and_the_sets_are_identical(self) -> None:
        tree = self.tree()
        wb = self.inv["writerBoundary"]
        run_files = sorted(tree.rs_files(f"{CLI}/commands/run"))
        cli = writer_scan(tree, run_files)
        cli_ext: set[str] = set()
        for r in wb["cli"]["perFile"]:
            c = cli.get(r["file"], collections.Counter())
            self.assertEqual(
                r["externalNames"],
                sorted(k for k in c if k not in LOCAL_CLI),
                r["file"],
            )
            self.assertEqual(r["externalCount"], len(r["externalNames"]))
            cli_ext |= set(r["externalNames"])
        py = writer_scan(tree, [f"{PY}/run.rs"])[f"{PY}/run.rs"]
        py_ext = {k for k in py if k not in LOCAL_PY}
        self.assertEqual(sorted(cli_ext), wb["cli"]["externalNames"])
        self.assertEqual(sorted(py_ext), wb["python"]["externalNames"])
        self.assertEqual(
            (wb["cli"]["externalCount"], wb["python"]["externalCount"]), (17, 17)
        )
        self.assertEqual((wb["cliOnly"], wb["pyOnly"]), ([], []))
        self.assertTrue(wb["identical"])
        per = {
            r["file"].rsplit("/", 1)[-1]: r["externalCount"]
            for r in wb["cli"]["perFile"]
        }
        self.assertEqual(
            (per["outputs.rs"], per["setup.rs"], per["simulation.rs"]), (13, 3, 1)
        )
        self.assertEqual(
            (per["mod.rs"], per["policy.rs"], per["training.rs"]), (0, 0, 0)
        )
        self.assertEqual(
            [o["name"] for o in wb["cli"]["localOrchestrators"]],
            [
                "write_training_outputs",
                "write_simulation_outputs",
                "write_sim_outputs_on_root",
            ],
        )
        for o in wb["cli"]["localOrchestrators"] + wb["python"]["localHelpers"]:
            text = tree.read_text(o["file"]).splitlines()
            self.assertRegex(text[o["line"] - 1], rf"fn {o['name']}\b", o["name"])
        self.assertEqual(len(wb["python"]["localHelpers"]), 6)
        self.assertEqual(wb["python"]["localHelpersSeen"], sorted(LOCAL_PY))
        raw = len(
            set(re.findall(r"write_[a-z_0-9]*\(", tree.read_text(f"{PY}/run.rs")))
        )
        self.assertEqual(wb["python"]["rawDistinctNames"], raw)
        self.assertEqual(raw, 23)

    def test_parity_enforcement_records_measured_coverage_per_layer(self) -> None:
        tree = self.tree()
        p = self.inv["parityEnforcement"]
        l1 = p["layer1_source"]
        self.assertEqual(l1["exit"], 0)
        self.assertIn("write functions in both paths", l1["stdout"])
        shared = sorted(set(l1["scriptSeesCli"]) & set(l1["scriptSeesPython"]))
        self.assertEqual(l1["sharedCount"], len(shared))
        self.assertGreaterEqual(l1["sharedCount"], l1["minSharedFloor"])
        cov = l1["coverageOfMeasuredWriters"]
        self.assertEqual(cov["measuredExternalNames"], 17)
        self.assertEqual(cov["coveredCount"] + len(cov["notCoveredByName"]), 17)
        self.assertEqual(cov["notCoveredByName"], ["write_scenario"])
        self.assertIn("SimulationParquetWriter", l1["scriptSeesCli"])
        self.assertEqual(
            l1["normalise"].get("write_checkpoint"), "write_policy_checkpoint"
        )
        script = tree.read_text("scripts/ci/check_python_parity.py")
        self.assertIn("def parse_imports", script)
        self.assertIn("--min-shared", script)
        for layer in p["layer2_runtime_python"]:
            self.assertTrue(tree.is_file(layer["file"]), layer["file"])
            self.assertEqual(
                layer["tests"],
                len(re.findall(r"^def test_", tree.read_text(layer["file"]), re.M)),
            )
        l3 = p["layer3_runtime_workspace"]
        self.assertEqual(
            l3["tests"], re.findall(r"^fn (\w+)\(", tree.read_text(l3["file"]), re.M)
        )
        ci = tree.read_text(".github/workflows/ci.yml").splitlines()
        for ref in p["ciReachability"]["layer2"]:
            path, line = ref.rsplit(":", 1)
            self.assertIn(
                "pytest crates/cobre-python/tests/",
                tree.read_text(path).splitlines()[int(line) - 1],
                ref,
            )
        m = re.search(
            r"\.github/workflows/ci\.yml:(\d+)", p["ciReachability"]["layer1_and_3"]
        )
        self.assertIsNotNone(m)
        assert m is not None
        self.assertIn("check_python_parity.py", ci[int(m.group(1)) - 1])
        claude = tree.read_text("CLAUDE.md").splitlines()
        self.assertIn("**Python parity**", claude[p["hardRule"]["line"] - 1])

    def test_test_surface_reproduces(self) -> None:
        tree = self.tree()
        ts = self.inv["testSurface"]
        bins = {b["path"]: b for b in ts["cobreCli"]["integrationBinaries"]}
        self.assertEqual(sorted(bins), sorted(tree.rs_files("crates/cobre-cli/tests")))
        self.assertEqual(len(bins), 14)
        for path, b in bins.items():
            self.assertEqual(b["testFns"], tree.read_text(path).count("#[test]"), path)
            self.assertEqual(b["lines"], tree.read_bytes(path).count(b"\n"), path)
        self.assertEqual(
            ts["cobreCli"]["binaryTestFns"], sum(b["testFns"] for b in bins.values())
        )
        inline = {
            f["path"][len(CLI) + 1 :]: f["inlineTestFns"]
            for f in self.files
            if f["crate"] == "cobre-cli" and f["inlineTestFns"]
        }
        self.assertEqual(
            ts["cobreCli"]["inlineTestFnsPerFile"],
            dict(sorted(inline.items(), key=lambda kv: (-kv[1], kv[0]))),
        )
        self.assertEqual(
            ts["cobreCli"]["totalTestFns"],
            ts["cobreCli"]["binaryTestFns"] + ts["cobreCli"]["inlineTestFns"],
        )
        py_files = sorted(
            p for p in tree.ls_files("crates/cobre-python/tests") if p.endswith(".py")
        )
        self.assertEqual(
            [t["path"] for t in ts["cobrePython"]["pytestFiles"]], py_files
        )
        for t in ts["cobrePython"]["pytestFiles"]:
            self.assertEqual(
                t["testFns"],
                len(re.findall(r"^def test_", tree.read_text(t["path"]), re.M)),
                t["path"],
            )
            self.assertEqual(
                t["lines"], tree.read_bytes(t["path"]).count(b"\n"), t["path"]
            )
        self.assertEqual(
            ts["cobrePython"]["testFunctions"],
            sum(t["testFns"] for t in ts["cobrePython"]["pytestFiles"]),
        )
        rust = {
            f["path"][len(PY) + 1 :]: f["inlineTestFns"]
            for f in self.files
            if f["crate"] == "cobre-python" and f["inlineTestFns"]
        }
        self.assertEqual(
            ts["cobrePython"]["rustTestFnsPerFile"],
            dict(sorted(rust.items(), key=lambda kv: (-kv[1], kv[0]))),
        )
        self.assertEqual(ts["cobrePython"]["rustTestFns"], sum(rust.values()))

    def test_ci_visibility_is_evidenced_on_the_baseline_workflow(self) -> None:
        tree = self.tree()
        ci = self.inv["testSurface"]["ciVisibility"]
        yml = tree.read_text(".github/workflows/ci.yml").splitlines()
        self.assertIn(
            "Run Rust tests for the bindings crate",
            yml[ci["rustTestsStep"]["stepNameLine"] - 1],
        )
        for n in ci["rustTestsStep"]["commandLines"]:
            self.assertIn(
                "cargo test --manifest-path crates/cobre-python/Cargo.toml", yml[n - 1]
            )
        self.assertIn("compiled and run in CI", ci["verdictAtBaseline"])
        self.assertIn("superseded", ci["verdictAtBaseline"])
        for doc in ci["staleDocs"]:
            self.assertIn(
                doc["heading"],
                tree.read_text(doc["file"]).splitlines()[doc["line"] - 1],
                doc["file"],
            )

    def test_supersessions_carry_old_new_and_command(self) -> None:
        tree = self.tree()
        sup = {s["fact"]: s for s in self.inv["supersessions"]}
        for s in sup.values():
            for key in ("fact", "inherited", "inheritedFrom", "baseline", "command"):
                self.assertIn(key, s, s["fact"])
            self.assertNotEqual(s["inherited"], s["baseline"], s["fact"])
        fc = sup["`StudyParams::from_config` non-test call sites"]
        self.assertEqual((fc["inherited"], fc["baseline"]), (1, 5))
        for site in fc["sites"]:
            self.assertIn(
                "StudyParams::from_config",
                tree.read_text(site["file"]).splitlines()[site["line"] - 1],
                site,
            )
        self.assertEqual(
            {s["file"] for s in fc["sites"]},
            {
                "crates/cobre-sddp/src/setup/mod.rs",
                f"{CLI}/commands/validate.rs",
                f"{CLI}/commands/broadcast.rs",
                f"{PY}/io.rs",
                f"{PY}/run.rs",
            },
        )
        names = sup["distinct `write_*(` names in cobre-python/src/run.rs"]
        self.assertEqual((names["inherited"], names["baseline"]), (26, 23))
        prod = sup["cobre-cli production (non-test) lines"]
        self.assertEqual(
            prod["inherited"], {"register 2026-08-22": 4522, "ticket @a136840d": 5419}
        )
        self.assertEqual(prod["baseline"], self.inv["crates"][0]["nonTestLines"])
        self.assertEqual(sup["Command enum variants"]["baseline"], 5)
        self.assertEqual(
            sup["parity gate shared writer names"]["baseline"],
            self.inv["parityEnforcement"]["layer1_source"]["sharedCount"],
        )

    def test_ticket_figure_deviations_are_real(self) -> None:
        dev = self.inv["ticketFigureDeviations"]["changed"]
        self.assertGreater(len(dev), 0)
        for d in dev:
            self.assertNotEqual(d["ticket"], d["measured"], d["figure"])
        figures = {d["figure"]: d for d in dev}
        self.assertEqual(
            (
                figures["crates.cobre-cli.srcFiles"]["ticket"],
                figures["crates.cobre-cli.srcFiles"]["measured"],
            ),
            (20, 18),
        )
        self.assertEqual(
            (figures["total.files"]["ticket"], figures["total.files"]["measured"]),
            (32, 30),
        )
        self.assertEqual(
            (
                figures["commandSurface.variants"]["ticket"],
                figures["commandSurface.variants"]["measured"],
            ),
            (7, 5),
        )
        self.assertEqual(figures["parityEnforcement.layer1.sharedCount"]["ticket"], 4)


class PriorRegisterTests(sc.StationCase):
    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (cls.station_dir() / "prior-register.md").read_text(encoding="utf-8")
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))

    def test_owned_ids_table_has_the_four_rows_pending(self) -> None:
        table = self.text.split("## Owned prior IDs", 1)[1].split("\n## ", 1)[0]
        for rid in PRIOR_IDS:
            self.assertIsNotNone(
                re.search(rf"^\| {rid} \| \d \| .*\| _pending_ \|$", table, re.M), rid
            )
        rows = [ln for ln in table.splitlines() if ln.startswith("| CD-")]
        self.assertEqual(len(rows), 4)
        self.assertIn("→", table)
        self.assertIn("cobre-io (L2)", table)

    def test_every_cited_id_exists_in_the_register(self) -> None:
        for rid in sorted(set(ID_RE.findall(self.text))):
            self.assertIn(f"**{rid} · ", self.register, rid)

    def test_every_anchor_resolves_at_the_baseline(self) -> None:
        tree = self.tree()
        anchors = sc.anchors_in(self.text)
        self.assertGreaterEqual(len(anchors), 12)
        for a in anchors:
            self.assertTrue(sc.anchor_exists(a, tree), a)
        self.assertIsNone(
            re.search(r"`crates/[^`]+\.rs:\d+`", self.text), "a line-only code anchor"
        )

    def test_check_anchors_exits_zero_over_every_section(self) -> None:
        base = self.baseline()
        titles = [
            ln[3:].strip() for ln in self.text.splitlines() if ln.startswith("## ")
        ]
        self.assertGreaterEqual(len(titles), 5)
        for title in titles:
            self.assertEqual(
                sc.run_checker(
                    "check-anchors.py",
                    title,
                    "--register",
                    str(self.artifact("prior-register.md")),
                    "--baseline",
                    base,
                ),
                0,
                title,
            )

    def test_destination_rule_normative_and_supersession_blocks(self) -> None:
        for heading in (
            "## Destination rule",
            "## Normative — not candidates",
            "## Supersessions",
            "## Do not re-raise",
            "## Re-derive",
        ):
            self.assertIn(heading, self.text)
        rule = self.text.split("## Destination rule", 1)[1].split("\n## ", 1)[0]
        self.assertIn("cobre-io (L2)", rule)
        self.assertIn("`conflicts`", rule)
        norm = self.text.split("## Normative — not candidates", 1)[1].split("\n## ", 1)[
            0
        ]
        for needle in (
            "Python-parity hard rule",
            "Reserved-seams check",
            "no entry for the facade crate",
            "Stale mirror entry",
            "Part-I I.5",
        ):
            self.assertIn(needle, norm)
        sup = self.text.split("## Supersessions", 1)[1].split("\n## ", 1)[0]
        rows = [
            ln
            for ln in sup.splitlines()
            if ln.startswith("| ")
            and not ln.startswith("| Fact")
            and not ln.startswith("| --")
        ]
        self.assertGreaterEqual(len(rows), 7)
        for needle in ("from_config", "| 1 | 5 ", "| 26 | 23 |", "4522", "5419"):
            self.assertIn(needle, sup)


OWNED = {"CD-025", "CD-029", "CD-002", "CD-009"}
NOT_OURS = {"CD-004", "CD-001", "CD-003-construction-hop", "CD-026", "CD-082"}
ALIGN = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
DISPOSITIONS = {"keep", "retire", "sharpen"}
DO_NOT_TOUCH = {"CD-008", "PD-001", "PD-004"}
PARITY_HALF_COMMIT = "fc81427a"
REPORT_REMOVAL_COMMIT = "797ba443"
TEN_NONES = "(None, None, None, None, None, None, None, None, None, None)"
POLICY_DIR_LINE = "let policy_dir = ctx.output_dir.join(&setup.policy_path);"
RUN_DOC = "Load a case directory, train a policy, and run simulation."
DECL_KINDS = r"(?:fn|struct|enum|trait|type|const|static|mod|impl)"


def render_probe(title: str, rows: list[dict[str, Any]], baseline: str) -> str:
    out = [f"## {title}", ""]
    for d in rows:
        anchors = [
            d["baselineAnchor"],
            *[a for a in d.get("anchors", []) if a != d["baselineAnchor"]],
        ]
        rid = d["id"].split("-construction-hop")[0]
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


def commit_is_ancestor_of(sha: str, pin: str) -> bool:
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=sc.REPO, check=False
    )
    if exists.returncode != 0:
        return False
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, pin], cwd=sc.REPO, check=False
        ).returncode
        == 0
    )


class WaveReverifyTests(sc.StationCase):
    """Wave 5 + Part-I I.5 re-verification (E06-2): wave-dispositions.json and partI-handoff.json.

    Every line the two artifacts record is re-measured on the pin's blobs; anchors are symbol-only
    and resolve through the same helper check-anchors.py uses.
    """

    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = sc.load_json(cls.station_dir() / "wave-dispositions.json")
        cls.handoff = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.by_id = {d["id"]: d for d in cls.env["dispositions"]}
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))

    def check_spans(self, tree: sc.Tree, spans: list[dict[str, Any]], ctx: str) -> None:
        for s in spans:
            text = tree.read_text(s["path"]).splitlines()
            at = [s["line"]] if "line" in s else list(s["lines"])
            needles = s.get("needles") or [s["needle"]] * len(at)
            self.assertEqual(len(needles), len(at), (ctx, s))
            for n, needle in zip(at, needles):
                self.assertIn(needle, text[n - 1], (ctx, s["path"], n))

    def test_roster_is_exactly_the_four_owned_ids_and_disowns_the_rest(self) -> None:
        ids = [d["id"] for d in self.env["dispositions"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), OWNED)
        self.assertEqual(self.env["ownedCount"], 4)
        self.assertEqual({n["id"] for n in self.env["notOwned"]}, NOT_OURS)
        for n in self.env["notOwned"]:
            self.assertTrue(n["owningStation"], n["id"])
        self.assertEqual(self.env["baseline"], self.baseline())
        self.assertEqual({d["wave"] for d in self.env["dispositions"]}, {4, 5, 7})
        for disp, members in self.env["byDisposition"].items():
            self.assertEqual(
                sorted(members),
                sorted(
                    d["id"]
                    for d in self.env["dispositions"]
                    if d["disposition"] == disp
                ),
                disp,
            )
        self.assertEqual(self.env["existenceChecks"], [])
        for rid in OWNED:
            self.assertIn(f"**{rid} · ", self.register)
            self.assertTrue(
                self.register.splitlines()[
                    self.by_id[rid]["registerLine"] - 1
                ].startswith(f"**{rid} · "),
                rid,
            )

    def test_every_anchor_is_symbol_only_and_resolves_at_the_baseline(self) -> None:
        tree = self.tree()
        for d in [*self.env["dispositions"], *self.env["notOwned"]]:
            with self.subTest(id=d["id"]):
                anchors = list(d.get("anchors", []))
                if "baselineAnchor" in d:
                    anchors.insert(0, d["baselineAnchor"])
                for a in anchors:
                    self.assertEqual(set(a), {"path", "symbol"}, a)
                    self.assertTrue(
                        sc.anchor_exists(f"`{a['path']}::{a['symbol']}`", tree), a
                    )
                if d["id"] in OWNED:
                    self.assertTrue(
                        d["priorAnchor"]["raw"], "provenance anchors dropped"
                    )
                    for u in d["unresolvedHistoricalSymbols"]:
                        self.assertFalse(
                            sc.anchor_exists(f"`{u['path']}::{u['symbol']}`", tree), u
                        )
                        self.assertTrue(u["why"])

    def test_anchor_drift_rows_re_measure_on_the_pin(self) -> None:
        tree = self.tree()
        for d in self.env["dispositions"]:
            self.assertTrue(d["anchorDrift"], d["id"])
            for row in d["anchorDrift"]:
                with self.subTest(id=d["id"], symbol=row["symbol"]):
                    text = tree.read_text(row["path"])
                    measured = decl_line(
                        text,
                        rf"^\s*(?:pub(?:\([^)]*\))?\s+)?{DECL_KINDS}\s+{row['symbol']}\b",
                    )
                    self.assertEqual(row["baselineLine"], measured)
                    self.assertIn(row["symbol"], row["resolvedBy"])
                    self.assertIn(self.baseline()[:8], row["resolvedBy"])
                    self.assertTrue(row["status"])
            self.check_spans(tree, d["spans"], d["id"])

    def test_dispositions_are_well_formed(self) -> None:
        for d in self.env["dispositions"]:
            with self.subTest(id=d["id"]):
                self.assertIn(d["disposition"], DISPOSITIONS)
                if d["disposition"] == "retire":
                    self.assertTrue(
                        commit_is_ancestor_of(d["resolvingCommit"], self.baseline())
                    )
                else:
                    self.assertIsNone(d["resolvingCommit"])
                    self.assertTrue(d["survivingClaim"])
                if d["disposition"] == "sharpen":
                    self.assertTrue(d["supersededClaim"] and d["survivingClaim"])
                self.assertIn(d["alignment"], ALIGN)
                h = d["alignmentHint"]
                self.assertEqual(h["value"], d["alignment"])
                self.assertRegex(h["citation"], r"Part (IV|V)\.\d")
                self.assertTrue(h["argument"])
                self.assertIn("beyond-sddp-generalization.md", d["alignmentCitation"])
                if h["value"] == "conflicts":
                    self.assertTrue(h["guardrailBreach"])
                for c in d["conflicts"]:
                    self.assertEqual(c["tag"], "conflicts")
                    self.assertFalse(c["presentedAsDirection"])
                    self.assertIn("Part IV", c["rule"])
                self.assertEqual(
                    d["originalRatingKept"],
                    d["recordedSeverity"] == d["reverifiedSeverity"],
                )
                if not d["originalRatingKept"]:
                    self.assertTrue(d["severityNote"])
                    self.assertTrue(
                        any(
                            "downgrade" in q["question"].lower()
                            for q in d["needsHuman"]
                        )
                    )
                for q in d["needsHuman"]:
                    self.assertEqual(q["forGate"], "E06-8")
                    self.assertTrue(q["recommended"])
                for o in d["retiredOverlaps"]:
                    self.assertTrue(o["justification"])
                    self.assertIn(o["kind"], ("same-entry", "anchor-overlap"))
        self.assertEqual(self.env["byDisposition"]["retire"], [])

    def test_cd_025_is_restated_as_the_cobre_io_entry_point(self) -> None:
        tree = self.tree()
        d = self.by_id["CD-025"]
        self.assertEqual(d["disposition"], "sharpen")
        self.assertEqual(d["wave"], 5)
        self.assertEqual((d["recordedSeverity"], d["reverifiedSeverity"]), ("B", "B"))
        sw = d["supersededWording"]
        self.assertIn(
            " ".join(sw["old"].split()), " ".join(self.register.split()), "not verbatim"
        )
        self.assertIn("(cobre-sddp or cobre-io)", sw["old"])
        self.assertIn("(cobre-sddp or cobre-io)", sw["oldAsQuotedByTicket"])
        self.assertEqual(sw["newDestination"], "cobre-io (L2)")
        self.assertIn("cobre-io", d["restatedFixShape"])
        self.assertIn("shared output-orchestration entry point", d["restatedFixShape"])
        self.assertEqual(
            d["phase0aRestatement"]["artifact"],
            "shared output-orchestration entry point",
        )
        for key in ("shape", "guardrail", "byteNeutralityBar", "anchorOwner"):
            self.assertTrue(d["phase0aRestatement"][key], key)
        self.assertIn("Engine enum stays at L4", d["phase0aRestatement"]["guardrail"])
        self.assertEqual(d["alignment"], "advances-0a")
        self.assertRegex(d["alignmentCitation"], r"V\.1")
        self.assertIn("wire ED outputs in both CLI and Python", d["alignmentCitation"])
        variants = [c["variant"] for c in d["conflicts"]]
        self.assertEqual(len(variants), 2)
        self.assertTrue(any("cobre-sddp" in v for v in variants))
        self.assertTrue(any("cobre-cli-local" in v for v in variants))
        self.assertNotIn("cobre-sddp", d["restatedFixShape"])
        self.assertEqual(d["partIRef"], "I.5")
        # the five helpers exist in run.rs and are called from study.rs only
        run_rs = tree.read_text(f"{PY}/run.rs")
        study = tree.read_text(f"{PY}/study.rs")
        for name in sorted(LOCAL_PY - {"write_training_artifacts"}):
            self.assertRegex(run_rs, rf"fn {name}\(")
            self.assertIn(f"{name}(", study)
        callers = [
            p
            for p in tree.rs_files(PY)
            if re.search(r"(?<!fn )\bwrite_[a-z_]+_if_any\(", tree.read_text(p))
        ]
        self.assertEqual(callers, [f"{PY}/study.rs"])
        counts = d["evidence"]["writeCallCounts"]
        cli = sum(
            len(re.findall(r"write_[a-z_0-9]*\(", tree.read_text(p)))
            for p in tree.rs_files(f"{CLI}/commands/run")
        )
        self.assertEqual(counts["cliRunPath"], cli)
        self.assertEqual(
            counts["pythonRunRs"], len(re.findall(r"write_[a-z_0-9]*\(", run_rs))
        )
        self.assertNotEqual((counts["cliRunPath"], counts["pythonRunRs"]), (41, 29))

    def test_cd_029_three_facts_and_the_parity_half_supersession(self) -> None:
        tree = self.tree()
        d = self.by_id["CD-029"]
        self.assertEqual(d["disposition"], "sharpen")
        phases = tree.read_text("crates/cobre-sddp/src/validate_phases.rs").splitlines()
        prose = next(s for s in d["spans"] if "four" in s.get("needle", ""))
        self.assertIn("four", phases[prose["line"] - 1])
        rows = [ln for ln in phases if re.match(r"^/// \| \[`\w+`\]", ln)]
        variants = [
            ln
            for ln in phases
            if re.match(r"^    (Config|Stochastic|HydroModels),$", ln)
        ]
        self.assertEqual((len(rows), len(variants)), (3, 3))
        validate = tree.read_text(f"{CLI}/commands/validate.rs")
        for fn in ("format_boundary_error", "reconcile_boundary", "run_boundary_check"):
            self.assertIsNotNone(re.search(rf"^fn {fn}\(", validate, re.M), fn)
        self.assertNotIn("PrepPhase::Boundary", validate)
        self.assertNotIn("Boundary,", "\n".join(phases))
        io_rs = tree.read_text(f"{PY}/io.rs").splitlines()
        boundary = sum(1 for ln in io_rs if "boundary" in ln)
        self.assertEqual(d["evidence"]["ioBoundaryCount"]["baseline"], boundary)
        self.assertGreater(boundary, 0)
        self.assertEqual(d["evidence"]["ioBoundaryCount"]["ticket"], 0)
        halves = {h["half"]: h for h in d["halves"]}
        self.assertEqual(set(halves), {"doc-drift", "boundary-bypass", "python-parity"})
        self.assertEqual(halves["python-parity"]["status"], "fixed")
        self.assertEqual(halves["python-parity"]["resolvingCommit"], PARITY_HALF_COMMIT)
        self.assertTrue(commit_is_ancestor_of(PARITY_HALF_COMMIT, self.baseline()))
        py_run = tree.read_text(f"{PY}/run.rs")
        self.assertRegex(py_run, r"fn reconcile_boundary_policy\(")
        for step in halves["boundary-bypass"]["sharedSteps"]:
            self.assertIn(step, validate, step)
            self.assertIn(step, py_run, step)
        self.assertIn("PrepPhase", d["restatedFixShape"])
        self.assertIn("cobre-sddp", d["restatedFixShape"])
        self.assertNotIn("cobre-io", d["restatedFixShape"])
        self.assertTrue(any("cobre-io" in c["variant"] for c in d["conflicts"]))
        self.assertTrue(any("cobre-io" in q["question"] for q in d["needsHuman"]))
        self.assertEqual(d["alignment"], "advances-0a")
        for symbol, path in d["evidence"]["helperOwners"].items():
            self.assertTrue(path.startswith("crates/cobre-sddp/"), symbol)
            self.assertTrue(sc.anchor_exists(f"`{path}::{symbol}`", tree), symbol)
        self.check_spans(tree, d["spans"], "CD-029")

    def test_cd_002_tuple_and_cd_009_sites_hold_at_the_pin(self) -> None:
        tree = self.tree()
        setup = tree.read_text(f"{CLI}/commands/run/setup.rs").splitlines()
        cd002 = self.by_id["CD-002"]
        self.assertEqual(len([ln for ln in setup if ln.strip() == TEN_NONES]), 1)
        self.assertEqual(len([ln for ln in setup if ln.strip() == "load_err,"]), 1)
        self.assertIn("pub(super) struct LoadBroadcastResult {", setup)
        slots = cd002["evidence"]["tupleSlots"]
        self.assertEqual(len(slots), 10)
        self.assertEqual((slots[0], slots[-1]), ("raw_system", "load_err"))
        self.assertEqual(cd002["waveDependency"]["wave"], 4)
        self.assertIn("Wave 4", cd002["waveDependency"]["registerRow"])
        self.assertIn("CD-002", self.register.splitlines()[1929 - 1])
        self.assertEqual(
            (
                cd002["recordedSeverity"],
                cd002["reverifiedSeverity"],
                cd002["originalRatingKept"],
            ),
            ("B", "C", False),
        )
        self.assertIn("RootLoadArtifacts", cd002["restatedFixShape"])
        self.assertEqual(cd002["alignment"], "neutral")
        self.assertEqual(cd002["conflicts"], [])
        policy = tree.read_text(f"{CLI}/commands/run/policy.rs").splitlines()
        cd009 = self.by_id["CD-009"]
        sites = [i for i, ln in enumerate(policy, 1) if ln.strip() == POLICY_DIR_LINE]
        guards = [
            i
            for i, ln in enumerate(policy, 1)
            if ln.strip() == "if !policy_dir.exists() {"
        ]
        self.assertEqual(len(sites), 3)
        self.assertEqual([s + 1 for s in sites], guards)
        span = next(s for s in cd009["spans"] if s["needle"] == POLICY_DIR_LINE)
        self.assertEqual(span["lines"], sites)
        self.assertEqual(len(set(cd009["evidence"]["hints"])), 3)
        for hint in cd009["evidence"]["hints"]:
            self.assertIn(
                " ".join(hint.split()),
                " ".join("\n".join(policy).replace("\\\n", "").split()),
            )
        self.assertEqual(
            (cd009["recordedSeverity"], cd009["reverifiedSeverity"]), ("C", "C")
        )
        self.assertIn("resolve_policy_dir", cd009["restatedFixShape"])
        self.assertEqual(cd009["conflicts"], [])
        self.assertEqual(cd009["alignment"], "neutral")

    def test_probes_pass_through_the_harness_checkers(self) -> None:
        base = self.baseline()
        rows = self.env["dispositions"]
        with tempfile.TemporaryDirectory(prefix="cli-python-wave-probe.") as tmp:
            stub = pathlib.Path(tmp) / "anchor-probe.md"
            title = "WAVE ANCHOR PROBE — cli-python (test)"
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
            live = [d for d in rows if d["disposition"] != "retire"]
            self.assertEqual(len(live), 4)
            stub_r = pathlib.Path(tmp) / "reraise-probe.md"
            title_r = "WAVE RERAISE PROBE — cli-python (test)"
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
        for probe in ("anchorProbe", "reraiseProbe"):
            self.assertEqual(self.env["probes"][probe]["exit"], 0, probe)
            self.assertIn(base, self.env["probes"][probe]["command"])

    def test_do_not_touch_list_has_no_owned_match(self) -> None:
        dnt = self.env["doNotTouch"]
        self.assertTrue(DO_NOT_TOUCH <= set(dnt["ids"]))
        self.assertEqual(dnt["matchesAmongOwned"], [])
        self.assertFalse(set(dnt["ids"]) & OWNED)
        line = int(dnt["source"].rsplit("L", 1)[1])
        para = self.register.splitlines()[line - 1]
        self.assertTrue(para.startswith("**Do-not-touch list"))
        for rid in DO_NOT_TOUCH:
            self.assertIn(rid, para)

    def test_ticket_deviations_are_real(self) -> None:
        dev = self.env["ticketDeviations"]
        self.assertGreaterEqual(len(dev), 12)
        for d in dev:
            self.assertNotEqual(d["ticket"], d["measured"], d["figure"])
            self.assertTrue(d["command"], d["figure"])
        figures = " ".join(d["figure"] for d in dev)
        self.assertIn("grep -c boundary", figures)
        self.assertIn("CD-029 destination", figures)
        self.assertIn("gitignored", figures)

    def test_part_i_handoff_rows_re_measure_and_name_the_alignment_epic(self) -> None:
        tree = self.tree()
        h = self.handoff
        self.assertEqual(h["baseline"], self.baseline())
        self.assertIn("partI-i5-queue.json", h["artifactAlias"])
        self.assertIn("partI-handoff.json", h["artifactAlias"])
        self.assertIn("alignment", h["destination"])
        rows = h["i5Queue"]
        refs = [r["ref"] for r in rows]
        self.assertEqual(len(refs), len(set(refs)))
        self.assertTrue(all(r.startswith("I.5-") for r in refs))
        for r in rows:
            with self.subTest(ref=r["ref"]):
                self.assertIn(r["disposition"], DISPOSITIONS)
                for key in (
                    "subClaim",
                    "priorFigure",
                    "baselineFigure",
                    "command",
                    "note",
                ):
                    self.assertTrue(r[key], key)
                if r["disposition"] != "keep":
                    self.assertNotEqual(r["priorFigure"], r["baselineFigure"])
                for a in r["anchors"]:
                    self.assertEqual(set(a), {"path", "symbol"}, a)
                    self.assertTrue(
                        sc.anchor_exists(f"`{a['path']}::{a['symbol']}`", tree), a
                    )
                self.check_spans(tree, r["spans"], r["ref"])
        by_ref = {r["ref"]: r for r in rows}
        self.assertEqual(by_ref["I.5-1a"]["disposition"], "retire")
        self.assertIn(RUN_DOC, by_ref["I.5-1a"]["baselineFigure"])
        main_rs = tree.read_text(f"{CLI}/main.rs")
        self.assertIn(f"/// {RUN_DOC}", main_rs)
        self.assertIsNone(re.search(r"sddp", main_rs, re.I))
        self.assertEqual(by_ref["I.5-2"]["disposition"], "keep")
        occ = sum(tree.read_text(p).count("cobre_sddp::") for p in tree.rs_files(CLI))
        files = sum(
            1 for p in tree.rs_files(CLI) if "cobre_sddp::" in tree.read_text(p)
        )
        self.assertIn(
            f"{occ} `cobre_sddp::` occurrences over {files} files",
            by_ref["I.5-3"]["baselineFigure"],
        )
        self.assertIn("~70", by_ref["I.5-3"]["priorFigure"])
        self.assertGreaterEqual(len(by_ref["I.5-3"]["distinctSymbols"]), 70)
        self.assertIn(REPORT_REMOVAL_COMMIT, by_ref["I.5-7"]["baselineFigure"])
        self.assertFalse(tree.exists(f"{CLI}/commands/report.rs"))
        counts = h["counts"]
        self.assertEqual(counts["i5Rows"], len(rows))
        self.assertEqual(
            (counts["retired"], counts["kept"], counts["sharpened"]),
            tuple(
                sum(1 for r in rows if r["disposition"] == k)
                for k in ("retire", "keep", "sharpen")
            ),
        )
        frag = h["item7Fragment"]
        self.assertEqual((frag["partIItem"], frag["partIRef"]), (7, "I.3-7"))
        self.assertEqual(frag["alignment"], "advances-0a")
        self.assertIn("alignment", frag["forEpic"])
        for cite, needle in (
            (frag["sharpenedPath"][0], "BroadcastConfig::from_config(&config)?"),
            (frag["sharpenedPath"][1], "pub(crate) fn from_config"),
            (frag["sharpenedPath"][2], "StudyParams::from_config(config)"),
            (frag["secondPath"][0], "StudyParams::from_config(&config),"),
            (frag["secondPath"][1], "StudySetup::new_with_boundary_requirements("),
            (frag["sddpSideCaller"], "StudyParams::from_config(config)?;"),
        ):
            path, line = re.match(r"^(\S+):(\d+)", cite).groups()  # type: ignore[union-attr]
            self.assertIn(
                needle, tree.read_text(path).splitlines()[int(line) - 1], cite
            )
        self.assertEqual(len(frag["nonTestCallSites"]), 5)
        sddp_side = [
            s
            for s in frag["nonTestCallSites"]
            if s["file"].startswith("crates/cobre-sddp/")
        ]
        self.assertEqual(len(sddp_side), 1)
        self.assertEqual(
            f"{sddp_side[0]['file']}:{sddp_side[0]['line']}", frag["sddpSideCaller"]
        )
        entries = h["entries"]
        self.assertEqual([e["partIRef"] for e in entries], [*refs, "I.3-7"])
        for e in entries:
            self.assertIn(e["disposition"], DISPOSITIONS)
            self.assertEqual(e["proposedPhase"], "0a")
            self.assertTrue(
                sc.anchor_exists(
                    f"`{e['baselineAnchor']['path']}::{e['baselineAnchor']['symbol']}`",
                    tree,
                )
            )
        self.assertFalse(any(e["partIRef"] == "I.3-6" for e in entries))
        proc = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "station_verify.py"),
                "genericity",
                str(sc.REPO),
                str(self.artifact("partI-handoff.json")),
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


LENSES = ("architecture", "performance", "over-engineering", "test-bloat")
SUBSTATIONS = ("S6a", "S6b", "S6c")
STATION_PREFIXES = ("crates/cobre-cli/", "crates/cobre-python/", "crates/cobre/")
TEST_DIRS = ("crates/cobre-cli/tests/", "crates/cobre-python/tests/")
LINE_ANCHOR_OK = {"crates/cobre/src/lib.rs", "crates/cobre/Cargo.toml"}
COUPLING_WORDS = re.compile(
    r"cobre_sddp|StudySetup|BroadcastConfig|from_config|new_with_boundary_requirements|PrepPhase|\bEngine\b",
    re.I,
)
ENFORCEMENT_WORDS = re.compile(
    r"parity gate|enforcement|coverage gap|check_python_parity", re.I
)
TIMING_NUMBER = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\b|×|%|(?:ms|µs|us|ns|s|sec|secs|seconds|minutes|min|speedup|faster|slower)\b)",
    re.I,
)
DIFF_MARKERS = ("---", "+++", "@@", "diff ", "```")


def log_section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


class CandidateEnvelopeTests(sc.StationCase):
    """The attacker fan-out (E06-3): four merged lens files and the dispatch log.

    Each lens file merges the three sub-surface cells; every anchor is re-resolved on the pin, the
    station field rules the ticket prescribes are asserted, and the log must record all twelve cells.
    """

    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.files = {
            lens: sc.load_json(cls.station_dir() / f"candidates-{lens}.json")
            for lens in LENSES
        }
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")
        cls.manifest = {
            m: r["id"] for r in cls.inv["substationRollup"] for m in r["members"]
        }
        cls.log = (cls.station_dir() / "attacker-log.md").read_text(encoding="utf-8")

    def anchor_allowed(self, a: dict[str, Any], subs: list[str], lens: str) -> bool:
        path = a["path"]
        if not path.startswith(STATION_PREFIXES):
            return False
        if path in self.manifest and self.manifest[path] in subs:
            return True
        if lens == "test-bloat" and path.startswith(TEST_DIRS):
            return True
        return "S6c" in subs and path == "crates/cobre-python/src/run.rs"

    def test_four_lens_files_validate_and_cover_every_sub_surface(self) -> None:
        for lens, env in self.files.items():
            with self.subTest(lens=lens):
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(sc.TOOLS / "validate-envelope.py"),
                        "--role",
                        "attacker",
                        "--station",
                        "cli-python",
                        str(self.artifact(f"candidates-{lens}.json")),
                    ],
                    cwd=sc.REPO,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(env["station"], "cli-python")
                self.assertEqual(env["stationLabel"], "cobre-cli+cobre-python+facade")
                self.assertEqual(env["lens"], lens)
                self.assertEqual(env["baseline"], self.baseline())
                self.assertEqual(env["cells"], [f"{lens}.{sub}" for sub in SUBSTATIONS])
                for sub in SUBSTATIONS:
                    covered = any(c["subSurface"] == sub for c in env["candidates"])
                    cleared = any(v["subSurface"] == sub for v in env["cleanVerdicts"])
                    self.assertTrue(
                        covered or cleared, f"{lens}: {sub} neither covered nor cleared"
                    )
                    self.assertEqual(
                        env["coverage"][sub],
                        "candidates" if covered else "clean-verdict",
                    )
                    self.assertEqual(
                        env["gate"][sub]["validator"], "pass", f"{lens}.{sub}"
                    )
                    if covered:
                        self.assertEqual(
                            env["gate"][sub]["checkAnchors"], "pass", f"{lens}.{sub}"
                        )
                for v in env["cleanVerdicts"]:
                    self.assertTrue(v["why"].strip())
                    self.assertGreaterEqual(
                        v["positives"], 1, "a clean cell needs a positive"
                    )
                self.assertTrue(env["candidates"] or env["positives"], "blank lens")

    def test_every_anchor_is_inside_the_station_and_resolves_at_the_pin(self) -> None:
        tree = self.tree()
        for lens, env in self.files.items():
            for c in env["candidates"]:
                with self.subTest(lens=lens, title=c["title"][:60]):
                    self.assertTrue(c["anchors"])
                    self.assertEqual(c["cell"], f"{lens}.{c['subSurface']}")
                    for a in c["anchors"]:
                        keys = set(a)
                        self.assertIn(keys, ({"path", "symbol"}, {"path", "line"}), a)
                        subs = c.get("subSurfaces") or [c["subSurface"]]
                        self.assertTrue(self.anchor_allowed(a, subs, lens), a)
                        if "line" in keys:
                            self.assertIn(a["path"], LINE_ANCHOR_OK, a)
                            anchor = f"`{a['path']}:{a['line']}`"
                        else:
                            anchor = f"`{a['path']}::{a['symbol']}`"
                        self.assertTrue(sc.anchor_exists(anchor, tree), a)
                    self.assertIn(c["alignmentHint"], ALIGN)
                    self.assertIn(c["proposedSeverity"], ("A", "B", "C"))
                    self.assertTrue(c["evidence"]["command"].strip())
                    fix = c["fixShape"]
                    self.assertGreaterEqual(len(fix), 20)
                    self.assertFalse(fix.lstrip().startswith(DIFF_MARKERS), fix[:40])
                    self.assertNotIn("```", fix)
                    if c.get("reRaiseOf"):
                        self.assertIn(c["reRaiseOf"], OWNED)
                    for d in env["dupOf"]:
                        self.assertTrue(d["dupOf"]["station"])
                        self.assertNotIn(
                            d["title"], [k["title"] for k in env["candidates"]]
                        )

    def test_station_field_rules_hold_per_lens(self) -> None:
        arch = self.files["architecture"]
        tagged = [c for c in arch["candidates"] if c.get("partIRef") == "I.5"]
        self.assertGreaterEqual(len(tagged), 1)
        for c in arch["candidates"]:
            claim = c["title"] + " " + c["fixShape"]
            if COUPLING_WORDS.search(claim):
                self.assertIn(c.get("partIRef"), ("I.5", "I.3-7"), c["title"][:60])
        for lens, env in self.files.items():
            for c in env["candidates"]:
                if c.get("waveRef"):
                    self.assertIn(c["waveRef"], ("CD-025", "CD-029"), c["title"][:60])
                if c.get("waveRef") == "CD-025":
                    fix = c["fixShape"].lower()
                    self.assertTrue(
                        "cobre-io" in fix or c["alignmentHint"] == "conflicts",
                        f"{lens}: CD-025 fix-shape names no cobre-io owner: {c['title'][:60]}",
                    )
        oe = self.files["over-engineering"]
        for c in oe["candidates"]:
            rsc = c.get("reservedSeamsCheck")
            self.assertIsInstance(rsc, dict, c["title"][:60])
            self.assertTrue(rsc["checked"], c["title"][:60])
            self.assertIn(rsc["result"], ("sanctioned", "not-found", "not-applicable"))
        mentioned = json.dumps(oe["candidates"]) + json.dumps(oe["positives"])
        for path in (
            "crates/cobre/src/lib.rs",
            "crates/cobre-cli/src/commands/broadcast.rs",
        ):
            self.assertIn(path, mentioned, f"over-engineering never reached {path}")
        perf = self.files["performance"]
        self.assertIn("no timing command", perf["timingRuns"])
        for c in perf["candidates"]:
            with self.subTest(title=c["title"][:60]):
                self.assertIs(c["measured"], False)
                self.assertEqual(c["unmeasured"]["tag"], "UNMEASURED")
                self.assertTrue(c["unmeasured"]["reason"].strip())
                self.assertIn(c["measurementRequest"]["layout"], ("4t", "2x2"))
                self.assertIn(
                    c["measurementRequest"]["claimType"],
                    ("single-process", "collective"),
                )
                self.assertEqual(c["queuedTo"], "perf-sweep")
                self.assertTrue(str(c["mechanism"]).strip())
                blob = " ".join(
                    [
                        c["title"],
                        str(c["mechanism"]),
                        c["fixShape"],
                        str(c["evidence"].get("reading", "")),
                    ]
                )
                self.assertIsNone(TIMING_NUMBER.search(blob), blob[:120])

    def test_enforcement_gap_candidates_carry_measured_evidence(self) -> None:
        seen = 0
        for env in self.files.values():
            for c in env["candidates"]:
                if ENFORCEMENT_WORDS.search(c["title"]):
                    seen += 1
                    ev = c["evidence"]
                    self.assertRegex(
                        ev["command"], r"git (show|grep|ls-tree)|grep|python3"
                    )
                    self.assertTrue(str(ev.get("output", "")).strip(), c["title"][:60])
                    self.assertTrue(str(ev.get("reading", "")).strip(), c["title"][:60])
        self.assertGreaterEqual(seen, 0)

    def test_dispatch_log_records_all_twelve_cells_and_no_blank_cell(self) -> None:
        for lens in LENSES:
            for sub in SUBSTATIONS:
                self.assertIn(
                    f"{lens}.{sub}", self.log, "a cell without a dispatch record"
                )
        matrix = log_section(self.log, "## Coverage matrix")
        rows = [
            ln
            for ln in matrix.splitlines()
            if ln.startswith("| ") and not ln.startswith("| -")
        ]
        self.assertEqual(len(rows), 5, "header + four lens rows")
        for ln in rows[1:]:
            self.assertNotIn("pending", ln)
            self.assertNotIn("0/0/0", ln)
            self.assertNotIn("FAIL", ln)
        self.assertEqual(
            sorted(ln.split("|")[1].strip() for ln in rows[1:]), sorted(LENSES)
        )
        self.assertIn("## Prior-register screen and re-routes", self.log)
        self.assertIn("No timing command was executed", self.log)
        records = json.loads(
            log_section(self.log, "## Per-cell records")
            .split("```json", 1)[1]
            .split("```", 1)[0]
        )
        self.assertEqual(
            set(records), {f"{lens}.{sub}" for lens in LENSES for sub in SUBSTATIONS}
        )
        for cell, r in records.items():
            self.assertEqual(r["validator"], "pass", cell)
            self.assertTrue(r["candidates"] or r["positives"], f"{cell} is blank")
            for d in r["dropped"]:
                self.assertIn(
                    d["action"],
                    {
                        "settled",
                        "sanctioned",
                        "dup-of",
                        "merged",
                        "anchor-missing",
                        "needs-human",
                    },
                )


INGEST_STATES = {
    "accepted",
    "rejected-anchor",
    "rejected-reserved-seam",
    "rejected-defender",
    "merged",
    "unresolved",
    "handed-off",
    "blocked",
}
DISMISSAL_BASES = {
    "sanctioned-seam",
    "premise-false-at-pin",
    "deliberate-and-documented",
    "contract",
    "cost-accepted-by-rule",
}
SANCTIONED_BY_CLOSED_SET = (
    "CLI/Python output orchestration hand-mirror",
    "Setup config-projection sprawl + CLI non-root reconstruction",
    "`#[allow(...)]` census — Load-bearing / Reserved-seam / Symmetry-or-test-retention classes",
    "Unwired config is reserved, not dead",
    "Umbrella crate reserved for a future single-dependency convenience re-export",
    "Python parity hard rule",
)
BYTE_NEUTRAL = {"asserted", "needs-rebaseline", "n/a"}
PROBE_TITLE = "INGEST ANCHOR PROBE — cli-python (2026-09, baseline)"
INGEST_LOG_HEADINGS = (
    "## Candidate census",
    "## Anchor rejections",
    "## Re-route and dup-of",
    "## Cleared (sanctioned)",
    "## Dup-of merges",
    "## Re-raise rejections",
    "## Blocked pending measurement",
    "## Measure-then-claim",
    "## Contract dismissals",
    "## Cross-lens overlaps — decisions",
    "## Defender summary",
    "## Per-candidate roster",
)
STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in into is it its no not of on or that the this to was were with without vs via per than then their there these those over under one two three four five six seven eight nine ten".split()
)
THIRD_LAYER = (
    "crates/cobre-cli/tests/python_parity_check.rs::python_parity_script_passes"
)


def claim_tokens(text: str) -> set[str]:
    return {
        w
        for w in re.split(r"[^\w-]+", text.replace("`", " ").lower())
        if len(w) >= 3 and w not in STOP_WORDS and not w.isdigit()
    }


def derived_state(entry: dict[str, Any]) -> str:
    d = entry["disposition"]
    if d == "dup-of":
        return "merged"
    if d == "anchor-missing":
        return "rejected-anchor"
    if d == "out-of-station":
        return "handed-off"
    if d == "blocked-pending-measurement":
        return "blocked"
    if entry["verdict"] == "confirmed":
        return "accepted"
    if entry["verdict"] == "dismissed":
        return (
            "rejected-reserved-seam"
            if entry["dismissalBasis"] == "sanctioned-seam"
            else "rejected-defender"
        )
    return "unresolved"


class IngestTests(sc.StationCase):
    """The ingest (E06-4): verdicts.json, ingest-log.md and enforcement-measurements.json.

    One verdict per candidateRef of the four lens files; every state is derivable from
    disposition + verdict + dismissalBasis; accepted anchors resolve at the pin; the two
    enforcement-gap verdicts carry the ingest-measured evidence; the measurements re-measure.
    """

    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = sc.load_json(cls.station_dir() / "verdicts.json")
        cls.verdicts = cls.doc["verdicts"]
        cls.meas = sc.load_json(cls.station_dir() / "enforcement-measurements.json")
        cls.log = (cls.station_dir() / "ingest-log.md").read_text(encoding="utf-8")
        cls.brief = (cls.station_dir() / "defender-prompt.md").read_text(
            encoding="utf-8"
        )
        cls.candidates = {}
        for lens in LENSES:
            counters = {sub: 0 for sub in SUBSTATIONS}
            for c in sc.load_json(cls.station_dir() / f"candidates-{lens}.json")[
                "candidates"
            ]:
                sub = c["subSurface"]
                cls.candidates[f"{sub}-{lens}-{counters[sub]:02d}"] = c
                counters[sub] += 1

    def defended(self) -> list[dict[str, Any]]:
        return [e for e in self.verdicts.values() if e["disposition"] == "defended"]

    def test_exactly_one_verdict_per_candidate_ref(self) -> None:
        self.assertEqual(set(self.verdicts), set(self.candidates))
        for ref, entry in self.verdicts.items():
            self.assertEqual(entry["candidateRef"], ref)
            self.assertEqual(entry["title"], self.candidates[ref]["title"])
            parts = re.fullmatch(
                r"(S6[abc])-(architecture|performance|over-engineering|test-bloat)-\d{2}",
                ref,
            )
            assert parts is not None, ref
            self.assertEqual(entry["subStation"], parts.group(1))
            self.assertEqual(entry["lens"], parts.group(2))
            self.assertEqual(
                entry.get("partIRef"), self.candidates[ref].get("partIRef")
            )
            own_wave = self.candidates[ref].get("waveRef")
            if entry.get("waveRef") != own_wave:
                self.assertIsNone(own_wave, ref)
                inherited = {
                    self.candidates[m].get("waveRef")
                    for m in entry.get("mergedFrom") or []
                }
                self.assertIn(entry["waveRef"], inherited, ref)
        self.assertEqual(self.doc["counts"]["received"], len(self.candidates))
        self.assertEqual(self.doc["baseline"], self.baseline())
        self.assertEqual(self.doc["baseline"], header_baseline())

    def test_every_verdict_has_a_known_state_and_the_counts_sum(self) -> None:
        counts = self.doc["counts"]
        states = []
        for entry in self.verdicts.values():
            with self.subTest(ref=entry["candidateRef"]):
                self.assertIn(entry["state"], INGEST_STATES)
                self.assertEqual(entry["state"], derived_state(entry))
                states.append(entry["state"])
                if entry["disposition"] == "defended" and entry["verdict"] is None:
                    self.assertTrue(entry["_needsHuman"])
        self.assertEqual(
            counts["received"],
            counts["malformed"]
            + counts["anchorRejected"]
            + counts["outOfStation"]
            + counts["dupOf"]
            + counts["blocked"]
            + counts["defended"],
        )
        self.assertEqual(
            counts["defended"],
            counts["confirmed"] + counts["dismissed"] + counts["unresolved"],
        )
        self.assertEqual(states.count("merged"), counts["dupOf"])
        self.assertEqual(states.count("accepted"), counts["confirmed"])
        self.assertEqual(states.count("unresolved"), counts["unresolved"])
        self.assertEqual(
            states.count("rejected-reserved-seam"), counts["sanctionedSeam"]
        )
        self.assertEqual(
            states.count("rejected-reserved-seam") + states.count("rejected-defender"),
            counts["dismissed"],
        )
        self.assertEqual(counts["unresolved"], 0, "a defender never resolved")

    def test_accepted_anchors_are_in_station_and_resolve_at_the_baseline(self) -> None:
        tree = self.tree()
        accepted = [e for e in self.verdicts.values() if e["state"] == "accepted"]
        self.assertTrue(accepted)
        for entry in accepted:
            for a in entry["anchors"]:
                with self.subTest(ref=entry["candidateRef"], anchor=a):
                    self.assertTrue(a["path"].startswith(STATION_PREFIXES), a)
                    anchor = (
                        f"`{a['path']}::{a['symbol']}`"
                        if "symbol" in a
                        else f"`{a['path']}:{a['line']}`"
                    )
                    self.assertTrue(sc.anchor_exists(anchor, tree), a)

    def test_anchor_probe_resolves_with_the_harness_checker(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(sc.TOOLS / "check-anchors.py"),
                PROBE_TITLE,
                "--register",
                str(self.artifact("anchor-probe.md")),
                "--baseline",
                self.baseline(),
                "--json",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            report["checked"], sum(len(e["anchors"]) for e in self.verdicts.values())
        )

    def test_dismissals_carry_a_basis_and_seams_cite_the_closed_set(self) -> None:
        for text in SANCTIONED_BY_CLOSED_SET:
            self.assertIn(text, self.brief)
        self.assertEqual(
            list(self.doc["sanctionedByClosedSet"]), list(SANCTIONED_BY_CLOSED_SET)
        )
        dismissed = [e for e in self.defended() if e["verdict"] == "dismissed"]
        self.assertTrue(dismissed)
        for entry in dismissed:
            with self.subTest(ref=entry["candidateRef"]):
                self.assertIn(entry["dismissalBasis"], DISMISSAL_BASES)
                self.assertTrue(str(entry["basisCitation"]).strip())
                self.assertIsNone(entry["survivingClaim"])
                self.assertEqual(entry["byteNeutral"], "n/a")
                if entry["dismissalBasis"] == "sanctioned-seam":
                    self.assertIn(entry["sanctionedBy"], SANCTIONED_BY_CLOSED_SET)
                else:
                    self.assertIsNone(entry["sanctionedBy"])
                if entry["dismissalBasis"] == "contract":
                    self.assertTrue(str(entry["contractCited"]).strip())
        for entry in self.defended():
            if entry["verdict"] == "confirmed":
                self.assertIsNone(entry["dismissalBasis"])
                self.assertIsNone(entry["sanctionedBy"])

    def test_confirmed_verdicts_narrow_the_title_and_tags_are_consistent(self) -> None:
        for entry in self.defended():
            with self.subTest(ref=entry["candidateRef"]):
                self.assertIn(entry["byteNeutral"], BYTE_NEUTRAL)
                self.assertIn(entry["alignmentHint"], ALIGN)
                self.assertIsInstance(entry["conflicts"], bool)
                self.assertEqual(
                    entry["conflicts"], entry["alignmentHint"] == "conflicts"
                )
                self.assertEqual(
                    bool(entry["conflictsRule"]),
                    entry["conflicts"],
                    entry["candidateRef"],
                )
                self.assertGreaterEqual(len(entry["argument"]), 120)
                blob = entry["argument"] + " " + (entry["survivingClaim"] or "")
                self.assertNotIn("```", blob)
                if entry["verdict"] == "confirmed":
                    claim = entry["survivingClaim"]
                    self.assertTrue(claim and len(claim) >= 40)
                    title_tokens = claim_tokens(entry["title"])
                    tokens = claim_tokens(claim)
                    self.assertNotEqual(title_tokens, tokens)
                    self.assertLess(
                        len(title_tokens & tokens) / len(title_tokens | tokens), 0.85
                    )
                if entry["lens"] == "performance":
                    self.assertIsNone(TIMING_NUMBER.search(blob), blob[:120])
                if entry["priorRelation"] == "sharpens":
                    self.assertIn(entry["priorId"], OWNED)

    def test_enforcement_gap_verdicts_carry_the_measured_evidence(self) -> None:
        source = self.meas["sourceLayer"]
        runtime = self.meas["runtimeLayer"]
        gap = [e for e in self.defended() if e["claimsEnforcementGap"]]
        self.assertEqual(
            sorted(e["candidateRef"] for e in gap), sorted(self.meas["claims"])
        )
        self.assertTrue(gap)
        for entry in gap:
            with self.subTest(ref=entry["candidateRef"]):
                me = entry["measuredEvidence"]
                self.assertIsInstance(me, dict)
                self.assertEqual(
                    set(me["sourceLayerNames"]["cli"]), set(source["cliNames"])
                )
                self.assertEqual(
                    set(me["sourceLayerNames"]["python"]), set(source["pythonNames"])
                )
                self.assertEqual(me["runtimeTest"]["outcome"], runtime["outcome"])
                self.assertEqual(me["runtimeTest"]["ciExecutes"], runtime["ciExecutes"])
                self.assertEqual(me["thirdLayer"], THIRD_LAYER)
                self.assertTrue(me["writerCallSites"])
        for entry in self.defended():
            if not entry["claimsEnforcementGap"]:
                self.assertIsNone(entry["measuredEvidence"], entry["candidateRef"])
        self.assertEqual(self.meas["blockedPendingMeasurement"], [])
        self.assertEqual(runtime["outcome"], "passed")
        self.assertEqual(runtime["ciExecutes"], "yes")

    def test_merged_entries_name_a_defended_survivor(self) -> None:
        merged = [e for e in self.verdicts.values() if e["state"] == "merged"]
        self.assertTrue(merged)
        for entry in merged:
            with self.subTest(ref=entry["candidateRef"]):
                self.assertIsNone(entry["verdict"])
                self.assertIn(entry["priorRelation"], {"restates", "intra-station"})
                self.assertTrue(str(entry["argument"]).strip(), "a fold needs a reason")
                if entry["priorRelation"] == "intra-station":
                    survivor = self.verdicts[entry["priorId"]]
                    self.assertEqual(survivor["disposition"], "defended")
                    self.assertIn(entry["candidateRef"], survivor["mergedFrom"])
                    self.assertNotEqual(survivor["lens"], entry["lens"])
                else:
                    self.assertIn(entry["priorId"], OWNED)
        for entry in self.defended():
            for ref in entry["mergedFrom"] or []:
                self.assertEqual(self.verdicts[ref]["priorId"], entry["candidateRef"])

    def test_ingest_log_has_one_roster_row_per_candidate(self) -> None:
        for heading in INGEST_LOG_HEADINGS:
            self.assertIn(heading, self.log)
        header = HEADER_BASELINE.search(self.log.replace("`", ""))
        assert header is not None
        self.assertEqual(header.group(1), header_baseline())
        roster = log_section(self.log, "## Per-candidate roster")
        refs = re.findall(
            r"^\| (S6[abc]-(?:architecture|performance|over-engineering|test-bloat)-\d{2}) \|",
            roster,
            re.M,
        )
        self.assertEqual(sorted(refs), sorted(self.candidates))
        self.assertEqual(len(refs), len(set(refs)))
        cleared = log_section(self.log, "## Cleared (sanctioned)")
        self.assertIn("crates/cobre/src/lib.rs", cleared)
        self.assertIn("commands/broadcast.rs", cleared)
        summary = log_section(self.log, "## Defender summary")
        self.assertIn("Retry history", summary)
        self.assertIn(str(self.doc["counts"]["confirmed"]) + " confirmed", summary)

    def test_enforcement_measurements_re_measure_at_the_pin(self) -> None:
        tree = self.tree()
        source = self.meas["sourceLayer"]
        proc = subprocess.run(
            [
                "python3",
                "scripts/ci/check_python_parity.py",
                "--max",
                "0",
                "--root",
                ".",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, source["exitCode"])
        self.assertIn(source["summaryLine"], proc.stdout + proc.stderr)
        self.assertEqual(source["sharedCount"], len(source["inBoth"]))
        self.assertGreaterEqual(source["sharedCount"], source["minSharedFloor"])
        writer = self.meas["writerSurface"]
        run_dir = tree.rs_files("crates/cobre-cli/src/commands/run")
        cli_calls = [
            m
            for p in run_dir
            for m in re.findall(r"write_[a-z_0-9]*\(", tree.read_text(p))
        ]
        py_calls = re.findall(
            r"write_[a-z_0-9]*\(", tree.read_text("crates/cobre-python/src/run.rs")
        )
        self.assertEqual(len(cli_calls), writer["cliCallSites"])
        self.assertEqual(cli_calls.count("write_line("), writer["cliTerminalWriteLine"])
        self.assertEqual(len(py_calls), writer["pythonCallSites"])
        self.assertEqual(
            sum(1 for c in py_calls if c.endswith("_if_any(")),
            writer["pythonIfAnyHelpers"],
        )
        ci_vis = self.meas["ciVisibility"]
        per_file = {
            p.rsplit("/", 1)[-1]: tree.read_text(p).count("#[test]")
            for p in tree.rs_files("crates/cobre-python/src")
            if "#[test]" in tree.read_text(p)
        }
        self.assertEqual(per_file, ci_vis["perFile"])
        self.assertEqual(sum(per_file.values()), ci_vis["cobrePythonRustTests"])
        ci = tree.read_text(".github/workflows/ci.yml")
        for needle in (
            "--require-cli-binary",
            "cargo build --release -p cobre-cli",
            "cargo test --manifest-path crates/cobre-python/Cargo.toml",
            "check_python_parity.py --max 0",
        ):
            self.assertIn(needle, ci)
        conftest = tree.read_text("crates/cobre-python/tests/conftest.py")
        self.assertIsNotNone(re.search(r"^def cli_binary\(", conftest, re.M))
        self.assertIsNotNone(
            re.search(
                r"^def resolve_cli_binary\(",
                tree.read_text("crates/cobre-python/tests/_cobre_cli.py"),
                re.M,
            )
        )


CAL_ID_RE = re.compile(r"^(CD|PD|OD|TD)-(\d{3})$")
ID_FLOORS = {"CD": 40, "PD": 6, "OD": 10, "TD": 1}
SEVERITIES = {"A", "B", "B (A-risk)", "C"}
LAYOUTS = {"4t", "2x2"}
REQUIRES = {"none", "enumerated", "external-library"}
ENTRY_HEADING = re.compile(r"^\*\*((?:CD|PD|OD|TD)-\d{3}) · Sev ", re.M)
SCAFFOLD_CLI = "## ★ QUALITY EVALUATION (2026-09, baseline a136840d) — cli-python"
STATION6_TITLE = "★ STATION 6 — cobre-cli / cobre-python / facade (2026-09, baseline)"
LENS_CLASS = {
    "architecture": "CD",
    "performance": "PD",
    "over-engineering": "OD",
    "test-bloat": "TD",
}
RANK = {"A": 3, "B (A-risk)": 2.5, "B": 2, "C": 1}
LENS_HEADINGS = (
    "#### Architecture (CD)",
    "#### Performance (PD)",
    "#### Over-engineering (OD)",
    "#### Test-suite bloat (TD)",
)
NON_ENTRY_HEADINGS = (
    "#### Wave-5 dispositions",
    "#### Part-I cross-references — item I.5 and the BroadcastConfig handoff",
    "#### Supersession notes",
    "#### Positives",
    "#### ↩︎ Cleared",
    "#### Queued out",
    "#### Owner gate — decisions",
    "#### Findings by lens",
)
OWNED_ALL_SHARPEN = {"CD-025", "CD-029", "CD-002", "CD-009"}
PARITY_LAYERS = (
    "check_python_parity.py",
    "test_cli_python_file_set_parity.py",
    "python_parity_check.rs",
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
    """E06-5: id assignment, house calibration, the four reused ids, the L2 rule, supersession notes, queues, section."""

    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

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
        cls.handoff = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.notes_md = (cls.station_dir() / "supersession-notes.md").read_text(
            encoding="utf-8"
        )
        cls.register = sc.BACKLOG.read_text(encoding="utf-8")
        cls.section = cls.register.split(SCAFFOLD_CLI, 1)[1].split("\n## ", 1)[0]
        cls.prior = register_before_station(cls.register, cls.section, "cli-python")
        cls.folds = {
            d["candidateRef"]
            for d in cls.cal["dupOf"]
            if d["foldedAt"] == "calibration"
        }

    def block(self, entry_id: str) -> str:
        return self.section.split(f"**{entry_id} · ", 1)[1].split("\n**", 1)[0]

    def sub_block(self, heading: str) -> str:
        return self.section.split(heading, 1)[1].split("\n#### ", 1)[0]

    def test_envelope_baseline_and_heading_levels(self) -> None:
        self.assertEqual(self.cal["station"], "cli-python")
        self.assertEqual(self.cal["baseline"], header_baseline())
        self.assertEqual(self.cal["baseline"], self.baseline())
        self.assertEqual(self.cal["sectionTitle"], STATION6_TITLE)
        self.assertEqual(self.section.count(f"\n### {STATION6_TITLE}\n"), 1)
        self.assertEqual(
            re.findall(r"^### .*$", self.section, re.M), [f"### {STATION6_TITLE}"]
        )
        for heading in NON_ENTRY_HEADINGS + LENS_HEADINGS:
            self.assertIn(heading, self.section, heading)
        for heading in LENS_HEADINGS:
            self.assertIsNotNone(
                re.search(rf"^{re.escape(heading)}\n\n_\d+ minted", self.section, re.M),
                heading,
            )
        gate = self.sub_block("#### Owner gate — decisions")
        if (self.station_dir() / "decisions.json").exists():
            self.assertIn("**Gate: RETURNED ", gate)
        else:
            self.assertIn("_(pending", gate)
        findings = self.section.index("#### Findings by lens")
        for heading in NON_ENTRY_HEADINGS[:-1]:
            self.assertLess(self.section.index(heading), findings, heading)

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
        self.assertEqual(set(self.by_ref) | self.folds, confirmed_new)
        self.assertTrue(self.folds <= confirmed_new)
        for ref, e in self.verdicts.items():
            if e.get("priorRelation") == "sharpens":
                self.assertNotIn(ref, self.by_ref, "a sharpening must never mint an id")
                deltas = {
                    d["candidateRef"] for d in self.reused[e["priorId"]]["e06_4Deltas"]
                }
                self.assertIn(ref, deltas, ref)
        self.assertEqual(
            {c["candidateRef"] for c in self.cal["cleared"] if c["candidateRef"]},
            {
                ref
                for ref, e in self.verdicts.items()
                if e["disposition"] == "defended" and e["verdict"] == "dismissed"
            },
        )
        ingest_dups = {
            d["candidateRef"] for d in self.cal["dupOf"] if d["foldedAt"] == "ingest"
        }
        self.assertEqual(
            ingest_dups,
            {ref for ref, e in self.verdicts.items() if e["disposition"] == "dup-of"},
        )
        for d in self.cal["dupOf"]:
            self.assertIsNone(d["assignedId"])
            if d["priorRelation"] == "intra-station":
                if d["survivorId"] in self.by_id:
                    self.assertIn(
                        d["candidateRef"], self.by_id[d["survivorId"]]["mergedFrom"]
                    )
                else:
                    self.assertIn(d["survivorId"], self.reused, d["candidateRef"])
                    self.assertIn(
                        d["candidateRef"], self.reused[d["survivorId"]]["dupOf"]
                    )
        for row in self.assigned:
            self.assertIsNone(row["priorId"])
            claim, title = (
                claim_tokens(row["survivingClaim"]),
                claim_tokens(row["title"]),
            )
            self.assertTrue(claim and claim != title, row["id"])
        self.assertNotIn(
            "S6a-architecture-01",
            self.by_ref,
            "the simulation-side mirror rides CD-025",
        )
        self.assertIn(
            "S6a-architecture-01",
            {d["candidateRef"] for d in self.reused["CD-025"]["e06_4Deltas"]},
        )

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
        self.assertEqual(
            sorted(ENTRY_HEADING.findall(self.section)), sorted(self.by_id)
        )
        self.assertEqual(set(self.reused), OWNED_ALL_SHARPEN)
        for rid in self.reused:
            self.assertNotIn(rid, ENTRY_HEADING.findall(self.section))
            self.assertIsNotNone(
                re.search(rf"^\| {rid} \| \d+ \| sharpen \|", self.section, re.M), rid
            )

    def test_severity_effort_confidence_alignment_and_the_house_precedents(
        self,
    ) -> None:
        for row in self.assigned:
            self.assertIn(row["severity"], SEVERITIES, row["id"])
            self.assertIn(row["alignmentHint"], ALIGN, row["id"])
            self.assertIn(row["effort"], ("S", "M", "L"))
            self.assertIn(row["confidence"], ("high", "med", "low"))
            self.assertTrue(row["category"])
            self.assertRegex(row["alignmentCites"], r"Part (IV|V)")
            if row["confidence"] == "high":
                cmd = row["evidence"]["command"]
                self.assertTrue(
                    not re.search(r"\bgrep -c\b|\bwc -l\b", cmd)
                    or re.search(r"git show|sed -n", cmd),
                    f"{row['id']}: confidence high on a grep-count-only evidence command",
                )
        mirrors = [
            r for r in self.assigned if "CD-025 precedent" in r["calibrationBasis"]
        ]
        self.assertTrue(mirrors)
        for r in mirrors:
            self.assertEqual(r["severity"], "B", r["id"])
        cd025 = self.reused["CD-025"]
        self.assertEqual(cd025["severity"], "B")
        self.assertEqual(cd025["alignmentHint"], "advances-0a")
        self.assertRegex(cd025["alignmentCites"] or "", r"Part (IV|V)")
        self.assertIn("hard rule", cd025["calibrationBasis"])
        self.assertIn("runtime file-set parity test", cd025["calibrationBasis"])
        self.assertEqual(self.reused["CD-029"]["severity"], "B")
        self.assertEqual(self.reused["CD-029"]["alignmentHint"], "advances-0a")
        disp = self.sub_block("#### Wave-5 dispositions")
        self.assertIn("SUPERSEDED", disp)
        self.assertIn("cobre-sddp or cobre-io", disp)
        self.assertIn("not A", disp)

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

    def test_l2_destination_rule_ran_over_every_fix_shape_and_no_row_conflicts(
        self,
    ) -> None:
        self.assertEqual(self.cal["conflicts"], [])
        self.assertEqual(self.cal["l2Rule"]["hits"], 0)
        for row in self.assigned:
            self.assertEqual(row["l2Rule"]["hits"], [], row["id"])
            self.assertNotEqual(row["alignmentHint"], "conflicts", row["id"])
            self.assertTrue(row["actionable"])
            self.assertIn(
                "- **L2 destination rule:**", self.block(row["id"]), row["id"]
            )
        for r in self.reused.values():
            self.assertTrue(r["actionable"])
            for v in r["conflictVariants"]:
                self.assertEqual(v["tag"], "conflicts")
                self.assertFalse(v["presentedAsDirection"])
                self.assertRegex(v["rule"], r"Part (IV|III|V)")

    def test_reused_ids_are_all_sharpen_with_linked_supersession_notes(self) -> None:
        note_ids = {n["id"] for n in self.cal["supersessionNotes"]}
        for rid, r in self.reused.items():
            self.assertEqual(r["disposition"], "sharpen", rid)
            self.assertTrue(r["supersededClaim"] and r["survivingClaim"], rid)
            self.assertTrue(
                r["supersessionNotes"], f"{rid}: sharpen with no supersession note"
            )
            for n in r["supersessionNotes"]:
                self.assertIn(n, note_ids)
                self.assertIn(f"## {n} — ", self.notes_md)
        for n in self.cal["supersessionNotes"]:
            body = self.notes_md.split(f"## {n['id']} — ", 1)[1].split("\n## ", 1)[0]
            for marker in (
                "- **Old figure (source):**",
                "- **New figure (baseline",
                "- **Command:** `",
            ):
                self.assertIn(marker, body, n["id"])
            self.assertTrue(n["old"] and n["new"] and n["command"], n["id"])
        self.assertIn("#### Supersession notes", self.section)

    def test_parity_gap_evidence_names_all_three_layers(self) -> None:
        disp = self.sub_block("#### Wave-5 dispositions")
        for layer in PARITY_LAYERS:
            self.assertIn(layer, disp, layer)
        gap = {
            ref
            for ref, e in self.verdicts.items()
            if e["disposition"] == "defended" and e["claimsEnforcementGap"]
        }
        self.assertTrue(gap)
        deltas = {d["candidateRef"] for d in self.reused["CD-025"]["e06_4Deltas"]}
        for ref in gap:
            self.assertTrue(ref in deltas or ref in self.by_ref, ref)
            if ref in self.by_ref:
                self.assertIn(
                    "- **Measured evidence (measure-then-claim):**",
                    self.block(self.by_ref[ref]["id"]),
                )

    def test_cleared_has_the_facade_and_ci_visibility_rows_and_no_id_for_them(
        self,
    ) -> None:
        cleared = self.sub_block("#### ↩︎ Cleared")
        self.assertIn("crates/cobre/src/lib.rs", cleared)
        self.assertIn("ARCHITECTURE.md:102-106", cleared)
        self.assertIn("Python-binding Rust tests invisible to CI", cleared)
        self.assertIn(":347", cleared)
        sanctioned = [c for c in self.cal["cleared"] if c["verdict"] == "sanctioned"]
        self.assertEqual(len(sanctioned), 2)
        for c in sanctioned:
            self.assertIsNone(c["id"])
        for row in self.assigned:
            for a in row["anchors"]:
                self.assertFalse(a["path"].startswith("crates/cobre/"), row["id"])
        for c in self.cal["cleared"]:
            if c["candidateRef"]:
                self.assertTrue(
                    c["dismissalBasis"] and c["basisCitation"], c["candidateRef"]
                )
                self.assertIn(f"**{c['candidateRef']} — ", cleared)

    def test_byte_neutrality_line_on_every_entry_and_rebaselines_held(self) -> None:
        for row in self.assigned:
            block = self.block(row["id"])
            self.assertIn("- **Byte-neutrality:**", block, row["id"])
            self.assertIn(
                row["byteNeutral"], ("asserted", "n/a", "needs-rebaseline"), row["id"]
            )
            if row["byteNeutral"] == "needs-rebaseline":
                self.assertIn("owner decision", block, row["id"])
            if row["byteNeutral"] == "asserted":
                self.assertIn("test_cli_python_file_set_parity.py", block, row["id"])
            self.assertIn(f"- **Baseline:** `{header_baseline()}`", block)
            m = re.search(
                r"^- \*\*Alignment:\*\* (\S+) \((?:provisional; Epic 9 adjudicates .*beyond-sddp-generalization\.md"
                r"|.*; station hint: (\S+), retagged \d{4}-\d{2}-\d{2} by alignment/alignment-ledger\.json\))",
                block,
                re.M,
            )
            self.assertIsNotNone(m, row["id"])
            assert m is not None
            self.assertEqual(m.group(2) or m.group(1), row["alignmentHint"])

    def test_checkers_exit_zero_with_the_exact_title_and_the_slug(self) -> None:
        for arg in (STATION6_TITLE, "cli-python"):
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
                self.assertNotIn("section-not-found", result.stdout + result.stderr)

    def test_perf_queue_only_sev_ab_pd_with_layout_call_sites_and_no_number(
        self,
    ) -> None:
        ab = {
            r["id"]
            for r in self.assigned
            if r["class"] == "PD" and r["severity"][0] in "AB"
        }
        queue = {q["id"]: q for q in self.perf["queue"]}
        self.assertEqual(set(queue), ab)
        self.assertTrue(ab)
        for qid, q in queue.items():
            self.assertIn(q["layout"], LAYOUTS, qid)
            self.assertIn(q["claimType"], ("single-process", "collective"))
            self.assertIn(q["requires"], REQUIRES, qid)
            self.assertEqual(q["status"], "UNMEASURED", qid)
            self.assertTrue(q["exercisingCallSites"], qid)
            self.assertTrue(q["byteNeutralFixShape"], qid)
            self.assertIsNone(q["measured"])
        for row in self.assigned:
            if row["class"] == "PD":
                self.assertIn(
                    row["claimType"], ("single-process", "collective"), row["id"]
                )
                body = self.block(row["id"]).split("\n", 1)[1]
                self.assertIsNone(TIMING_NUMBER.search(body), row["id"])
        self.assertIsNone(
            TIMING_NUMBER.search(json.dumps(self.perf["queue"])),
            "perf-queue carries a timing figure",
        )
        self.assertIn("perf-queue.json", self.section)
        self.assertEqual(self.perf["baseline"], header_baseline())

    def test_td_queue_every_td_with_yardstick_corpus_figure_and_no_new_fixture_crate(
        self,
    ) -> None:
        tds = {r["id"] for r in self.assigned if r["class"] == "TD"}
        self.assertEqual({q["id"] for q in self.td["queue"]}, tds)
        self.assertTrue(tds)
        for q in self.td["queue"]:
            self.assertTrue(q["yardstick"].startswith("§"), q["id"])
            self.assertTrue(q["claim"] and q["corpusFigure"], q["id"])
            fix = q["fixShape"]
            if "fixture crate" in fix.lower():
                self.assertIsNotNone(
                    re.search(
                        r"\b(no|not|never|without|rather than|instead of)\b[^.]{0,60}fixture crate",
                        fix,
                        re.I,
                    ),
                    f"{q['id']}: proposes a new fixture crate",
                )
        for f in self.td["calibrationFolds"]:
            self.assertIn(f["survivorId"], tds)
            self.assertIn(f["folded"], self.folds)
        self.assertEqual(self.td["baseline"], header_baseline())
        self.assertIn("td-queue.json", self.section)

    def test_part_i_block_carries_the_i5_queue_and_the_broadcastconfig_handoff(
        self,
    ) -> None:
        block = self.sub_block(
            "#### Part-I cross-references — item I.5 and the BroadcastConfig handoff"
        )
        rows = re.findall(r"^\| (I\.5-\w+) \|", block, re.M)
        self.assertEqual(
            sorted(rows), sorted(r["ref"] for r in self.handoff["i5Queue"])
        )
        self.assertIn("BroadcastConfig", block)
        self.assertIn("CD-004", block)
        for row in self.assigned:
            if row["partIRef"] == "I.5":
                self.assertIn(row["id"], block)
                self.assertIn("alignment", row["queuedTo"])


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
    "heading-resolution",
    "figures",
    "cross-check",
    "supersession",
    "wave-5-dispositions",
    "i5-handoff",
    "no-timing",
    "read-only-snapshot",
    "py-build",
    "ci-visibility",
)
DATED_TITLE = SCAFFOLD_CLI[3:]
FIGURES_HEADER = "figure\texpected\tmeasured\tstatus\tcommand"
FIGURE_FLOOR = 17
TREE_TOKEN = "<tree@077dbe2c>"


class SectionVerifyTests(sc.StationCase):
    """Executable proof of the station verification (E06-6).

    verify-station.sh runs this module, so nothing here may invoke it (recursion); the
    station driver verify-figures.sh is run once with --no-shared for the same reason, and
    the rendered verification.md is read when it exists.
    """

    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver = cls.station_dir() / "verify-figures.sh"
        cls.tsv_path = cls.station_dir() / "figures.tsv"
        cls.committed_tsv = (
            cls.tsv_path.read_text(encoding="utf-8") if cls.tsv_path.exists() else None
        )
        cls.driver_run = cls.run_driver(DATED_TITLE, "--no-shared")
        cls.tsv = cls.tsv_path.read_text(encoding="utf-8")
        cls.inv = sc.load_json(cls.station_dir() / "inventory.json")

    @classmethod
    def run_driver(cls, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(cls.driver), *args],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_checkers_exit_zero_over_the_dated_title_and_the_slug(self) -> None:
        for arg in (DATED_TITLE, "cli-python"):
            for tool in ("check-anchors.py", "check-reraise.py", "fields-check.py"):
                self.assertEqual(sc.run_checker(tool, arg), 0, f"{tool} {arg}")

    def test_no_tracked_path_is_modified(self) -> None:
        self.assertEqual(sc.tracked_modifications(), [])

    def test_inventory_module_set_equals_the_rs_files_at_the_baseline(self) -> None:
        tree = self.tree()
        listed = [f["path"] for f in self.inv["files"]]
        self.assertEqual(sorted(listed), sorted(set(listed)))
        expected = sorted(p for root in ROOTS for p in tree.rs_files(root))
        self.assertEqual(sorted(listed), expected)
        for f in self.inv["files"]:
            self.assertEqual(f["lines"], sc.raw_lines(f["path"], tree), f["path"])

    def test_verify_figures_no_shared_exits_zero_with_every_check_passing(self) -> None:
        out = self.driver_run.stdout
        self.assertEqual(self.driver_run.returncode, 0, out + self.driver_run.stderr)
        self.assertIn("## Station-specific checks — cli-python", out)
        self.assertIsNotNone(
            re.search(r"^\| shared-verifier \| .* \| - \| SKIP \|$", out, re.M)
        )
        for check in STATION_CHECKS:
            self.assertIsNotNone(
                re.search(rf"^\| {check} \| .* \| 0 \| PASS \|$", out, re.M), check
            )
        self.assertNotIn("| FAIL |", out)
        self.assertNotIn("Failures:", out)
        self.assertIn("Carried-in dirty tracked path: none at 077dbe2c", out)
        self.assertIn("verify-figures.sh cli-python: PASS", self.driver_run.stdout)

    def test_figures_tsv_all_ok_byte_stable_and_free_of_scratch_paths(self) -> None:
        lines = self.tsv.splitlines()
        self.assertEqual(lines[0], FIGURES_HEADER)
        rows = [line.split("\t") for line in lines[1:]]
        self.assertGreaterEqual(len(rows), FIGURE_FLOOR)
        self.assertEqual(len({r[0] for r in rows}), len(rows))
        for name, expected, measured, status, command in rows:
            self.assertEqual(status, "OK", name)
            self.assertEqual(expected, measured, name)
            self.assertNotIn("/tmp/", command, name)
            self.assertNotIn("verify-figures-cli-python.", command, name)
        commands = "\n".join(r[4] for r in rows)
        self.assertIn(TREE_TOKEN, commands)
        for name in (
            "src_files",
            "src_lines",
            "src_nontest_lines",
            "sddp_refs",
            "sddp_ref_files",
            "sddp_use_lines",
            "cli_writer_calls",
            "py_writer_calls",
            "cli_external_writer_names",
            "py_external_writer_names",
            "parity_shared_names",
            "parity_exit",
            "from_config_nontest_sites",
            "cli_test_binaries",
            "py_pytest_files",
            "py_rust_tests",
            "facade_lib_lines",
            "py_cargo_check",
            "ci_cargo_test_steps",
        ):
            self.assertIn(name, {r[0] for r in rows})
        if self.committed_tsv is not None:
            self.assertEqual(self.tsv, self.committed_tsv)

    def test_bare_station_6_is_refused_as_a_selector(self) -> None:
        for arg in ("STATION 6", "Station 6"):
            result = self.run_driver(arg, "--no-shared")
            self.assertEqual(result.returncode, 2, arg)
            self.assertIn("FAIL heading-resolution", result.stderr)
            self.assertEqual(result.stdout, "")
        bad = self.run_driver("--bogus")
        self.assertEqual(bad.returncode, 2)

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
        self.assertIn("## Station-specific checks — cli-python", report)
        self.assertIsNotNone(
            re.search(r"^\| shared-verifier \| .* \| 0 \| PASS \|$", report, re.M)
        )
        for check in STATION_CHECKS:
            self.assertIsNotNone(
                re.search(rf"^\| {check} \| .* \| 0 \| PASS \|$", report, re.M), check
            )
        self.assertNotIn("| FAIL |", report)
        self.assertNotIn("Failures:", report)
        self.assertIn("Superseded figures (old → new", report)
        for sn in ("SN-01", "SN-02", "SN-03", "SN-06"):
            self.assertIn(sn, report)
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
}
HOLD_VERBS = {"take-the-alternative", "override-with-rationale", "reject", "defer"}
GATE_HEADINGS = (
    "### 1.1 Wave-5 dispositions",
    "### 1.2 Part-I I.5 dispositions",
    "### 1.3 New entries by house severity",
    "### 1.4 Holds",
    "### 1.5 Cleared and dup-of",
    "### 1.6 Queues handed on",
    "### 1.7 Handoffs the alignment epic consumes",
    "### 1.8 Worker needs-human items",
    "## 2. Round plan",
    "## 3. Decision record",
    "### 3.1 Wave-5 dispositions",
    "### 3.2 Part-I I.5 dispositions",
    "### 3.3 New entries",
    "### 3.4 Holds",
    "### 3.5 Worker needs-human answers",
    "## 4. Handoffs after the gate",
)
HOLD_ROUNDS = (
    "R4-hold-CD-025-cobre-sddp-home",
    "R4-hold-CD-025-cli-local-helper",
    "R4-hold-CD-029-cobre-io-home",
    "R4-hold-CD-029-two-front-end-copies",
)
GATE_TIMING = (
    re.compile(
        r"(?<![\w.\-/])\d+(?:\.\d+)?\s?(?:ns|µs|us|ms|s|secs?|seconds?|mins?|minutes?|h|hrs?|hours?)\b"
    ),
    re.compile(
        r"(?<![\w.\-/])\d+(?:\.\d+)?\s?(?:x|×)\s?(?:faster|slower|speed-?ups?)\b", re.I
    ),
)


def table_rows(block: str, first_cell: str) -> list[list[str]]:
    return [
        [c.strip() for c in ln.strip().strip("|").split("|")]
        for ln in block.splitlines()
        if ln.startswith(first_cell)
    ]


class GateTests(sc.StationCase):
    """The owner gate (E06-7): gate.md, decisions.json, the queue stamps, alignment-queue.json,
    the BACKLOG owner-gate block and the marker.

    The gate ticket runs this class after recording the owner decision; the class asserts the
    recorded state and never asks a question or writes a file.
    """

    SLUG = "cli-python"
    SECTION_TITLE = "cli-python"

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate = cls.station_dir().joinpath("gate.md").read_text(encoding="utf-8")
        cls.rec = sc.load_json(cls.station_dir() / "decisions.json")
        cls.cal = sc.load_json(cls.station_dir() / "calibration.json")
        cls.wave = sc.load_json(cls.station_dir() / "wave-dispositions.json")
        cls.handoff = sc.load_json(cls.station_dir() / "partI-handoff.json")
        cls.perf = sc.load_json(cls.station_dir() / "perf-queue.json")
        cls.td = sc.load_json(cls.station_dir() / "td-queue.json")
        cls.align = sc.load_json(cls.station_dir() / "alignment-queue.json")
        cls.register = "\n".join(backlog_parse.read_register(sc.BACKLOG))
        cls.section = cls.register.split(SCAFFOLD_CLI, 1)[1].split("\n## ", 1)[0]
        cls.owner_gate = cls.section.split("#### Owner gate — decisions", 1)[1].split(
            "\n#### ", 1
        )[0]
        cls.new = [d for d in cls.rec["decisions"] if d["kind"] == "new-entry"]
        cls.prior = [
            d for d in cls.rec["decisions"] if d["kind"] == "prior-disposition"
        ]
        cls.i5 = [d for d in cls.rec["decisions"] if d["kind"] == "i5-subclaim"]

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
        rows = [r for r in table_rows(prior, "| ") if re.match(r"^\d+$", r[0])]
        self.assertEqual([r[1] for r in rows], ["CD-025", "CD-029", "CD-002", "CD-009"])
        for r in rows:
            self.assertEqual(r[3], "sharpen", r[1])
            self.assertIn("→", r[5], r[1])
            self.assertRegex(r[4], r"^`crates/[^`]+::[A-Za-z_]+`$", r[1])
        cd025 = next(r for r in rows if r[1] == "CD-025")
        self.assertIn("cobre-io", cd025[6])
        self.assertIn("cobre-sddp or cobre-io", cd025[7])
        for layer in PARITY_LAYERS:
            self.assertIn(layer.split("/")[-1], cd025[10])
        i5 = self.gate.split(GATE_HEADINGS[1], 1)[1].split("\n### ", 1)[0]
        i5_rows = [r for r in table_rows(i5, "| ") if re.match(r"^\d+$", r[0])]
        self.assertEqual(
            [r[1] for r in i5_rows[:-1]], [q["ref"] for q in self.handoff["i5Queue"]]
        )
        self.assertEqual(i5_rows[-1][1], "I.3-7 fragment")
        for r in i5_rows:
            self.assertIn(r[3].split(" ")[0], {"keep", "retire", "sharpen"}, r[1])
            self.assertRegex(r[6], r"^`.+`", f"{r[1]}: no re-measure command")
        self.assertNotIn("(measured-by:", self.gate)

    def test_round_plan_keeps_holds_out_of_batches(self) -> None:
        plan = self.gate.split("## 2. Round plan", 1)[1].split("\n## ", 1)[0]
        rows = [r for r in table_rows(plan, "| R") if re.match(r"^R\d", r[0])]
        kinds = [r[1] for r in rows]
        self.assertEqual(
            kinds[:4],
            [
                "prior-disposition",
                "prior-severity",
                "prior-severity",
                "prior-disposition",
            ],
        )
        first_batch = kinds.index("severity-batch")
        self.assertGreater(first_batch, 3)
        holds = [r for r in rows if r[1] == "conflicts-hold"]
        self.assertEqual([r[0] for r in holds], list(HOLD_ROUNDS))
        for r in rows:
            self.assertLessEqual(len(r[3].split("·")), 4, r[0])
            if r[1] == "severity-batch":
                self.assertNotIn("variant", r[2])
                self.assertNotRegex(r[2], r"\bCD-025\b|\bCD-029\b")
            if r[1] == "conflicts-hold":
                self.assertIn("take-the-alternative", r[3])
                self.assertIn("override-with-rationale", r[3])
        self.assertEqual(sum(1 for k in kinds if k == "needs-human"), 37)
        self.assertEqual(kinds[-2:], ["handoff-confirm", "record-shape"])

    def test_decisions_cover_every_prior_id_i5_row_and_calibrated_entry(self) -> None:
        wave_ids = {d["id"] for d in self.wave["dispositions"]}
        new_ids = {r["id"] for r in self.cal["assigned"]}
        self.assertEqual({d["id"] for d in self.prior}, wave_ids)
        self.assertEqual(len(self.prior), 4)
        self.assertEqual(
            [d["id"] for d in self.i5],
            [q["ref"] for q in self.handoff["i5Queue"]] + ["I.3-7 fragment"],
        )
        self.assertEqual({d["id"] for d in self.new}, new_ids)
        for d in self.rec["decisions"]:
            self.assertIn(d["decision"], DECISION_VERBS, d["id"])
            self.assertTrue(d["rationale"].strip(), d["id"])
            if (
                d["kind"] == "prior-disposition"
                and d.get("ownerDisposition") == "retire"
            ):
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
        for d in self.i5:
            self.assertTrue(d["command"], d["id"])
            self.assertIn(d["ownerDisposition"], {"keep", "retire", "sharpen"})
        cd002 = next(d for d in self.prior if d["id"] == "CD-002")
        self.assertEqual((cd002["reviewerSeverity"], cd002["newSeverity"]), ("B", "C"))
        cd029 = next(d for d in self.prior if d["id"] == "CD-029")
        self.assertEqual(cd029["newSeverity"], "B")
        for d in self.prior:
            self.assertEqual(d["ownerDisposition"], "sharpen", d["id"])
            self.assertEqual(d["decision"], "ratify-as-presented", d["id"])
        gap = [d for d in self.rec["decisions"] if d.get("enforcementLayers")]
        self.assertEqual([d["id"] for d in gap], ["CD-025"])
        self.assertEqual(
            sorted(p.split("/")[-1] for p in gap[0]["enforcementLayers"]),
            sorted(PARITY_LAYERS),
        )
        for h in self.rec["holds"]:
            self.assertIn(h["decision"], HOLD_VERBS, h["id"])
        self.assertEqual([h["round"] for h in self.rec["holds"]], list(HOLD_ROUNDS))
        answered = [n for n in self.rec["needsHuman"] if n.get("answer", "").strip()]
        self.assertEqual(len(answered), len(self.rec["needsHuman"]))
        self.assertEqual(len(answered), 37)
        counts = self.rec["counts"]
        verbs = [d["decision"] for d in self.new]
        self.assertEqual(counts["needsHumanAnswered"], 37)
        self.assertEqual(counts["accepted"], verbs.count("accept"))
        self.assertEqual(counts["downgraded"], verbs.count("downgrade"))
        self.assertEqual(counts["rejected"], verbs.count("reject"))
        self.assertEqual(counts["deferred"], verbs.count("defer"))
        self.assertEqual(counts["presented"], len(self.rec["decisions"]))
        self.assertEqual(counts["priorSharpened"], 4)
        self.assertEqual(counts["holdsClosedByAlternative"], len(self.rec["holds"]))
        self.assertTrue(self.rec["returned"])
        self.assertEqual(self.rec["decision"], "ratified")
        self.assertEqual(self.rec["handoffConfirmation"]["decision"], "confirm")

    def test_backlog_carries_the_ratified_line_the_decision_columns_and_the_marker(
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
        wave = self.section.split("#### Wave-5 dispositions", 1)[1].split("\n#### ", 1)[
            0
        ]
        header = next(ln for ln in wave.splitlines() if ln.startswith("| ID | Wave |"))
        self.assertTrue(header.rstrip().endswith("| Decision |"))
        rows = table_rows(wave, "| CD-")
        self.assertEqual([r[0] for r in rows], ["CD-025", "CD-029", "CD-002", "CD-009"])
        for cells in rows:
            self.assertIn("ratified (R0a-wave5)", cells[-1], cells[0])
        self.assertIn("C (reviewer: B)", next(r for r in rows if r[0] == "CD-002")[-1])
        part_i = self.section.split("#### Part-I cross-references", 1)[1].split(
            "\n#### ", 1
        )[0]
        i5_header = next(
            ln for ln in part_i.splitlines() if ln.startswith("| I.5 row |")
        )
        self.assertTrue(i5_header.rstrip().endswith("| Owner decision |"))
        i5_rows = table_rows(part_i, "| I.5-")
        self.assertEqual(len(i5_rows), 8)
        for cells in i5_rows:
            self.assertEqual(cells[-1], "ratified (R0b-i5)", cells[0])
        self.assertIn(f"released {GATE_DATE}", part_i)
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
        for d in self.new:
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
            if d["decision"] == "defer":
                row = next(
                    ln
                    for ln in self.owner_gate.splitlines()
                    if ln.startswith(f"| {d['id']} | defer |")
                )
                self.assertIn("trigger: ", row)
        self.assertIn("**Cleared by this gate (do not re-raise):**", self.owner_gate)
        self.assertIn("no id promoted", self.owner_gate)
        for p in self.prior:
            self.assertEqual(
                self.owner_gate.count(
                    f"| {p['id']} | {p['wave']} | sharpen | ratified |"
                ),
                1,
                p["id"],
            )

    def test_owner_decisions_sit_beneath_the_alignment_field(self) -> None:
        with_directions = [d for d in self.new if d["directions"]]
        self.assertGreaterEqual(len(with_directions), 15)
        for d in with_directions:
            block = self.block(d["id"])
            align_at = block.index("- **Alignment:**")
            for direction in d["directions"]:
                line = f"- **Owner decision ({GATE_DATE}, {direction['round']}):** {direction['choice']}"
                self.assertIn(line, block, d["id"])
                self.assertGreater(block.index(line), align_at, d["id"])
        deferred = [d for d in self.new if d["decision"] == "defer"]
        self.assertEqual([d["id"] for d in deferred], ["OD-043"])
        block = self.block("OD-043")
        self.assertIn(
            f"- **Owner decision ({GATE_DATE}, R2-sevC → R5-nh-34):** defer — trigger: ",
            block,
        )

    def test_held_variants_closed_and_nothing_overridden(self) -> None:
        self.assertEqual(self.cal["conflicts"], [])
        held = {
            r["id"]
            for r in self.cal["reusedIds"] + self.cal["assigned"]
            if r.get("alignmentHint") == "conflicts"
        }
        self.assertEqual(held, set())
        variants = [
            (d["id"], v["variant"])
            for d in self.wave["dispositions"]
            for v in d.get("conflicts") or []
        ]
        self.assertEqual(
            [(h["priorId"], h["variant"]) for h in self.rec["holds"]], variants
        )
        for h in self.rec["holds"]:
            self.assertEqual(h["decision"], "take-the-alternative")
            self.assertFalse(h["idPromoted"])
        self.assertEqual(
            [d for d in self.rec["decisions"] if d.get("overrideRationale")], []
        )
        for h in self.align["holds"]:
            self.assertTrue(h["status"].startswith("CLOSED"))

    def test_queues_and_handoffs_reflect_the_ratified_state(self) -> None:
        kept = {
            d["id"]
            for d in self.new
            if (d.get("perfQueue") or {}).get("effect") == "kept"
        }
        self.assertEqual({q["id"] for q in self.perf["queue"]}, kept)
        self.assertEqual(self.perf["gate"]["rowsStruck"], [])
        self.assertEqual(sorted(self.perf["gate"]["rowsKept"]), sorted(kept))
        for q in self.perf["queue"]:
            self.assertEqual(q["status"], "UNMEASURED")
            self.assertIn(q["layout"], LAYOUTS)
        td_kept = {
            d["id"]
            for d in self.new
            if (d.get("tdQueue") or {}).get("effect") == "kept"
        }
        self.assertEqual({q["id"] for q in self.td["queue"]}, td_kept)
        self.assertEqual(self.td["gate"]["rowsStruck"], [])
        self.assertEqual(sorted(self.td["gate"]["rowsKept"]), sorted(td_kept))
        decided = {d["id"] for d in self.rec["decisions"]}
        self.assertTrue({r["id"] for r in self.align["rows"]} <= decided)
        for rid in ("CD-089", "CD-091", "CD-095", "CD-025", "CD-029", "I.3-7 fragment"):
            self.assertIn(rid, {r["id"] for r in self.align["rows"]})
        self.assertEqual(self.align["i5Handoff"], "released")
        self.assertEqual(
            self.owner_gate.count("Owner gate"), 0
        )  # the queue notes live in the Queued-out block, not the gate block
        queued_out = self.section.split("#### Queued out", 1)[1].split("\n#### ", 1)[0]
        self.assertEqual(queued_out.count(f"Owner gate {GATE_DATE}"), 2)
        self.assertIn(f"Released at the owner gate {GATE_DATE}", queued_out)
        hits = [m.group(0) for rx in GATE_TIMING for m in rx.finditer(self.gate)]
        self.assertEqual(hits, [])

    def test_cd_002_downgrade_is_written_in_place(self) -> None:
        self.assertEqual(
            self.register.count(
                "**CD-002 · Sev C · bad-abstraction · effort S–M · confidence high**"
            ),
            1,
        )
        self.assertNotIn("**CD-002 · Sev B ·", self.register)
        entry = self.register.split(
            "**CD-002 · Sev C · bad-abstraction · effort S–M · confidence high**", 1
        )[1].split("\n\n", 1)[0]
        self.assertIn(f"**Status ({GATE_DATE}):** downgraded B → C", entry)
        self.assertIn(
            "| CD-002 | 4 | sharpen | ratified | C (reviewer: B) |", self.owner_gate
        )


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
                ".github",
                "Cargo.toml",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
