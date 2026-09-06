"""core-io station tests: inventory census, partition proof, prior register, clean tree.

Run from the repository root:
    python3 -m unittest discover -s plans/architecture-debt-audit/stations/core-io/tests -p 'test_*.py'
Later core+io tickets append their own test classes here; the station verification runs the module.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

ID_RE = re.compile(r"\b(?:CD|PD|OD|TD)-\d{3}\b")
REGISTER_ID_RE = re.compile(r"^(?:(?:CD|PD|OD|TD)-\d{3}|reserved-seam-census|mirror:.+)$")
HEADING_RE = re.compile(r"^### (?P<title>.+?) — (?P<disp>KEEP|RETIRE|SHARPEN|OUT OF STATION)\b")
FIELD_RE = re.compile(r"^- \*\*(?P<label>[^*]+?):\*\*\s*(?P<value>.*)$")
OUT_OF_STATION_IDS = {"CD-031", "CD-039", "CD-025", "CD-029", "CD-004", "CD-001"}


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, cwd=sc.REPO, capture_output=True, text=True, check=True).stdout


def prior_register_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in text.splitlines():
        head = HEADING_RE.match(line)
        if head:
            entries.append({"title": head.group("title"), "disposition": head.group("disp")})
            continue
        field = FIELD_RE.match(line)
        if field and entries:
            entries[-1][field.group("label")] = field.group("value")
    return entries


class InventoryTests(sc.StationCase):
    SLUG = "core-io"
    SECTION_TITLE = "core-io"

    @classmethod
    def setUpClass(cls):
        cls.inv = sc.load_json(cls.STATION_DIR / "inventory.json" if cls.STATION_DIR else sc.STATIONS / cls.SLUG / "inventory.json")

    def modules(self):
        return [m for crate in self.inv["crates"].values() for m in crate["modules"]]

    def test_inventory_parses_and_carries_the_envelope(self):
        for key in ("station", "baseline", "producedAt", "commands", "crates", "integrationBinaries", "subStations", "partition"):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "core-io")
        for key in ("dirs", "topLevel", "linesRaw", "linesNonTest", "integrationBinaries"):
            self.assertIn(key, self.inv["commands"])

    def test_baseline_is_the_register_pin_and_matches_head_under_the_drift_rule(self):
        pin = backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG))
        self.assertEqual(self.inv["baseline"], pin)
        self.assertTrue(backlog_parse.baseline_matches_head(self.inv["baseline"]),
                        f"baseline-drift pinned={pin} head={backlog_parse.head_sha()}")

    def test_every_module_path_exists_exactly_once(self):
        paths = [m["path"] for m in self.modules()]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertTrue((sc.REPO / path).exists(), path)
        expected_dirs = {
            "crates/cobre-core/src/" + d for d in ("constraints", "entities", "model", "model/temporal", "model/resolved", "stats", "system", "topology")
        } | {
            "crates/cobre-io/src/" + d for d in ("config", "constraints", "extensions", "output", "output/policy", "resolution", "scenarios", "scenarios/estimation", "system", "validation", "validation/semantic")
        }
        self.assertEqual({m["path"] for m in self.modules() if m["kind"] == "directory"}, expected_dirs)
        top = set(sh("find crates/cobre-core/src crates/cobre-io/src -maxdepth 1 -name '*.rs'").split())
        self.assertEqual({m["path"] for m in self.modules() if m["kind"] == "file"}, top)

    def test_each_module_carries_both_counts_and_matches_the_tree(self):
        for m in self.modules():
            with self.subTest(path=m["path"]):
                self.assertIn("linesRaw", m)
                self.assertIn("linesNonTest", m)
                self.assertIn("commandRef", m)
                if m["kind"] == "directory":
                    wc = int(sh(f"find {m['path']} -name '*.rs' -print0 | xargs -0 wc -l | tail -1").split()[0])
                else:
                    wc = int(sh(f"wc -l < {m['path']}"))
                self.assertEqual(m["linesRaw"], wc)
                self.assertEqual(m["linesNonTest"], sc.non_test_lines(m["path"]))
                self.assertLessEqual(m["linesNonTest"], m["linesRaw"])

    def test_crate_totals_reconcile_with_wc_and_loc_stats(self):
        expected = {"cobre-core": (18671, 9228), "cobre-io": (104066, 45915)}
        for crate, (raw, non_test) in expected.items():
            with self.subTest(crate=crate):
                block = self.inv["crates"][crate]
                top = [m for m in block["modules"] if m["path"].count("/") == 3]
                self.assertEqual(sum(m["linesRaw"] for m in top), block["linesRaw"])
                self.assertEqual(block["linesRaw"], raw)
                self.assertEqual(block["reconciliation"]["linesRaw"], sc.raw_lines(f"crates/{crate}/src"))
                self.assertEqual(sum(m["linesNonTest"] for m in top), block["linesNonTest"])
                self.assertEqual(block["linesNonTest"], non_test)
                self.assertEqual(block["linesNonTest"], sc.loc_stats(crate)["prod_all"])

    def test_integration_binaries_are_rs_scoped_and_explain_the_naive_count(self):
        bins = self.inv["integrationBinaries"]
        for crate, count in (("cobre-core", 2), ("cobre-io", 12)):
            with self.subTest(crate=crate):
                self.assertEqual(bins[crate]["count"], count)
                self.assertEqual(len(bins[crate]["files"]), count)
                self.assertEqual(int(sh(bins[crate]["command"])), count)
                self.assertEqual(sorted(pathlib.Path(p).name for p in sh(f"find crates/{crate}/tests -maxdepth 1 -name '*.rs'").split()),
                                 bins[crate]["files"])
        self.assertEqual(bins["cobre-io"]["naiveCount"], 14)
        self.assertIn("fixtures/", bins["cobre-io"]["note"])
        self.assertIn("helpers/", bins["cobre-io"]["note"])
        self.assertIn("13", bins["cobre-io"]["note"])
        self.assertEqual(bins["combined"]["count"], 14)

    def test_partition_covers_every_path_exactly_once(self):
        subs = self.inv["subStations"]
        self.assertEqual(set(subs), {"A", "B", "C", "D"})
        assigned = [p for s in subs.values() for p in s["paths"]]
        self.assertEqual(len(assigned), len(set(assigned)), "path assigned twice")
        tree = set(sh("find crates/cobre-core/src crates/cobre-io/src -maxdepth 1").split()) - {"crates/cobre-core/src", "crates/cobre-io/src"}
        tree = {p for p in tree if p.endswith(".rs") or (sc.REPO / p).is_dir()}
        tree |= set(sh("find crates/cobre-core/src crates/cobre-io/src -mindepth 1 -type d").split())
        self.assertEqual(set(assigned), tree)
        for m in self.modules():
            owners = [k for k, s in subs.items() if m["path"] in s["paths"]]
            self.assertEqual(owners, [m["subStation"]], m["path"])
        self.assertTrue(self.inv["partition"]["isPartition"])
        self.assertEqual(subs["A"]["crate"], "cobre-core")
        for key in ("B", "C", "D"):
            self.assertEqual(subs[key]["crate"], "cobre-io")
            self.assertGreater(subs[key]["linesRaw"], 0)


class PriorRegisterTests(sc.StationCase):
    SLUG = "core-io"
    SECTION_TITLE = "core-io"

    @classmethod
    def setUpClass(cls):
        cls.text = (sc.STATIONS / cls.SLUG / "prior-register.md").read_text(encoding="utf-8")
        cls.entries = prior_register_entries(cls.text)

    def test_every_entry_has_id_anchor_status_and_disposition(self):
        self.assertGreaterEqual(len(self.entries), 10)
        for e in self.entries:
            with self.subTest(entry=e["title"]):
                for label in ("Register ID", "Anchor", "Status", "Disposition", "Reason"):
                    self.assertIn(label, e, f"missing {label}")
                self.assertRegex(e["Register ID"], REGISTER_ID_RE)
                self.assertRegex(e["Disposition"], r"^(KEEP|RETIRE|SHARPEN|OUT OF STATION)\b")
                anchors = sc.anchors_in(e["Anchor"])
                self.assertTrue(anchors, "no backticked anchor")
                for a in anchors:
                    self.assertTrue(sc.anchor_exists(a), f"anchor does not resolve: {a}")

    def test_cd_010_is_retired_because_resolved_with_live_anchors(self):
        cd010 = [e for e in self.entries if e["Register ID"] == "CD-010"]
        self.assertEqual(len(cd010), 1)
        e = cd010[0]
        self.assertTrue(e["Disposition"].startswith("RETIRE"))
        self.assertIn("RESOLVED", e["Status"])
        for a in ("`crates/cobre-io/src/output/policy/records.rs:52`", "`crates/cobre-io/src/output/policy/records.rs:29-31`",
                  "`crates/cobre-io/src/output/policy/records.rs::family`"):
            self.assertIn(a, e["Anchor"])
        self.assertIn("dictionary.rs:31-38", e.get("Residue to re-check, not re-raise", ""))

    def test_reserved_seams_are_sanctioned_with_mirror_citations(self):
        seams = {e["title"]: e for e in self.entries if e["Register ID"] == "reserved-seam-census"}
        lip = next(e for t, e in seams.items() if "LipschitzConfig.mode" in t)
        self.assertIn("`crates/cobre-io/src/config/training.rs:531`", lip["Anchor"])
        self.assertIn("`LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`", lip["Status"])
        self.assertIn("dismissable", lip["Disposition"])
        hydro = next(e for t, e in seams.items() if "filling_target_violation_cost" in t)
        self.assertIn("Verified NOT reserved", hydro["Status"])
        self.assertIn("consumed, not reserved", hydro["Disposition"])
        self.assertIn("dismissable", hydro["Disposition"])

    def test_out_of_station_items_name_the_owning_epic_and_stay_off_the_dnr_list(self):
        out = [e for e in self.entries if e["Disposition"].startswith("OUT OF STATION")]
        ids = {e["Register ID"] for e in out}
        self.assertTrue({"CD-039", "CD-025", "CD-029"} <= ids, ids)
        for e in out:
            self.assertRegex(e["Disposition"], r"Epic \d+")
        c17 = next(e for e in out if "warn_dropped_source_couplings" in e["Anchor"])
        self.assertIn("policy_load.rs::warn_dropped_source_couplings", c17["Anchor"])
        dnr = self.text.split("\n## Do not re-raise\n", 1)[1]
        for item in dnr.strip().splitlines():
            for fid in ID_RE.findall(item):
                self.assertNotIn(fid, {"CD-039", "CD-025", "CD-029", "CD-004", "CD-001"}, item)
            self.assertNotIn("BoundaryStateRequirements", item)
            self.assertNotIn("warn_dropped_source_couplings", item)

    def test_do_not_re_raise_list_is_last_and_tokenizable(self):
        self.assertIn("\n## Do not re-raise\n", self.text)
        tail = self.text.split("\n## Do not re-raise\n", 1)[1]
        self.assertNotIn("\n## ", tail, "the do-not-re-raise list must be the final section")
        bullets = [l for l in tail.strip().splitlines() if l.startswith("- ")]
        self.assertGreaterEqual(len(bullets), 8)
        self.assertEqual(len(bullets), len([l for l in tail.strip().splitlines() if l.strip()]))
        for b in bullets:
            self.assertRegex(b, r"`[^`]+`", "each item carries a backticked path or symbol for check-reraise tokenization")


class CleanTreeTests(sc.StationCase):
    SLUG = "core-io"

    def test_no_tracked_file_modified(self):
        self.assertEqual(sc.tracked_modifications(), [])
        untracked = sh("git status --porcelain --untracked-files=all").splitlines()
        for line in untracked:
            self.assertTrue(line.startswith("?? plans/architecture-debt-audit/"), line)
