"""cobre-stochastic station tests: inventory census, four-way partition, prior register.

Run from the repository root:
    python3 -m unittest plans/architecture-debt-audit/stations/stochastic/tests/test_station.py
Later cobre-stochastic tickets append their own stage classes here; the station
verification runs the module.
"""

from __future__ import annotations

import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "tools"))

from lib import backlog_parse  # noqa: E402
from lib import station_checks as sc  # noqa: E402

CFG_TEST = re.compile(r"^#\[cfg\(test\)\]")
SUB_IDS = {"par", "sampling", "tree-noise", "seam"}
EXPECTED_PARTITION = {"par": 21, "sampling": 8, "tree-noise": 16, "seam": 6}
DISPOSITIONS = {"keep", "retire", "not-ours", "cross-reference", "resolved", "sharpen"}
# backticked repo path with optional :line or ::symbol (a superset of ANCHOR_LOC_RE that
# also tolerates a trailing directory anchor with no extension).
ANCHOR_TOKEN = re.compile(
    r"`(?P<path>(?:crates|scripts|docs|plans|schemas|examples|tests|\.github|\.claude)/[\w./-]+?"
    r"(?:\.(?:rs|toml|md|json|fbs|sh|py|yml|yaml|txt|csv|lock))?)"
    r"(?:::(?P<sym>[A-Za-z_]\w*)|:(?P<line>\d+))?`"
)


def raw_lines(rel: str) -> int:
    return (sc.REPO / rel).read_bytes().count(b"\n")


def non_test_lines(rel: str) -> int:
    n = 0
    for line in (
        (sc.REPO / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if CFG_TEST.match(line):
            break
        n += 1
    return n


class InventoryTests(sc.StationCase):
    SLUG = "stochastic"

    @classmethod
    def setUpClass(cls):
        cls.inv = sc.load_json(
            (cls.STATION_DIR or sc.STATIONS / cls.SLUG) / "inventory.json"
        )

    def test_inventory_parses_and_carries_the_envelope(self):
        for key in (
            "station",
            "baseline",
            "crate",
            "totals",
            "files",
            "subStations",
            "partitionCheck",
            "corrections",
            "layeringEvidence",
        ):
            self.assertIn(key, self.inv, f"inventory.json missing {key}")
        self.assertEqual(self.inv["station"], "stochastic")
        self.assertEqual(self.inv["crate"], "cobre-stochastic")

    def test_baseline_is_the_register_pin_and_matches_head_under_the_drift_rule(self):
        pin = backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG))
        self.assertTrue(self.inv["baseline"].startswith(pin[:8]))
        self.assertTrue(
            backlog_parse.baseline_matches_head(self.inv["baseline"]),
            "HEAD's evaluated surfaces must be byte-identical to the pin",
        )

    def test_every_file_path_exists_and_line_counts_reproduce(self):
        for f in self.inv["files"]:
            p = sc.REPO / f["path"]
            self.assertTrue(p.is_file(), f"missing {f['path']}")
            self.assertEqual(
                f["rawLines"], raw_lines(f["path"]), f"{f['path']}: rawLines != wc -l"
            )
            expected_nt = 0 if f.get("isSiblingTest") else non_test_lines(f["path"])
            self.assertEqual(
                f["nonTestLines"], expected_nt, f"{f['path']}: nonTestLines drift"
            )
            self.assertEqual(
                f["hasInlineTests"],
                any(
                    CFG_TEST.match(ln)
                    for ln in p.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                ),
                f"{f['path']}: hasInlineTests drift",
            )

    def test_totals_reconcile_with_re_measurement(self):
        t = self.inv["totals"]
        files = self.inv["files"]
        self.assertEqual(t["srcFiles"], 51)
        self.assertEqual(t["srcFiles"], len(files))
        self.assertEqual(t["nonTestLines"], 35198)
        self.assertEqual(t["nonTestLines"], sum(f["nonTestLines"] for f in files))
        self.assertEqual(t["inlineTestFiles"], 35)
        self.assertEqual(
            t["inlineTestFiles"], sum(1 for f in files if f["hasInlineTests"])
        )
        self.assertEqual((t["integrationBinaries"], t["integrationLines"]), (8, 4096))
        self.assertEqual(sum(i["lines"] for i in self.inv["integration"]), 4096)
        for field in (
            "srcFiles",
            "nonTestLines",
            "inlineTestFiles",
            "integrationBinaries",
            "integrationLines",
        ):
            self.assertTrue(
                t["commands"].get(field), f"totals.{field} has no reproducing command"
            )

    def test_partition_is_total_and_disjoint(self):
        pc = self.inv["partitionCheck"]
        self.assertEqual(pc["sum"], 51)
        self.assertEqual(pc["duplicates"], [])
        self.assertEqual(pc["unassigned"], [])
        counts = {s["id"]: s["files"] for s in self.inv["subStations"]}
        self.assertEqual(counts, EXPECTED_PARTITION)
        self.assertEqual({s["id"] for s in self.inv["subStations"]}, SUB_IDS)
        seam = next(f for f in self.inv["files"] if f["path"].endswith("/error.rs"))
        self.assertEqual(
            seam["subStation"], "seam", "error.rs must be swept with the seam"
        )

    def test_corrections_split_the_external_rs_claim(self):
        ext = next(c for c in self.inv["corrections"] if "external.rs" in c["claim"])
        self.assertEqual(ext["measured"]["rawLines"], 3200)
        self.assertEqual(ext["measured"]["nonTestLines"], 888)
        self.assertFalse(ext["measured"]["isLargest"])
        sob = next(
            c
            for c in self.inv["corrections"]
            if "sobol_directions.rs" in (c.get("measured", {}).get("path") or "")
        )
        self.assertEqual(sob["measured"]["rawLines"], 21229)
        self.assertTrue(sob["measured"]["generated"])
        for c in self.inv["corrections"]:
            self.assertTrue(
                c.get("command"), "each correction must carry its measuring command"
            )

    def test_layering_evidence_is_l2_to_l1_consistent(self):
        le = next(
            e for e in self.inv["layeringEvidence"] if e["symbol"] == "season_cast"
        )
        self.assertEqual((le["direction"], le["verdict"]), ("L2->L1", "consistent"))
        sites = set(le["sites"])
        for expected in (
            "crates/cobre-io/src/scenarios/estimation.rs:70",
            "crates/cobre-io/src/validation/semantic/inflow_seeding.rs:22",
            "crates/cobre-io/src/validation/semantic/thermal.rs:17",
            "crates/cobre-io/src/validation/semantic/scenarios.rs:12",
        ):
            self.assertIn(expected, sites)


class PriorRegisterTests(sc.StationCase):
    SLUG = "stochastic"

    @classmethod
    def setUpClass(cls):
        text = (
            (cls.STATION_DIR or sc.STATIONS / cls.SLUG) / "prior-register.md"
        ).read_text(encoding="utf-8")
        cls.entries = cls._split(text)

    @staticmethod
    def _split(text: str) -> list[dict]:
        entries: list[dict] = []
        current: dict | None = None
        for line in text.splitlines():
            if line.startswith("### "):
                current = {"title": line[4:].strip(), "body": []}
                entries.append(current)
            elif current is not None and not line.startswith("## "):
                current["body"].append(line)
        for e in entries:
            e["text"] = "\n".join(e["body"])
        return entries

    def test_seven_entries(self):
        self.assertEqual(len(self.entries), 7, [e["title"] for e in self.entries])

    def test_each_entry_has_disposition_reraisekey_and_a_resolving_anchor(self):
        for e in self.entries:
            body = e["text"]
            disp = re.search(r"(?mi)^-\s+\*\*Disposition\*\*:\s*([a-z-]+)", body)
            self.assertIsNotNone(disp, f"{e['title']}: no Disposition line")
            self.assertIn(
                disp.group(1),
                DISPOSITIONS,
                f"{e['title']}: bad disposition {disp.group(1)!r}",
            )
            self.assertRegex(
                body,
                r"(?mi)^-\s+\*\*reraiseKey\*\*:\s*`",
                f"{e['title']}: no reraiseKey",
            )
            anchors = [m for m in ANCHOR_TOKEN.finditer(body)]
            self.assertTrue(anchors, f"{e['title']}: no backticked baseline anchor")
            self.assertTrue(
                any(self._anchor_resolves(m) for m in anchors),
                f"{e['title']}: no baseline anchor resolves in the tree",
            )

    @staticmethod
    def _anchor_resolves(m: re.Match) -> bool:
        path = m.group("path")
        p = sc.REPO / path
        if m.group("sym") or m.group("line"):
            return sc.anchor_exists(m.group(0).strip("`"))
        return p.exists()

    def test_cd_001_is_resolved_with_live_anchors_and_an_alignment_hint(self):
        cd = next((e for e in self.entries if "CD-001" in e["title"]), None)
        self.assertIsNotNone(cd, "CD-001 entry missing")
        body = cd["text"]
        self.assertRegex(body, r"(?mi)^-\s+\*\*Disposition\*\*:\s*resolved")
        for anchor in (
            "crates/cobre-sddp/src/setup/stochastic_pipeline.rs:424",
            "crates/cobre-cli/src/commands/run/setup.rs:409",
            "crates/cobre-stochastic/src/context.rs:601",
        ):
            self.assertTrue(
                sc.anchor_exists(f"`{anchor}`"), f"CD-001 anchor unresolved: {anchor}"
            )
        self.assertIn("docs/design/reserved-seams-and-deferred-debt.md", body)
        self.assertRegex(
            body, r"[Aa]lignment hint", "CD-001 must carry the L1-homing alignment hint"
        )
        self.assertRegex(
            body, r"[Nn]o attacker may raise", "CD-001 must forbid a fresh re-raise"
        )

    def test_out_of_station_entries_name_the_owning_station(self):
        for e in self.entries:
            disp = re.search(r"(?mi)^-\s+\*\*Disposition\*\*:\s*([a-z-]+)", e["text"])
            if disp and disp.group(1) == "not-ours":
                self.assertRegex(
                    e["text"],
                    r"(?mi)^-\s+\*\*Owning station",
                    f"{e['title']}: not-ours without an owning station",
                )


if __name__ == "__main__":
    unittest.main()
