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
REGISTER_ID_RE = re.compile(
    r"^(?:(?:CD|PD|OD|TD)-\d{3}|reserved-seam-census|mirror:.+)$"
)
HEADING_RE = re.compile(
    r"^### (?P<title>.+?) — (?P<disp>KEEP|RETIRE|SHARPEN|OUT OF STATION)\b"
)
FIELD_RE = re.compile(r"^- \*\*(?P<label>[^*]+?):\*\*\s*(?P<value>.*)$")
OUT_OF_STATION_IDS = {"CD-031", "CD-039", "CD-025", "CD-029", "CD-004", "CD-001"}


def sh(cmd: str) -> str:
    return subprocess.run(
        cmd, shell=True, cwd=sc.REPO, capture_output=True, text=True, check=True
    ).stdout


def prior_register_entries(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in text.splitlines():
        head = HEADING_RE.match(line)
        if head:
            entries.append(
                {"title": head.group("title"), "disposition": head.group("disp")}
            )
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
        cls.inv = sc.load_json(
            cls.STATION_DIR / "inventory.json"
            if cls.STATION_DIR
            else sc.STATIONS / cls.SLUG / "inventory.json"
        )

    def modules(self):
        return [m for crate in self.inv["crates"].values() for m in crate["modules"]]

    def test_inventory_parses_and_carries_the_envelope(self):
        for key in (
            "station",
            "baseline",
            "producedAt",
            "commands",
            "crates",
            "integrationBinaries",
            "subStations",
            "partition",
        ):
            self.assertIn(key, self.inv)
        self.assertEqual(self.inv["station"], "core-io")
        for key in (
            "dirs",
            "topLevel",
            "linesRaw",
            "linesNonTest",
            "integrationBinaries",
        ):
            self.assertIn(key, self.inv["commands"])

    def test_baseline_is_the_register_pin_and_matches_head_under_the_drift_rule(self):
        pin = backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG))
        self.assertEqual(self.inv["baseline"], pin)
        self.assertTrue(
            backlog_parse.baseline_matches_head(self.inv["baseline"]),
            f"baseline-drift pinned={pin} head={backlog_parse.head_sha()}",
        )

    def test_every_module_path_exists_exactly_once(self):
        paths = [m["path"] for m in self.modules()]
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            self.assertTrue((sc.REPO / path).exists(), path)
        expected_dirs = {
            "crates/cobre-core/src/" + d
            for d in (
                "constraints",
                "entities",
                "model",
                "model/temporal",
                "model/resolved",
                "stats",
                "system",
                "topology",
            )
        } | {
            "crates/cobre-io/src/" + d
            for d in (
                "config",
                "constraints",
                "extensions",
                "output",
                "output/policy",
                "resolution",
                "scenarios",
                "scenarios/estimation",
                "system",
                "validation",
                "validation/semantic",
            )
        }
        self.assertEqual(
            {m["path"] for m in self.modules() if m["kind"] == "directory"},
            expected_dirs,
        )
        top = set(
            sh(
                "find crates/cobre-core/src crates/cobre-io/src -maxdepth 1 -name '*.rs'"
            ).split()
        )
        self.assertEqual(
            {m["path"] for m in self.modules() if m["kind"] == "file"}, top
        )

    def test_each_module_carries_both_counts_and_matches_the_tree(self):
        for m in self.modules():
            with self.subTest(path=m["path"]):
                self.assertIn("linesRaw", m)
                self.assertIn("linesNonTest", m)
                self.assertIn("commandRef", m)
                if m["kind"] == "directory":
                    wc = int(
                        sh(
                            f"find {m['path']} -name '*.rs' -print0 | xargs -0 wc -l | tail -1"
                        ).split()[0]
                    )
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
                self.assertEqual(
                    block["reconciliation"]["linesRaw"],
                    sc.raw_lines(f"crates/{crate}/src"),
                )
                self.assertEqual(
                    sum(m["linesNonTest"] for m in top), block["linesNonTest"]
                )
                self.assertEqual(block["linesNonTest"], non_test)
                self.assertEqual(block["linesNonTest"], sc.loc_stats(crate)["prod_all"])

    def test_integration_binaries_are_rs_scoped_and_explain_the_naive_count(self):
        bins = self.inv["integrationBinaries"]
        for crate, count in (("cobre-core", 2), ("cobre-io", 12)):
            with self.subTest(crate=crate):
                self.assertEqual(bins[crate]["count"], count)
                self.assertEqual(len(bins[crate]["files"]), count)
                self.assertEqual(int(sh(bins[crate]["command"])), count)
                self.assertEqual(
                    sorted(
                        pathlib.Path(p).name
                        for p in sh(
                            f"find crates/{crate}/tests -maxdepth 1 -name '*.rs'"
                        ).split()
                    ),
                    bins[crate]["files"],
                )
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
        tree = set(
            sh("find crates/cobre-core/src crates/cobre-io/src -maxdepth 1").split()
        ) - {"crates/cobre-core/src", "crates/cobre-io/src"}
        tree = {p for p in tree if p.endswith(".rs") or (sc.REPO / p).is_dir()}
        tree |= set(
            sh(
                "find crates/cobre-core/src crates/cobre-io/src -mindepth 1 -type d"
            ).split()
        )
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
        cls.text = (sc.STATIONS / cls.SLUG / "prior-register.md").read_text(
            encoding="utf-8"
        )
        cls.entries = prior_register_entries(cls.text)

    def test_every_entry_has_id_anchor_status_and_disposition(self):
        self.assertGreaterEqual(len(self.entries), 10)
        for e in self.entries:
            with self.subTest(entry=e["title"]):
                for label in (
                    "Register ID",
                    "Anchor",
                    "Status",
                    "Disposition",
                    "Reason",
                ):
                    self.assertIn(label, e, f"missing {label}")
                self.assertRegex(e["Register ID"], REGISTER_ID_RE)
                self.assertRegex(
                    e["Disposition"], r"^(KEEP|RETIRE|SHARPEN|OUT OF STATION)\b"
                )
                anchors = sc.anchors_in(e["Anchor"])
                self.assertTrue(anchors, "no backticked anchor")
                for a in anchors:
                    self.assertTrue(
                        sc.anchor_exists(a), f"anchor does not resolve: {a}"
                    )

    def test_cd_010_is_retired_because_resolved_with_live_anchors(self):
        cd010 = [e for e in self.entries if e["Register ID"] == "CD-010"]
        self.assertEqual(len(cd010), 1)
        e = cd010[0]
        self.assertTrue(e["Disposition"].startswith("RETIRE"))
        self.assertIn("RESOLVED", e["Status"])
        for a in (
            "`crates/cobre-io/src/output/policy/records.rs:52`",
            "`crates/cobre-io/src/output/policy/records.rs:29-31`",
            "`crates/cobre-io/src/output/policy/records.rs::family`",
        ):
            self.assertIn(a, e["Anchor"])
        self.assertIn(
            "dictionary.rs:31-38", e.get("Residue to re-check, not re-raise", "")
        )

    def test_reserved_seams_are_sanctioned_with_mirror_citations(self):
        seams = {
            e["title"]: e
            for e in self.entries
            if e["Register ID"] == "reserved-seam-census"
        }
        lip = next(e for t, e in seams.items() if "LipschitzConfig.mode" in t)
        self.assertIn("`crates/cobre-io/src/config/training.rs:531`", lip["Anchor"])
        self.assertIn(
            "`LipschitzConfig.mode` and its enclosing `UpperBoundEvaluationConfig`",
            lip["Status"],
        )
        self.assertIn("dismissable", lip["Disposition"])
        hydro = next(
            e for t, e in seams.items() if "filling_target_violation_cost" in t
        )
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
                self.assertNotIn(
                    fid, {"CD-039", "CD-025", "CD-029", "CD-004", "CD-001"}, item
                )
            self.assertNotIn("BoundaryStateRequirements", item)
            self.assertNotIn("warn_dropped_source_couplings", item)

    def test_do_not_re_raise_list_is_last_and_tokenizable(self):
        self.assertIn("\n## Do not re-raise\n", self.text)
        tail = self.text.split("\n## Do not re-raise\n", 1)[1]
        self.assertNotIn(
            "\n## ", tail, "the do-not-re-raise list must be the final section"
        )
        bullets = [l for l in tail.strip().splitlines() if l.startswith("- ")]
        self.assertGreaterEqual(len(bullets), 8)
        self.assertEqual(
            len(bullets), len([l for l in tail.strip().splitlines() if l.strip()])
        )
        for b in bullets:
            self.assertRegex(
                b,
                r"`[^`]+`",
                "each item carries a backticked path or symbol for check-reraise tokenization",
            )


class CleanTreeTests(sc.StationCase):
    SLUG = "core-io"

    def test_no_tracked_file_modified(self):
        self.assertEqual(sc.tracked_modifications(), [])
        untracked = sh("git status --porcelain --untracked-files=all").splitlines()
        for line in untracked:
            self.assertTrue(line.startswith("?? plans/architecture-debt-audit/"), line)


OWNED_PART_I = {"I.3-1", "I.3-2", "I.3-3", "I.3-4", "I.3-6", "I.3-7"}
PHASES = {"0a", "0b", "1"}


def validate_partI_envelope(env: dict) -> list[str]:
    """Structural validation of partI-handoff.json; every failure names the offending partIRef."""
    bad: list[str] = []
    base = env.get("baseline", "")
    refs = [d.get("partIRef") for d in env.get("dispositions", [])]
    if sorted(refs) != sorted(OWNED_PART_I):
        bad.append(f"owned set mismatch: {sorted(refs)} != {sorted(OWNED_PART_I)}")
    for d in env.get("dispositions", []):
        ref, disp = d.get("partIRef", "?"), d.get("disposition")
        if disp not in ("keep", "retire", "sharpen"):
            bad.append(f"{ref}: bad disposition {disp!r}")
        if (
            disp in ("retire", "sharpen")
            and not str(d.get("changedSinceV012", "")).strip()
        ):
            bad.append(f"{ref}: {disp} without changedSinceV012")
        if disp == "sharpen":
            claim = str(d.get("survivingClaim", "")).strip()
            if not claim:
                bad.append(f"{ref}: sharpen without survivingClaim")
            elif claim == str(d.get("v012Title", "")).strip():
                bad.append(
                    f"{ref}: survivingClaim is not narrower than the v0.12 title"
                )
        if disp == "retire" and not d.get("resolvingCommit"):
            bad.append(f"{ref}: retire without resolvingCommit")
        if d.get("proposedPhase") not in PHASES:
            bad.append(f"{ref}: bad proposedPhase {d.get('proposedPhase')!r}")
        if not str(d.get("alignmentDestination", "")).strip():
            bad.append(f"{ref}: no alignmentDestination")
        anchor = d.get("baselineAnchor") or {}
        path, symbol = anchor.get("path"), anchor.get("symbol")
        if (
            not path
            or subprocess.run(
                ["git", "show", f"{base}:{path}"], cwd=sc.REPO, capture_output=True
            ).returncode
        ):
            bad.append(f"{ref}: anchor path does not resolve at {base[:12]}: {path}")
        elif symbol and not sc.anchor_exists(f"`{path}::{symbol}`"):
            bad.append(f"{ref}: symbol {symbol} unresolved in {path}")
    handoffs = env.get("handoffs", [])
    if len(handoffs) != 2:
        bad.append(
            f"expected exactly 2 out-of-station handoffs for I.3-7, found {len(handoffs)}"
        )
    for h in handoffs:
        ref = f"handoff {h.get('partIRef')}/{h.get('owningStation')}"
        anchor = h.get("baselineAnchor") or {}
        if not sc.anchor_exists(f"`{anchor.get('path')}::{anchor.get('symbol')}`"):
            bad.append(f"{ref}: anchor unresolved {anchor}")
        if h.get("proposedPhase") not in PHASES:
            bad.append(f"{ref}: bad proposedPhase")
        if not str(h.get("alignmentDestination", "")).strip():
            bad.append(f"{ref}: no alignmentDestination")
    return bad


class PartIHandoffTests(sc.StationCase):
    SLUG = "core-io"
    SECTION_TITLE = "core-io"

    @classmethod
    def setUpClass(cls):
        cls.env = sc.load_json(sc.STATIONS / cls.SLUG / "partI-handoff.json")
        cls.by_ref = {d["partIRef"]: d for d in cls.env["dispositions"]}

    def test_envelope_validates(self):
        self.assertEqual(validate_partI_envelope(self.env), [])

    def test_contains_exactly_the_owned_items(self):
        self.assertEqual(set(self.by_ref), OWNED_PART_I)
        self.assertEqual(len(self.env["dispositions"]), 6)
        self.assertEqual(
            self.env["baseline"],
            backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG)),
        )

    def test_each_disposition_is_keep_retire_or_sharpen_with_the_required_evidence(
        self,
    ):
        for ref, d in self.by_ref.items():
            with self.subTest(ref=ref):
                self.assertIn(d["disposition"], ("keep", "retire", "sharpen"))
                anchor = d["baselineAnchor"]
                if d["disposition"] == "retire":
                    sha = d.get("resolvingCommit", "")
                    self.assertEqual(
                        subprocess.run(
                            ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=sc.REPO
                        ).returncode,
                        0,
                        ref,
                    )
                else:
                    self.assertTrue(
                        sc.anchor_exists(f"`{anchor['path']}::{anchor['symbol']}`"),
                        anchor,
                    )
                    self.assertEqual(
                        anchor["line"],
                        self._decl_line(anchor["path"], anchor["symbol"]),
                    )
                self.assertIn("Epic 9", d["alignmentDestination"])

    @staticmethod
    def _decl_line(path: str, symbol: str) -> int:
        pattern = re.compile(
            r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(fn|struct|enum|trait|type|const|static|mod|impl)\s+"
            + re.escape(symbol)
            + r"\b"
        )
        for i, line in enumerate(
            (sc.REPO / path).read_text(encoding="utf-8").splitlines(), 1
        ):
            if pattern.match(line):
                return i
        return -1

    def test_item_3_rename_is_sharpened_not_rejected(self):
        d = self.by_ref["I.3-3"]
        self.assertEqual(d["disposition"], "sharpen")
        self.assertEqual(
            d["baselineAnchor"]["path"], "crates/cobre-core/src/model/horizon.rs"
        )
        self.assertEqual(d["baselineAnchor"]["symbol"], "HorizonGraph")
        self.assertIn("PolicyGraph", d["changedSinceV012"])
        self.assertEqual(d["forwardBackwardHits"], 0)
        self.assertIn("horizon.rs:23", d["survivingClaim"])
        self.assertIn("horizon.rs:26-27", d["survivingClaim"])
        self.assertIn("system/mod.rs:92", d["survivingClaim"])

    def test_item_6_oracle_is_recorded_with_its_reach_limits(self):
        d = self.by_ref["I.3-6"]
        self.assertEqual(d["disposition"], "sharpen")
        self.assertEqual(d["oracle"]["exit"], 0)
        self.assertIn(
            "check-infra-genericity.sh:74 EXCLUDED_FILES=()",
            d["oracle"]["excludedFiles"],
        )
        self.assertEqual(len(d["oracle"]["reachLimits"]), 2)
        for needle in ("records.rs:98", "records.rs:176", "policy.fbs:140"):
            self.assertIn(needle, d["survivingClaim"])

    def test_item_1_is_sharpened_with_the_widened_surface(self):
        d = self.by_ref["I.3-1"]
        self.assertEqual(d["disposition"], "sharpen")
        for f in (
            "inflow_history",
            "external_scenarios",
            "external_load_scenarios",
            "external_ncs_scenarios",
        ):
            self.assertIn(f, d["wideningFields"])
            self.assertIn(f, d["changedSinceV012"])
        self.assertIn("system/mod.rs:111-117", d["changedSinceV012"])
        self.assertIn("125-131", d["changedSinceV012"])

    def test_item_7_is_split_by_crate_ownership(self):
        d = self.by_ref["I.3-7"]
        self.assertEqual(d["owningStation"], "core-io")
        self.assertEqual(
            (d["baselineAnchor"]["path"], d["baselineAnchor"]["symbol"]),
            ("crates/cobre-io/src/config/mod.rs", "Config"),
        )
        self.assertTrue(d["field"].startswith("training"))
        handoffs = {h["owningStation"]: h for h in self.env["handoffs"]}
        self.assertEqual(set(handoffs), {"sddp", "cli"})
        self.assertEqual(
            (
                handoffs["sddp"]["baselineAnchor"]["path"],
                handoffs["sddp"]["baselineAnchor"]["symbol"],
            ),
            ("crates/cobre-sddp/src/setup/params.rs", "from_config"),
        )
        self.assertEqual(
            (
                handoffs["cli"]["baselineAnchor"]["path"],
                handoffs["cli"]["baselineAnchor"]["symbol"],
            ),
            ("crates/cobre-cli/src/commands/broadcast.rs", "BroadcastConfig"),
        )
        self.assertEqual(handoffs["cli"]["baselineAnchor"]["visibility"], "pub(crate)")
        for h in handoffs.values():
            self.assertEqual(h["partIRef"], "I.3-7")


LENS_ORDER = ("architecture", "perf", "over-engineering", "test-bloat")
DISPOSITIONS = {
    "defended",
    "anchor-missing",
    "sanctioned",
    "dup-of",
    "re-raise",
    "out-of-station",
}
VERDICTS = {"confirmed", "dismissed", None}


def candidate_refs() -> dict[str, dict]:
    """Re-derive the `<sub>-<lens>-<nn>` ref of every candidate in candidates-A..D.json.

    Numbering is per (subStation, lens) in file/array order — the same rule the ingest
    used to key verdicts.json, so a drift between the source files and the verdict ledger
    fails loudly here rather than silently dropping a candidate.
    """
    import collections

    out: dict[str, dict] = {}
    per: dict[tuple[str, str], int] = collections.defaultdict(int)
    for sub in "ABCD":
        doc = sc.load_json(sc.STATIONS / "core-io" / f"candidates-{sub}.json")
        for cand in doc["candidates"]:
            lens = cand["lens"]
            nn = per[(sub, lens)]
            per[(sub, lens)] += 1
            out[f"{sub}-{lens}-{nn:02d}"] = cand
    return out


def anchor_resolves(anchor: dict) -> bool:
    """An anchor resolves if its symbol resolves as a declaration/field OR its line is in range."""
    path = anchor["path"]
    if anchor.get("symbol") and sc.anchor_exists(f"`{path}::{anchor['symbol']}`"):
        return True
    if anchor.get("line") is not None and sc.anchor_exists(
        f"`{path}:{anchor['line']}`"
    ):
        return True
    return False


class IngestTests(sc.StationCase):
    SLUG = "core-io"

    def setUp(self):
        self.verdicts = sc.load_json(self.artifact("verdicts.json"))
        self.candidates = candidate_refs()

    def test_one_verdict_per_candidate(self):
        vkeys = set(self.verdicts["verdicts"])
        ckeys = set(self.candidates)
        self.assertEqual(
            vkeys,
            ckeys,
            f"verdicts != candidates: missing {ckeys - vkeys}, extra {vkeys - ckeys}",
        )
        self.assertEqual(len(self.verdicts["verdicts"]), len(self.candidates))
        self.assertEqual(self.verdicts["counts"]["received"], len(self.candidates))

    def test_disposition_and_verdict_vocabulary(self):
        for ref, e in self.verdicts["verdicts"].items():
            self.assertIn(
                e["disposition"],
                DISPOSITIONS,
                f"{ref}: bad disposition {e['disposition']!r}",
            )
            self.assertIn(
                e.get("verdict"), VERDICTS, f"{ref}: bad verdict {e.get('verdict')!r}"
            )
            if e["disposition"] == "defended":
                self.assertIn(
                    e["verdict"],
                    ("confirmed", "dismissed"),
                    f"{ref}: defended must carry a verdict",
                )

    def test_confirmed_carries_narrower_surviving_claim(self):
        for ref, e in self.verdicts["verdicts"].items():
            if e.get("verdict") == "confirmed":
                claim = (e.get("survivingClaim") or "").strip()
                self.assertTrue(claim, f"{ref}: confirmed without survivingClaim")
                self.assertNotEqual(
                    claim,
                    self.candidates[ref]["title"].strip(),
                    f"{ref}: survivingClaim is not narrower than the candidate title",
                )

    def test_dismissed_carries_argument(self):
        for ref, e in self.verdicts["verdicts"].items():
            if e.get("verdict") == "dismissed":
                self.assertTrue(
                    (e.get("argument") or "").strip(),
                    f"{ref}: dismissed without argument",
                )

    def test_accepted_candidate_anchors_resolve(self):
        for ref, e in self.verdicts["verdicts"].items():
            if e["disposition"] == "defended" and e.get("verdict") == "confirmed":
                for anchor in self.candidates[ref]["anchors"]:
                    self.assertTrue(
                        anchor_resolves(anchor),
                        f"{ref}: anchor does not resolve through anchor_exists: {anchor}",
                    )

    def test_sanctioned_cites_mirror_entry(self):
        mirror = (
            sc.REPO / "docs" / "design" / "reserved-seams-and-deferred-debt.md"
        ).read_text(encoding="utf-8")
        for ref, e in self.verdicts["verdicts"].items():
            if e["disposition"] == "sanctioned":
                cite = e.get("sanctionedBy") or ""
                self.assertTrue(cite, f"{ref}: sanctioned without sanctionedBy")
                self.assertIn(
                    "reserved-seams-and-deferred-debt.md",
                    cite,
                    f"{ref}: sanctionedBy must cite the mirror",
                )
                self.assertTrue(
                    any(
                        seam in mirror
                        for seam in ("LipschitzConfig", "transit_bucket_topology")
                    ),
                    "mirror is missing its reserved-seam register",
                )

    def test_merged_names_surviving_candidate(self):
        for ref, e in self.verdicts["verdicts"].items():
            if e["disposition"] == "dup-of":
                self.assertIn(
                    e.get("mergedInto"),
                    self.verdicts["verdicts"],
                    f"{ref}: dup-of must name a surviving candidate id",
                )

    def test_ingest_log_has_one_row_per_candidate(self):
        text = self.artifact("ingest-log.md").read_text(encoding="utf-8")
        marker = "## Per-candidate roster"
        self.assertIn(marker, text, "ingest-log.md has no per-candidate roster")
        roster = text[text.index(marker) :]
        for ref in self.candidates:
            self.assertEqual(
                roster.count(f"| {ref} |"),
                1,
                f"{ref}: expected exactly one roster row in ingest-log.md",
            )

    def test_counts_are_consistent(self):
        counts = self.verdicts["counts"]
        vals = list(self.verdicts["verdicts"].values())
        self.assertEqual(
            counts["confirmed"], sum(1 for v in vals if v.get("verdict") == "confirmed")
        )
        self.assertEqual(
            counts["dismissed"], sum(1 for v in vals if v.get("verdict") == "dismissed")
        )
        self.assertEqual(
            counts["defended"], sum(1 for v in vals if v["disposition"] == "defended")
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-partI":
        failures = validate_partI_envelope(
            sc.load_json(sc.STATIONS / "core-io" / "partI-handoff.json")
        )
        for f in failures:
            print("FAIL", f)
        sys.exit(1 if failures else 0)
    unittest.main()
