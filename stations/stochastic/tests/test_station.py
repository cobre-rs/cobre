"""cobre-stochastic station tests: inventory census, four-way partition, prior register.

Run from the repository root:
    python3 -m unittest plans/architecture-debt-audit/stations/stochastic/tests/test_station.py
Later cobre-stochastic tickets append their own stage classes here; the station
verification runs the module.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
import types
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

LENSES = {"architecture", "over-engineering", "performance", "test-bloat"}
LENS_ORDER = ("architecture", "over-engineering", "performance", "test-bloat")
SUB_ORDER = ("par", "sampling", "tree-noise", "seam")
SEV = {"A", "B", "C"}
ALIGN = {"advances-0a", "advances-0b", "advances-1", "neutral", "conflicts"}
SCOPE = "crates/cobre-stochastic/"
INGEST_DISPOSITIONS = {
    "defended",
    "anchor-missing",
    "sanctioned",
    "dup-of",
    "re-raise",
    "out-of-station",
}
VERDICTS = {"confirmed", "dismissed", None}


def load_validate_envelope() -> types.ModuleType:
    path = (
        pathlib.Path(__file__).resolve().parents[3] / "tools" / "validate-envelope.py"
    )
    spec = importlib.util.spec_from_file_location("validate_envelope", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def anchor_resolves(anchor: dict) -> bool:
    """Resolve an anchor by symbol OR line (core-io semantics) against HEAD == baseline."""
    path = anchor["path"]
    if anchor.get("symbol") and sc.anchor_exists(f"`{path}::{anchor['symbol']}`"):
        return True
    if anchor.get("line") is not None and sc.anchor_exists(
        f"`{path}:{anchor['line']}`"
    ):
        return True
    return False


def candidate_refs() -> dict[str, dict]:
    """Re-derive the ingest `<sub>-<lens>-<nn>` key of every candidate in candidates-*.json.

    Numbering is per (subStation, lens) in file/array order — the same rule the ingest used to
    key verdicts.json, so a drift between the source files and the verdict ledger fails loudly
    here rather than silently dropping a candidate.
    """
    import collections

    out: dict[str, dict] = {}
    per: dict[tuple[str, str], int] = collections.defaultdict(int)
    for sub in SUB_ORDER:
        doc = sc.load_json(sc.STATIONS / "stochastic" / f"candidates-{sub}.json")
        for cand in doc["candidates"]:
            lens = cand["lens"]
            nn = per[(sub, lens)]
            per[(sub, lens)] += 1
            out[f"{sub}-{lens}-{nn:02d}"] = cand
    return out


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


class CandidateEnvelopeTests(sc.StationCase):
    """The attacker fan-out: 16 raw cells merged into four candidates-<sub>.json files."""

    SLUG = "stochastic"

    @classmethod
    def setUpClass(cls):
        cls.dir = cls.STATION_DIR or sc.STATIONS / cls.SLUG
        cls.ve = load_validate_envelope()
        cls.docs = {
            sub: sc.load_json(cls.dir / f"candidates-{sub}.json") for sub in SUB_IDS
        }
        cls.raw = {
            (lens, sub): sc.load_json(cls.dir / "raw" / f"sto-{lens}-{sub}.json")
            for lens in LENSES
            for sub in SUB_IDS
        }

    def test_four_files_carry_the_merged_envelope_and_the_pinned_baseline(self):
        pin = backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG))
        for sub, doc in self.docs.items():
            for key in (
                "station",
                "subStation",
                "baseline",
                "candidates",
                "positives",
                "_needsHuman",
                "lenses",
            ):
                self.assertIn(key, doc, f"candidates-{sub}.json missing {key}")
            self.assertEqual(doc["station"], "stochastic")
            self.assertEqual(
                doc["subStation"], sub, f"candidates-{sub}.json subStation"
            )
            self.assertTrue(doc["baseline"].startswith(pin[:8]))
            self.assertTrue(
                backlog_parse.baseline_matches_head(doc["baseline"]),
                f"candidates-{sub}.json baseline drifted from HEAD's evaluated surfaces",
            )

    def test_candidate_refs_are_unique_and_lens_sub_consistent(self):
        seen: set[str] = set()
        pat = re.compile(
            r"sto-(?P<lens>[a-z-]+)-(?P<sub>par|sampling|tree-noise|seam)-\d{2}$"
        )
        for sub, doc in self.docs.items():
            for c in doc["candidates"]:
                ref = c["candidateRef"]
                self.assertNotIn(ref, seen, f"duplicate candidateRef {ref}")
                seen.add(ref)
                m = pat.match(ref)
                self.assertIsNotNone(m, f"malformed candidateRef {ref}")
                self.assertEqual(m.group("sub"), sub, f"{ref}: sub != file {sub}")
                self.assertEqual(m.group("lens"), c["lens"], f"{ref}: lens mismatch")
                self.assertIn(
                    c["lens"], LENSES, f"{ref}: lens {c['lens']!r} not in set"
                )

    def test_each_lens_group_passes_the_single_lens_validator(self):
        for sub, doc in self.docs.items():
            for lens in LENSES:
                group = [c for c in doc["candidates"] if c["lens"] == lens]
                env = {
                    "station": "stochastic",
                    "subStation": sub,
                    "baseline": doc["baseline"],
                    "lens": lens,
                    "candidates": group,
                    "positives": [],
                    "_needsHuman": [],
                }
                errs: list[str] = []
                self.ve.validate_attacker(env, errs, "stochastic")
                self.assertEqual(
                    errs, [], f"{lens}/{sub} candidates fail shape: {errs}"
                )

    def test_every_anchor_is_in_scope_and_resolves_at_baseline(self):
        for doc in self.docs.values():
            for c in doc["candidates"]:
                self.assertTrue(c["anchors"], f"{c['candidateRef']}: no anchors")
                for a in c["anchors"]:
                    self.assertTrue(
                        a["path"].startswith(SCOPE),
                        f"{c['candidateRef']}: out-of-scope anchor {a['path']}",
                    )
                    self.assertTrue(
                        anchor_resolves(a),
                        f"{c['candidateRef']}: anchor does not resolve: {a}",
                    )

    def test_lenses_map_reconciles_with_candidate_counts(self):
        for sub, doc in self.docs.items():
            lens_map = doc["lenses"]
            self.assertEqual(
                set(lens_map),
                {f"sto-{lens}-{sub}" for lens in LENSES},
                f"candidates-{sub}.json lenses map keys",
            )
            for lens in LENSES:
                got = sum(1 for c in doc["candidates"] if c["lens"] == lens)
                self.assertEqual(
                    lens_map[f"sto-{lens}-{sub}"],
                    got,
                    f"candidates-{sub}.json lenses[{lens}] != candidate count",
                )
            self.assertEqual(sum(lens_map.values()), len(doc["candidates"]))

    def test_raw_cells_present_and_content_preserved_in_merge(self):
        pin = backlog_parse.parse_baseline(backlog_parse.read_register(sc.BACKLOG))
        for (lens, sub), raw in self.raw.items():
            self.assertTrue(
                raw["baseline"].startswith(pin[:8]), f"sto-{lens}-{sub} baseline"
            )
            merged = {
                c["candidateRef"]: c
                for c in self.docs[sub]["candidates"]
                if c["lens"] == lens
            }
            self.assertEqual(
                len(raw.get("candidates") or []),
                self.docs[sub]["lenses"][f"sto-{lens}-{sub}"],
                f"sto-{lens}-{sub}: raw count != lenses map",
            )
            for i, rc in enumerate(raw.get("candidates") or []):
                ref = f"sto-{lens}-{sub}-{i:02d}"
                self.assertIn(ref, merged, f"{ref} absent from candidates-{sub}.json")
                mc = merged[ref]
                self.assertEqual(
                    rc["title"], mc["title"], f"{ref}: title changed in merge"
                )
                self.assertEqual(
                    rc["anchors"], mc["anchors"], f"{ref}: anchors changed in merge"
                )

    def test_positives_and_needs_human_are_lens_stamped(self):
        for sub, doc in self.docs.items():
            for p in doc["positives"]:
                self.assertIn(
                    p.get("lens"), LENSES, f"{sub}: positive without lens stamp"
                )
                self.assertTrue(
                    str(p.get("subject", "")).strip(), f"{sub}: positive subject"
                )
            for h in doc["_needsHuman"]:
                self.assertIn(
                    h.get("lens"), LENSES, f"{sub}: _needsHuman without lens stamp"
                )

    def test_no_candidate_claims_an_unadjudicated_reraise(self):
        for doc in self.docs.values():
            for c in doc["candidates"]:
                if "reRaiseOf" in c:
                    self.assertIsNone(
                        c["reRaiseOf"],
                        f"{c['candidateRef']}: a non-null reRaiseOf must be screened at ingest",
                    )

    def test_attacker_log_records_the_matrix_and_the_prior_register_screen(self):
        text = (self.dir / "attacker-log.md").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?i)prior-register screen")
        for sub, doc in self.docs.items():
            for lens in LENSES:
                worker = f"sto-{lens}-{sub}"
                row = re.search(
                    rf"\|\s*{re.escape(worker)}\s*\|[^|]*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|",
                    text,
                )
                self.assertIsNotNone(
                    row, f"attacker-log.md has no Workers row for {worker}"
                )
                self.assertEqual(
                    int(row.group(1)),
                    doc["lenses"][worker],
                    f"attacker-log.md {worker} count != candidates-{sub}.json",
                )
        for ref in (
            "sto-architecture-par-00",
            "sto-performance-seam-00",
            "sto-over-engineering-seam-00",
            "sto-test-bloat-seam-01",
        ):
            self.assertIn(ref, text, f"attacker-log.md screen omits adjudicated {ref}")


class IngestTests(sc.StationCase):
    """The defender pass merged into verdicts.json + ingest-log.md + anchor-probe.md."""

    SLUG = "stochastic"

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
                INGEST_DISPOSITIONS,
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

    def test_every_verdict_carries_reasoning_and_alignment(self):
        for ref, e in self.verdicts["verdicts"].items():
            self.assertGreaterEqual(
                len((e.get("argument") or "").strip()),
                120,
                f"{ref}: argument is too short to be defender reasoning",
            )
            self.assertIn(
                e.get("alignmentHint"),
                ALIGN,
                f"{ref}: bad alignmentHint {e.get('alignmentHint')!r}",
            )

    def test_accepted_candidate_anchors_resolve(self):
        for ref, e in self.verdicts["verdicts"].items():
            if e["disposition"] == "defended" and e.get("verdict") == "confirmed":
                for anchor in self.candidates[ref]["anchors"]:
                    self.assertTrue(
                        anchor_resolves(anchor),
                        f"{ref}: anchor does not resolve through anchor_exists: {anchor}",
                    )

    def test_part_i_refs_are_preserved_from_the_candidates(self):
        for ref, e in self.verdicts["verdicts"].items():
            cand_ref = self.candidates[ref].get("partIRef")
            if cand_ref:
                self.assertEqual(
                    e.get("partIRef"),
                    cand_ref,
                    f"{ref}: partIRef dropped or changed at ingest",
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
                self.assertIn("LipschitzConfig", mirror)

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

    def test_anchor_probe_has_one_block_per_candidate(self):
        text = self.artifact("anchor-probe.md").read_text(encoding="utf-8")
        for ref in self.candidates:
            self.assertEqual(
                text.count(f"### {ref} "), 1, f"{ref}: expected one anchor-probe block"
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
        self.assertEqual(counts["received"], len(vals))


if __name__ == "__main__":
    unittest.main()
