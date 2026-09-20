"""Performance-sweep tests (measurements/).

PD004ProfileTests binds the PD-004 measurement artifacts to the sweep protocol: the run
directory records the deck, layout and worker budget the claim-type table derived; runs.tsv
carries one warm-up plus three timed rows and median.txt is their median; the profile and its
disposition exist; and the disposition records a verdict from the allowed set citing the
materiality threshold it was judged against. Later layout tickets append their classes here.

Deviations from the ticket's test prose, recorded here rather than silently met: the epic named
a `perf.data` digest and a flamegraph SVG, but perf-run.sh deletes its scratch (so no perf.data
survives) and no flamegraph renderer is installed, so the deliverable is perf.txt (a `perf report`
text) and the machine-readable disposition.json; the case is `mar-26-enumerated` on the 2t layout,
not the 4t/cobre_reduzido_2 the generic block template quotes; the measured pin is the register
pin 077dbe2c, not the scaffold pin a136840d.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from lib import station_checks as sc  # noqa: E402

MEAS = sc.AUDIT / "measurements"
PD004 = MEAS / "PD-004"
PD005 = MEAS / "PD-005"
UNMEASURED_REASONS = {
    "timeout",
    "unexercised-path",
    "mpi-unavailable",
    "case-infeasible",
}
VERDICTS = {"material", "not-material", "unmeasured"}


class PD004ProfileTests(unittest.TestCase):
    """E10-2: the PD-004 profile, its reduction and its terminal disposition."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = dict(
            ln.split("\t", 1)
            for ln in (PD004 / "env.txt").read_text(encoding="utf-8").splitlines()
            if "\t" in ln
        )
        cls.disp = sc.load_json(PD004 / "disposition.json")
        cls.median_txt = dict(
            ln.split("\t", 1)
            for ln in (PD004 / "median.txt").read_text(encoding="utf-8").splitlines()
            if "\t" in ln
        )
        cls.runs = [
            ln.split("\t")
            for ln in (PD004 / "runs.tsv").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        cls.register = sc.BACKLOG.read_text(encoding="utf-8")

    def test_run_dir_records_the_derived_deck_layout_and_worker_budget(self) -> None:
        self.assertEqual(self.env["layout"], "2t")
        self.assertTrue(self.env["deck"].endswith("cobre-mar-26-rv2-reduced"))
        self.assertEqual(self.env["threads"], "2")
        self.assertLessEqual(
            int(self.env["threads"]),
            2,
            "enumerated deck is owner-limited to two workers",
        )
        self.assertEqual(self.env["taskset_mask"], "0,2")
        self.assertEqual(self.env["cargo_profile"], "profiling")
        self.assertEqual(self.env["perf_event_paranoid"], "2")
        self.assertIn("powersave", self.env["governor"])
        self.assertRegex(self.env["baseline_sha"], r"^077dbe2c")
        cmd = (PD004 / "cmd.txt").read_text(encoding="utf-8")
        self.assertIn("--threads 2", " ".join(cmd.split()))
        self.assertIn("--comm-backend local", " ".join(cmd.split()))
        self.assertIn("taskset", cmd)

    def test_runs_tsv_is_one_warmup_plus_three_timed_and_median_matches(self) -> None:
        roles = [r[1] for r in self.runs]
        self.assertEqual(roles.count("warmup"), 1)
        self.assertEqual(roles.count("timed"), 3)
        timed = sorted(float(r[2]) for r in self.runs if r[1] == "timed")
        self.assertEqual(len(timed), 3)
        recorded = float(self.median_txt["median_s"])
        self.assertAlmostEqual(
            recorded,
            timed[1],
            places=3,
            msg="median.txt is not the median of the three timed rows",
        )
        self.assertAlmostEqual(self.disp["medianSeconds"], timed[1], places=3)

    def test_profile_and_disposition_artifacts_exist(self) -> None:
        for name in (
            "cmd.txt",
            "runs.tsv",
            "median.txt",
            "env.txt",
            "perf.txt",
            "disposition.json",
        ):
            self.assertTrue((PD004 / name).is_file(), name)
        perf = (PD004 / "perf.txt").read_text(encoding="utf-8")
        self.assertRegex(perf, r"Overhead|Samples", "perf.txt is not a perf report")
        self.assertIn(
            "LBR inclusive attribution", perf, "the call-graph supplement is missing"
        )

    def test_disposition_verdict_is_allowed_and_cites_its_threshold(self) -> None:
        d = self.disp
        self.assertIn(d["verdict"], VERDICTS)
        if d["verdict"] == "unmeasured":
            self.assertIn(d["unmeasuredReason"], UNMEASURED_REASONS)
        else:
            self.assertIsNone(d["unmeasuredReason"])
        self.assertRegex(d["materialityRule"], r"3%.*phase wall|1%.*samples")
        self.assertTrue(d["materialityFinding"])
        self.assertFalse(
            d["kernelFramesAttributed"],
            "no cost may be attributed to kernel frames at paranoid=2",
        )
        if d["verdict"] == "material":
            self.assertTrue(d["fixShapePromoted"])
            self.assertTrue(
                d["byteNeutral"], "a promoted fix-shape is byte-neutral by construction"
            )
        else:
            self.assertFalse(d["fixShapePromoted"])
        self.assertEqual(d["alignment"], "neutral")

    def test_register_entry_carries_the_measurement_and_do_not_touch_is_amended(
        self,
    ) -> None:
        self.assertRegex(
            self.register, r"\*\*→ MEASURED \(2026-09-20, baseline `077dbe2c`"
        )
        self.assertIn("Verdict: not-material", self.register)
        # the do-not-touch block no longer defers PD-004, and its neighbours are untouched
        lines = self.register.splitlines()
        start = next(
            i for i, ln in enumerate(lines) if ln.startswith("**Do-not-touch list")
        )
        block = "\n".join(lines[start : start + 3])
        self.assertNotIn("deferred", block)
        self.assertIn("CD-008 (retracted)", block)
        self.assertIn("PD-001 (refuted)", block)
        self.assertIn("sanctioned reserved-seam census", block)
        self.assertRegex(
            block, r"PD-004\s*\n?\s*\(\*\*not-material\*\* at baseline `077dbe2c`"
        )

    def test_pd005_residual_recorded_by_absence_without_reopening(self) -> None:
        self.assertTrue((PD005 / "perf.txt").is_file())
        self.assertTrue((PD005 / "cmd.txt").is_file())
        p5 = (PD005 / "perf.txt").read_text(encoding="utf-8")
        self.assertRegex(p5, r"nested_ub_recursion\s+0 samples")
        self.assertIn("RE-CONFIRMS BY ABSENCE", p5)
        self.assertIn("Re-confirmed by absence", self.register)
        self.assertNotIn("→ REOPENED", self.register)

    def test_read_only_worktree_beyond_the_plan_tree(self) -> None:
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "crates",
                "docs",
                "scripts",
                ".github",
                "schemas",
                "Cargo.toml",
                "Cargo.lock",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        self.assertEqual(
            dirty, "", f"tracked source touched by a read-only sweep: {dirty}"
        )


SWEEP4T = MEAS / "SWEEP-4T"
CLAIM_TABLE = MEAS / "claim-table.json"
WORK4T = MEAS / "_4t"
BOUND_4T = float((MEAS / "CAL" / "median.txt").read_text().strip())
CASE_INFEASIBLE_IDS = {"PD-032", "PD-033", "PD-034", "PD-049", "PD-050", "PD-051"}


def _timed_walls(runs_tsv: pathlib.Path) -> tuple[list[float], int]:
    rows = [
        ln.split("\t")
        for ln in runs_tsv.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    warmups = sum(1 for r in rows if r[1] == "warmup")
    timed = sorted(float(r[2]) for r in rows if r[1] == "timed")
    return timed, warmups


class Layout4tTests(unittest.TestCase):
    """E10-3: the single-process 4t sweep (shared SWEEP-4T recording) + the 2t PD-047 recording.

    Deviations from the step-8 prose, recorded rather than silently met: (a) the owner approved
    ONE shared 4t recording (measurements/SWEEP-4T) for all 27 4t claims instead of 27 per-ID
    sweeps of the identical deck/layout, so per-claim run directories carry a verdict.json + a
    perf.txt excerpt citing the shared recording, not their own runs.tsv/median; (b) perf-run.sh
    deletes its scratch on exit, so no perf.data digest survives — the deliverable is the perf.txt
    report; (c) the measured pin is the register pin 077dbe2c, not the scaffold pin a136840d;
    (d) six claims (CLP-backend PD-032/033/034, cobre-python PD-049/050/051) are UNMEASURED /
    case-infeasible because their symbols are absent from the pinned HiGHS CLI binary.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.table = sc.load_json(CLAIM_TABLE)
        cls.rows_4t = [
            r for r in cls.table["rows"] if r["layout"] == "4t" and not r.get("parked")
        ]
        cls.pd047 = next(r for r in cls.table["rows"] if r["id"] == "PD-047")

    def _env(self, d: pathlib.Path) -> dict[str, str]:
        return dict(
            ln.split("\t", 1)
            for ln in (d / "env.txt").read_text(encoding="utf-8").splitlines()
            if "\t" in ln
        )

    def test_shared_4t_recording_is_one_warmup_plus_three_timed_under_bound(
        self,
    ) -> None:
        timed, warmups = _timed_walls(SWEEP4T / "runs.tsv")
        self.assertEqual(warmups, 1)
        self.assertEqual(len(timed), 3)
        recorded = float((SWEEP4T / "median.txt").read_text().split()[0])
        self.assertAlmostEqual(recorded, timed[1], places=3)
        for w in timed:
            self.assertLessEqual(
                w, 3 * BOUND_4T, "a timed wall exceeded 3x the 4t protocol bound"
            )
        env = self._env(SWEEP4T)
        self.assertTrue(env["deck"].endswith("cobre_reduzido"))
        self.assertEqual(env["threads"], "4")
        self.assertEqual(env["taskset_mask"], "0,2,4,6")
        for cpu in env["taskset_mask"].split(","):
            self.assertNotIn(int(cpu), range(16, 20), "an E-core entered a timed run")
        self.assertEqual(env["cargo_profile"], "profiling")
        self.assertRegex(env["baseline_sha"], r"^077dbe2c")
        self.assertRegex(
            (SWEEP4T / "perf.txt").read_text(), r"Overhead|Samples|%\s+cobre"
        )

    def test_pd047_2t_recording_shape(self) -> None:
        d = MEAS / "PD-047"
        timed, warmups = _timed_walls(d / "runs.tsv")
        self.assertEqual(warmups, 1)
        self.assertEqual(len(timed), 3)
        env = self._env(d)
        self.assertTrue(env["deck"].endswith("cobre-mar-26-rv2-reduced"))
        self.assertEqual(env["threads"], "2")
        self.assertEqual(env["taskset_mask"], "0,2")
        self.assertTrue((d / "perf.txt").is_file())

    def test_every_measured_claim_has_a_verdict_in_the_allowed_set(self) -> None:
        for row in [*self.rows_4t, self.pd047]:
            v = sc.load_json(MEAS / row["id"] / "verdict.json")
            self.assertIn(v["verdict"], VERDICTS, row["id"])
            if v["verdict"] == "unmeasured":
                self.assertIn(v["unmeasuredReason"], UNMEASURED_REASONS, row["id"])
                self.assertIsNone(v.get("medianSeconds"), row["id"])
            else:
                self.assertIsNotNone(v.get("selfPct"), row["id"])
            self.assertTrue(v["byteNeutral"], row["id"])
            self.assertEqual(v["alignment"], "neutral", row["id"])
            self.assertTrue((MEAS / row["id"] / "perf.txt").is_file(), row["id"])

    def test_case_infeasible_claims_carry_evidence_and_needs_human(self) -> None:
        for cid in CASE_INFEASIBLE_IDS:
            v = sc.load_json(MEAS / cid / "verdict.json")
            self.assertEqual(v["verdict"], "unmeasured", cid)
            self.assertEqual(v["unmeasuredReason"], "case-infeasible", cid)
            self.assertTrue(v["evidence"].strip(), cid)
            self.assertTrue(v["_needsHuman"], cid)
            self.assertIsNone(v["medianSeconds"], cid)

    def test_material_verdict_would_cite_its_threshold_and_drop_kernel_frames(
        self,
    ) -> None:
        # no 4t claim measured material on this LP-solve-bound deck; assert the discipline holds
        for row in self.rows_4t:
            v = sc.load_json(MEAS / row["id"] / "verdict.json")
            if v["verdict"] == "unmeasured":
                continue
            self.assertRegex(v["materialityRule"], r"1%.*samples|3%.*phase wall")
            self.assertIsInstance(v["kernelFramesDropped"], int)
            if v["verdict"] == "material":
                self.assertTrue(v["byteNeutral"])

    def test_claim_table_4t_rows_and_pd047_flipped_others_untouched(self) -> None:
        for row in self.table["rows"]:
            if row["id"] in {
                "PD-008",
                "PD-017",
            }:  # collective 2x2, left for that ticket
                self.assertFalse(row.get("measured"), row["id"])
            elif row.get("parked"):  # PD-004 / PD-005-residual
                continue
            elif row["layout"] == "4t" or row["id"] == "PD-047":
                self.assertTrue(row.get("measured"), row["id"])
                self.assertIn(
                    row.get("verdict"), {"material", "not-material", "unmeasured"}
                )

    def test_handoff_worklist_symbolcheck_and_unmeasured_exist(self) -> None:
        handoff = sc.load_json(WORK4T / "handoff.json")
        measured_ids = {r["id"] for r in self.rows_4t} | {"PD-047"}
        self.assertEqual({c["id"] for c in handoff["claims"]}, measured_ids)
        self.assertTrue((WORK4T / "worklist.json").is_file())
        self.assertTrue((WORK4T / "symbol-check.tsv").is_file())
        unmeasured = (WORK4T / "unmeasured.md").read_text()
        for cid in CASE_INFEASIBLE_IDS:
            self.assertIn(cid, unmeasured)

    def test_read_only_worktree_beyond_the_plan_tree(self) -> None:
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "crates",
                "docs",
                "scripts",
                ".github",
                "schemas",
                "Cargo.toml",
                "Cargo.lock",
            ],
            cwd=sc.REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        self.assertEqual(
            dirty, "", f"tracked source touched by a read-only sweep: {dirty}"
        )


if __name__ == "__main__":
    unittest.main()
