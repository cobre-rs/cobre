#!/usr/bin/env python3
"""analyze_solver_iterations.py.

Reusable analyzer for `training/solver/iterations.parquet`, the per-opening
per-stage solver instrumentation parquet produced by the SDDP training loop
after epic-04a.

Reports generated:
  1. Global phase totals (forward / backward / lower_bound).
  2. Backward/forward wall-time ratio (overall and excluding iter 1 warm-up).
  3. Per-opening backward breakdown (piv/solve, ms/solve, share of totals).
  4. Per-iteration ω=0 vs ω>0 trend (chain warm-start decay curve).
  5. Stage-bucket homogeneity check.
  6. Basis-reconstruction counter health (preserved / new_tight / new_slack /
     demotions / consistency_failures).
  7. Retry histogram summary.
  8. Back-of-envelope savings ceilings for Step 2 (cut-tightness classification)
     and Step 3 (per-opening basis store).

Usage:
    python3 scripts/analyze_solver_iterations.py <case-output-dir>
    python3 scripts/analyze_solver_iterations.py ~/git/cobre-bridge/example/convertido_arch/output
    python3 scripts/analyze_solver_iterations.py --parquet path/to/iterations.parquet

Exit codes:
  0 -- analysis produced
  2 -- prerequisite missing (parquet not found, polars unavailable)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import polars as pl
except ImportError:
    print(
        "ERROR: polars is not available. Run `uv pip install polars` in the active venv.",
        file=sys.stderr,
    )
    sys.exit(2)


RULE = "=" * 72
SUB_RULE = "-" * 72


def _f(x: object) -> float:
    """Coerce a polars scalar (possibly Decimal/None) to float."""
    if x is None:
        return 0.0
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _i(x: object) -> int:
    """Coerce a polars scalar (possibly None) to int."""
    if x is None:
        return 0
    try:
        return int(float(x))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class Paths:
    iterations: Path
    retry_histogram: Path | None
    cut_selection: Path | None
    timing: Path | None


def _with_companions(iters: Path) -> Paths:
    """Look for companion parquets (cut_selection, timing, retry) near iters."""
    training_dir = iters.parents[1] if len(iters.parents) >= 2 else iters.parent
    retry = iters.with_name("retry_histogram.parquet")
    cut_sel = training_dir / "cut_selection" / "iterations.parquet"
    timing = training_dir / "timing" / "iterations.parquet"
    return Paths(
        iterations=iters,
        retry_histogram=retry if retry.exists() else None,
        cut_selection=cut_sel if cut_sel.exists() else None,
        timing=timing if timing.exists() else None,
    )


def _resolve_paths(args: argparse.Namespace) -> Paths:
    """Resolve the parquet paths from either --parquet or a case output dir."""
    if args.parquet is not None:
        iters = Path(args.parquet).expanduser().resolve()
        return _with_companions(iters)

    root = Path(args.case_dir).expanduser().resolve()
    candidates = [
        root / "training" / "solver" / "iterations.parquet",
        root / "output" / "training" / "solver" / "iterations.parquet",
        root / "iterations.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return _with_companions(candidate)

    print(
        f"ERROR: iterations.parquet not found under {root}. "
        "Tried training/solver/, output/training/solver/, and the root itself.",
        file=sys.stderr,
    )
    sys.exit(2)


def _header(title: str) -> None:
    print(RULE)
    print(title)
    print(RULE)


def _report_totals(t: pl.DataFrame) -> None:
    _header("1. Global phase totals")
    per_phase = (
        t.group_by("phase")
        .agg(
            pl.col("lp_solves").sum().alias("solves"),
            pl.col("simplex_iterations").sum().alias("pivots"),
            pl.col("solve_time_ms").sum().alias("total_ms"),
        )
        .sort("phase")
    )
    header = f"{'phase':>12}  {'solves':>10}  {'pivots':>14}  {'total_ms':>14}  {'piv/solve':>10}  {'ms/solve':>10}"
    print(header)
    print(SUB_RULE)
    for row in per_phase.iter_rows(named=True):
        solves = _i(row["solves"]) or 1
        pivots = _i(row["pivots"])
        total_ms = _f(row["total_ms"])
        print(
            f"{row['phase']:>12}  {solves:>10}  {pivots:>14}  "
            f"{total_ms:>14.1f}  {pivots / solves:>10.1f}  "
            f"{total_ms / solves:>10.3f}"
        )
    print()


def _report_ratio(t: pl.DataFrame) -> None:
    _header("2. Backward / forward wall-time ratio")
    fw_ms = _f(t.filter(pl.col("phase") == "forward")["solve_time_ms"].sum())
    bw_ms = _f(t.filter(pl.col("phase") == "backward")["solve_time_ms"].sum())
    ratio = bw_ms / fw_ms if fw_ms > 0 else float("nan")
    fw_ms_ss = _f(
        t.filter((pl.col("phase") == "forward") & (pl.col("iteration") > 1))[
            "solve_time_ms"
        ].sum()
    )
    bw_ms_ss = _f(
        t.filter((pl.col("phase") == "backward") & (pl.col("iteration") > 1))[
            "solve_time_ms"
        ].sum()
    )
    ratio_ss = bw_ms_ss / fw_ms_ss if fw_ms_ss > 0 else float("nan")
    print(f"Overall:                {ratio:>6.2f} x")
    print(f"Excluding iter 1 (SS):  {ratio_ss:>6.2f} x")
    print()


def _report_per_opening(t: pl.DataFrame) -> None:
    _header("3. Per-opening backward breakdown (aggregated across iters/stages)")
    bw = t.filter(pl.col("phase") == "backward")
    total_piv = _f(bw["simplex_iterations"].sum()) or 1.0
    total_ms = _f(bw["solve_time_ms"].sum()) or 1.0
    per_op = (
        bw.group_by("opening")
        .agg(
            (pl.col("simplex_iterations").sum() / pl.col("lp_solves").sum()).alias(
                "piv/solve"
            ),
            (pl.col("solve_time_ms").sum() / pl.col("lp_solves").sum()).alias(
                "ms/solve"
            ),
            pl.col("simplex_iterations").sum().alias("piv"),
            pl.col("solve_time_ms").sum().alias("ms"),
            pl.col("basis_set_time_ms").sum().alias("basis_set_ms"),
        )
        .sort("opening")
    )
    print(
        f"{'ω':>3}  {'piv/solve':>10}  {'ms/solve':>10}  {'%piv':>8}  {'%ms':>8}  {'basis_set_ms':>14}"
    )
    print(SUB_RULE)
    for row in per_op.iter_rows(named=True):
        piv_per_solve = _f(row["piv/solve"])
        ms_per_solve = _f(row["ms/solve"])
        piv = _f(row["piv"])
        ms = _f(row["ms"])
        basis_set_ms = _f(row["basis_set_ms"])
        print(
            f"{row['opening']:>3}  {piv_per_solve:>10.1f}  {ms_per_solve:>10.3f}  "
            f"{100 * piv / total_piv:>7.1f}%  {100 * ms / total_ms:>7.1f}%  "
            f"{basis_set_ms:>14.3f}"
        )
    p0 = _f(per_op.filter(pl.col("opening") == 0)["piv/solve"][0])
    pg_mean = _f(per_op.filter(pl.col("opening") > 0)["piv/solve"].mean())
    if pg_mean > 0:
        print(f"\nω=0 piv/solve ratio vs ω>0 mean: {p0 / pg_mean:.2f} x")
    print()


def _report_per_iteration(t: pl.DataFrame) -> None:
    _header("4. Per-iteration ω=0 vs ω>0 trend (chain warm-start dynamics)")
    bw = t.filter(pl.col("phase") == "backward")
    fw = t.filter(pl.col("phase") == "forward")
    is_zero = pl.col("opening") == 0
    per_iter = (
        bw.group_by("iteration")
        .agg(
            (
                pl.col("simplex_iterations").filter(is_zero).sum()
                / pl.col("lp_solves").filter(is_zero).sum()
            ).alias("ω=0 piv/solve"),
            (
                pl.col("simplex_iterations").filter(~is_zero).sum()
                / pl.col("lp_solves").filter(~is_zero).sum()
            ).alias("ω>0 piv/solve"),
            pl.col("solve_time_ms").filter(is_zero).sum().alias("ω=0 ms"),
            pl.col("solve_time_ms").filter(~is_zero).sum().alias("ω>0 ms"),
            pl.col("solve_time_ms").sum().alias("bw_total_ms"),
        )
        .sort("iteration")
    )
    fw_per_iter = (
        fw.group_by("iteration")
        .agg(pl.col("solve_time_ms").sum().alias("fw_total_ms"))
        .sort("iteration")
    )
    merged = per_iter.join(fw_per_iter, on="iteration", how="left")
    print(
        f"{'iter':>5}  {'ω=0 piv':>8}  {'ω>0 piv':>8}  {'gap':>5}  "
        f"{'ω=0 ms':>10}  {'ω>0 ms':>10}  {'bw ms':>10}  {'fw ms':>10}  {'bw/fw':>7}"
    )
    print(SUB_RULE)
    for row in merged.iter_rows(named=True):
        fw_ms = _f(row["fw_total_ms"]) or 1.0
        piv0 = _f(row["ω=0 piv/solve"])
        pivg = _f(row["ω>0 piv/solve"])
        ms0 = _f(row["ω=0 ms"])
        msg = _f(row["ω>0 ms"])
        bw_total = _f(row["bw_total_ms"])
        print(
            f"{row['iteration']:>5}  {piv0:>8.1f}  {pivg:>8.1f}  "
            f"{piv0 - pivg:>5.0f}  {ms0:>10.0f}  {msg:>10.0f}  "
            f"{bw_total:>10.0f}  {fw_ms:>10.0f}  {bw_total / fw_ms:>7.2f}x"
        )
    print()


def _report_stage_buckets(t: pl.DataFrame) -> None:
    _header("5. Stage-bucket homogeneity (early / middle / late thirds)")
    bw = t.filter(pl.col("phase") == "backward")
    max_stage = _i(bw["stage"].max())
    if max_stage <= 0:
        print("(no backward rows)")
        print()
        return
    cut_a = max_stage // 3
    cut_b = 2 * max_stage // 3

    bucket_expr = (
        pl.when(pl.col("stage") < cut_a)
        .then(pl.lit("early"))
        .when(pl.col("stage") < cut_b)
        .then(pl.lit("middle"))
        .otherwise(pl.lit("late"))
        .alias("bucket")
    )
    is_zero = pl.col("opening") == 0
    bw_bucketed = bw.with_columns(bucket_expr)
    split = (
        bw_bucketed.group_by("bucket")
        .agg(
            (
                pl.col("simplex_iterations").filter(is_zero).sum()
                / pl.col("lp_solves").filter(is_zero).sum()
            ).alias("ω=0 piv/solve"),
            (
                pl.col("simplex_iterations").filter(~is_zero).sum()
                / pl.col("lp_solves").filter(~is_zero).sum()
            ).alias("ω>0 piv/solve"),
            pl.col("solve_time_ms").sum().alias("total_ms"),
        )
        .sort("bucket")
    )
    print(f"{'bucket':>10}  {'ω=0 piv':>8}  {'ω>0 piv':>8}  {'total_ms':>12}")
    print(SUB_RULE)
    for row in split.iter_rows(named=True):
        print(
            f"{row['bucket']:>10}  {_f(row['ω=0 piv/solve']):>8.1f}  "
            f"{_f(row['ω>0 piv/solve']):>8.1f}  {_f(row['total_ms']):>12.1f}"
        )
    print(f"\nBoundary stages: early < {cut_a} <= middle < {cut_b} <= late")
    print()


def _report_basis_health(t: pl.DataFrame) -> None:
    _header("6. Basis-reconstruction counter health (ω=0 only)")
    bw = t.filter(pl.col("phase") == "backward")
    o0 = bw.filter(pl.col("opening") == 0)
    per_iter = (
        o0.group_by("iteration")
        .agg(
            pl.col("basis_offered").sum().alias("offered"),
            pl.col("basis_consistency_failures").sum().alias("cons_fail"),
            pl.col("basis_preserved").sum().alias("preserved"),
            pl.col("basis_new_tight").sum().alias("new_tight"),
            pl.col("basis_new_slack").sum().alias("new_slack"),
            pl.col("basis_demotions").sum().alias("demotions"),
            pl.col("lp_solves").sum().alias("solves"),
        )
        .sort("iteration")
    )
    header = f"{'iter':>5}  {'offered':>8}  {'cons_f':>6}  {'preserved':>10}  {'new_tight':>10}  {'new_slack':>10}  {'demote':>8}"
    print(header)
    print(SUB_RULE)
    for row in per_iter.iter_rows(named=True):
        print(
            f"{row['iteration']:>5}  {row['offered']:>8}  {row['cons_fail']:>6}  "
            f"{row['preserved']:>10}  {row['new_tight']:>10}  {row['new_slack']:>10}  "
            f"{row['demotions']:>8}"
        )

    gt0 = bw.filter(pl.col("opening") > 0)
    nonzero_cols = [
        c
        for c in (
            "basis_offered",
            "basis_consistency_failures",
            "basis_preserved",
            "basis_new_tight",
            "basis_new_slack",
            "basis_demotions",
        )
        if (gt0[c].sum() or 0) > 0
    ]
    if nonzero_cols:
        print(f"\nWARNING: ω>0 rows have non-zero basis counters: {nonzero_cols}")
    else:
        print(
            "\nω>0 basis counters all zero — expected, confirms setBasis is ω=0-only."
        )
    print()


def _report_retries(t: pl.DataFrame, retry_path: Path | None) -> None:
    _header("7. Retry / failure summary")
    per_phase = (
        t.group_by("phase")
        .agg(
            pl.col("lp_solves").sum().alias("solves"),
            pl.col("lp_successes").sum().alias("successes"),
            pl.col("lp_retries").sum().alias("retries"),
            pl.col("lp_failures").sum().alias("failures"),
            pl.col("retry_attempts").sum().alias("retry_attempts"),
        )
        .sort("phase")
    )
    header = f"{'phase':>12}  {'solves':>10}  {'success':>10}  {'retries':>10}  {'failures':>10}  {'attempts':>10}"
    print(header)
    print(SUB_RULE)
    for row in per_phase.iter_rows(named=True):
        print(
            f"{row['phase']:>12}  {row['solves']:>10}  {row['successes']:>10}  "
            f"{row['retries']:>10}  {row['failures']:>10}  {row['retry_attempts']:>10}"
        )
    if retry_path is not None and retry_path.exists():
        rh = pl.read_parquet(retry_path)
        if len(rh) == 0:
            print("\nretry_histogram.parquet is empty — no retries triggered.")
        else:
            print("\nretry_histogram.parquet (first 20 rows):")
            print(rh.head(20))
    else:
        print("\nNo retry_histogram.parquet found alongside iterations.parquet.")
    print()


def _report_sizing(t: pl.DataFrame) -> None:
    _header("8. Step-2 / Step-3 savings ceilings (back-of-envelope)")
    bw = t.filter(pl.col("phase") == "backward")
    is_zero = pl.col("opening") == 0
    o0 = bw.filter(is_zero)
    og = bw.filter(~is_zero)
    o0_ms = _f(o0["solve_time_ms"].sum())
    og_ms = _f(og["solve_time_ms"].sum())
    o0_piv = _f(o0["simplex_iterations"].sum())
    og_piv = _f(og["simplex_iterations"].sum())
    o0_solves = _f(o0["lp_solves"].sum()) or 1.0
    og_solves = _f(og["lp_solves"].sum()) or 1.0
    total_ms = _f(bw["solve_time_ms"].sum()) or 1.0

    o0_piv_per_solve = o0_piv / o0_solves
    og_piv_per_solve = og_piv / og_solves
    o0_ms_per_piv = o0_ms / o0_piv if o0_piv > 0 else 0.0

    step2_ceiling_ms = 0.0
    if o0_piv_per_solve > og_piv_per_solve:
        step2_piv_saved = (o0_piv_per_solve - og_piv_per_solve) * o0_solves
        step2_ceiling_ms = step2_piv_saved * o0_ms_per_piv

    step3_lite_ms = 0.4 * o0_ms
    step3_full_conservative_ms = 0.2 * og_ms
    step3_full_aspirational_ms = 0.4 * og_ms

    rows = [
        (
            "Step 2 (cut classification, ω=0 only)",
            step2_ceiling_ms,
            "ω=0 matches ω>0 piv/solve",
        ),
        (
            "Step 3-lite (per-opening basis, ω=0 only)",
            step3_lite_ms,
            "40% ω=0 reduction",
        ),
        (
            "Step 3-full conservative (per-opening basis, ω>0)",
            step3_full_conservative_ms,
            "20% ω>0 reduction",
        ),
        (
            "Step 3-full aspirational (per-opening basis, ω>0)",
            step3_full_aspirational_ms,
            "40% ω>0 reduction",
        ),
    ]
    print(f"{'Lever':<50}  {'saved ms':>10}  {'% bw':>7}  {'assumption':<28}")
    print(SUB_RULE)
    for label, saved, note in rows:
        print(f"{label:<50}  {saved:>10.1f}  {100 * saved / total_ms:>6.1f}%  {note}")

    print("\nCurrent state:")
    all_piv = o0_piv + og_piv or 1.0
    print(
        f"  ω=0 share of backward: {100 * o0_ms / total_ms:>5.1f}% ms, {100 * o0_piv / all_piv:>5.1f}% pivots"
    )
    print(
        f"  ω>0 share of backward: {100 * og_ms / total_ms:>5.1f}% ms, {100 * og_piv / all_piv:>5.1f}% pivots"
    )
    print(
        f"  ω=0 piv/solve: {o0_piv_per_solve:.1f}  |  ω>0 piv/solve: {og_piv_per_solve:.1f}"
    )
    print()


def _report_timing_breakdown(timing_path: Path | None) -> None:
    _header("9. Training-loop wall-time breakdown (timing/iterations.parquet)")
    if timing_path is None:
        print("(no timing/iterations.parquet — run training to generate it)\n")
        return
    ti = pl.read_parquet(timing_path)
    if len(ti) == 0:
        print("(timing parquet is empty)\n")
        return

    total_fw = _f(ti["forward_wall_ms"].sum())
    total_bw = _f(ti["backward_wall_ms"].sum())
    total_cs = _f(ti["cut_selection_ms"].sum())
    total_allreduce = _f(ti["mpi_allreduce_ms"].sum())
    total_cut_sync = _f(ti["cut_sync_ms"].sum())
    total_lb = _f(ti["lower_bound_ms"].sum())
    total_state_exchange = _f(ti["state_exchange_ms"].sum())
    total_batch_build = _f(ti["cut_batch_build_ms"].sum())
    total_bwd_setup = _f(ti["bwd_setup_ms"].sum())
    total_bwd_imbalance = _f(ti["bwd_load_imbalance_ms"].sum())
    total_bwd_sched = _f(ti["bwd_scheduling_overhead_ms"].sum())
    total_fwd_setup = _f(ti["fwd_setup_ms"].sum())
    total_fwd_imbalance = _f(ti["fwd_load_imbalance_ms"].sum())
    total_fwd_sched = _f(ti["fwd_scheduling_overhead_ms"].sum())
    total_overhead = _f(ti["overhead_ms"].sum())

    wall = total_fw + total_bw + total_cs + total_allreduce + total_cut_sync + total_lb
    wall = wall + total_state_exchange + total_batch_build + total_overhead

    def pct(v: float) -> float:
        return 100 * v / wall if wall > 0 else 0.0

    print(f"{'Component':<30}  {'ms':>12}  {'%':>7}")
    print(SUB_RULE)
    rows = [
        ("forward_wall_ms", total_fw),
        ("backward_wall_ms", total_bw),
        ("  bwd_setup_ms", total_bwd_setup),
        ("  bwd_load_imbalance_ms", total_bwd_imbalance),
        ("  bwd_scheduling_overhead_ms", total_bwd_sched),
        ("  fwd_setup_ms", total_fwd_setup),
        ("  fwd_load_imbalance_ms", total_fwd_imbalance),
        ("  fwd_scheduling_overhead_ms", total_fwd_sched),
        ("cut_selection_ms", total_cs),
        ("cut_batch_build_ms", total_batch_build),
        ("cut_sync_ms", total_cut_sync),
        ("mpi_allreduce_ms", total_allreduce),
        ("state_exchange_ms", total_state_exchange),
        ("lower_bound_ms", total_lb),
        ("overhead_ms", total_overhead),
    ]
    for label, ms in rows:
        print(f"{label:<30}  {ms:>12.0f}  {pct(ms):>6.1f}%")

    if total_bw > 0:
        print(
            f"\nbwd_load_imbalance as fraction of backward_wall: {100 * total_bwd_imbalance / total_bw:.1f}%"
        )
    print()


def _corr(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation; returns nan if variance is zero."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sxx = sum(v * v for v in dx)
    syy = sum(v * v for v in dy)
    sxy = sum(a * b for a, b in zip(dx, dy, strict=True))
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx**0.5 * syy**0.5)


def _report_cut_selection(solver_t: pl.DataFrame, cut_path: Path | None) -> None:
    _header("10. Cut-selection × solver correlation")
    if cut_path is None:
        print("(no cut_selection/iterations.parquet — skipping)\n")
        return
    cs = pl.read_parquet(cut_path)
    if len(cs) == 0:
        print("(cut_selection parquet is empty)\n")
        return

    bw = solver_t.filter(pl.col("phase") == "backward")
    is_zero = pl.col("opening") == 0
    bw_agg = bw.group_by(["iteration", "stage"]).agg(
        (
            pl.col("simplex_iterations").filter(is_zero).sum()
            / pl.col("lp_solves").filter(is_zero).sum()
        ).alias("o0_piv"),
        (
            pl.col("simplex_iterations").filter(~is_zero).sum()
            / pl.col("lp_solves").filter(~is_zero).sum()
        ).alias("og_piv"),
        pl.col("solve_time_ms").sum().alias("solve_ms"),
        pl.col("simplex_iterations").sum().alias("piv"),
    )

    # NOTE: cut selection at iteration N sets up the LP for iteration N+1.
    # Align (cs iter) with (bw iter = cs_iter + 1) for proper cause→effect correlation.
    cs_shift = cs.with_columns((pl.col("iteration") + 1).alias("iteration"))
    joined = bw_agg.join(cs_shift, on=["iteration", "stage"], how="inner")
    if len(joined) == 0:
        print("(could not align cut selection with backward rows — skipping)\n")
        return

    print(f"Rows joined: {len(joined)}  (cut_sel iter shifted +1 to match next bw)")
    print()
    print("Per-iteration cut-count evolution (post-selection):")
    per_iter_cs = (
        cs.group_by("iteration")
        .agg(
            pl.col("cuts_populated").mean().alias("avg_populated"),
            pl.col("cuts_active_before").mean().alias("avg_before"),
            pl.col("cuts_deactivated").mean().alias("avg_deactivated"),
            pl.col("cuts_active_after").mean().alias("avg_after"),
            pl.col("selection_time_ms").sum().alias("total_sel_ms"),
        )
        .sort("iteration")
    )
    print(
        f"{'iter':>5}  {'populated':>10}  {'active_bef':>10}  {'deact':>8}  {'active_aft':>10}  {'sel_ms':>10}"
    )
    print(SUB_RULE)
    for row in per_iter_cs.iter_rows(named=True):
        print(
            f"{row['iteration']:>5}  {_f(row['avg_populated']):>10.1f}  "
            f"{_f(row['avg_before']):>10.1f}  {_f(row['avg_deactivated']):>8.1f}  "
            f"{_f(row['avg_after']):>10.1f}  {_f(row['total_sel_ms']):>10.3f}"
        )

    print()
    print(
        "Correlations: (cut-set properties at iter k) ↔ (solver behavior at iter k+1)"
    )
    cols_x = [
        "cuts_populated",
        "cuts_active_before",
        "cuts_deactivated",
        "cuts_active_after",
        "selection_time_ms",
    ]
    cols_y = [
        ("o0_piv", "ω=0 piv/solve"),
        ("og_piv", "ω>0 piv/solve"),
        ("piv", "total piv"),
    ]
    print(f"{'driver':<22}  " + "  ".join(f"{label:<18}" for _, label in cols_y))
    print(SUB_RULE)
    for xcol in cols_x:
        if xcol not in joined.columns:
            continue
        xs = [_f(v) for v in joined[xcol].to_list()]
        row_vals = []
        for ycol, _ in cols_y:
            ys = [_f(v) for v in joined[ycol].to_list()]
            row_vals.append(_corr(xs, ys))
        row_str = "  ".join(f"{v:<18.3f}" for v in row_vals)
        print(f"{xcol:<22}  {row_str}")

    print()
    print("First-iter-after-selection comparison — does heavy deactivation hurt ω=0?")
    # Split into high/low deactivation buckets and compare ω=0 piv
    q = joined["cuts_deactivated"].quantile(0.75)
    hi = joined.filter(pl.col("cuts_deactivated") >= q)
    lo = joined.filter(pl.col("cuts_deactivated") < q)
    if len(hi) > 0 and len(lo) > 0:
        print(
            f"  HIGH deactivation (>= p75 = {_f(q):.0f}): n={len(hi)}, "
            f"ω=0 piv/solve = {_f(hi['o0_piv'].mean()):.1f}, ω>0 = {_f(hi['og_piv'].mean()):.1f}"
        )
        print(
            f"  LOW  deactivation (<  p75): n={len(lo)}, "
            f"ω=0 piv/solve = {_f(lo['o0_piv'].mean()):.1f}, ω>0 = {_f(lo['og_piv'].mean()):.1f}"
        )

    # Active-after buckets
    q_after = joined["cuts_active_after"].quantile(0.75)
    hi_a = joined.filter(pl.col("cuts_active_after") >= q_after)
    lo_a = joined.filter(pl.col("cuts_active_after") < q_after)
    if len(hi_a) > 0 and len(lo_a) > 0:
        print(
            f"\n  LARGE active set (>= p75 = {_f(q_after):.0f}): n={len(hi_a)}, "
            f"ω=0 piv/solve = {_f(hi_a['o0_piv'].mean()):.1f}, ω>0 = {_f(hi_a['og_piv'].mean()):.1f}"
        )
        print(
            f"  SMALL active set (<  p75): n={len(lo_a)}, "
            f"ω=0 piv/solve = {_f(lo_a['o0_piv'].mean()):.1f}, ω>0 = {_f(lo_a['og_piv'].mean()):.1f}"
        )
    print()


def _report_dimensions(t: pl.DataFrame) -> None:
    _header("0. Data dimensions")
    bw = t.filter(pl.col("phase") == "backward")
    fw = t.filter(pl.col("phase") == "forward")
    n_iter = t["iteration"].n_unique()
    n_stage_bw = bw["stage"].n_unique()
    n_stage_fw = fw["stage"].n_unique()
    n_open_bw = bw["opening"].n_unique()
    trial_points_bw = 0
    n_bw_rows = len(bw)
    if n_bw_rows > 0:
        trial_points_bw = round(_f(bw["lp_solves"].sum()) / n_bw_rows)
    print(f"  iterations: {n_iter}")
    print(f"  stages (backward): {n_stage_bw}  |  stages (forward): {n_stage_fw}")
    print(f"  openings (backward): {n_open_bw}")
    print(f"  trial points per (iter, stage, ω): {trial_points_bw}")
    print(f"  total rows: {len(t)}  (backward={len(bw)}, forward={len(fw)})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze training/solver/iterations.parquet for SDDP performance diagnostics.",
    )
    parser.add_argument(
        "case_dir",
        nargs="?",
        default=None,
        help="Path to a case output dir (auto-discovers training/solver/iterations.parquet).",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Direct path to an iterations.parquet file (overrides case_dir).",
    )
    args = parser.parse_args()

    if args.case_dir is None and args.parquet is None:
        parser.error("Provide either case_dir or --parquet")

    paths = _resolve_paths(args)
    print(f"Reading {paths.iterations}\n")

    t = pl.read_parquet(paths.iterations)
    if len(t) == 0:
        print("ERROR: iterations.parquet is empty.", file=sys.stderr)
        return 2

    _report_dimensions(t)
    _report_totals(t)
    _report_ratio(t)
    _report_per_opening(t)
    _report_per_iteration(t)
    _report_stage_buckets(t)
    _report_basis_health(t)
    _report_retries(t, paths.retry_histogram)
    _report_sizing(t)
    _report_timing_breakdown(paths.timing)
    _report_cut_selection(t, paths.cut_selection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
