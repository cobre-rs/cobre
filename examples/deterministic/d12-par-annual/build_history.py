#!/usr/bin/env python3
"""Generate synthetic inflow history for the d12-par-annual deterministic test fixture.

Produces ``scenarios/inflow_history.parquet`` containing 360 monthly inflow
observations for two hydros (hydro_id 0 and 1), spanning 1979-01-01 through
2008-12-01 (30 years × 12 months = 360 rows per hydro, 720 rows total).

The data is designed so that the PAR-A estimation path succeeds:

- 360 chronological observations per hydro (well above the 13-observation
  minimum required by ``estimate_annual_seasonal_stats``).
- 30 observations per (hydro, season) pair (above the ``min_observations_per_season = 4``
  threshold configured in ``config.json``).
- Seasonal sinusoidal mean with PAR(1) residuals; the annual component
  estimator will produce non-trivial ``ψ`` coefficients.
- 30 years provides stable PACF order selection (threshold ≈ 0.113), preventing
  explosive high-order coefficients that would cause LP infeasibility.

Determinism guarantee
---------------------
The RNG is seeded with ``seed=42`` via ``numpy.random.default_rng``.  Parquet
is written with ``zstd`` compression (no row-group metadata randomness).
Re-running this script produces byte-identical output on the same platform and
numpy version.

Usage::

    python examples/deterministic/d12-par-annual/build_history.py

The parquet file is committed to the repository; regeneration is only needed
if the fixture specification changes.
"""

from __future__ import annotations

import pathlib
from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ── Parameters ────────────────────────────────────────────────────────────────

RNG_SEED: int = 42
N_YEARS: int = 30  # 30 × 12 = 360 observations per hydro
START_YEAR: int = 1979
HYDRO_IDS: list[int] = [0, 1]

# Seasonal means (m3/s) — one per month (January=0 … December=11)
SEASONAL_MEANS_H0: list[float] = [
    220.0,  # January
    200.0,  # February
    190.0,  # March
    170.0,  # April
    155.0,  # May
    145.0,  # June
    140.0,  # July
    145.0,  # August
    160.0,  # September
    180.0,  # October
    195.0,  # November
    210.0,  # December
]

SEASONAL_MEANS_H1: list[float] = [
    110.0,  # January
    100.0,  # February
    95.0,  # March
    85.0,  # April
    78.0,  # May
    73.0,  # June
    70.0,  # July
    73.0,  # August
    80.0,  # September
    90.0,  # October
    98.0,  # November
    105.0,  # December
]

# Seasonal standard deviations (m3/s)
SEASONAL_STDS_H0: list[float] = [
    40.0,
    36.0,
    34.0,
    30.0,
    28.0,
    26.0,
    24.0,
    26.0,
    29.0,
    33.0,
    37.0,
    39.0,
]

SEASONAL_STDS_H1: list[float] = [
    20.0,
    18.0,
    17.0,
    15.0,
    14.0,
    13.0,
    12.0,
    13.0,
    14.5,
    16.5,
    18.5,
    19.5,
]

# PAR(1) coefficient (same for all seasons and both hydros)
PHI1: float = 0.5


# ── Simulation ────────────────────────────────────────────────────────────────


def simulate_par1(
    n_years: int,
    seed: int,
    seasonal_means: list[float],
    seasonal_stds: list[float],
) -> np.ndarray:
    """Simulate a PAR(1) process for ``n_years * 12`` months.

    Model: ``a_t = mu_m(t) + phi1*(a_{t-1} - mu_{m(t-1)}) + sigma_m(t)*eps_t``

    Returns a 1-D float64 array of length ``n_years * 12``.
    Values are clipped to a minimum of 5 m3/s for physical positivity.
    """
    rng = np.random.default_rng(seed)
    n_total = n_years * 12
    inflows = np.zeros(n_total, dtype=np.float64)

    # Initialise burn-in lag with the December seasonal mean.
    a_prev = seasonal_means[11]
    m_prev = 11

    for t in range(n_total):
        m = t % 12
        mu_t = seasonal_means[m]
        mu_prev = seasonal_means[m_prev]
        sigma_t = seasonal_stds[m]
        noise = rng.standard_normal() * sigma_t * (1.0 - PHI1**2) ** 0.5
        val = mu_t + PHI1 * (a_prev - mu_prev) + noise
        inflows[t] = max(val, 5.0)
        a_prev = inflows[t]
        m_prev = m

    return inflows


def build_dates(n_years: int, start_year: int) -> list[int]:
    """Return a list of ``date32`` day-offsets from the Unix epoch.

    One entry per month, from ``start_year-01-01`` through
    ``(start_year + n_years - 1)-12-01``.
    """
    epoch = date(1970, 1, 1)
    offsets: list[int] = []
    year = start_year
    month = 1
    for _ in range(n_years * 12):
        offsets.append((date(year, month, 1) - epoch).days)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return offsets


def main() -> None:
    """Generate and write ``scenarios/inflow_history.parquet``."""
    script_dir = pathlib.Path(__file__).parent
    out_path = script_dir / "scenarios" / "inflow_history.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    date_offsets = build_dates(N_YEARS, START_YEAR)

    # Each hydro uses an independent RNG stream derived from the global seed so
    # that hydro-0 and hydro-1 inflows are independent realisations.
    rng_root = np.random.default_rng(RNG_SEED)
    seeds = rng_root.integers(0, 2**31, size=len(HYDRO_IDS)).tolist()

    rows_hydro_id: list[int] = []
    rows_date: list[int] = []
    rows_value: list[float] = []

    means_by_hydro = [SEASONAL_MEANS_H0, SEASONAL_MEANS_H1]
    stds_by_hydro = [SEASONAL_STDS_H0, SEASONAL_STDS_H1]

    for hydro_idx, hydro_id in enumerate(HYDRO_IDS):
        inflows = simulate_par1(
            N_YEARS,
            int(seeds[hydro_idx]),
            means_by_hydro[hydro_idx],
            stds_by_hydro[hydro_idx],
        )
        rows_hydro_id.extend([hydro_id] * len(inflows))
        rows_date.extend(date_offsets)
        rows_value.extend(inflows.tolist())

    table = pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "date": pa.array(rows_date, type=pa.date32()),
            "value_m3s": pa.array(rows_value, type=pa.float64()),
        }
    )

    # zstd compression: consistent with the Rust parquet crate's default
    # (the crate is built with "zstd" but not "snap").
    pq.write_table(table, out_path, compression="zstd")

    n_hydros = len(HYDRO_IDS)
    n_obs = N_YEARS * 12
    print(
        f"Wrote {len(rows_hydro_id)} rows "
        f"({n_hydros} hydros × {n_obs} months) to {out_path}"
    )


if __name__ == "__main__":
    main()
