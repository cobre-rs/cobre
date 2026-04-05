#!/usr/bin/env python3.14
"""Generate synthetic PAR(2) inflow history for D26 deterministic test case.

This script produces `scenarios/inflow_history.parquet` with 40 years of
monthly inflows for hydro 0, following a PAR(2) process with true coefficients
phi_1=0.6, phi_2=0.35 for all 12 seasons. The data is designed so that PACF
order selection reliably picks order 2 (PACF at lag 2 ~ 0.35 > 1.96/sqrt(40)
~ 0.31).

Run once to regenerate (not needed during CI — the parquet file is committed).
"""

import pathlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date, timedelta

# ── PAR(2) process parameters ─────────────────────────────────────────────────

# True AR coefficients (same for all 12 seasons)
PHI1 = 0.6
PHI2 = 0.35

# Seasonal means (m3/s), one per month (Jan=0 ... Dec=11)
SEASONAL_MEANS = [
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

# Seasonal standard deviations (m3/s), one per month
SEASONAL_STDS = [
    45.0,  # January
    40.0,  # February
    38.0,  # March
    35.0,  # April
    32.0,  # May
    30.0,  # June
    28.0,  # July
    30.0,  # August
    33.0,  # September
    37.0,  # October
    42.0,  # November
    44.0,  # December
]

# ── Simulation settings ───────────────────────────────────────────────────────

RNG_SEED = 42
N_YEARS = 40  # 40 years × 12 months = 480 observations per entity
START_YEAR = 1981  # 1981-01-01 through 2020-12-01
HYDRO_ID = 0

# ── Simulate PAR(2) process ───────────────────────────────────────────────────


def simulate_par2(n_years: int, seed: int) -> np.ndarray:
    """Simulate a PAR(2) process for n_years × 12 months.

    The model is:
        a_t = mu_m(t) + phi1*(a_{t-1} - mu_{m(t-1)}) + phi2*(a_{t-2} - mu_{m(t-2)}) + sigma_m(t)*eps_t

    Returns a 1-D array of length n_years * 12 with inflow values in m3/s.
    All values are clipped to a minimum of 5 m3/s to ensure physical positivity.
    """
    rng = np.random.default_rng(seed)
    n_total = n_years * 12

    # Pre-compute: residual noise scale factors. The residual variance for a
    # PAR(p) process is sigma^2 * (1 - sum(phi_k * rho_k)). We use approximate
    # residual_ratio = sqrt(1 - phi1^2 - phi2^2 - 2*phi1*phi2) as a simplification
    # (exact for diagonal correlation); in practice the ratio is < 1.
    residual_ratio = np.sqrt(max(1.0 - PHI1**2 - PHI2**2 - 2 * PHI1 * PHI2 * 0.0, 0.01))

    inflows = np.zeros(n_total)

    # Initialise with seasonal means for the two burn-in lags.
    # Month indices for positions -2 and -1:
    m_prev2 = 10  # November (season_id=10)
    m_prev1 = 11  # December (season_id=11)
    a_prev2 = SEASONAL_MEANS[m_prev2]
    a_prev1 = SEASONAL_MEANS[m_prev1]

    for t in range(n_total):
        m = t % 12  # current season (0=Jan, ..., 11=Dec)
        m_prev1_idx = (t - 1) % 12 if t >= 1 else m_prev1
        m_prev2_idx = (t - 2) % 12 if t >= 2 else m_prev2

        mu_t = SEASONAL_MEANS[m]
        mu_t1 = SEASONAL_MEANS[m_prev1_idx] if t >= 1 else SEASONAL_MEANS[m_prev1]
        mu_t2 = SEASONAL_MEANS[m_prev2_idx] if t >= 2 else SEASONAL_MEANS[m_prev2]
        sigma_t = SEASONAL_STDS[m]

        noise = rng.standard_normal() * sigma_t * residual_ratio

        a_prev1_actual = inflows[t - 1] if t >= 1 else a_prev1
        a_prev2_actual = inflows[t - 2] if t >= 2 else a_prev2

        val = (
            mu_t
            + PHI1 * (a_prev1_actual - mu_t1)
            + PHI2 * (a_prev2_actual - mu_t2)
            + noise
        )
        # Physical inflows must be positive. Clip to 5 m3/s floor.
        inflows[t] = max(val, 5.0)

    return inflows


def main() -> None:
    script_dir = pathlib.Path(__file__).parent
    out_path = script_dir / "scenarios" / "inflow_history.parquet"

    inflows = simulate_par2(N_YEARS, RNG_SEED)

    # ── Verify PAR(2) is identifiable from the simulated data ────────────────
    # Quick autocorrelation check: lag-2 partial autocorrelation should be > 0.
    # This is a sanity check only; the full PACF selection happens in Rust.
    lag1_corr = np.corrcoef(inflows[1:], inflows[:-1])[0, 1]
    lag2_corr = np.corrcoef(inflows[2:], inflows[:-2])[0, 1]
    print(f"Lag-1 autocorrelation: {lag1_corr:.4f}")
    print(f"Lag-2 autocorrelation: {lag2_corr:.4f}")
    print(f"Min inflow: {inflows.min():.2f} m3/s")
    print(f"Mean inflow: {inflows.mean():.2f} m3/s")
    print(f"Max inflow: {inflows.max():.2f} m3/s")

    # ── Build date sequence: 1981-01-01 through 2020-12-01 ───────────────────
    dates = []
    year = START_YEAR
    month = 1
    for _ in range(N_YEARS * 12):
        dates.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1

    assert len(dates) == len(inflows), f"{len(dates)} != {len(inflows)}"
    assert dates[-1] == date(START_YEAR + N_YEARS - 1, 12, 1), (
        f"Last date should be {START_YEAR + N_YEARS - 1}-12-01, got {dates[-1]}"
    )

    # ── Write parquet file ────────────────────────────────────────────────────
    hydro_ids = [HYDRO_ID] * len(inflows)
    date32_values = [(d - date(1970, 1, 1)).days for d in dates]

    table = pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "date": pa.array(date32_values, type=pa.date32()),
            "value_m3s": pa.array(inflows.tolist(), type=pa.float64()),
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use zstd compression — the Rust parquet crate is built with the
    # "zstd" feature but NOT the "snap" (snappy) feature.
    pq.write_table(table, out_path, compression="zstd")
    print(f"\nWrote {len(inflows)} rows to {out_path}")


if __name__ == "__main__":
    main()
