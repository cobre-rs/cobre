"""
Generate Parquet scenario files for d30-pattern-d-monthly-quarterly test case.

Run with: python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - inflow_ar_coefficients.parquet
  - load_seasonal_stats.parquet
  - inflow_history.parquet

Case: 1 bus (B0), 1 hydro (H0).
  - 6 monthly study stages (Jan-Jun 2024, stage_id 0-5)
  - 4 quarterly study stages (Q3 2024, Q4 2024, Q1 2025, Q2 2025, stage_id 6-9)
  - Custom SeasonMap: 12 monthly seasons (id 0-11) + 4 quarterly seasons (id 12-15)
  - PAR(1) with psi=0.5 and residual_std_ratio=0.85 for all 10 study stages
  - OutOfSample noise

The inflow_seasonal_stats includes the pre-study stage (id=-1) and all 10 study
stages. The inflow_ar_coefficients covers all 10 study stages.
Load is deterministic: mean=50.0 MW, std=0.0 for all 10 study stages.
The inflow_history covers 5 years of monthly observations (Jan 2019 - Dec 2023)
with seasonal variation, used by the aggregation pipeline.
"""

import os
import math
from datetime import date
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), mean_m3s (FLOAT64), std_m3s (FLOAT64)
#
# Covers pre-study stage (id=-1) and study stages 0..9.
# mean_m3s=100.0, std_m3s=20.0 for all stages.

stats_hydro_ids = [0] * 11
stats_stage_ids = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
mean_m3s        = [100.0] * 11
std_m3s         = [20.0] * 11

stats_schema = pa.schema([
    pa.field("hydro_id", pa.int32(),   nullable=False),
    pa.field("stage_id", pa.int32(),   nullable=False),
    pa.field("mean_m3s", pa.float64(), nullable=False),
    pa.field("std_m3s",  pa.float64(), nullable=False),
])

stats_table = pa.table(
    {
        "hydro_id": pa.array(stats_hydro_ids, type=pa.int32()),
        "stage_id": pa.array(stats_stage_ids, type=pa.int32()),
        "mean_m3s": pa.array(mean_m3s,        type=pa.float64()),
        "std_m3s":  pa.array(std_m3s,         type=pa.float64()),
    },
    schema=stats_schema,
)

stats_path = os.path.join(script_dir, "inflow_seasonal_stats.parquet")
pq.write_table(stats_table, stats_path, compression="zstd")
print(f"wrote {len(stats_table)} rows -> {stats_path}")

# ── inflow_ar_coefficients.parquet ────────────────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), lag (INT32),
#         coefficient (FLOAT64), residual_std_ratio (FLOAT64)
#
# PAR(1): 1 lag per stage, coefficient=0.5, residual_std_ratio=0.85.
# Covers all 10 study stages (0..9).

ar_hydro_ids          = [0] * 10
ar_stage_ids          = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ar_lags               = [1] * 10
ar_coefficients       = [0.5] * 10
ar_residual_std_ratio = [0.85] * 10

ar_schema = pa.schema([
    pa.field("hydro_id",          pa.int32(),   nullable=False),
    pa.field("stage_id",          pa.int32(),   nullable=False),
    pa.field("lag",               pa.int32(),   nullable=False),
    pa.field("coefficient",       pa.float64(), nullable=False),
    pa.field("residual_std_ratio",pa.float64(), nullable=False),
])

ar_table = pa.table(
    {
        "hydro_id":           pa.array(ar_hydro_ids,          type=pa.int32()),
        "stage_id":           pa.array(ar_stage_ids,          type=pa.int32()),
        "lag":                pa.array(ar_lags,               type=pa.int32()),
        "coefficient":        pa.array(ar_coefficients,       type=pa.float64()),
        "residual_std_ratio": pa.array(ar_residual_std_ratio, type=pa.float64()),
    },
    schema=ar_schema,
)

ar_path = os.path.join(script_dir, "inflow_ar_coefficients.parquet")
pq.write_table(ar_table, ar_path, compression="zstd")
print(f"wrote {len(ar_table)} rows -> {ar_path}")

# ── load_seasonal_stats.parquet ───────────────────────────────────────────────
#
# Schema: bus_id (INT32), stage_id (INT32), mean_mw (FLOAT64), std_mw (FLOAT64)
#
# Deterministic load: mean=50.0 MW, std=0.0 for all 10 study stages.

load_bus_ids   = [0] * 10
load_stage_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
load_mean_mw   = [50.0] * 10
load_std_mw    = [0.0] * 10

load_schema = pa.schema([
    pa.field("bus_id",   pa.int32(),   nullable=False),
    pa.field("stage_id", pa.int32(),   nullable=False),
    pa.field("mean_mw",  pa.float64(), nullable=False),
    pa.field("std_mw",   pa.float64(), nullable=False),
])

load_table = pa.table(
    {
        "bus_id":   pa.array(load_bus_ids,   type=pa.int32()),
        "stage_id": pa.array(load_stage_ids, type=pa.int32()),
        "mean_mw":  pa.array(load_mean_mw,   type=pa.float64()),
        "std_mw":   pa.array(load_std_mw,    type=pa.float64()),
    },
    schema=load_schema,
)

load_path = os.path.join(script_dir, "load_seasonal_stats.parquet")
pq.write_table(load_table, load_path, compression="zstd")
print(f"wrote {len(load_table)} rows -> {load_path}")

# ── inflow_history.parquet ────────────────────────────────────────────────────
#
# Schema: hydro_id (INT32), date (DATE32), value_m3s (FLOAT64)
#
# 5 years of monthly observations (Jan 2019 - Dec 2023), 60 rows total.
# Values vary seasonally: 80 + 40 * sin(month * pi / 6) where month is 1-based.
# This provides the monthly history for the aggregation pipeline to aggregate
# monthly observations into quarterly stats.
#
# Dates use Date32 encoding: days since 1970-01-01.

epoch = date(1970, 1, 1)
history_hydro_ids = []
history_dates = []
history_values = []

for year in range(2019, 2024):  # 2019..2023 inclusive
    for month in range(1, 13):  # 1..12 inclusive
        d = date(year, month, 1)
        history_hydro_ids.append(0)
        history_dates.append((d - epoch).days)
        # Seasonal variation: 80 + 40 * sin(month * pi / 6)
        history_values.append(80.0 + 40.0 * math.sin(month * math.pi / 6.0))

assert len(history_hydro_ids) == 60, f"Expected 60 history rows, got {len(history_hydro_ids)}"

history_schema = pa.schema([
    pa.field("hydro_id", pa.int32(),   nullable=False),
    pa.field("date",     pa.date32(),  nullable=False),
    pa.field("value_m3s",pa.float64(), nullable=False),
])

history_table = pa.table(
    {
        "hydro_id":  pa.array(history_hydro_ids, type=pa.int32()),
        "date":      pa.array(history_dates,     type=pa.date32()),
        "value_m3s": pa.array(history_values,    type=pa.float64()),
    },
    schema=history_schema,
)

history_path = os.path.join(script_dir, "inflow_history.parquet")
pq.write_table(history_table, history_path, compression="zstd")
print(f"wrote {len(history_table)} rows -> {history_path}")

print("done.")
