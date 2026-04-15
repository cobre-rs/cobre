"""
Generate Parquet scenario files for d29-pattern-c-weekly-par test case.

Run with: python3 generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - inflow_ar_coefficients.parquet
  - load_seasonal_stats.parquet

Case: 1 bus (B0), 1 hydro (H0), 4 weekly stages all with season_id=0 (January 2024),
PAR(1) with psi=0.5 and residual_std_ratio=0.85, OutOfSample noise.

The inflow_seasonal_stats includes the pre-study stage (id=-1) and all 4 study stages.
The inflow_ar_coefficients covers the 4 study stages (PAR coefficients apply to study
stages only; pre-study stage is the lag initialization point).
Load is deterministic: mean=50.0 MW, std=0.0 for all 4 study stages.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), mean_m3s (FLOAT64), std_m3s (FLOAT64)
#
# Covers pre-study stage (id=-1) and study stages 0..3.
# mean_m3s=100.0, std_m3s=20.0 for all stages.

stats_hydro_ids = [0, 0, 0, 0, 0]
stats_stage_ids = [-1, 0, 1, 2, 3]
mean_m3s        = [100.0, 100.0, 100.0, 100.0, 100.0]
std_m3s         = [20.0,  20.0,  20.0,  20.0,  20.0]

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
# Covers all 4 study stages (0..3).

ar_hydro_ids          = [0, 0, 0, 0]
ar_stage_ids          = [0, 1, 2, 3]
ar_lags               = [1, 1, 1, 1]
ar_coefficients       = [0.5, 0.5, 0.5, 0.5]
ar_residual_std_ratio = [0.85, 0.85, 0.85, 0.85]

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
# Deterministic load: mean=50.0 MW, std=0.0 for all 4 study stages.

load_bus_ids   = [0, 0, 0, 0]
load_stage_ids = [0, 1, 2, 3]
load_mean_mw   = [50.0, 50.0, 50.0, 50.0]
load_std_mw    = [0.0,  0.0,  0.0,  0.0]

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

print("done.")
