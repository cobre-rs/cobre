"""
Generate Parquet scenario files for d28-decomp-weekly-monthly test case.

Run with: uv tool run --with pyarrow -- python3 generate_parquet.py

Produces:
  - external_inflow_scenarios.parquet
  - inflow_seasonal_stats.parquet
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

# ── external_inflow_scenarios.parquet ─────────────────────────────────────────
#
# Schema: stage_id (INT32), scenario_id (INT32), hydro_id (INT32), value_m3s (FLOAT64)
#
# Layout:
#   stages 0-4: 1 scenario each (scenario_id=0), hydro_id=0
#   stage 5:    5 scenarios (scenario_id=0..4), hydro_id=0
#
# Deterministic values in 40-80 m3/s range (non-trivial, all distinct):
#   stage 0 scenario 0: 45.0
#   stage 1 scenario 0: 55.0
#   stage 2 scenario 0: 60.0
#   stage 3 scenario 0: 50.0
#   stage 4 scenario 0: 65.0
#   stage 5 scenario 0: 40.0
#   stage 5 scenario 1: 50.0
#   stage 5 scenario 2: 60.0
#   stage 5 scenario 3: 70.0
#   stage 5 scenario 4: 80.0
#
# Total: 10 rows (5 weekly x 1 + 1 monthly x 5)

stage_ids    = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5]
scenario_ids = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
hydro_ids    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
values_m3s   = [45.0, 55.0, 60.0, 50.0, 65.0, 40.0, 50.0, 60.0, 70.0, 80.0]

external_schema = pa.schema([
    pa.field("stage_id",    pa.int32(),   nullable=False),
    pa.field("scenario_id", pa.int32(),   nullable=False),
    pa.field("hydro_id",    pa.int32(),   nullable=False),
    pa.field("value_m3s",   pa.float64(), nullable=False),
])

external_table = pa.table(
    {
        "stage_id":    pa.array(stage_ids,    type=pa.int32()),
        "scenario_id": pa.array(scenario_ids, type=pa.int32()),
        "hydro_id":    pa.array(hydro_ids,    type=pa.int32()),
        "value_m3s":   pa.array(values_m3s,   type=pa.float64()),
    },
    schema=external_schema,
)

external_path = os.path.join(script_dir, "external_inflow_scenarios.parquet")
pq.write_table(external_table, external_path, compression="zstd")
print(f"wrote {len(external_table)} rows -> {external_path}")

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id (INT32), stage_id (INT32), mean_m3s (FLOAT64), std_m3s (FLOAT64)
#
# One row per (hydro, stage) pair: hydro_id=0, stage_id=0..5
# mean_m3s=60.0, std_m3s=15.0 for all stages

stats_hydro_ids = [0, 0, 0, 0, 0, 0]
stats_stage_ids = [0, 1, 2, 3, 4, 5]
mean_m3s        = [60.0, 60.0, 60.0, 60.0, 60.0, 60.0]
std_m3s         = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]

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

print("done.")
