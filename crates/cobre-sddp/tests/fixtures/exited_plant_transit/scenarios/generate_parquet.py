"""
Generate Parquet scenario files for the exited_plant_transit fixture (water in
transit toward a plant that retires before it arrives).

Run with: .venv/bin/python generate_parquet.py

Produces:
  - inflow_seasonal_stats.parquet
  - load_seasonal_stats.parquet

Determinism: std_m3s = 0.0 and std_mw = 0.0 on every row, so each realized
inflow/load equals its mean exactly (no per-opening noise) and the simulated
trajectory is bit-reproducible.

Topology
========
Cascade U (id 0) -> J (id 1) -> D (id 2). U is run-of-river and carries
travel_time_hours = 360.0 on its arc to J (720 h stages -> k_0 = k_1 = 1/2).
J has exit_stage_id = 1: it operates at stage 0 and is retired at stage 1, so
the half of U's stage-0 release still in transit when J retires must reach D.
U's stage-0 inflow is 25 m3/s (64.8 hm3); every other inflow is zero. Demand
is 200 MWh per stage, carried by one thermal at 10 $/MWh.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

script_dir = os.path.dirname(os.path.abspath(__file__))

N_STAGES = 2

# ── inflow_seasonal_stats.parquet ─────────────────────────────────────────────
#
# Schema: hydro_id INT32, stage_id INT32, mean_m3s FLOAT64, std_m3s FLOAT64
#
# U's stage-0 inflow of 25 m3/s is the only water in the system; every other
# hydro and stage has zero inflow.
U_STAGE0_MEAN_M3S = 25.0

inflow_hydro_ids = []
inflow_stage_ids = []
inflow_mean_m3s = []
inflow_std_m3s = []
for hydro_id in (0, 1, 2):
    for stage_id in range(N_STAGES):
        inflow_hydro_ids.append(hydro_id)
        inflow_stage_ids.append(stage_id)
        inflow_mean_m3s.append(
            U_STAGE0_MEAN_M3S if (hydro_id, stage_id) == (0, 0) else 0.0
        )
        inflow_std_m3s.append(0.0)

inflow_schema = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_m3s", pa.float64(), nullable=False),
        pa.field("std_m3s", pa.float64(), nullable=False),
    ]
)

inflow_table = pa.table(
    {
        "hydro_id": pa.array(inflow_hydro_ids, type=pa.int32()),
        "stage_id": pa.array(inflow_stage_ids, type=pa.int32()),
        "mean_m3s": pa.array(inflow_mean_m3s, type=pa.float64()),
        "std_m3s": pa.array(inflow_std_m3s, type=pa.float64()),
    },
    schema=inflow_schema,
)

inflow_path = os.path.join(script_dir, "inflow_seasonal_stats.parquet")
pq.write_table(inflow_table, inflow_path, compression="zstd")
print(f"wrote {len(inflow_table)} rows -> {inflow_path}")

# ── load_seasonal_stats.parquet ───────────────────────────────────────────────
#
# Schema: bus_id INT32, stage_id INT32, mean_mw FLOAT64, std_mw FLOAT64
#
# One bus (id 0); 200 MWh per 720 h stage -> mean_mw = 200/720; std 0.0
# (deterministic). One FLAT block per stage, so mean_mw is the whole-stage load.
LOAD_MEAN_MW = 200.0 / 720.0

load_bus_ids = [0] * N_STAGES
load_stage_ids = list(range(N_STAGES))
load_mean_mw = [LOAD_MEAN_MW] * N_STAGES
load_std_mw = [0.0] * N_STAGES

load_schema = pa.schema(
    [
        pa.field("bus_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("mean_mw", pa.float64(), nullable=False),
        pa.field("std_mw", pa.float64(), nullable=False),
    ]
)

load_table = pa.table(
    {
        "bus_id": pa.array(load_bus_ids, type=pa.int32()),
        "stage_id": pa.array(load_stage_ids, type=pa.int32()),
        "mean_mw": pa.array(load_mean_mw, type=pa.float64()),
        "std_mw": pa.array(load_std_mw, type=pa.float64()),
    },
    schema=load_schema,
)

load_path = os.path.join(script_dir, "load_seasonal_stats.parquet")
pq.write_table(load_table, load_path, compression="zstd")
print(f"wrote {len(load_table)} rows -> {load_path}")

print("done.")
