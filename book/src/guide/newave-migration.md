# Coming from NEWAVE?

This guide helps users familiar with NEWAVE understand how Cobre maps to
familiar concepts and how to convert existing cases.

## Entity Mapping

| NEWAVE                          | Cobre                   | File                                   |
| ------------------------------- | ----------------------- | -------------------------------------- |
| Usina hidrelétrica (`hidr.dat`) | `HydroPlant`            | `system/hydros.json`                   |
| Usina térmica (`term.dat`)      | `ThermalUnit`           | `system/thermals.json`                 |
| Subsistema (`sistema.dat`)      | `Bus`                   | `system/buses.json`                    |
| Interligação (`sis_int.dat`)    | `Exchange`              | `system/exchanges.json`                |
| Fontes não controláveis         | `NonControllableSource` | `system/non_controllable_sources.json` |
| Séries de afluência             | PAR(p) inflow models    | `scenarios/inflow_history.parquet`     |
| Configuração de execução        | Config                  | `config.json` + `stages.json`          |

## Key Differences

- **Input format**: Cobre uses JSON for entity definitions and Parquet for
  time series data, replacing the fixed-width text files used by NEWAVE.
- **Production models**: Cobre supports constant-productivity and four-piece
  hyperplane approximation (FPHA) for variable-head hydro plants.
- **Stochastic modeling**: PAR(p) coefficients can be estimated from historical
  inflow records or supplied directly.
- **Output format**: Results are written as Hive-partitioned Parquet files,
  directly readable from Python (Polars, Pandas) or any Arrow-compatible tool.

## Converting a NEWAVE Case

[Placeholder: `cobre-bridge convert newave <path>` -- the conversion tool is
under development in the `cobre-bridge` repository.]

## Bounds Comparison

[Placeholder: comparison results between Cobre and NEWAVE for reference cases
will be published when the validation pipeline is complete (see roadmap).]
