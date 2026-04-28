//! Multi-hydro PAR(p)-A order parity vs NEWAVE.
//!
//! For every hydro in `inflow_history.parquet`, runs cobre's conditional FACP
//! pipeline (NEWAVE-style population stds, absolute-year alignment) and emits
//! the selected order per calendar month. The caller pipes the output to a
//! diff-friendly TSV/CSV for comparison against NEWAVE's parpvaz.dat orders.
//!
//! Run:
//! ```text
//! cargo run --release -p cobre-sddp --example dump_facp_all_hydros \
//!   -- /path/to/convertido_parpa > /tmp/cobre_orders.tsv
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

use std::env;
use std::path::PathBuf;

use chrono::Datelike;
use cobre_io::scenarios::parse_inflow_history;
use cobre_stochastic::par::fitting::{
    classify_history, conditional_facp_partitioned, select_order_pacf_annual,
};

const N_SEASONS: usize = 12;
const MAX_ORDER: usize = 6;
const Z_ALPHA: f64 = 1.96;

fn pop_stats(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let mean = values.iter().sum::<f64>() / n as f64;
    #[allow(clippy::cast_precision_loss)]
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    (mean, var.sqrt())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let case = PathBuf::from(args.get(1).cloned().expect("usage: <case_dir>"));

    let rows = parse_inflow_history(&case.join("scenarios/inflow_history.parquet"))
        .expect("read inflow_history");

    // Header: hydro_id, then 12 selected orders for calendar months 1..12.
    println!("hydro_id\tmin_year\tmax_year\tn_obs_first_month\torders");

    // Group by hydro_id once.
    use std::collections::BTreeMap;
    let mut by_hydro: BTreeMap<i32, Vec<&_>> = BTreeMap::new();
    for r in &rows {
        by_hydro.entry(r.hydro_id.0).or_default().push(r);
    }

    for (hid, mut hist) in by_hydro {
        hist.sort_unstable_by_key(|r| r.date);

        // Restrict to NEWAVE's reporting window 1931-2022 (parpvaz.dat).
        let hist: Vec<_> = hist
            .into_iter()
            .filter(|r| (1931..=2022).contains(&r.date.year()))
            .collect();
        if hist.is_empty() {
            continue;
        }
        let min_year = hist.first().unwrap().date.year();
        let max_year = hist.last().unwrap().date.year();

        // Z buckets per calendar month.
        let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); N_SEASONS];
        for r in &hist {
            let s = r.date.month0() as usize;
            obs_by_season[s].push(r.value_m3s);
        }

        // A buckets per target-month season.
        let all_pairs: Vec<(chrono::NaiveDate, f64)> =
            hist.iter().map(|r| (r.date, r.value_m3s)).collect();
        let mut annual_by_season: Vec<Vec<f64>> = vec![Vec::new(); N_SEASONS];
        for i in 0..all_pairs.len().saturating_sub(12) {
            let target = all_pairs[i + 11].0;
            let s = target.month0() as usize;
            let m: f64 = all_pairs[i..i + 12].iter().map(|(_, v)| *v).sum::<f64>() / 12.0;
            annual_by_season[s].push(m);
        }

        // Year starts.
        let mut z_year_starts: Vec<i32> = vec![i32::MAX; N_SEASONS];
        for r in &hist {
            let s = r.date.month0() as usize;
            let y = r.date.year();
            if y < z_year_starts[s] {
                z_year_starts[s] = y;
            }
        }
        let mut a_year_starts: Vec<i32> = vec![i32::MAX; N_SEASONS];
        for i in 0..all_pairs.len().saturating_sub(12) {
            let target = all_pairs[i + 11].0;
            let s = target.month0() as usize;
            let y = target.year();
            if y < a_year_starts[s] {
                a_year_starts[s] = y;
            }
        }

        // Population stats per season — apply NEWAVE TIPO override.
        let stats: Vec<(f64, f64)> = obs_by_season
            .iter()
            .map(|v| match classify_history(v).stats_override() {
                Some(forced) => forced,
                None => pop_stats(v),
            })
            .collect();
        let ann_stats: Vec<(f64, f64)> = annual_by_season.iter().map(|v| pop_stats(v)).collect();

        let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();
        let ann_refs: Vec<&[f64]> = annual_by_season.iter().map(Vec::as_slice).collect();

        // For each calendar month (cobre season index = month - 1), compute order.
        let mut orders = [0_usize; 12];
        let n_obs_jan = obs_by_season[0].len();
        for season in 0..N_SEASONS {
            let n_obs = obs_by_season[season].len();
            let prev_season = (season + N_SEASONS - 1) % N_SEASONS;
            // Mirror cobre's degenerate-bucket guard.
            if stats[season].1 == 0.0
                || n_obs < 2
                || annual_by_season[prev_season].is_empty()
                || ann_stats[prev_season].1 == 0.0
            {
                orders[season] = 0;
                continue;
            }
            let facp = conditional_facp_partitioned(
                season,
                MAX_ORDER,
                N_SEASONS,
                &obs_refs,
                &stats,
                &z_year_starts,
                &ann_refs,
                &ann_stats,
                &a_year_starts,
            );
            orders[season] = select_order_pacf_annual(&facp, n_obs, Z_ALPHA).selected_order;
        }

        let orders_str = orders
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        println!("{hid}\t{min_year}\t{max_year}\t{n_obs_jan}\t{orders_str}");
    }
}
