//! Dump cobre's conditional FACP values for CAMARGOS in `convertido_parpa`.
//!
//! Run:
//! ```text
//! cargo run --example dump_facp_camargos -p cobre-sddp --release \
//!   -- /path/to/convertido_parpa
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use chrono::Datelike;
use cobre_core::EntityId;
use cobre_io::scenarios::parse_inflow_history;
use cobre_stochastic::par::fitting::{
    conditional_facp_partitioned, periodic_pacf, select_order_pacf_annual,
};

const CAMARGOS_ID: EntityId = EntityId(0);
const N_SEASONS: usize = 12;
const MAX_ORDER: usize = 6;
const Z_ALPHA: f64 = 1.96;

fn main() {
    let args: Vec<String> = env::args().collect();
    let case = PathBuf::from(args.get(1).cloned().expect("usage: <case_dir>"));

    let hist_path = case.join("scenarios/inflow_history.parquet");
    let rows = parse_inflow_history(&hist_path).expect("read inflow_history");

    let cam_rows: Vec<_> = rows.iter().filter(|r| r.hydro_id == CAMARGOS_ID).collect();
    println!("CAMARGOS rows: {}", cam_rows.len());
    println!(
        "Date range: {} to {}",
        cam_rows
            .first()
            .map(|r| r.date.to_string())
            .unwrap_or_default(),
        cam_rows
            .last()
            .map(|r| r.date.to_string())
            .unwrap_or_default()
    );

    // Restrict to NEWAVE's window 1931-2022 to match parpvaz.dat.
    let cam_rows: Vec<_> = cam_rows
        .into_iter()
        .filter(|r| (1931..=2022).contains(&r.date.year()))
        .collect();
    println!("After 1931..=2022 filter: {} rows", cam_rows.len());

    // Group by calendar month (use month-1 as season index 0..11).
    let mut obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); N_SEASONS];
    for r in &cam_rows {
        let m = r.date.month0() as usize;
        obs_by_season[m].push(r.value_m3s);
    }
    for (m, v) in obs_by_season.iter().enumerate() {
        println!(
            "  month {}: n={} (sample) first={:.2} last={:.2}",
            m + 1,
            v.len(),
            v[0],
            v[v.len() - 1]
        );
    }

    // Stats: cobre uses sample std (n-1). NEWAVE uses population std (n).
    // We compute BOTH for comparison.
    let mut stats_sample: Vec<(f64, f64)> = Vec::with_capacity(N_SEASONS);
    let mut stats_pop: Vec<(f64, f64)> = Vec::with_capacity(N_SEASONS);
    for v in &obs_by_season {
        let n = v.len();
        #[allow(clippy::cast_precision_loss)]
        let mean = v.iter().sum::<f64>() / n as f64;
        #[allow(clippy::cast_precision_loss)]
        let var_s = if n >= 2 {
            v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let var_p = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        stats_sample.push((mean, var_s.sqrt()));
        stats_pop.push((mean, var_p.sqrt()));
    }

    // Build A_{t-1} groups: A at PDF time-index of group[i+11] = mean(group[i..i+12]).
    // Following cobre convention: annual_obs_by_season[s] holds A values whose
    // PDF time-index falls in season s.
    let mut all_obs: Vec<(chrono::NaiveDate, f64)> =
        cam_rows.iter().map(|r| (r.date, r.value_m3s)).collect();
    all_obs.sort_unstable_by_key(|(d, _)| *d);

    let mut annual_obs_by_season: Vec<Vec<f64>> = vec![Vec::new(); N_SEASONS];
    for i in 0..all_obs.len().saturating_sub(12) {
        let target_date = all_obs[i + 11].0;
        let season = target_date.month0() as usize;
        let mean_a: f64 = all_obs[i..i + 12].iter().map(|(_, v)| v).sum::<f64>() / 12.0;
        annual_obs_by_season[season].push(mean_a);
    }

    let mut annual_stats_sample: Vec<(f64, f64)> = Vec::with_capacity(N_SEASONS);
    let mut annual_stats_pop: Vec<(f64, f64)> = Vec::with_capacity(N_SEASONS);
    for v in &annual_obs_by_season {
        let n = v.len();
        #[allow(clippy::cast_precision_loss)]
        let mean = v.iter().sum::<f64>() / n.max(1) as f64;
        #[allow(clippy::cast_precision_loss)]
        let var_s = if n >= 2 {
            v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        #[allow(clippy::cast_precision_loss)]
        let var_p = if n >= 1 {
            v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64
        } else {
            0.0
        };
        annual_stats_sample.push((mean, var_s.sqrt()));
        annual_stats_pop.push((mean, var_p.sqrt()));
    }

    let obs_refs: Vec<&[f64]> = obs_by_season.iter().map(Vec::as_slice).collect();
    let annual_obs_refs: Vec<&[f64]> = annual_obs_by_season.iter().map(Vec::as_slice).collect();

    println!("\n=== Conditional FACP per month (cobre formula, sample stds) ===");
    let nw_pacf: HashMap<usize, [f64; 6]> = HashMap::from([
        (1, [0.3236, -0.0769, -0.0050, -0.1912, 0.0478, 0.0868]),
        (2, [0.3891, -0.0686, -0.2467, 0.0325, -0.1910, 0.0387]),
        (3, [0.4737, 0.0998, 0.0153, 0.0316, 0.0587, -0.1532]),
        (4, [0.6216, 0.1709, 0.0051, 0.2233, 0.1217, 0.0299]),
        (5, [0.8863, -0.0197, 0.2687, 0.0228, 0.1372, 0.0280]),
        (6, [0.7256, 0.2609, 0.0701, 0.0007, 0.1293, -0.0524]),
        (7, [0.8461, 0.2280, 0.0891, 0.2584, 0.0225, -0.0999]),
        (8, [0.7648, -0.3432, 0.0604, 0.0167, 0.1494, -0.0478]),
        (9, [0.5964, 0.4944, 0.2145, -0.3872, -0.0045, -0.1574]),
        (10, [0.5980, 0.1622, 0.3569, 0.1250, 0.2227, -0.0399]),
        (11, [0.5027, 0.2206, -0.1808, -0.0064, -0.1677, 0.0148]),
        (12, [0.3727, 0.2015, 0.0004, 0.0959, -0.0853, -0.0513]),
    ]);

    let n_obs = obs_by_season[0].len();
    #[allow(clippy::cast_precision_loss)]
    let threshold = Z_ALPHA / (n_obs as f64).sqrt();
    println!("Threshold (z=1.96, n={n_obs}): {threshold:.4}\n");

    println!("month |  k=1 cobre | k=1 NW    |  k=2 cobre | k=2 NW    |  k=3 cobre | k=3 NW    |  k=4 cobre | k=4 NW    |  k=5 cobre | k=5 NW    |  k=6 cobre | k=6 NW    | order_cobre | order_NW_orig");
    let nw_orig_orders = [1, 3, 1, 4, 3, 2, 4, 2, 4, 5, 2, 1];

    for season in 0..N_SEASONS {
        let m = season + 1;
        let facp_sample = conditional_facp_partitioned(
            season,
            MAX_ORDER,
            N_SEASONS,
            &obs_refs,
            &stats_sample,
            &annual_obs_refs,
            &annual_stats_sample,
        );
        let order_sample = select_order_pacf_annual(&facp_sample, n_obs, Z_ALPHA).selected_order;
        let nw = nw_pacf[&m];

        let mut row = format!("{:5} |", m);
        for k in 0..6 {
            let cv = facp_sample.get(k).copied().unwrap_or(f64::NAN);
            row.push_str(&format!(" {:>+10.4} | {:>+9.4} |", cv, nw[k]));
        }
        row.push_str(&format!(
            " {:>11} | {:>13}",
            order_sample, nw_orig_orders[season]
        ));
        println!("{}", row);
    }

    println!("\n=== Same with population stds (NEWAVE convention) ===");
    for season in 0..N_SEASONS {
        let m = season + 1;
        let facp_pop = conditional_facp_partitioned(
            season,
            MAX_ORDER,
            N_SEASONS,
            &obs_refs,
            &stats_pop,
            &annual_obs_refs,
            &annual_stats_pop,
        );
        let order_pop = select_order_pacf_annual(&facp_pop, n_obs, Z_ALPHA).selected_order;
        let nw = nw_pacf[&m];

        let mut row = format!("{:5} |", m);
        for k in 0..6 {
            let cv = facp_pop.get(k).copied().unwrap_or(f64::NAN);
            row.push_str(&format!(" {:>+10.4} | {:>+9.4} |", cv, nw[k]));
        }
        row.push_str(&format!(
            " {:>11} | {:>13}",
            order_pop, nw_orig_orders[season]
        ));
        println!("{}", row);
    }

    println!("\n=== Classical (no annual) periodic_pacf ===");
    for season in 0..N_SEASONS {
        let m = season + 1;
        let pacf = periodic_pacf(season, MAX_ORDER, N_SEASONS, &obs_refs, &stats_pop);
        let mut row = format!("{:5} |", m);
        for k in 0..MAX_ORDER {
            let v = pacf.get(k).copied().unwrap_or(f64::NAN);
            row.push_str(&format!(" {:>+10.4} |", v));
        }
        println!("{}", row);
    }
}
