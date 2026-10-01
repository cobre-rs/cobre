//! Single owner of the stage block-hours facts derived from `Stage::blocks`.

use cobre_core::{Block, Stage};

use crate::indexer::BlockIdx;

/// Per-hour conversion factor from m³/s to hm³:
/// `seconds_per_hour / m³_per_hm³ = 3600 / 1_000_000`. Callers multiply by
/// `Block::duration_hours`: `volume_hm3 = flow_m3s * M3S_TO_HM3 * duration_hours`.
pub(crate) const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;

/// zeta is the summed block hours times [`M3S_TO_HM3`], never the sum of the
/// per-block `tau` factors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockClock<'a> {
    blocks: &'a [Block],
    total_hours: f64,
}

impl<'a> BlockClock<'a> {
    pub(crate) fn new(stage: &'a Stage) -> Self {
        Self {
            blocks: &stage.blocks,
            total_hours: stage.total_hours(),
        }
    }

    pub(crate) fn total_hours(self) -> f64 {
        self.total_hours
    }

    pub(crate) fn n_blks(self) -> usize {
        self.blocks.len()
    }

    pub(crate) fn zeta(self) -> f64 {
        self.total_hours * M3S_TO_HM3
    }

    pub(crate) fn tau(self, blk: BlockIdx) -> f64 {
        self.blocks[blk.get()].duration_hours * M3S_TO_HM3
    }

    pub(crate) fn hours(self, blk: BlockIdx) -> f64 {
        self.blocks[blk.get()].duration_hours
    }
}

#[cfg(test)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };

    use super::{BlockClock, M3S_TO_HM3};
    use crate::indexer::BlockIdx;

    fn stage_with_hours(hours: [f64; 3]) -> Stage {
        Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: hours
                .into_iter()
                .enumerate()
                .map(|(index, duration_hours)| Block {
                    index,
                    name: format!("BLK{index}"),
                    duration_hours,
                })
                .collect(),
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    #[test]
    fn zeta_multiplies_the_summed_hours_not_the_summed_block_factors() {
        let stage = stage_with_hours([0.1, 0.2, 0.3]);
        let clock = BlockClock::new(&stage);
        let expected = (0.1_f64 + 0.2 + 0.3) * M3S_TO_HM3;
        assert_eq!(clock.zeta().to_bits(), expected.to_bits());

        let summed_taus =
            clock.tau(BlockIdx::new(0)) + clock.tau(BlockIdx::new(1)) + clock.tau(BlockIdx::new(2));
        assert_ne!(clock.zeta().to_bits(), summed_taus.to_bits());
    }

    #[test]
    fn n_blks_is_the_stage_block_count() {
        let stage = stage_with_hours([0.1, 0.2, 0.3]);
        let clock = BlockClock::new(&stage);
        assert_eq!(clock.n_blks(), 3);
    }

    #[test]
    fn total_hours_is_the_left_to_right_block_sum() {
        let stage = stage_with_hours([0.1, 0.2, 0.3]);
        let clock = BlockClock::new(&stage);
        assert_eq!(
            clock.total_hours().to_bits(),
            (0.1_f64 + 0.2 + 0.3).to_bits()
        );
    }

    #[test]
    fn tau_and_hours_read_the_named_block() {
        let stage = stage_with_hours([0.1, 0.2, 0.3]);
        let clock = BlockClock::new(&stage);
        assert_eq!(
            clock.tau(BlockIdx::new(1)).to_bits(),
            (0.2_f64 * M3S_TO_HM3).to_bits()
        );
        assert_eq!(clock.hours(BlockIdx::new(2)), 0.3);
    }
}
