//! Constants below were captured on the pre-change code; a diff in them means the
//! forward noise stream moved.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]

use cobre_core::{EntityId, NoiseMethod, SamplingScheme};
use cobre_stochastic::{
    ClassDimensions, DecomposedCorrelation, SampleRequest, build_forward_sampler,
    generate_opening_tree, tree::OpeningTreeGenerationInputs,
};

mod common;
use common::{
    build_test_ctx, build_test_system, correlated_correlation_model, identity_correlation_model,
    make_sampler_config, method_stage, stages_from_system, tables_for,
};

const GOLDEN_SEQUENCE: [f64; 48] = [
    -0.080_725_442_736_187_18,
    0.342_169_703_875_618_6,
    1.849_818_605_734_19,
    -1.108_196_839_820_201_8,
    -0.779_542_349_566_169_5,
    -0.295_284_160_939_788_5,
    0.576_407_613_977_412_9,
    1.194_651_590_486_514,
    -0.652_782_348_056_772_2,
    0.004_955_670_217_361_959,
    0.696_519_746_226_396_6,
    -2.881_805_549_003_158_4,
    -2.459_925_019_377_221_3,
    -0.668_281_316_820_787_3,
    0.017_417_691_321_618_904,
    0.680_724_290_146_252_7,
    -8.21,
    -8.21,
    -8.21,
    -0.430_727_299_975_868_17,
    0.0,
    0.430_727_299_975_868_06,
    -8.21,
    -0.764_709_674_990_941_9,
    -8.21,
    -8.21,
    0.0,
    -0.430_727_299_975_868_17,
    -8.21,
    0.430_727_299_975_868_06,
    0.0,
    -1.220_640_349_639_383,
    -1.072_385_873_667_138,
    -0.157_124_096_842_055_58,
    0.087_711_240_490_644_9,
    0.770_845_438_329_447_1,
    1.006_628_194_622_461_3,
    -1.303_805_206_386_656_5,
    -0.167_933_641_235_421_45,
    0.542_772_246_963_319_6,
    0.157_086_514_183_187_86,
    -1.949_857_445_865_825_8,
    -0.090_633_019_333_538_7,
    -0.331_743_552_456_976_84,
    -0.879_316_021_632_609_9,
    0.877_941_307_496_606_9,
    0.891_338_480_484_778_2,
    0.337_851_871_717_413_1,
];

const GOLDEN_WIDE_S0_O0: [f64; 70] = [
    0.261_943_978_666_802_36,
    0.127_889_353_586_245_6,
    -0.397_279_971_931_077_9,
    0.260_598_250_741_210_85,
    0.105_779_504_081_037_67,
    0.893_746_933_495_428_1,
    -1.343_253_373_302_420_8,
    0.092_521_056_368_748_44,
    -1.238_486_988_715_376_2,
    0.554_255_718_755_035,
    0.653_610_230_622_547_7,
    -1.041_250_121_454_886_6,
    0.386_666_049_609_057_3,
    1.133_666_403_097_759_2,
    -0.226_238_634_032_527_7,
    0.225_323_484_452_808_36,
    1.261_331_954_543_671_9,
    -0.450_374_834_499_717_1,
    1.230_461_538_878_261_4,
    0.292_739_676_474_220_17,
    -0.420_218_021_805_944_1,
    -1.654_856_993_187_656,
    1.504_837_998_773_783_4,
    -0.002_252_242_451_901_694_7,
    0.229_162_099_307_601_4,
    -0.104_673_117_545_754_37,
    -1.503_430_227_024_682_9,
    -0.005_976_751_182_632_97,
    -0.807_785_330_003_612_1,
    -0.729_196_429_459_687_3,
    -1.478_225_681_275_69,
    -1.385_178_214_711_543,
    -0.019_444_656_016_684_07,
    -0.728_736_399_628_431_7,
    -0.366_729_314_351_177,
    1.055_206_087_526_587_4,
    0.381_122_526_742_886_77,
    0.031_861_444_545_165_92,
    -1.458_715_231_306_354_5,
    0.170_958_400_988_427_68,
    -0.168_848_052_028_692_9,
    0.061_222_412_313_849_78,
    -1.094_028_115_684_667_8,
    -0.164_298_889_510_819_17,
    0.649_241_502_060_317_3,
    0.812_822_186_738_101_7,
    0.190_802_024_947_911_87,
    0.550_820_451_612_900_2,
    -0.276_802_046_353_518_57,
    -1.085_787_901_578_905_1,
    0.479_195_526_656_396_8,
    1.035_696_150_711_604_3,
    0.734_861_037_432_464_4,
    -0.171_897_587_695_379_36,
    -0.768_803_273_593_813_2,
    0.373_641_904_424_883_7,
    -0.050_268_087_882_327_825,
    0.071_530_581_330_186_04,
    -0.754_801_503_344_352_4,
    -0.402_241_416_296_409_8,
    -0.498_269_241_895_616_55,
    1.035_545_800_847_415,
    -0.003_801_806_202_421_025_7,
    -0.242_196_127_856_020_15,
    -0.560_809_407_557_846,
    -0.469_451_756_654_139_5,
    0.169_197_703_157_331_49,
    0.341_386_821_571_796_9,
    1.005_807_869_934_130_6,
    -0.274_053_317_089_533_95,
];
const GOLDEN_WIDE_S0_O2_D0: f64 = 0.441_985_571_180_052_8;
const GOLDEN_WIDE_S0_O2_D69: f64 = -0.110_289_738_555_546_02;
const GOLDEN_WIDE_S1_O1_D0: f64 = 0.311_586_211_850_554_1;
const GOLDEN_WIDE_S1_O1_D69: f64 = 0.653_769_318_292_650_7;

#[test]
fn out_of_sample_point_method_golden_sequence() {
    let system = build_test_system(
        &[
            NoiseMethod::QmcSobol,
            NoiseMethod::QmcHalton,
            NoiseMethod::Lhs,
        ],
        identity_correlation_model(&[1, 2]),
    );
    let ctx = build_test_ctx(&system, Some(99));
    let stages = stages_from_system(&system);
    let sampler = build_forward_sampler(make_sampler_config(
        SamplingScheme::OutOfSample,
        &ctx,
        &stages,
    ))
    .unwrap();
    let dim = ctx.dim();
    let total_scenarios: u32 = 4;

    let mut noise_buf = vec![0.0f64; dim];
    let mut corr_scratch = vec![0.0f64; 2 * dim];

    let mut idx = 0usize;
    for stage_idx in 0..3_usize {
        for noise_group_id in 0..2_u32 {
            // This sweep probes the same stage_idx under two synthetic
            // noise_group_id values — a table is keyed by stage_idx alone, so
            // it must be rebuilt per noise_group_id here to stay bit-identical
            // to the golden's pre-table per-draw derivation.
            let mut groups = vec![0u32; 3];
            groups[stage_idx] = noise_group_id;
            let tables = tables_for(&sampler, 0, total_scenarios, &groups);
            for scenario in 0..total_scenarios {
                let result = sampler
                    .sample(SampleRequest {
                        iteration: 0,
                        scenario,
                        stage: u32::try_from(stage_idx).unwrap(),
                        stage_idx,
                        noise_buf: &mut noise_buf,
                        corr_scratch: &mut corr_scratch,
                        total_scenarios,
                        noise_group_id,
                        node_opening_offset: 0,
                        node_opening_len: 0,
                        pinned_scenario: None,
                        tables: &tables,
                    })
                    .unwrap();

                let slice = result.as_slice();
                for (d, &v) in slice.iter().enumerate() {
                    assert_eq!(
                        v, GOLDEN_SEQUENCE[idx],
                        "stage_idx={stage_idx} noise_group_id={noise_group_id} \
                         scenario={scenario} dim={d}"
                    );
                    idx += 1;
                }
            }
        }
    }
}

#[test]
fn wide_correlation_group_opening_tree_golden() {
    let entity_ids: Vec<i32> = (1..=70).collect();
    let entity_order: Vec<EntityId> = entity_ids.iter().copied().map(EntityId).collect();
    let dims = ClassDimensions {
        n_hydros: 70,
        n_load_buses: 0,
        n_ncs: 0,
    };
    let corr = DecomposedCorrelation::build(
        &correlated_correlation_model(&entity_ids, 0.4),
        &entity_order,
        dims,
    )
    .unwrap();

    let stages = vec![
        method_stage(0, 0, 3, NoiseMethod::Saa),
        method_stage(1, 1, 3, NoiseMethod::Saa),
    ];

    let tree = generate_opening_tree(
        42,
        &stages,
        &corr,
        &entity_order,
        dims,
        &OpeningTreeGenerationInputs::default(),
    )
    .unwrap();

    let s0_o0 = tree.opening(0, 0);
    for (d, &v) in s0_o0.iter().enumerate() {
        assert_eq!(v, GOLDEN_WIDE_S0_O0[d], "stage=0 opening=0 dim={d}");
    }

    assert_eq!(
        tree.opening(0, 2)[0],
        GOLDEN_WIDE_S0_O2_D0,
        "stage=0 opening=2 dim=0"
    );
    assert_eq!(
        tree.opening(0, 2)[69],
        GOLDEN_WIDE_S0_O2_D69,
        "stage=0 opening=2 dim=69"
    );
    assert_eq!(
        tree.opening(1, 1)[0],
        GOLDEN_WIDE_S1_O1_D0,
        "stage=1 opening=1 dim=0"
    );
    assert_eq!(
        tree.opening(1, 1)[69],
        GOLDEN_WIDE_S1_O1_D69,
        "stage=1 opening=1 dim=69"
    );
}
