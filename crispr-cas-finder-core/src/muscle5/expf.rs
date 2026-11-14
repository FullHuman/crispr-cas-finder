//! `expf` as the C library the MUSCLE binary runs against computes it.
//!
//! MUSCLE turns log posteriors into probabilities with `expf`
//! (calcposteriorflat.cpp `CalcPostFlat`), and whether a probability reaches
//! the 0.01 that keeps it in the sparse matrix can hang on its last bit. The
//! macOS `expf` is not correctly rounded (about one result in a thousand
//! differs from the nearest `float`), so neither the standard library's
//! `exp` nor a correctly rounded one will do: this follows the same steps,
//! in double precision with fused multiply-adds, and gives the same result
//! for every `float` argument MUSCLE passes (the tests check a sample, a
//! native run checked all of them).
//!
//! The method is the usual one: with k = round(x * 128 / ln 2) and
//! r = x * 128 / ln 2 - k, exp(x) = 2^(k/128) * (1 + r * (C1 + C0 * r)),
//! where 2^(k/128) comes from a table of 128 correctly rounded powers.

const INV_LN2_N: f64 = f64::from_bits(0x4067_1547_652b_82fe);
const SHIFT: f64 = f64::from_bits(0x4338_0000_0000_0000);
const C0: f64 = f64::from_bits(0x3eee_bfbd_ff30_d656);
const C1: f64 = f64::from_bits(0x3f76_2e44_53e1_0dae);

/// 2^(i/128) rounded to a `double`, as its bits, minus `i << 45` (so that
/// adding `k << 45` puts k / 128 in the exponent field).
const POWERS: [u64; 128] = [
    0x3ff0_0000_0000_0000,
    0x3fef_f63d_a9fb_3335,
    0x3fef_ec9a_3e77_8061,
    0x3fef_e315_e86e_7f85,
    0x3fef_d9b0_d315_8574,
    0x3fef_d06b_29dd_f6de,
    0x3fef_c745_1875_9bc8,
    0x3fef_be3e_cac6_f383,
    0x3fef_b558_6cf9_890f,
    0x3fef_ac92_2b72_47f7,
    0x3fef_a3ec_32d3_d1a2,
    0x3fef_9b66_affe_d31b,
    0x3fef_9301_d012_5b51,
    0x3fef_8abd_c06c_31cc,
    0x3fef_829a_aea9_2de0,
    0x3fef_7a98_c8a5_8e51,
    0x3fef_72b8_3c7d_517b,
    0x3fef_6af9_388c_8dea,
    0x3fef_635b_eb6f_cb75,
    0x3fef_5be0_8404_5cd4,
    0x3fef_5487_3168_b9aa,
    0x3fef_4d50_22fc_d91d,
    0x3fef_463b_8862_8cd6,
    0x3fef_3f49_917d_dc96,
    0x3fef_387a_6e75_6238,
    0x3fef_31ce_4fb2_a63f,
    0x3fef_2b45_65e2_7cdd,
    0x3fef_24df_e1f5_6381,
    0x3fef_1e9d_f51f_dee1,
    0x3fef_187f_d0da_d990,
    0x3fef_1285_a6e4_030b,
    0x3fef_0caf_a93e_2f56,
    0x3fef_06fe_0a31_b715,
    0x3fef_0170_fc4c_d831,
    0x3fee_fc08_b264_16ff,
    0x3fee_f6c5_5f92_9ff1,
    0x3fee_f1a7_373a_a9cb,
    0x3fee_ecae_6d05_d866,
    0x3fee_e7db_34e5_9ff7,
    0x3fee_e32d_c313_a8e5,
    0x3fee_dea6_4c12_3422,
    0x3fee_da45_04ac_801c,
    0x3fee_d60a_21f7_2e2a,
    0x3fee_d1f5_d950_a897,
    0x3fee_ce08_6061_892d,
    0x3fee_ca41_ed1d_0057,
    0x3fee_c6a2_b5c1_3cd0,
    0x3fee_c32a_f0d7_d3de,
    0x3fee_bfda_d536_2a27,
    0x3fee_bcb2_99fd_dd0d,
    0x3fee_b9b2_769d_2ca7,
    0x3fee_b6da_a2cf_6642,
    0x3fee_b42b_569d_4f82,
    0x3fee_b1a4_ca5d_920f,
    0x3fee_af47_36b5_27da,
    0x3fee_ad12_d497_c7fd,
    0x3fee_ab07_dd48_5429,
    0x3fee_a926_8a59_46b7,
    0x3fee_a76f_15ad_2148,
    0x3fee_a5e1_b976_dc09,
    0x3fee_a47e_b03a_5585,
    0x3fee_a346_34cc_c320,
    0x3fee_a238_8255_2225,
    0x3fee_a155_d44c_a973,
    0x3fee_a09e_667f_3bcd,
    0x3fee_a012_750b_dabf,
    0x3fee_9fb2_3c65_1a2f,
    0x3fee_9f7d_f951_9484,
    0x3fee_9f75_e8ec_5f74,
    0x3fee_9f9a_48a5_8174,
    0x3fee_9feb_5642_67c9,
    0x3fee_a069_4fde_5d3f,
    0x3fee_a114_73eb_0187,
    0x3fee_a1ed_0130_c132,
    0x3fee_a2f3_36cf_4e62,
    0x3fee_a427_543e_1a12,
    0x3fee_a589_994c_ce13,
    0x3fee_a71a_4623_c7ad,
    0x3fee_a8d9_9b44_92ed,
    0x3fee_aac7_d98a_6699,
    0x3fee_ace5_422a_a0db,
    0x3fee_af32_16b5_448c,
    0x3fee_b1ae_9915_7736,
    0x3fee_b45b_0b91_ffc6,
    0x3fee_b737_b0cd_c5e5,
    0x3fee_ba44_cbc8_520f,
    0x3fee_bd82_9fde_4e50,
    0x3fee_c0f1_70ca_07ba,
    0x3fee_c491_82a3_f090,
    0x3fee_c863_19e3_2323,
    0x3fee_cc66_7b5d_e565,
    0x3fee_d09b_ec4a_2d33,
    0x3fee_d503_b23e_255d,
    0x3fee_d99e_1330_b358,
    0x3fee_de6b_5579_fdbf,
    0x3fee_e36b_bfd3_f37a,
    0x3fee_e89f_995a_d3ad,
    0x3fee_ee07_298d_b666,
    0x3fee_f3a2_b84f_15fb,
    0x3fee_f972_8de5_593a,
    0x3fee_ff76_f2fb_5e47,
    0x3fef_05b0_30a1_064a,
    0x3fef_0c1e_904b_c1d2,
    0x3fef_12c2_5bd7_1e09,
    0x3fef_199b_dd85_529c,
    0x3fef_20ab_5fff_d07a,
    0x3fef_27f1_2e57_d14b,
    0x3fef_2f6d_9406_e7b5,
    0x3fef_3720_dcef_9069,
    0x3fef_3f0b_555d_c3fa,
    0x3fef_472d_4a07_897c,
    0x3fef_4f87_080d_89f2,
    0x3fef_5818_dcfb_a487,
    0x3fef_60e3_16c9_8398,
    0x3fef_69e6_03db_3285,
    0x3fef_7321_f301_b460,
    0x3fef_7c97_337b_9b5f,
    0x3fef_8646_14f5_a129,
    0x3fef_902e_e78b_3ff6,
    0x3fef_9a51_fbc7_4c83,
    0x3fef_a4af_a2a4_90da,
    0x3fef_af48_2d8e_67f1,
    0x3fef_ba1b_ee61_5a27,
    0x3fef_c52b_376b_ba97,
    0x3fef_d076_5b6e_4540,
    0x3fef_dbfd_ad9c_be14,
    0x3fef_e7c1_819e_90d8,
    0x3fef_f3c2_2b8f_71f1,
];

/// e^x for |x| < 128, bit for bit as macOS `expf` returns it.
pub(super) fn expf(x: f32) -> f32 {
    let x = f64::from(x);
    let z = INV_LN2_N.mul_add(x, SHIFT);
    let k = z.to_bits();
    let r = INV_LN2_N.mul_add(x, -(z - SHIFT));
    let scale = f64::from_bits(POWERS[(k & 0x7f) as usize].wrapping_add(k << 45));
    (C0.mul_add(r, C1) * r).mul_add(scale, scale) as f32
}
