// simd/forward_filter.rs — SIMD Forward filter using portable SIMD (std::simd)
//
// Uses Simd<f32, 4> for 4-way parallel Forward in probability (odds-ratio) space
// with Farrar-striped layout. Sparse rescaling prevents single-precision overflow.
//
// Parser mode: O(M+L) memory — one MDI row reused, plus special state
// scores per position for posterior decoding of domain structure.

use super::oprofile::*;
use std::simd::prelude::*;

/// Result of the Forward filter in parser mode.
pub struct ForwardResult {
    pub score: f32,
    pub xmx: Vec<[f32; 5]>,      // [E,N,J,B,C] per position 0..L
    pub scale_factors: Vec<f32>, // scale factor at each position 0..L
    pub totscale: f64,
}

/// Run the Forward filter (parser mode) on a digitized sequence.
///
/// `dsq` is 0-indexed: residues at dsq[0..l].
pub fn forward_filter(dsq: &[u8], l: usize, om: &OptimizedProfile) -> ForwardResult {
    let q = om.q4;
    let zero_v = f32x4::splat(0.0);
    // Transition layout is invariant across sequence rows.
    let dd_base = NTSC_PER_Q * q;
    let regular_tfv = &om.tfv[..dd_base * 4];

    // Single MDI row (parser mode — O(M) memory)
    let mut mmx = vec![f32x4::splat(0.0); q];
    let mut imx = vec![f32x4::splat(0.0); q];
    let mut dmx = vec![f32x4::splat(0.0); q];

    let mut xmx = vec![[0.0f32; 5]; l + 1];
    let mut scale_factors = vec![1.0f32; l + 1];
    let mut totscale: f64 = 0.0;

    // Init row 0
    let mut x_e: f32 = 0.0;
    let mut x_n: f32 = 1.0;
    let mut x_j: f32 = 0.0;
    let mut x_b: f32 = om.xf[XST_N][XTR_MOVE];
    let mut x_c: f32 = 0.0;

    xmx[0] = [x_e, x_n, x_j, x_b, x_c];

    for i in 0..l {
        let xi = dsq[i] as usize;
        let rsc = if xi < om.kp { &om.rfv[xi] } else { &om.rfv[0] };

        let mut dcv = f32x4::splat(0.0);
        let mut xe_v = f32x4::splat(0.0);
        let xb_v = f32x4::splat(x_b);

        // Right-shift previous M/D/I rows by 1 lane
        let mut mpv = shr_f32x4(zero_v, mmx[q - 1]);
        let mut dpv = shr_f32x4(zero_v, dmx[q - 1]);
        let mut ipv = shr_f32x4(zero_v, imx[q - 1]);

        // Each q-block contains seven adjacent SIMD transition vectors.
        // Chunking exposes that layout directly and removes the incrementing
        // transition index and its repeated scale/address calculations.
        for (qi, (rsc_q, tsc_q)) in rsc
            .as_chunks::<4>()
            .0
            .iter()
            .zip(regular_tfv.as_chunks::<{ NTSC_PER_Q * 4 }>().0.iter())
            .enumerate()
        {
            let rsc_v = f32x4::from_slice(rsc_q);
            let tbm_v = f32x4::from_slice(&tsc_q[0..4]);
            let tmm_v = f32x4::from_slice(&tsc_q[4..8]);
            let tim_v = f32x4::from_slice(&tsc_q[8..12]);
            let tdm_v = f32x4::from_slice(&tsc_q[12..16]);

            // M(i,q) = (B*tBM + M(i-1,q-1)*tMM + I(i-1,q-1)*tIM + D(i-1,q-1)*tDM) * rsc
            let mut sv = xb_v * tbm_v;
            sv += mpv * tmm_v;
            sv += ipv * tim_v;
            sv += dpv * tdm_v;
            sv *= rsc_v;

            xe_v += sv;

            // Save prev values, overwrite with current
            mpv = mmx[qi];
            dpv = dmx[qi];
            ipv = imx[qi];

            mmx[qi] = sv;
            dmx[qi] = dcv;

            // D->D carry: D(i,q+1) partial via M->D
            let tmd_v = f32x4::from_slice(&tsc_q[16..20]);
            dcv = sv * tmd_v;

            // I(i,q) = M(i-1,q)*tMI + I(i-1,q)*tII
            let tmi_v = f32x4::from_slice(&tsc_q[20..24]);
            let tii_v = f32x4::from_slice(&tsc_q[24..28]);
            imx[qi] = mpv * tmi_v + ipv * tii_v;
        }

        // DD passes: propagate D->D transitions across the stripe boundary
        // First pass
        dcv = shr_f32x4(zero_v, dcv);

        // The main loop has already written its initially-zero carry to
        // dmx[0], so clearing that element again would be a redundant store.
        for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
            let dd_v = f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
            *dmx_qi += dcv;
            dcv = *dmx_qi * dd_v;
        }

        // Additional DD passes (up to 3 more, for convergence)
        if om.m < 100 {
            for _pass in 1..4 {
                dcv = shr_f32x4(zero_v, dcv);
                for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                    let dd_v =
                        f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                    *dmx_qi += dcv;
                    dcv *= dd_v;
                }
            }
        } else {
            for _pass in 1..4 {
                dcv = shr_f32x4(zero_v, dcv);
                let mut any_change = false;
                for (qi, dmx_qi) in dmx.iter_mut().enumerate() {
                    let dd_v =
                        f32x4::from_slice(&om.tfv[(dd_base + qi) * 4..(dd_base + qi) * 4 + 4]);
                    let old = *dmx_qi;
                    let new_val = old + dcv;
                    if new_val.simd_gt(old).any() {
                        any_change = true;
                    }
                    *dmx_qi = new_val;
                    dcv *= dd_v;
                }
                if !any_change {
                    break;
                }
            }
        }

        // Add D contributions to xE
        for &v in &dmx {
            xe_v += v;
        }

        // Horizontal sum of xe_v
        x_e = xe_v.reduce_sum();

        // Special states
        x_n *= om.xf[XST_N][XTR_LOOP];
        x_c = x_c * om.xf[XST_C][XTR_LOOP] + x_e * om.xf[XST_E][XTR_MOVE];
        x_j = x_j * om.xf[XST_J][XTR_LOOP] + x_e * om.xf[XST_E][XTR_LOOP];
        x_b = x_j * om.xf[XST_J][XTR_MOVE] + x_n * om.xf[XST_N][XTR_MOVE];

        // Sparse rescaling to prevent overflow
        if x_e > 1.0e4 {
            let inv = 1.0 / x_e;
            x_n *= inv;
            x_c *= inv;
            x_j *= inv;
            x_b *= inv;

            let inv_v = f32x4::splat(inv);
            for qi in 0..q {
                mmx[qi] *= inv_v;
                dmx[qi] *= inv_v;
                imx[qi] *= inv_v;
            }
            scale_factors[i + 1] = x_e;
            totscale += (x_e as f64).ln();
            x_e = 1.0;
        }

        xmx[i + 1] = [x_e, x_n, x_j, x_b, x_c];
    }

    let score = if l > 0 && x_c > 0.0 {
        totscale as f32 + (x_c * om.xf[XST_C][XTR_MOVE]).ln()
    } else {
        f32::NEG_INFINITY
    };

    ForwardResult {
        score,
        xmx,
        scale_factors,
        totscale,
    }
}

/// Right-shift a f32x4 by 1 lane: new = [0.0, v[0], v[1], v[2]].
/// This is the Farrar shift that propagates the last lane of the previous
/// q-vector into the first lane of the current one.
#[inline(always)]
fn shr_f32x4(zero: Simd<f32, 4>, v: Simd<f32, 4>) -> Simd<f32, 4> {
    // Reuse the caller's loop-invariant zero vector instead of constructing
    // another splat at every source-level shift.
    simd_swizzle!(v, zero, [4, 0, 1, 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::background::BackgroundModel;
    use crate::modelconfig;
    use crate::profile::Profile;
    use crate::test_helpers::*;

    fn setup_profile_and_om(m: usize, l: usize) -> (Profile, OptimizedProfile, Vec<u8>) {
        let abc = Alphabet::amino();
        let mut rng = crate::rng::XorShift64::new(42);
        let hmm = hmm_sample(&mut rng, m, &abc);
        let bg = BackgroundModel::new(&abc);
        let mut gm = Profile::new(m, &abc);
        modelconfig::profile_config(&hmm, &bg, &mut gm, l, crate::config::SearchMode::Local);
        modelconfig::reconfigure_length(&mut gm, l);

        let mut om = OptimizedProfile::from_profile(&gm);
        om.reconfigure_length(l);

        let dsq = random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, l);
        (gm, om, dsq)
    }

    #[test]
    fn test_simd_forward_produces_finite_score() {
        let (_, om, dsq) = setup_profile_and_om(20, 50);
        let result = forward_filter(&dsq, dsq.len(), &om);
        assert!(
            result.score.is_finite(),
            "SIMD forward score should be finite, got {}",
            result.score
        );
    }

    #[test]
    fn test_simd_forward_xmx_length() {
        let (_, om, dsq) = setup_profile_and_om(20, 50);
        let result = forward_filter(&dsq, dsq.len(), &om);
        assert_eq!(result.xmx.len(), dsq.len() + 1);
        assert_eq!(result.scale_factors.len(), dsq.len() + 1);
    }
}
