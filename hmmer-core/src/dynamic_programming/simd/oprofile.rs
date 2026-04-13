// simd/oprofile.rs — Optimized (SIMD-striped) score profile
//
// Port of HMMER's P7_OPROFILE using Farrar's striped layout.
//
// Model positions 1..M are interleaved across Q SIMD vectors.
// Position k maps to vector q = (k-1) % Q, lane z = (k-1) / Q.
//
// Three precision tiers:
//   MSV:     u8   in 16-lane vectors (Q16 = max(2, ceil((M-1)/16) + 1))
//   Viterbi: i16  in 8-lane vectors  (Q8  = max(2, ceil((M-1)/8)  + 1))
//   Forward: f32  in 4-lane vectors  (Q4  = max(2, ceil((M-1)/4)  + 1))

use crate::config::*;
use crate::profile::Profile;

// ---------- layout helpers ----------

/// Number of u8x16 vectors for MSV.
#[inline]
pub fn nqb(m: usize) -> usize {
    2.max(m.saturating_sub(1) / 16 + 1)
}

/// Number of i16x8 vectors for Viterbi.
#[inline]
pub fn nqw(m: usize) -> usize {
    2.max(m.saturating_sub(1) / 8 + 1)
}

/// Number of f32x4 vectors for Forward.
#[inline]
pub fn nqf(m: usize) -> usize {
    2.max(m.saturating_sub(1) / 4 + 1)
}

// Transition indices within the per-q block (7 regular + DD at end)
pub const TSC_BM: usize = 0;
pub const TSC_MM: usize = 1;
pub const TSC_IM: usize = 2;
pub const TSC_DM: usize = 3;
pub const TSC_MD: usize = 4;
pub const TSC_MI: usize = 5;
pub const TSC_II: usize = 6;
pub const NTSC_PER_Q: usize = 7;

// Special state indices
pub const XST_E: usize = 0;
pub const XST_N: usize = 1;
pub const XST_J: usize = 2;
pub const XST_C: usize = 3;
pub const NUM_XSTATES: usize = 4;

pub const XTR_MOVE: usize = 0;
pub const XTR_LOOP: usize = 1;

/// SIMD-optimized score profile (Farrar-striped layout).
#[derive(Debug, Clone)]
pub struct OptimizedProfile {
    pub m: usize,
    pub l: usize,
    pub nj: f32,

    pub kp: usize, // full alphabet size

    pub q16: usize,
    pub q8: usize,
    pub q4: usize,

    // --- MSV (u8) ---
    pub rbv: Vec<Vec<u8>>, // [kp][q16 * 16]
    pub tbm_b: u8,
    pub tec_b: u8,
    pub tjb_b: u8,
    pub scale_b: f32,
    pub base_b: u8,
    pub bias_b: u8,

    // --- Viterbi (i16) ---
    pub rwv: Vec<Vec<i16>>, // [kp][q8 * 8]
    pub twv: Vec<i16>,      // [(7*q8 + q8) * 8]
    pub xw: [[i16; 2]; NUM_XSTATES],
    pub scale_w: f32,
    pub base_w: i16,
    pub ddbound_w: i16,

    // --- Forward (f32 odds ratios) ---
    pub rfv: Vec<Vec<f32>>, // [kp][q4 * 4] — match emission odds
    pub riv: Vec<Vec<f32>>, // [kp][q4 * 4] — insert emission odds
    pub tfv: Vec<f32>,      // [(7*q4 + q4) * 4]
    pub xf: [[f32; 2]; NUM_XSTATES],

    // --- Log-space f32x4 (MSV/Viterbi exact scores) ---
    pub rlv: Vec<Vec<f32>>, // [kp][q4 * 4] — match emission log-scores
    pub tlv: Vec<f32>,      // [(7*q4 + q4) * 4] — transition log-scores

    // E-value parameters
    pub ev_params: [f32; NUM_EV_PARAMS],
    pub mode: SearchMode,
    pub max_length: usize,
}

impl OptimizedProfile {
    pub fn new(m: usize, kp: usize) -> Self {
        let q16 = nqb(m);
        let q8 = nqw(m);
        let q4 = nqf(m);

        OptimizedProfile {
            m,
            l: 0,
            nj: 0.0,
            kp,
            q16,
            q8,
            q4,
            rbv: vec![vec![0u8; q16 * 16]; kp],
            tbm_b: 0,
            tec_b: 0,
            tjb_b: 0,
            scale_b: 0.0,
            base_b: 0,
            bias_b: 0,
            rwv: vec![vec![0i16; q8 * 8]; kp],
            twv: vec![-32768i16; (NTSC_PER_Q * q8 + q8) * 8],
            xw: [[0i16; 2]; NUM_XSTATES],
            scale_w: 0.0,
            base_w: 0,
            ddbound_w: -32768,
            rfv: vec![vec![0.0f32; q4 * 4]; kp],
            riv: vec![vec![0.0f32; q4 * 4]; kp],
            tfv: vec![0.0f32; (NTSC_PER_Q * q4 + q4) * 4],
            xf: [[0.0f32; 2]; NUM_XSTATES],
            rlv: vec![vec![f32::NEG_INFINITY; q4 * 4]; kp],
            tlv: vec![f32::NEG_INFINITY; (NTSC_PER_Q * q4 + q4) * 4],
            ev_params: [EV_PARAM_UNSET; NUM_EV_PARAMS],
            mode: SearchMode::Local,
            max_length: 0,
        }
    }

    /// Build an OptimizedProfile from a generic Profile.
    pub fn from_profile(gm: &Profile) -> Self {
        let kp = gm.alphabet.full_size;
        let mut om = OptimizedProfile::new(gm.num_nodes, kp);
        om.mode = gm.mode;
        om.l = gm.target_length;
        om.nj = gm.expected_j_uses;
        om.max_length = gm.max_length;
        om.ev_params = gm.ev_params;

        msv_conversion(gm, &mut om);
        vit_conversion(gm, &mut om);
        fwd_conversion(gm, &mut om);
        log_conversion(gm, &mut om);

        om
    }

    /// Reconfigure length model for target length L.
    pub fn reconfigure_length(&mut self, l: usize) {
        // MSV
        self.tjb_b = unbiased_byteify(self.scale_b, (3.0 / (l as f32 + 3.0)).ln());

        // Forward & Viterbi
        let pmove = (2.0 + self.nj) / (l as f32 + 2.0 + self.nj);
        let ploop = 1.0 - pmove;

        self.xf[XST_N][XTR_LOOP] = ploop;
        self.xf[XST_N][XTR_MOVE] = pmove;
        self.xf[XST_C][XTR_LOOP] = ploop;
        self.xf[XST_C][XTR_MOVE] = pmove;
        self.xf[XST_J][XTR_LOOP] = ploop;
        self.xf[XST_J][XTR_MOVE] = pmove;

        self.xw[XST_N][XTR_MOVE] = wordify(self.scale_w, pmove.ln());
        self.xw[XST_C][XTR_MOVE] = wordify(self.scale_w, pmove.ln());
        self.xw[XST_J][XTR_MOVE] = wordify(self.scale_w, pmove.ln());
        self.xw[XST_N][XTR_LOOP] = 0;
        self.xw[XST_C][XTR_LOOP] = 0;
        self.xw[XST_J][XTR_LOOP] = 0;

        self.l = l;
    }

    /// Reconfigure specials for unihit (single domain) mode at length Ld.
    /// Used during domain envelope rescoring.
    pub fn reconfig_unihit(&mut self, ld: usize) {
        let ld_f = ld as f32;
        let pmove = 1.0 / (ld_f + 1.0);
        let ploop = ld_f / (ld_f + 1.0);

        self.xf[XST_N][XTR_LOOP] = ploop;
        self.xf[XST_N][XTR_MOVE] = pmove;
        self.xf[XST_C][XTR_LOOP] = ploop;
        self.xf[XST_C][XTR_MOVE] = pmove;
        self.xf[XST_E][XTR_MOVE] = 1.0; // exp(0) = 1
        self.xf[XST_E][XTR_LOOP] = 0.0; // exp(-inf) = 0
        self.xf[XST_J][XTR_LOOP] = 0.0; // exp(-inf) = 0
        self.xf[XST_J][XTR_MOVE] = 0.0; // exp(-inf) = 0
    }

    /// Restore specials for multihit mode at length L.
    pub fn reconfig_multihit(&mut self, l: usize) {
        let pmove = (2.0 + self.nj) / (l as f32 + 2.0 + self.nj);
        let ploop = 1.0 - pmove;

        self.xf[XST_N][XTR_LOOP] = ploop;
        self.xf[XST_N][XTR_MOVE] = pmove;
        self.xf[XST_C][XTR_LOOP] = ploop;
        self.xf[XST_C][XTR_MOVE] = pmove;
        self.xf[XST_J][XTR_LOOP] = ploop;
        self.xf[XST_J][XTR_MOVE] = pmove;
        self.xf[XST_E][XTR_MOVE] = 0.5_f32; // exp(ln(0.5))
        self.xf[XST_E][XTR_LOOP] = 0.5_f32;
    }
}

// ---------- Score conversion helpers ----------

fn biased_byteify(scale: f32, bias: u8, sc: f32) -> u8 {
    let s = -(scale * sc).round();
    let b = if s > 255.0 - bias as f32 {
        255.0
    } else {
        s + bias as f32
    };
    b as u8
}

fn unbiased_byteify(scale: f32, sc: f32) -> u8 {
    let s = -(scale * sc).round();
    if s > 255.0 { 255 } else { s as u8 }
}

fn wordify(scale: f32, sc: f32) -> i16 {
    let s = (scale * sc).round();
    s.clamp(-32768.0, 32767.0) as i16
}

// ---------- MSV conversion ----------

fn msv_conversion(gm: &Profile, om: &mut OptimizedProfile) {
    let m = gm.num_nodes;
    let nq = om.q16;
    let abc_k = gm.alphabet.canonical_size;
    let kp = om.kp;

    // Determine max emission score → bias
    let mut max_sc = 0.0f32;
    for x in 0..abc_k {
        let rsc = gm.residue_scores_for(x);
        for k in 1..=m {
            let sc = rsc[k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize];
            if sc > max_sc {
                max_sc = sc;
            }
        }
    }
    om.scale_b = 3.0 / std::f32::consts::LN_2;
    om.base_b = 190;
    om.bias_b = unbiased_byteify(om.scale_b, -max_sc);

    // Striped match emission scores
    for x in 0..kp {
        let rsc = gm.residue_scores_for(x);
        let rsc_len = rsc.len();
        for q in 0..nq {
            let k_base = q + 1; // k = k_base + z * nq
            for z in 0..16 {
                let k = k_base + z * nq;
                let idx = q * 16 + z;
                if k <= m {
                    let score_idx = k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize;
                    if score_idx < rsc_len {
                        om.rbv[x][idx] = biased_byteify(om.scale_b, om.bias_b, rsc[score_idx]);
                    } else {
                        om.rbv[x][idx] = 255;
                    }
                } else {
                    om.rbv[x][idx] = 255; // -inf in biased uint8
                }
            }
        }
    }

    // Transition costs
    om.tbm_b = unbiased_byteify(
        om.scale_b,
        (2.0 / (gm.num_nodes as f32 * (gm.num_nodes as f32 + 1.0))).ln(),
    );
    om.tec_b = unbiased_byteify(om.scale_b, 0.5f32.ln());
    om.tjb_b = unbiased_byteify(om.scale_b, (3.0 / (gm.target_length as f32 + 3.0)).ln());
}

// ---------- Viterbi conversion ----------

fn vit_conversion(gm: &Profile, om: &mut OptimizedProfile) {
    let m = gm.num_nodes;
    let nq = om.q8;
    let kp = om.kp;
    let tsc = gm.transition_scores_raw();

    om.scale_w = 500.0 / std::f32::consts::LN_2;
    om.base_w = 12000;

    // Striped match emission scores
    for x in 0..kp {
        let rsc = gm.residue_scores_for(x);
        let rsc_len = rsc.len();
        for q in 0..nq {
            let k_base = q + 1;
            for z in 0..8 {
                let k = k_base + z * nq;
                let idx = q * 8 + z;
                if k <= m {
                    let score_idx = k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize;
                    if score_idx < rsc_len {
                        om.rwv[x][idx] = wordify(om.scale_w, rsc[score_idx]);
                    } else {
                        om.rwv[x][idx] = -32768;
                    }
                } else {
                    om.rwv[x][idx] = -32768;
                }
            }
        }
    }

    // Transition scores: 7 per q-block, then DD
    let tsc_len = tsc.len();
    let mut j = 0usize;

    // Map from archive t-index to current PTsc enum
    let t_map: [(usize, i32, i16); 7] = [
        (PTsc::BeginToMatch.idx(), -1, 0),
        (PTsc::MatchToMatch.idx(), -1, 0),
        (PTsc::InsertToMatch.idx(), -1, 0),
        (PTsc::DeleteToMatch.idx(), -1, 0),
        (PTsc::MatchToDelete.idx(), 0, 0),
        (PTsc::MatchToInsert.idx(), 0, 0),
        (PTsc::InsertToInsert.idx(), 0, -1),
    ];

    for q in 0..nq {
        let k_base = q + 1;
        for &(tg, kb_offset, maxval) in &t_map {
            for z in 0..8 {
                let kb = (k_base as i32 + kb_offset + z as i32 * nq as i32) as usize;
                let idx = j * 8 + z;
                let tsc_idx = kb * PROFILE_NUM_TRANSITIONS + tg;
                if kb < m && tsc_idx < tsc_len {
                    let val = wordify(om.scale_w, tsc[tsc_idx]);
                    om.twv[idx] = if val <= maxval { val } else { maxval };
                } else {
                    om.twv[idx] = -32768;
                }
            }
            j += 1;
        }
    }

    // DD transitions at end
    for q in 0..nq {
        let k_base = q + 1;
        for z in 0..8 {
            let k = k_base + z * nq;
            let idx = j * 8 + z;
            let tsc_idx = k * PROFILE_NUM_TRANSITIONS + PTsc::DeleteToDelete.idx();
            if k < m && tsc_idx < tsc_len {
                om.twv[idx] = wordify(om.scale_w, tsc[tsc_idx]);
            } else {
                om.twv[idx] = -32768;
            }
        }
        j += 1;
    }

    // Special state transitions
    let ss = &gm.special_scores;
    om.xw[XST_E][XTR_LOOP] = wordify(om.scale_w, ss.e_loop);
    om.xw[XST_E][XTR_MOVE] = wordify(om.scale_w, ss.e_move);
    om.xw[XST_N][XTR_MOVE] = wordify(om.scale_w, ss.n_move);
    om.xw[XST_N][XTR_LOOP] = 0;
    om.xw[XST_C][XTR_MOVE] = wordify(om.scale_w, ss.c_move);
    om.xw[XST_C][XTR_LOOP] = 0;
    om.xw[XST_J][XTR_MOVE] = wordify(om.scale_w, ss.j_move);
    om.xw[XST_J][XTR_LOOP] = 0;

    // DD bound for lazy-F evaluation
    om.ddbound_w = -32768;
    for k in 2..m.saturating_sub(1) {
        let dd_idx = k * PROFILE_NUM_TRANSITIONS + PTsc::DeleteToDelete.idx();
        let dm_idx = (k + 1) * PROFILE_NUM_TRANSITIONS + PTsc::DeleteToMatch.idx();
        let bm_idx = (k + 1) * PROFILE_NUM_TRANSITIONS + PTsc::BeginToMatch.idx();
        if dd_idx < tsc_len && dm_idx < tsc_len && bm_idx < tsc_len {
            let dd = wordify(om.scale_w, tsc[dd_idx]) as i32;
            let dm = wordify(om.scale_w, tsc[dm_idx]) as i32;
            let bm = wordify(om.scale_w, tsc[bm_idx]) as i32;
            let ddtmp = dd + dm - bm;
            if ddtmp > om.ddbound_w as i32 {
                om.ddbound_w = ddtmp.min(32767) as i16;
            }
        }
    }
}

// ---------- Forward/Backward (f32 odds) conversion ----------

fn fwd_conversion(gm: &Profile, om: &mut OptimizedProfile) {
    let m = gm.num_nodes;
    let nq = om.q4;
    let kp = om.kp;
    let tsc = gm.transition_scores_raw();
    let tsc_len = tsc.len();

    // Striped match emission odds ratios
    for x in 0..kp {
        let rsc = gm.residue_scores_for(x);
        let rsc_len = rsc.len();
        for q in 0..nq {
            let k_base = q + 1;
            for z in 0..4 {
                let k = k_base + z * nq;
                let idx = q * 4 + z;
                if k <= m {
                    let match_idx = k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize;
                    let ins_idx = k * PROFILE_NUM_EMISSIONS + PRsc::InsertScore as usize;
                    if match_idx < rsc_len {
                        om.rfv[x][idx] = rsc[match_idx].exp();
                    } else {
                        om.rfv[x][idx] = 0.0;
                    }
                    if ins_idx < rsc_len && k < m {
                        om.riv[x][idx] = rsc[ins_idx].exp();
                    } else {
                        om.riv[x][idx] = 0.0;
                    }
                } else {
                    om.rfv[x][idx] = 0.0;
                    om.riv[x][idx] = 0.0;
                }
            }
        }
    }

    // Transition odds ratios: 7 per q-block, then DD
    let t_map: [(usize, i32); 7] = [
        (PTsc::BeginToMatch.idx(), -1),
        (PTsc::MatchToMatch.idx(), -1),
        (PTsc::InsertToMatch.idx(), -1),
        (PTsc::DeleteToMatch.idx(), -1),
        (PTsc::MatchToDelete.idx(), 0),
        (PTsc::MatchToInsert.idx(), 0),
        (PTsc::InsertToInsert.idx(), 0),
    ];

    let mut j = 0usize;
    for q in 0..nq {
        let k_base = q + 1;
        for &(tg, kb_offset) in &t_map {
            for z in 0..4 {
                let kb = (k_base as i32 + kb_offset + z as i32 * nq as i32) as usize;
                let idx = j * 4 + z;
                let tsc_idx = kb * PROFILE_NUM_TRANSITIONS + tg;
                if kb < m && tsc_idx < tsc_len {
                    om.tfv[idx] = tsc[tsc_idx].exp();
                } else {
                    om.tfv[idx] = 0.0;
                }
            }
            j += 1;
        }
    }

    // DD transitions at end
    for q in 0..nq {
        let k_base = q + 1;
        for z in 0..4 {
            let k = k_base + z * nq;
            let idx = j * 4 + z;
            let tsc_idx = k * PROFILE_NUM_TRANSITIONS + PTsc::DeleteToDelete.idx();
            if k < m && tsc_idx < tsc_len {
                om.tfv[idx] = tsc[tsc_idx].exp();
            } else {
                om.tfv[idx] = 0.0;
            }
        }
        j += 1;
    }

    // Special state transitions (probability space)
    let ss = &gm.special_scores;
    om.xf[XST_E][XTR_LOOP] = ss.e_loop.exp();
    om.xf[XST_E][XTR_MOVE] = ss.e_move.exp();
    om.xf[XST_N][XTR_LOOP] = ss.n_loop.exp();
    om.xf[XST_N][XTR_MOVE] = ss.n_move.exp();
    om.xf[XST_C][XTR_LOOP] = ss.c_loop.exp();
    om.xf[XST_C][XTR_MOVE] = ss.c_move.exp();
    om.xf[XST_J][XTR_LOOP] = ss.j_loop.exp();
    om.xf[XST_J][XTR_MOVE] = ss.j_move.exp();
}

// ---------- Log-space f32x4 (exact scores for MSV/Viterbi) ----------

fn log_conversion(gm: &Profile, om: &mut OptimizedProfile) {
    let m = gm.num_nodes;
    let nq = om.q4;
    let kp = om.kp;
    let tsc = gm.transition_scores_raw();
    let tsc_len = tsc.len();
    let neg_inf = f32::NEG_INFINITY;

    // Striped match emission log-scores (NOT exponentiated)
    for x in 0..kp {
        let rsc = gm.residue_scores_for(x);
        let rsc_len = rsc.len();
        for q in 0..nq {
            let k_base = q + 1;
            for z in 0..4 {
                let k = k_base + z * nq;
                let idx = q * 4 + z;
                if k <= m {
                    let score_idx = k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize;
                    if score_idx < rsc_len {
                        om.rlv[x][idx] = rsc[score_idx];
                    } else {
                        om.rlv[x][idx] = neg_inf;
                    }
                } else {
                    om.rlv[x][idx] = neg_inf;
                }
            }
        }
    }

    // Transition log-scores: same Farrar layout as tfv
    let t_map: [(usize, i32); 7] = [
        (PTsc::BeginToMatch.idx(), -1),
        (PTsc::MatchToMatch.idx(), -1),
        (PTsc::InsertToMatch.idx(), -1),
        (PTsc::DeleteToMatch.idx(), -1),
        (PTsc::MatchToDelete.idx(), 0),
        (PTsc::MatchToInsert.idx(), 0),
        (PTsc::InsertToInsert.idx(), 0),
    ];

    let mut j = 0usize;
    for q in 0..nq {
        let k_base = q + 1;
        for &(tg, kb_offset) in &t_map {
            for z in 0..4 {
                let kb = (k_base as i32 + kb_offset + z as i32 * nq as i32) as usize;
                let idx = j * 4 + z;
                let tsc_idx = kb * PROFILE_NUM_TRANSITIONS + tg;
                if kb < m && tsc_idx < tsc_len {
                    om.tlv[idx] = tsc[tsc_idx];
                } else {
                    om.tlv[idx] = neg_inf;
                }
            }
            j += 1;
        }
    }

    // DD transitions at end
    for q in 0..nq {
        let k_base = q + 1;
        for z in 0..4 {
            let k = k_base + z * nq;
            let idx = j * 4 + z;
            let tsc_idx = k * PROFILE_NUM_TRANSITIONS + PTsc::DeleteToDelete.idx();
            if k < m && tsc_idx < tsc_len {
                om.tlv[idx] = tsc[tsc_idx];
            } else {
                om.tlv[idx] = neg_inf;
            }
        }
        j += 1;
    }
}
