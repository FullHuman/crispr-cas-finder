// simd/oprofile.rs — Optimized (SIMD-striped) score profile
//
// Port of HMMER's P7_OPROFILE using Farrar's striped layout.
//
// Model positions 1..M are interleaved across Q SIMD vectors.
// Position k maps to vector q = (k-1) % Q, lane z = (k-1) / Q.
//
// Two f32 precision tiers:
//   Forward: f32  in 4-lane vectors  (Q4  = max(2, ceil((M-1)/4)  + 1)) — odds space
//   Log:     f32  in 4-lane vectors  (Q4)  — log-space for MSV/Viterbi filters

use super::{i16_msv, i16_viterbi};
use crate::hmmer_core::config::*;
use crate::hmmer_core::profile::Profile;

// ---------- layout helpers ----------

/// Number of f32x4 vectors for Forward / log-space tiers.
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

    pub q4: usize,

    // --- Forward (f32 odds ratios) ---
    pub rfv: Vec<Vec<f32>>, // [kp][q4 * 4] — match emission odds
    pub riv: Vec<Vec<f32>>, // [kp][q4 * 4] — insert emission odds
    pub tfv: Vec<f32>,      // [(7*q4 + q4) * 4]
    pub xf: [[f32; 2]; NUM_XSTATES],

    // --- Log-space f32x4 (MSV/Viterbi exact scores) ---
    pub rlv: Vec<Vec<f32>>, // [kp][q4 * 4] — match emission log-scores
    pub tlv: Vec<f32>,      // [(7*q4 + q4) * 4] — transition log-scores

    // --- i16x8 MSV over-estimate (see i16_msv.rs) ---
    pub q8: usize,
    pub msv_scale: f64,           // integer units per nat
    pub msv_units: Vec<Vec<i16>>, // [kp][q8 * 8] — match emissions, rounded up
    pub vit_units: Vec<i16>,      // [(7*q8 + q8) * 8] — transitions, rounded up
    pub vit_min_begin_units: i16, // smallest finite begin transition

    // E-value parameters
    pub ev_params: [f32; NUM_EV_PARAMS],
    pub mode: SearchMode,
    pub max_length: usize,
}

impl OptimizedProfile {
    pub fn new(m: usize, kp: usize) -> Self {
        let q4 = nqf(m);

        OptimizedProfile {
            m,
            l: 0,
            nj: 0.0,
            kp,
            q4,
            rfv: vec![vec![0.0f32; q4 * 4]; kp],
            riv: vec![vec![0.0f32; q4 * 4]; kp],
            tfv: vec![0.0f32; (NTSC_PER_Q * q4 + q4) * 4],
            xf: [[0.0f32; 2]; NUM_XSTATES],
            rlv: vec![vec![f32::NEG_INFINITY; q4 * 4]; kp],
            tlv: vec![f32::NEG_INFINITY; (NTSC_PER_Q * q4 + q4) * 4],
            q8: i16_msv::vector_count(m),
            msv_scale: i16_msv::scale_for(m),
            msv_units: vec![vec![i16::MIN; i16_msv::vector_count(m) * i16_msv::LANES]; kp],
            vit_units: vec![i16::MIN; (NTSC_PER_Q + 1) * i16_msv::vector_count(m) * i16_msv::LANES],
            vit_min_begin_units: 0,
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

        fwd_conversion(gm, &mut om);
        log_conversion(gm, &mut om);
        msv_units_conversion(gm, &mut om);
        vit_units_conversion(gm, &mut om);

        om
    }

    /// Reconfigure length model for target length L.
    pub fn reconfigure_length(&mut self, l: usize) {
        let pmove = (2.0 + self.nj) / (l as f32 + 2.0 + self.nj);
        let ploop = 1.0 - pmove;

        self.xf[XST_N][XTR_LOOP] = ploop;
        self.xf[XST_N][XTR_MOVE] = pmove;
        self.xf[XST_C][XTR_LOOP] = ploop;
        self.xf[XST_C][XTR_MOVE] = pmove;
        self.xf[XST_J][XTR_LOOP] = ploop;
        self.xf[XST_J][XTR_MOVE] = pmove;

        self.l = l;
    }

    /// Reconfigure specials for unihit (single domain) mode at length Ld.
    /// Used during domain envelope rescoring.
    pub fn reconfig_unihit(&mut self, l: usize) {
        let l = l as f32;
        let pmove = 2.0 / (l + 2.0);
        let ploop = 1.0 - pmove;

        self.xf[XST_N][XTR_LOOP] = ploop;
        self.xf[XST_N][XTR_MOVE] = pmove;
        self.xf[XST_C][XTR_LOOP] = ploop;
        self.xf[XST_C][XTR_MOVE] = pmove;
        self.xf[XST_E][XTR_MOVE] = 1.0; // exp(0) = 1
        self.xf[XST_E][XTR_LOOP] = 0.0; // exp(-inf) = 0
        self.xf[XST_J][XTR_LOOP] = 0.0; // exp(-inf) = 0
        self.xf[XST_J][XTR_MOVE] = 0.0; // exp(-inf) = 0
        self.nj = 0.0;
    }

    /// Restore specials for multihit mode at length L.
    pub fn reconfig_multihit(&mut self, l: usize) {
        let pmove = 3.0 / (l as f32 + 3.0);
        let ploop = 1.0 - pmove;

        self.xf[XST_N][XTR_LOOP] = ploop;
        self.xf[XST_N][XTR_MOVE] = pmove;
        self.xf[XST_C][XTR_LOOP] = ploop;
        self.xf[XST_C][XTR_MOVE] = pmove;
        self.xf[XST_J][XTR_LOOP] = ploop;
        self.xf[XST_J][XTR_MOVE] = pmove;
        self.xf[XST_E][XTR_MOVE] = 0.5_f32; // exp(ln(0.5))
        self.xf[XST_E][XTR_LOOP] = 0.5_f32;
        self.nj = 1.0;
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

// ---------- i16x8 match emissions for the MSV over-estimate ----------

fn msv_units_conversion(gm: &Profile, om: &mut OptimizedProfile) {
    let m = gm.num_nodes;
    let nq = om.q8;
    for x in 0..om.kp {
        let rsc = gm.residue_scores_for(x);
        for q in 0..nq {
            for z in 0..i16_msv::LANES {
                let k = q + 1 + z * nq;
                let score_idx = k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore as usize;
                if k <= m && score_idx < rsc.len() {
                    om.msv_units[x][q * i16_msv::LANES + z] =
                        i16_msv::emission_units(om.msv_scale, rsc[score_idx]);
                }
            }
        }
    }
}

// ---------- i16x8 transitions for the Viterbi over-estimate ----------

/// Same striped layout as `tlv`, with eight lanes: seven transitions per
/// vector, then the delete-to-delete transitions.
fn vit_units_conversion(gm: &Profile, om: &mut OptimizedProfile) {
    let m = gm.num_nodes;
    let nq = om.q8;
    let lanes = i16_msv::LANES;
    let tsc = gm.transition_scores_raw();
    let units = |index: usize| {
        tsc.get(index).map_or(i16::MIN, |&score| {
            i16_viterbi::transition_units(om.msv_scale, score)
        })
    };

    let t_map: [(usize, usize); NTSC_PER_Q] = [
        (PTsc::BeginToMatch.idx(), 1),
        (PTsc::MatchToMatch.idx(), 1),
        (PTsc::InsertToMatch.idx(), 1),
        (PTsc::DeleteToMatch.idx(), 1),
        (PTsc::MatchToDelete.idx(), 0),
        (PTsc::MatchToInsert.idx(), 0),
        (PTsc::InsertToInsert.idx(), 0),
    ];
    let mut min_begin = i16::MAX;
    let mut j = 0usize;
    for q in 0..nq {
        for (slot, &(kind, back)) in t_map.iter().enumerate() {
            for z in 0..lanes {
                // Transitions into node k are stored at node k - 1.
                let kb = q + 1 + z * nq - back;
                let value = if kb < m {
                    units(kb * PROFILE_NUM_TRANSITIONS + kind)
                } else {
                    i16::MIN
                };
                om.vit_units[j * lanes + z] = value;
                if slot == TSC_BM && value > i16::MIN {
                    min_begin = min_begin.min(value);
                }
            }
            j += 1;
        }
    }
    for q in 0..nq {
        for z in 0..lanes {
            let k = q + 1 + z * nq;
            om.vit_units[j * lanes + z] = if k < m {
                units(k * PROFILE_NUM_TRANSITIONS + PTsc::DeleteToDelete.idx())
            } else {
                i16::MIN
            };
        }
        j += 1;
    }
    om.vit_min_begin_units = if min_begin == i16::MAX { 0 } else { min_begin };
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
