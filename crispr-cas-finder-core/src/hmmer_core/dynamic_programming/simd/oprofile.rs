// simd/oprofile.rs — Optimized (SIMD-striped) score profile
//
// Port of HMMER's P7_OPROFILE using Farrar's striped layout.
//
// Model positions 1..M are interleaved across Q SIMD vectors.
// Position k maps to vector q = (k-1) % Q, lane z = (k-1) / Q.
//
// Three sets of tables:
//   Forward: f32 odds ratios in 4-lane vectors (Q4 = max(2, ceil((M-1)/4) + 1))
//   MSV:     HMMER's byte costs (msv_filter.rs)
//   Viterbi: HMMER's 16-bit words (viterbi_filter.rs)

use super::int_msv::Isa;
use super::msv_filter::MsvBytes;
use super::viterbi_filter::ViterbiWords;
use crate::hmmer_core::config::*;
use crate::hmmer_core::profile::Profile;

// ---------- layout helpers ----------

/// Number of f32x4 vectors of the Forward tables.
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

    // --- Instruction set of the integer kernels (see int_msv.rs) ---
    // Private: the kernels compiled for it may only run where it is
    // available, which `new_with_isa` checks.
    isa: Isa,

    // --- HMMER's own filter tables: the scores its MSV and Viterbi filters
    // return decide the filter cascade (msv_filter.rs, viterbi_filter.rs) ---
    pub msv_bytes: MsvBytes,
    pub viterbi_words: ViterbiWords,

    // E-value parameters
    pub ev_params: [f32; NUM_EV_PARAMS],
    pub mode: SearchMode,
    pub max_length: usize,
}

impl OptimizedProfile {
    /// An empty profile of `m` nodes over an alphabet of `kp` symbols, for
    /// the integer kernels of `isa`, or of the portable kernels when this
    /// processor lacks it.
    pub fn new_with_isa(m: usize, kp: usize, isa: Isa) -> Self {
        let isa = if isa.is_available() {
            isa
        } else {
            Isa::Portable
        };
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
            isa,
            msv_bytes: MsvBytes::default(),
            viterbi_words: ViterbiWords::default(),
            ev_params: [EV_PARAM_UNSET; NUM_EV_PARAMS],
            mode: SearchMode::Local,
            max_length: 0,
        }
    }

    /// Build an OptimizedProfile from a generic Profile, for the instruction
    /// set this processor runs a model of its length best with: the Forward
    /// tables and those of HMMER's MSV and Viterbi filters.
    pub fn from_profile(gm: &Profile) -> Self {
        Self::from_profile_with_isa(gm, Isa::for_model(gm.num_nodes))
    }

    /// [`OptimizedProfile::from_profile`] for the integer kernels of `isa`,
    /// or of the portable kernels when this processor lacks it. The scores
    /// those kernels return do not depend on the instruction set.
    pub fn from_profile_with_isa(gm: &Profile, isa: Isa) -> Self {
        let kp = gm.alphabet.full_size;
        let mut om = OptimizedProfile::new_with_isa(gm.num_nodes, kp, isa);
        om.mode = gm.mode;
        om.l = gm.target_length;
        om.nj = gm.expected_j_uses;
        om.max_length = gm.max_length;
        om.ev_params = gm.ev_params;

        fwd_conversion(gm, &mut om);
        om.msv_bytes = MsvBytes::from_profile(gm, kp, om.isa);
        om.viterbi_words = ViterbiWords::from_profile(gm, kp, om.isa.lanes::<i16>());

        om
    }

    /// The instruction set the integer kernels run with.
    #[inline]
    pub fn isa(&self) -> Isa {
        self.isa
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
        self.msv_bytes.set_length(l);

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
