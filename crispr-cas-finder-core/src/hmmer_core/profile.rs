// p7_profile.rs - Scoring profile and its implicit model
//
// Port of src/p7_profile.c

use crate::hmmer_core::alphabet::Alphabet;
use crate::hmmer_core::config::*;
use std::sync::Arc;

/// Special-state transition scores (E, N, J, C × loop/move).
#[derive(Debug, Clone, Copy)]
pub struct SpecialScores {
    pub e_loop: f32,
    pub e_move: f32,
    pub n_loop: f32,
    pub n_move: f32,
    pub j_loop: f32,
    pub j_move: f32,
    pub c_loop: f32,
    pub c_move: f32,
}

impl Default for SpecialScores {
    fn default() -> Self {
        Self {
            e_loop: 0.0,
            e_move: 0.0,
            n_loop: 0.0,
            n_move: 0.0,
            j_loop: 0.0,
            j_move: 0.0,
            c_loop: 0.0,
            c_move: 0.0,
        }
    }
}

impl SpecialScores {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A scoring profile, configured from an HMM for local alignment and a target length.
///
/// Corresponds to P7_PROFILE in the C code.
///
/// The two large score arrays (`transition_scores`, `residue_scores`) are stored behind an
/// `Arc` so that `clone()` — used to create per-thread copies in parallel search — is cheap:
/// it bumps two reference counts instead of copying ~76 KB.  The arrays are written only
/// during `profile_config()`, before any clone happens.  The mutable accessor methods use
/// `Arc::get_mut()` and will panic if called after the Arc has been shared.
#[derive(Debug)]
pub struct Profile {
    transition_scores: Arc<Vec<f32>>, // transitions [0..M-1][0..p7P_NTRANS-1], flat
    residue_scores: Arc<Vec<f32>>, // emissions, residue-major flat: [residue * rsc_stride + k * P7P_NR + score_type]
    residue_score_stride: usize,   // (m+1) * P7P_NR per residue
    pub special_scores: SpecialScores, // special transitions

    pub target_length: usize, // current target seq length
    pub num_nodes: usize,     // number of nodes
    pub expected_j_uses: f32, // expected # of J uses

    // Info copied from parent HMM
    pub ev_params: [f32; NUM_EV_PARAMS],
    pub model_composition: [f32; MAX_CANONICAL_ALPHABET],

    pub alphabet: Alphabet,
}

impl Profile {
    /// Create a new profile for M nodes and the given alphabet.
    pub fn new(m: usize, abc: &Alphabet) -> Self {
        let kp = abc.full_size;
        let rsc_stride = (m + 1) * PROFILE_NUM_EMISSIONS;
        Profile {
            transition_scores: Arc::new(vec![0.0; m * PROFILE_NUM_TRANSITIONS]),
            residue_scores: Arc::new(vec![0.0; kp * rsc_stride]),
            residue_score_stride: rsc_stride,
            special_scores: SpecialScores::new(),
            target_length: 0,
            num_nodes: m,
            expected_j_uses: 0.0,
            ev_params: [EV_PARAM_UNSET; NUM_EV_PARAMS],
            model_composition: [COMPOSITION_UNSET; MAX_CANONICAL_ALPHABET],
            alphabet: abc.clone(),
        }
    }

    // -- Transition score accessors --

    /// Mutable transition scores for node k.  Panics if the score array is shared.
    #[inline]
    pub fn transition_scores_for_node_mut(&mut self, k: usize) -> &mut [f32] {
        let ts = Arc::get_mut(&mut self.transition_scores)
            .expect("Profile::transition_scores_for_node_mut called after clone");
        let start = (k - 1) * PROFILE_NUM_TRANSITIONS;
        &mut ts[start..start + PROFILE_NUM_TRANSITIONS]
    }

    /// Get a specific transition score.
    #[inline]
    pub fn transition_score_at(&self, k: usize, t: PTsc) -> f32 {
        self.transition_scores[(k - 1) * PROFILE_NUM_TRANSITIONS + t as usize]
    }

    /// The full tsc array as a slice (for DP inner loops that cache a reference).
    #[inline]
    pub fn transition_scores_raw(&self) -> &[f32] {
        &self.transition_scores
    }

    /// Ensure tsc has room for m nodes.  Panics if the score array is shared.
    pub fn resize_transition_scores(&mut self, m: usize) {
        if self.transition_scores.len() < m * PROFILE_NUM_TRANSITIONS {
            let ts = Arc::get_mut(&mut self.transition_scores)
                .expect("Profile::resize_transition_scores called after clone");
            ts.resize(m * PROFILE_NUM_TRANSITIONS, 0.0);
        }
    }

    // -- Residue emission score accessors --

    /// All emission scores for a given residue (match+insert for all nodes).
    /// Returns slice of length (M+1)*P7P_NR.
    #[inline]
    pub fn residue_scores_for(&self, x: usize) -> &[f32] {
        let start = x * self.residue_score_stride;
        &self.residue_scores[start..start + self.residue_score_stride]
    }

    /// Mutable emission scores for a given residue.  Panics if the score array is shared.
    #[inline]
    pub fn residue_scores_for_mut(&mut self, x: usize) -> &mut [f32] {
        let rs = Arc::get_mut(&mut self.residue_scores)
            .expect("Profile::residue_scores_for_mut called after clone");
        let start = x * self.residue_score_stride;
        &mut rs[start..start + self.residue_score_stride]
    }

    /// Ensure rsc has room for kp residues × (m+1) nodes.  Panics if the score array is shared.
    pub fn resize_residue_scores(&mut self, kp: usize, m: usize) {
        let stride = (m + 1) * PROFILE_NUM_EMISSIONS;
        self.residue_score_stride = stride;
        let needed = kp * stride;
        if self.residue_scores.len() < needed {
            let rs = Arc::get_mut(&mut self.residue_scores)
                .expect("Profile::resize_residue_scores called after clone");
            rs.resize(needed, 0.0);
        }
    }

    // -- Length model reconfiguration --

    /// Configure the length model for a target of length `target_length`.
    ///
    /// Sets N/C loop/move scores from the length distribution, and J loop/move
    /// if the profile is in multihit mode (`expected_j_uses > 0`).
    pub fn config_length_model(&mut self, target_length: usize) {
        let l = target_length as f32;
        let pmove = (2.0 + self.expected_j_uses) / (l + 2.0 + self.expected_j_uses);
        let ploop = 1.0 - pmove;
        self.special_scores.n_loop = ploop.ln();
        self.special_scores.n_move = pmove.ln();
        self.special_scores.c_loop = ploop.ln();
        self.special_scores.c_move = pmove.ln();
        if self.expected_j_uses > 0.0 {
            self.special_scores.j_loop = ploop.ln();
            self.special_scores.j_move = pmove.ln();
        }
        self.target_length = target_length;
    }

    /// Reconfigure the length model for a new target length.
    pub fn reconfigure_length(&mut self, target_length: usize) {
        self.config_length_model(target_length);
    }

    /// Reconfigure to unihit mode for a domain of length `target_length`.
    ///
    /// Port of `p7_ReconfigUnihit()`.
    pub fn reconfig_unihit(&mut self, target_length: usize) {
        let l = target_length as f32;
        let pmove = 2.0 / (l + 2.0);
        let ploop = 1.0 - pmove;
        self.special_scores.n_loop = ploop.ln();
        self.special_scores.n_move = pmove.ln();
        self.special_scores.c_loop = ploop.ln();
        self.special_scores.c_move = pmove.ln();
        self.special_scores.e_move = 0.0;
        self.special_scores.e_loop = f32::NEG_INFINITY;
        self.special_scores.j_loop = f32::NEG_INFINITY;
        self.special_scores.j_move = f32::NEG_INFINITY;
        self.expected_j_uses = 0.0;
        self.target_length = target_length;
    }

    /// Reconfigure to multihit mode for a target sequence of length `target_length`.
    ///
    /// Port of `p7_ReconfigMultihit()`.
    pub fn reconfig_multihit(&mut self, target_length: usize) {
        let l = target_length as f32;
        let pmove = 3.0 / (l + 3.0);
        let ploop = 1.0 - pmove;
        self.special_scores.n_loop = ploop.ln();
        self.special_scores.n_move = pmove.ln();
        self.special_scores.c_loop = ploop.ln();
        self.special_scores.c_move = pmove.ln();
        self.special_scores.j_loop = ploop.ln();
        self.special_scores.j_move = pmove.ln();
        self.special_scores.e_move = (0.5f32).ln();
        self.special_scores.e_loop = (0.5f32).ln();
        self.expected_j_uses = 1.0;
        self.target_length = target_length;
    }
}

/// Manual `Clone` impl: the two large score Vecs are shared via `Arc` (ref-count bump only),
/// while the small mutable fields are copied.  This makes per-thread profile clones cheap.
impl Clone for Profile {
    fn clone(&self) -> Self {
        Profile {
            transition_scores: Arc::clone(&self.transition_scores),
            residue_scores: Arc::clone(&self.residue_scores),
            residue_score_stride: self.residue_score_stride,
            special_scores: self.special_scores,
            target_length: self.target_length,
            num_nodes: self.num_nodes,
            expected_j_uses: self.expected_j_uses,
            ev_params: self.ev_params,
            model_composition: self.model_composition,
            alphabet: self.alphabet.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create() {
        let abc = Alphabet::amino();
        let gm = Profile::new(100, &abc);
        assert_eq!(gm.num_nodes, 100);
        assert_eq!(
            gm.transition_scores_raw().len(),
            100 * PROFILE_NUM_TRANSITIONS
        );
    }
}
