// modelconfig.rs - Configuring a profile from an HMM
//
// Port of src/modelconfig.c

use std::f32::consts::LN_2;

use crate::alphabet::Alphabet;
use crate::background::BackgroundModel;
use crate::config::*;
use crate::hmm::Hmm;
use crate::profile::Profile;

/// Configure a search profile from an HMM.
pub fn profile_config(
    hmm: &Hmm,
    background: &BackgroundModel,
    profile: &mut Profile,
    target_length: usize,
    mode: SearchMode,
) {
    let m = hmm.num_nodes;
    let full_alphabet_size = hmm.alphabet.full_size;

    profile.mode = mode;
    profile.num_nodes = m;
    profile.max_length = hmm.max_length.unwrap_or(0);

    copy_metadata(hmm, profile);

    // Ensure score arrays are large enough before writing any scores.
    profile.resize_transition_scores(m);
    profile.resize_residue_scores(full_alphabet_size, m);

    configure_entry_scores(hmm, profile, mode);
    configure_e_state(profile, mode);
    configure_transition_scores(hmm, profile);
    configure_match_emissions(hmm, background, profile);
    configure_insert_emissions(profile, full_alphabet_size);

    // Force reconfiguration of the length model.
    profile.target_length = 0;
    config_length_model(profile, target_length);
}

/// Copy all metadata fields from an HMM into a profile.
fn copy_metadata(hmm: &Hmm, profile: &mut Profile) {
    profile.name = hmm.name.clone();
    profile.accession = hmm.accession.clone();
    profile.description = hmm.description.clone();
    profile.reference_annotation = hmm.reference_annotation.clone();
    profile.model_mask = hmm.model_mask.clone();
    profile.consensus_structure = hmm.consensus_structure.clone();
    profile.consensus = hmm.consensus.clone();

    profile.ev_params = hmm
        .ev_params
        .map_or([EV_PARAM_UNSET; NUM_EV_PARAMS], |e| e.to_array());
    profile.score_cutoffs = hmm.cutoffs.to_array();

    if let Some(ref compo) = hmm.model_composition {
        let n = MAX_CANONICAL_ALPHABET.min(compo.len());
        profile.model_composition[..n].copy_from_slice(&compo[..n]);
    }
}

/// Set begin-to-match entry scores (B->M_k transitions).
fn configure_entry_scores(hmm: &Hmm, profile: &mut Profile, mode: SearchMode) {
    if mode.is_local() {
        configure_local_entry_scores(hmm, profile);
    } else {
        configure_glocal_entry_scores(hmm, profile);
    }
}

/// Local mode: occupancy-weighted entry probabilities.
///
/// Entry score for node k is log( occ[k] / sum_i(occ[i] * (M - i + 1)) ).
/// Falls back to a uniform distribution if the occupancy sum is zero.
fn configure_local_entry_scores(hmm: &Hmm, profile: &mut Profile) {
    let m = hmm.num_nodes;
    let (occ, _) = hmm.calculate_occupancy();

    let occupancy_sum: f32 = occ[1..=m]
        .iter()
        .enumerate()
        .map(|(i, &occ_k)| occ_k * (m - i) as f32) // index i here == k-1, so (M - k + 1) == (M - i)
        .sum();

    if occupancy_sum > 0.0 {
        for (k, &occ_k) in occ.iter().enumerate().skip(1).take(m) {
            profile.transition_scores_for_node_mut(k)[PTsc::BeginToMatch.idx()] =
                (occ_k / occupancy_sum).ln();
        }
    } else {
        // Uniform fallback when all occupancies are zero.
        let uniform_entry_prob = 2.0 / (m * (m + 1)) as f32;
        let log_uniform = uniform_entry_prob.ln();
        for k in 1..=m {
            profile.transition_scores_for_node_mut(k)[PTsc::BeginToMatch.idx()] = log_uniform;
        }
    }
}

/// Glocal mode: left wing retraction entry scores.
///
/// Entry to node k goes through the silent delete path D_1..D_{k-1}.
fn configure_glocal_entry_scores(hmm: &Hmm, profile: &mut Profile) {
    let m = hmm.num_nodes;
    let t0 = hmm.transitions(0);
    let md0 = t0[HTransition::MatchToDelete as usize];

    // B->M_1: probability of taking the match branch at node 0
    profile.transition_scores_for_node_mut(1)[PTsc::BeginToMatch.idx()] = (1.0 - md0).ln();

    // B->M_{k+1} for k in 1..M-1: accumulated wing retraction score
    let mut wing_score = md0.ln();
    for k in 1..m {
        let t_k = hmm.transitions(k);
        profile.transition_scores_for_node_mut(k + 1)[PTsc::BeginToMatch.idx()] =
            wing_score + t_k[HTransition::DeleteToMatch.idx()].ln();
        wing_score += t_k[HTransition::DeleteToDelete.idx()].ln();
    }
}

/// Configure the E-state loop/move scores and expected J uses.
fn configure_e_state(profile: &mut Profile, mode: SearchMode) {
    if mode.is_multihit() {
        profile.special_scores.e_move = -LN_2;
        profile.special_scores.e_loop = -LN_2;
        profile.expected_j_uses = 1.0;
    } else {
        profile.special_scores.e_move = 0.0;
        profile.special_scores.e_loop = f32::NEG_INFINITY;
        profile.expected_j_uses = 0.0;
    }
}

/// Set log-probability transition scores for nodes k = 1..M-1.
fn configure_transition_scores(hmm: &Hmm, profile: &mut Profile) {
    let m = hmm.num_nodes;
    for k in 1..m {
        let t = hmm.transitions(k);
        let ts = profile.transition_scores_for_node_mut(k + 1);
        ts[PTsc::MatchToMatch.idx()] = safe_ln(t[HTransition::MatchToMatch.idx()]);
        ts[PTsc::MatchToInsert.idx()] = safe_ln(t[HTransition::MatchToInsert.idx()]);
        ts[PTsc::MatchToDelete.idx()] = safe_ln(t[HTransition::MatchToDelete.idx()]);
        ts[PTsc::InsertToMatch.idx()] = safe_ln(t[HTransition::InsertToMatch.idx()]);
        ts[PTsc::InsertToInsert.idx()] = safe_ln(t[HTransition::InsertToInsert.idx()]);
        ts[PTsc::DeleteToMatch.idx()] = safe_ln(t[HTransition::DeleteToMatch.idx()]);
        ts[PTsc::DeleteToDelete.idx()] = safe_ln(t[HTransition::DeleteToDelete.idx()]);
    }
}

/// Set log-odds match emission scores for nodes k = 1..M.
///
/// Canonical residue score: log( p(x | M_k) / p(x | background) ).
/// Degenerate residue score: background-frequency-weighted average over the
/// contributing canonical residues.
/// Gap, nonresidue, and missing characters are set to -∞.
fn configure_match_emissions(hmm: &Hmm, background: &BackgroundModel, profile: &mut Profile) {
    let m = hmm.num_nodes;
    let abc = &hmm.alphabet;
    let full_alphabet_size = abc.full_size;

    for k in 1..=m {
        let mat_k = hmm.match_emissions(k);
        let scores = match_emission_scores(abc, background, mat_k, full_alphabet_size);

        for (x, &s) in scores.iter().enumerate() {
            profile.residue_scores_for_mut(x)[k * PROFILE_NUM_EMISSIONS + PRsc::MatchScore.idx()] =
                s;
        }
    }
}

/// Compute the full-alphabet match emission score vector for one HMM node.
///
/// Returns a Vec of length `full_alphabet_size` with log-odds scores.
fn match_emission_scores(
    abc: &Alphabet,
    background: &BackgroundModel,
    mat_k: &[f32],
    full_alphabet_size: usize,
) -> Vec<f32> {
    let mut scores = vec![f32::NEG_INFINITY; full_alphabet_size];

    // Canonical residues: log-odds ratio.
    for (x, s) in scores[..abc.canonical_size].iter_mut().enumerate() {
        let emission = mat_k[x];
        let bg_freq = background.residue_frequencies[x];
        if emission > 0.0 && bg_freq > 0.0 {
            *s = (emission / bg_freq).ln();
        }
        // else stays NEG_INFINITY (covers both zero-emission and zero-background cases)
    }

    // Gap character (index canonical_size) stays NEG_INFINITY.

    // Degenerate residues: expected score weighted by background frequencies.
    let degen_range = (abc.canonical_size + 1)..full_alphabet_size.saturating_sub(2);
    for x in degen_range {
        scores[x] = degen_expected_score(abc, background, x, &scores[..abc.canonical_size]);
    }

    // Nonresidue (second-to-last) and missing data (last) stay NEG_INFINITY.
    // Already set by initialisation; explicit assignments aid readability.
    if full_alphabet_size >= 2 {
        scores[full_alphabet_size - 2] = f32::NEG_INFINITY;
    }
    scores[full_alphabet_size - 1] = f32::NEG_INFINITY;

    scores
}

/// Compute the expected match emission score for a degenerate residue symbol.
///
/// Normalises by the sum of background frequencies for the contributing
/// canonical residues, so the result is a proper weighted average.
fn degen_expected_score(
    abc: &Alphabet,
    background: &BackgroundModel,
    x: usize,
    canonical_scores: &[f32],
) -> f32 {
    let degen = abc.degen_row(x);
    let bg = &background.residue_frequencies;

    let (weighted_sum, norm) = degen
        .iter()
        .zip(bg.iter())
        .zip(canonical_scores.iter())
        .filter(|&((&d, _), _)| d)
        .fold((0.0f32, 0.0f32), |(sum, n), ((_, &f), &s)| {
            (sum + f * s, n + f)
        });

    if norm > 0.0 {
        weighted_sum / norm.max(1e-30)
    } else {
        f32::NEG_INFINITY
    }
}

/// Set insert emission scores for all nodes.
///
/// Insert emissions are hardwired to 0.0 (identical to background), except
/// I_M (insert at last node) and special characters (gap, nonresidue, missing),
/// which are -∞.
fn configure_insert_emissions(profile: &mut Profile, full_alphabet_size: usize) {
    let m = profile.num_nodes;
    let canonical_size = profile.alphabet.canonical_size;

    for x in 0..full_alphabet_size {
        let rsc = profile.residue_scores_for_mut(x);
        for k in 1..m {
            rsc[k * PROFILE_NUM_EMISSIONS + PRsc::InsertScore.idx()] = 0.0;
        }
        // I_M is impossible.
        rsc[m * PROFILE_NUM_EMISSIONS + PRsc::InsertScore.idx()] = f32::NEG_INFINITY;
    }

    // Gap, nonresidue, and missing characters cannot be inserted; force -∞.
    for k in 1..=m {
        let offset = k * PROFILE_NUM_EMISSIONS + PRsc::InsertScore.idx();
        profile.residue_scores_for_mut(canonical_size)[offset] = f32::NEG_INFINITY;
        if full_alphabet_size >= 2 {
            profile.residue_scores_for_mut(full_alphabet_size - 2)[offset] = f32::NEG_INFINITY;
        }
        profile.residue_scores_for_mut(full_alphabet_size - 1)[offset] = f32::NEG_INFINITY;
    }
}

#[inline]
fn safe_ln(x: f32) -> f32 {
    if x > 0.0 { x.ln() } else { f32::NEG_INFINITY }
}

/// Configure the length model for a target of length L.
pub fn config_length_model(profile: &mut Profile, target_length: usize) {
    profile.config_length_model(target_length);
}

/// Reconfigure the length model of an already-configured profile.
pub fn reconfigure_length(profile: &mut Profile, target_length: usize) {
    profile.reconfigure_length(target_length);
}

/// Reconfigure profile for unihit mode with target length L.
///
/// Port of p7_ReconfigUnihit() in C HMMER.
/// Sets E->C = 1.0, E->J = 0, and adjusts N/C loop for length L.
pub fn reconfig_unihit(profile: &mut Profile, target_length: usize) {
    profile.reconfig_unihit(target_length);
}

/// Reconfigure profile for multihit mode with target length L.
///
/// Port of p7_ReconfigMultihit() in C HMMER.
pub fn reconfig_multihit(profile: &mut Profile, target_length: usize) {
    profile.reconfig_multihit(target_length);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;

    #[test]
    fn test_length_model() {
        let abc = Alphabet::amino();
        let mut profile = Profile::new(10, &abc);
        profile.expected_j_uses = 1.0;
        config_length_model(&mut profile, 400);
        assert_eq!(profile.target_length, 400);
        assert!(profile.special_scores.n_loop < 0.0);
        assert!(profile.special_scores.n_move < 0.0);
    }
}
