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
    let model_length = hmm.num_nodes;
    let full_alphabet_size = hmm.alphabet.full_size;

    profile.mode = mode;
    profile.num_nodes = model_length;
    profile.max_length = hmm.max_length.unwrap_or(0);

    copy_metadata(hmm, profile);
    configure_model_composition(hmm, background, profile);

    // Ensure score arrays are large enough before writing any scores.
    profile.resize_transition_scores(model_length);
    profile.resize_residue_scores(full_alphabet_size, model_length);

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
}

/// Copy or derive the model's average canonical-residue composition.
///
/// HMMER uses this vector to configure the composition-bias filter. Models
/// produced by modern HMMER normally carry a COMPO line; deriving it keeps
/// programmatically constructed and older models usable as search queries.
fn configure_model_composition(hmm: &Hmm, background: &BackgroundModel, profile: &mut Profile) {
    profile.model_composition.fill(COMPOSITION_UNSET);
    let canonical_size = hmm.alphabet.canonical_size.min(MAX_CANONICAL_ALPHABET);

    if let Some(composition) = hmm
        .model_composition
        .as_deref()
        .filter(|composition| composition.len() >= canonical_size)
    {
        profile.model_composition[..canonical_size].copy_from_slice(&composition[..canonical_size]);
    } else {
        let (occupancy, _) = hmm.calculate_occupancy();
        let occupancy_sum: f32 = occupancy[1..=hmm.num_nodes].iter().sum();

        if occupancy_sum > 0.0 {
            for residue in 0..canonical_size {
                let weighted_sum: f32 = (1..=hmm.num_nodes)
                    .map(|node| occupancy[node] * hmm.match_emissions(node)[residue])
                    .sum();
                profile.model_composition[residue] = weighted_sum / occupancy_sum;
            }
        } else {
            profile.model_composition[..canonical_size]
                .copy_from_slice(&background.residue_frequencies[..canonical_size]);
        }
    }

    // Keep small text-format rounding errors from changing the HMM's total
    // probability mass, and fall back safely if the supplied vector is bad.
    let composition = &mut profile.model_composition[..canonical_size];
    let sum: f32 = composition.iter().sum();
    if sum.is_finite()
        && sum > 0.0
        && composition
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    {
        composition.iter_mut().for_each(|value| *value /= sum);
    } else {
        composition.copy_from_slice(&background.residue_frequencies[..canonical_size]);
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
    let model_length = hmm.num_nodes;
    let (occupancy, _) = hmm.calculate_occupancy();

    let occupancy_sum: f32 = occupancy[1..=model_length]
        .iter()
        .enumerate()
        .map(|(index, &occupancy_at_node)| occupancy_at_node * (model_length - index) as f32) // index here == k-1, so (M - k + 1) == (M - index)
        .sum();

    if occupancy_sum > 0.0 {
        for (node_index, &occupancy_at_node) in
            occupancy.iter().enumerate().skip(1).take(model_length)
        {
            profile.transition_scores_for_node_mut(node_index)[PTsc::BeginToMatch.idx()] =
                (occupancy_at_node / occupancy_sum).ln();
        }
    } else {
        // Uniform fallback when all occupancies are zero.
        let uniform_entry_prob = 2.0 / (model_length * (model_length + 1)) as f32;
        let log_uniform = uniform_entry_prob.ln();
        for node_index in 1..=model_length {
            profile.transition_scores_for_node_mut(node_index)[PTsc::BeginToMatch.idx()] =
                log_uniform;
        }
    }
}

/// Glocal mode: left wing retraction entry scores.
///
/// Entry to node k goes through the silent delete path D_1..D_{k-1}.
fn configure_glocal_entry_scores(hmm: &Hmm, profile: &mut Profile) {
    let model_length = hmm.num_nodes;
    let t0 = hmm.transitions(0);
    let md0 = t0[HTransition::MatchToDelete as usize];

    // B->M_1: probability of taking the match branch at node 0
    profile.transition_scores_for_node_mut(1)[PTsc::BeginToMatch.idx()] = (1.0 - md0).ln();

    // B->M_{k+1} for k in 1..M-1: accumulated wing retraction score
    let mut wing_score = md0.ln();
    for node_index in 1..model_length {
        let node_transitions = hmm.transitions(node_index);
        profile.transition_scores_for_node_mut(node_index + 1)[PTsc::BeginToMatch.idx()] =
            wing_score + node_transitions[HTransition::DeleteToMatch.idx()].ln();
        wing_score += node_transitions[HTransition::DeleteToDelete.idx()].ln();
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
    let model_length = hmm.num_nodes;
    for node_index in 1..model_length {
        let node_transitions = hmm.transitions(node_index);
        let transition_scores = profile.transition_scores_for_node_mut(node_index + 1);
        transition_scores[PTsc::MatchToMatch.idx()] =
            safe_ln(node_transitions[HTransition::MatchToMatch.idx()]);
        transition_scores[PTsc::MatchToInsert.idx()] =
            safe_ln(node_transitions[HTransition::MatchToInsert.idx()]);
        transition_scores[PTsc::MatchToDelete.idx()] =
            safe_ln(node_transitions[HTransition::MatchToDelete.idx()]);
        transition_scores[PTsc::InsertToMatch.idx()] =
            safe_ln(node_transitions[HTransition::InsertToMatch.idx()]);
        transition_scores[PTsc::InsertToInsert.idx()] =
            safe_ln(node_transitions[HTransition::InsertToInsert.idx()]);
        transition_scores[PTsc::DeleteToMatch.idx()] =
            safe_ln(node_transitions[HTransition::DeleteToMatch.idx()]);
        transition_scores[PTsc::DeleteToDelete.idx()] =
            safe_ln(node_transitions[HTransition::DeleteToDelete.idx()]);
    }
}

/// Set log-odds match emission scores for nodes k = 1..M.
///
/// Canonical residue score: log( p(x | M_k) / p(x | background) ).
/// Degenerate residue score: background-frequency-weighted average over the
/// contributing canonical residues.
/// Gap, nonresidue, and missing characters are set to -∞.
fn configure_match_emissions(hmm: &Hmm, background: &BackgroundModel, profile: &mut Profile) {
    let model_length = hmm.num_nodes;
    let alphabet = &hmm.alphabet;
    let full_alphabet_size = alphabet.full_size;

    for node_index in 1..=model_length {
        let match_probabilities = hmm.match_emissions(node_index);
        let scores = match_emission_scores(
            alphabet,
            background,
            match_probabilities,
            full_alphabet_size,
        );

        for (residue_index, &score) in scores.iter().enumerate() {
            profile.residue_scores_for_mut(residue_index)
                [node_index * PROFILE_NUM_EMISSIONS + PRsc::MatchScore.idx()] = score;
        }
    }
}

/// Compute the full-alphabet match emission score vector for one HMM node.
///
/// Returns a Vec of length `full_alphabet_size` with log-odds scores.
fn match_emission_scores(
    alphabet: &Alphabet,
    background: &BackgroundModel,
    match_probabilities: &[f32],
    full_alphabet_size: usize,
) -> Vec<f32> {
    let mut scores = vec![f32::NEG_INFINITY; full_alphabet_size];

    // Canonical residues: log-odds ratio.
    for (residue_index, score) in scores[..alphabet.canonical_size].iter_mut().enumerate() {
        let emission_probability = match_probabilities[residue_index];
        let background_frequency = background.residue_frequencies[residue_index];
        if emission_probability > 0.0 && background_frequency > 0.0 {
            *score = (emission_probability / background_frequency).ln();
        }
        // else stays NEG_INFINITY (covers both zero-emission and zero-background cases)
    }

    // Gap character (index canonical_size) stays NEG_INFINITY.

    // Degenerate residues: expected score weighted by background frequencies.
    let degenerate_residue_range =
        (alphabet.canonical_size + 1)..full_alphabet_size.saturating_sub(2);
    for residue_index in degenerate_residue_range {
        scores[residue_index] = degen_expected_score(
            alphabet,
            background,
            residue_index,
            &scores[..alphabet.canonical_size],
        );
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
    alphabet: &Alphabet,
    background: &BackgroundModel,
    residue_index: usize,
    canonical_scores: &[f32],
) -> f32 {
    let degenerate_row = alphabet.degen_row(residue_index);
    let background_frequencies = &background.residue_frequencies;

    let (weighted_sum, normalization) = degenerate_row
        .iter()
        .zip(background_frequencies.iter())
        .zip(canonical_scores.iter())
        .filter(|&((&d, _), _)| d)
        .fold(
            (0.0f32, 0.0f32),
            |(sum, normalizer), ((_, &frequency), &score)| {
                (sum + frequency * score, normalizer + frequency)
            },
        );

    if normalization > 0.0 {
        weighted_sum / normalization.max(1e-30)
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
    let model_length = profile.num_nodes;
    let canonical_size = profile.alphabet.canonical_size;

    for residue_index in 0..full_alphabet_size {
        let residue_scores = profile.residue_scores_for_mut(residue_index);
        for node_index in 1..model_length {
            residue_scores[node_index * PROFILE_NUM_EMISSIONS + PRsc::InsertScore.idx()] = 0.0;
        }
        // I_M is impossible.
        residue_scores[model_length * PROFILE_NUM_EMISSIONS + PRsc::InsertScore.idx()] =
            f32::NEG_INFINITY;
    }

    // Gap, nonresidue, and missing characters cannot be inserted; force -∞.
    for node_index in 1..=model_length {
        let insert_score_offset = node_index * PROFILE_NUM_EMISSIONS + PRsc::InsertScore.idx();
        profile.residue_scores_for_mut(canonical_size)[insert_score_offset] = f32::NEG_INFINITY;
        if full_alphabet_size >= 2 {
            profile.residue_scores_for_mut(full_alphabet_size - 2)[insert_score_offset] =
                f32::NEG_INFINITY;
        }
        profile.residue_scores_for_mut(full_alphabet_size - 1)[insert_score_offset] =
            f32::NEG_INFINITY;
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
        let amino_alphabet = Alphabet::amino();
        let mut profile = Profile::new(10, &amino_alphabet);
        profile.expected_j_uses = 1.0;
        config_length_model(&mut profile, 400);
        assert_eq!(profile.target_length, 400);
        assert!(profile.special_scores.n_loop < 0.0);
        assert!(profile.special_scores.n_move < 0.0);
    }
}
