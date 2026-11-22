// p7_domaindef.rs - Domain definition (identification of domains in a sequence)
//
// Port of src/p7_domaindef.c

use crate::hmmer_core::alphabet::Alphabet;
use crate::hmmer_core::config::MAX_FULL_ALPHABET;
use crate::hmmer_core::domain::Domain;
use crate::hmmer_core::dynamic_programming::odds_decode::OddsTables;
use crate::hmmer_core::dynamic_programming::optimal_accuracy::OaDecoder;
use crate::hmmer_core::dynamic_programming::simd::oprofile::OptimizedProfile;
use crate::hmmer_core::dynamic_programming::simd::striped_forward::{
    PathState, PathStep, SamplingFailed, StripedForward,
};
use crate::hmmer_core::pipeline::spensemble::{ClusterRules, SegmentPairEnsemble};
use crate::hmmer_core::profile::Profile;
use crate::hmmer_core::rng::EaselFastRng;
use crate::hmmer_core::score_matrix::ScoreMatrix;

/// Pooled buffers for per-domain rescoring, reused across domains to avoid repeated allocation.
#[derive(Debug)]
pub struct RescoringBuffers {
    posterior_matrix: ScoreMatrix,
    oa_decoder: OaDecoder,
    ensemble: EnsembleBuffers,
}

impl RescoringBuffers {
    pub fn new(num_nodes: usize) -> Self {
        RescoringBuffers {
            posterior_matrix: ScoreMatrix::new(num_nodes, 1).unwrap(),
            oa_decoder: OaDecoder::new(num_nodes),
            ensemble: EnsembleBuffers::default(),
        }
    }
}

/// Buffers for resolving a multi-domain region by sampling: nothing is
/// allocated before the first such region, and they are reused after it.
#[derive(Debug, Default)]
struct EnsembleBuffers {
    /// Created at the first region, with the configured seed.
    rng: Option<EaselFastRng>,
    /// Forward matrix of the region, every row.
    forward: StripedForward,
    /// One sampled path.
    path: Vec<PathStep>,
    /// Residues emitted by each node in one domain of a path, striped.
    usage: Vec<f32>,
    /// The domains of the sampled paths.
    pairs: SegmentPairEnsemble,
}

/// Immutable configuration parameters for domain definition.
#[derive(Debug, Clone)]
pub struct DomainConfig {
    pub region_threshold: f32,
    pub cluster_threshold: f32,
    pub envelope_threshold: f32,
    pub nsamples: usize,
    pub min_overlap: f32,
    pub of_smaller: bool,
    pub max_diagdiff: usize,
    pub min_posterior: f32,
    pub min_endpointp: f32,
    /// Restart the random sequence at every multi-domain region, so that a
    /// region's envelopes do not depend on the regions sampled before it.
    pub do_reseeding: bool,
    /// Seed of the generator that samples tracebacks: `hmmsearch --seed`.
    /// Zero stands for the default, 42 (see `EaselFastRng::new`).
    pub seed: u32,
}

impl Default for DomainConfig {
    fn default() -> Self {
        DomainConfig {
            region_threshold: 0.25,
            cluster_threshold: 0.10,
            envelope_threshold: 0.20,
            nsamples: 200,
            min_overlap: 0.8,
            of_smaller: true,
            max_diagdiff: 4,
            min_posterior: 0.25,
            min_endpointp: 0.02,
            do_reseeding: true,
            seed: EaselFastRng::DEFAULT_SEED,
        }
    }
}

/// Per-sequence reusable workspace buffers for domain definition.
///
/// Filled by backward decoding (begin_totals, exit_totals, model_occupancy),
/// then consumed by `by_posterior_heuristics` to identify domain regions.
#[derive(Debug, Default)]
pub struct DomainWorkspace {
    pub model_occupancy: Vec<f32>,
    pub begin_totals: Vec<f32>,
    pub exit_totals: Vec<f32>,
    pub null2_scores: Vec<f32>,
}

impl DomainWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn grow_to(&mut self, l: usize) {
        let needed = l + 1;
        if self.model_occupancy.len() >= needed {
            return;
        }
        self.model_occupancy.resize(needed, 0.0);
        self.begin_totals.resize(needed, 0.0);
        self.exit_totals.resize(needed, 0.0);
        self.null2_scores.resize(needed, 0.0);
    }

    /// Define domains by posterior heuristics.
    ///
    /// Given Forward/Backward matrices and domain decoding already done,
    /// identify domain regions and score/align each one.
    ///
    /// Profile and OptimizedProfile are `&mut` only because this method
    /// reconfigures them per-envelope (save/restore). No inner function
    /// receives mutable profile or OM references.
    #[allow(clippy::too_many_arguments)]
    pub fn by_posterior_heuristics(
        &mut self,
        config: &DomainConfig,
        digital_sequence: &[u8],
        seq_n: usize,
        profile: &mut Profile,
        buf: &mut RescoringBuffers,
        om: &mut OptimizedProfile,
        odds: &OddsTables,
    ) -> DomainResult {
        self.null2_scores.fill(0.0);

        let mut result = DomainResult {
            domains: Vec::new(),
            num_regions: 0,
            num_clustered: 0,
            num_overlaps: 0,
            num_envelopes: 0,
        };

        for (region_i, region_j) in find_regions(self, config, seq_n) {
            result.num_regions += 1;

            // Save profile + OM state, reconfig for isolated domain
            let save_l = profile.target_length;
            let save_nj = profile.expected_j_uses;

            // A region that may hold several domains is split into
            // envelopes by sampling. Should the sampling fail, the region is
            // scored as the single envelope it would otherwise be.
            let mut resolved = false;
            if is_multidomain_region(self, config, region_i, region_j) {
                result.num_clustered += 1;
                om.reconfig_multihit(save_l);
                let envelopes = self.region_trace_ensemble(
                    config,
                    digital_sequence,
                    region_i,
                    region_j,
                    om,
                    &profile.alphabet,
                    &mut buf.ensemble,
                );
                match envelopes {
                    Ok(num_envelopes) => {
                        resolved = true;
                        profile.reconfig_unihit(save_l);
                        om.reconfig_unihit(save_l);
                        let mut last_end = 0usize;
                        for index in 0..num_envelopes {
                            let envelope = buf.ensemble.pairs.significant()[index];
                            let (start, end) = (envelope.i as usize, envelope.j as usize);
                            // HMMER counts an envelope that starts inside
                            // the previous one, and scores it all the same.
                            if start <= last_end {
                                result.num_overlaps += 1;
                            }
                            result.num_envelopes += 1;
                            let domain = rescore_isolated_domain(
                                digital_sequence,
                                profile,
                                start,
                                end,
                                buf,
                                om,
                                odds,
                                false,
                            );
                            // An envelope whose decoding fails gives no
                            // domain.
                            if let Some((mut dom, _)) = domain {
                                // The sampled paths have set the null2
                                // scores of the whole region already.
                                let mut domcorrection = 0.0f32;
                                for &score in &self.null2_scores[start..=end] {
                                    domcorrection += score;
                                }
                                dom.domain_correction = domcorrection;
                                result.domains.push(dom);
                                last_end = end;
                            }
                        }
                    }
                    Err(SamplingFailed) => self.null2_scores[region_i..=region_j].fill(0.0),
                }
            }

            if !resolved {
                // Upstream keeps the length model configured for the complete
                // target while forcing a single-hit parse of each envelope.
                // The envelope length is only the DP row count.
                profile.reconfig_unihit(save_l);
                om.reconfig_unihit(save_l);

                // Inner functions receive &Profile and &OptimizedProfile (immutable)
                let domain = rescore_isolated_domain(
                    digital_sequence,
                    profile,
                    region_i,
                    region_j,
                    buf,
                    om,
                    odds,
                    true,
                );

                if let Some((mut dom, null2)) = domain {
                    // Accumulate null2 scores using the returned null2 vector
                    debug_assert!(region_j <= seq_n);
                    debug_assert!(region_j < self.null2_scores.len());
                    let null2 = null2.expect("the expectation null2 was asked for");
                    let mut domcorrection = 0.0f32;
                    for pos in region_i..=region_j {
                        let x = digital_sequence[pos - 1] as usize;
                        if x < null2.len() {
                            let null2_log_correction = null2[x].ln();
                            self.null2_scores[pos] += null2_log_correction;
                            domcorrection += null2_log_correction;
                        }
                    }
                    dom.domain_correction = domcorrection;
                    result.domains.push(dom);
                }

                // HMMER counts candidate envelopes even when isolated rescoring fails.
                result.num_envelopes += 1;
            }

            // Restore profile + OM state
            if save_nj > 0.0 {
                profile.reconfig_multihit(save_l);
                om.reconfig_multihit(save_l);
            } else {
                profile.reconfigure_length(save_l);
                om.reconfigure_length(save_l);
            }
        }

        result
    }

    /// Resolve region `ireg..=jreg`, which may hold several domains, into
    /// envelopes: `region_trace_ensemble()` of p7_domaindef.c.
    ///
    /// `om` is configured multihit for the whole target. Tracebacks of the
    /// region's Forward matrix are sampled and the domains they contain
    /// clustered; the consensus envelopes are left in `buffers.pairs`, by
    /// start, and their number is returned. The null2 score of every residue
    /// of the region is set from the same tracebacks: the log of the average,
    /// over them, of the odds ratio of the domain the residue falls in, or of
    /// one where it falls in none.
    #[allow(clippy::too_many_arguments)]
    fn region_trace_ensemble(
        &mut self,
        config: &DomainConfig,
        digital_sequence: &[u8],
        ireg: usize,
        jreg: usize,
        om: &OptimizedProfile,
        alphabet: &Alphabet,
        buffers: &mut EnsembleBuffers,
    ) -> Result<usize, SamplingFailed> {
        let region = &digital_sequence[ireg - 1..jreg];
        let region_length = region.len();
        // scores[pos - 1] belongs to residue `pos` of the region, 1-based.
        let scores = &mut self.null2_scores[ireg..=jreg];
        scores.fill(0.0);

        let rng = buffers
            .rng
            .get_or_insert_with(|| EaselFastRng::new(config.seed));
        if config.do_reseeding {
            rng.reseed(config.seed);
        }

        buffers.forward.fill(region, om)?;
        buffers.pairs.reuse();
        let q = om.q4;
        buffers.usage.clear();
        buffers.usage.resize(q * 4, 0.0);
        let mut null2 = [1.0f32; MAX_FULL_ALPHABET];
        let null2 = &mut null2[..alphabet.full_size];
        let offset = (ireg - 1) as i32;

        for sample in 0..config.nsamples {
            buffers.forward.sample_path(rng, om, &mut buffers.path)?;

            // The path was sampled from its end; read it from its start.
            // `pos` is the first residue that has not received its term.
            let mut pos = 1usize;
            let (mut seq_from, mut seq_to, mut hmm_from, mut hmm_to) = (0u32, 0u32, 0u32, 0u32);
            let mut emitted = 0usize;
            for step in buffers.path.iter().rev() {
                match step.state {
                    PathState::B => {
                        seq_from = 0;
                        hmm_from = 0;
                        emitted = 0;
                        buffers.usage.fill(0.0);
                    }
                    PathState::Match | PathState::Insert => {
                        if step.state == PathState::Match {
                            if seq_from == 0 {
                                seq_from = step.residue;
                            }
                            if hmm_from == 0 {
                                hmm_from = step.node;
                            }
                            seq_to = step.residue;
                            hmm_to = step.node;
                        }
                        // HMMER books a residue an insert state emitted on
                        // the match state of its node.
                        let node = step.node as usize - 1;
                        buffers.usage[(node % q) * 4 + node / q] += 1.0;
                        emitted += 1;
                    }
                    PathState::E => {
                        buffers.pairs.add(
                            sample,
                            seq_from as i32 + offset,
                            seq_to as i32 + offset,
                            hmm_from as i32,
                            hmm_to as i32,
                        );
                        crate::hmmer_core::null2::null2_by_trace(
                            om,
                            alphabet,
                            &mut buffers.usage,
                            emitted,
                            null2,
                        );
                        // Outside domains the null2 model is the null model.
                        // As in HMMER, the first residue of a domain is
                        // counted with the residues before it.
                        while pos <= seq_from as usize {
                            scores[pos - 1] += 1.0;
                            pos += 1;
                        }
                        while pos <= seq_to as usize {
                            scores[pos - 1] += null2[region[pos - 1] as usize];
                            pos += 1;
                        }
                    }
                    _ => {}
                }
            }
            while pos <= region_length {
                scores[pos - 1] += 1.0;
                pos += 1;
            }
        }

        let num_samples = config.nsamples as f32;
        for score in scores.iter_mut() {
            *score = (*score / num_samples).ln();
        }

        buffers.pairs.cluster(&ClusterRules {
            min_overlap: config.min_overlap,
            of_smaller: config.of_smaller,
            max_diagdiff: config.max_diagdiff as i32,
            min_posterior: config.min_posterior,
            min_endpointp: config.min_endpointp,
        });
        Ok(buffers.pairs.remove_dominated())
    }
}

/// Output of domain definition by posterior heuristics.
#[derive(Debug, Default)]
pub struct DomainResult {
    pub domains: Vec<Domain>,
    pub num_regions: usize,
    pub num_clustered: usize,
    pub num_overlaps: usize,
    pub num_envelopes: usize,
}

/// Find candidate regions using HMMER's posterior occupancy heuristics.
///
/// There is intentionally no post-loop flush. This matches HMMER 3.4: a valid
/// region reaching the C-terminus closes at `j = L` because the final exit
/// posterior is subtracted from model occupancy at that position.
fn find_regions(
    workspace: &DomainWorkspace,
    config: &DomainConfig,
    seq_n: usize,
) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut region_start: Option<usize> = None;
    let mut triggered = false;

    for j in 1..=seq_n {
        if !triggered {
            if workspace.model_occupancy[j]
                - (workspace.begin_totals[j] - workspace.begin_totals[j - 1])
                < config.cluster_threshold
                || region_start.is_none()
            {
                region_start = Some(j);
            }
            if workspace.model_occupancy[j] >= config.region_threshold {
                triggered = true;
            }
        } else if workspace.model_occupancy[j]
            - (workspace.exit_totals[j] - workspace.exit_totals[j - 1])
            < config.cluster_threshold
        {
            regions.push((
                region_start.expect("triggered implies region_start is set"),
                j,
            ));
            region_start = None;
            triggered = false;
        }
    }

    regions
}

fn is_multidomain_region(
    workspace: &DomainWorkspace,
    config: &DomainConfig,
    i: usize,
    j: usize,
) -> bool {
    let mut max = -1.0f32;
    for z in i..=j {
        let en = f32::min(
            workspace.exit_totals[z] - workspace.exit_totals[i - 1],
            workspace.begin_totals[j] - workspace.begin_totals[z - 1],
        );
        max = f32::max(max, en);
    }
    max >= config.envelope_threshold
}

/// Rescore an isolated domain envelope [i..j].
///
/// Profile and OptimizedProfile must already be reconfigured for the domain length.
/// Returns the scored Domain after posterior and OA decoding succeed, with
/// the null2 odds ratios of the envelope by the expectation method when
/// `compute_null2` asks for them.
#[allow(clippy::too_many_arguments)]
fn rescore_isolated_domain(
    digital_sequence: &[u8],
    profile: &Profile,
    i: usize,
    j: usize,
    buf: &mut RescoringBuffers,
    om: &OptimizedProfile,
    odds: &OddsTables,
    compute_null2: bool,
) -> Option<(Domain, Option<Vec<f32>>)> {
    use crate::hmmer_core::dynamic_programming::simd::forward_filter;

    let domain_length = j - i + 1;
    let dsq_sub = &digital_sequence[i - 1..j];

    let forward_score = forward_filter::forward_filter(dsq_sub, domain_length, om);

    crate::hmmer_core::dynamic_programming::odds_decode::decode_odds_space(
        dsq_sub,
        profile,
        odds,
        forward_score,
        &mut buf.posterior_matrix,
        None,
    )
    .ok()?;

    finish_domain_scoring(
        profile,
        i,
        j,
        forward_score,
        &buf.posterior_matrix,
        &mut buf.oa_decoder,
        compute_null2,
    )
}

/// Shared tail of domain scoring: null2, optimal accuracy, traceback, domain creation.
///
/// Returns the scored Domain along with the null2 odds ratios, when asked
/// for. The `domain_correction` field is left at 0.0; the caller sets it.
fn finish_domain_scoring(
    profile: &Profile,
    i: usize,
    j: usize,
    forward_score: f32,
    posterior_matrix: &ScoreMatrix,
    oa_decoder: &mut OaDecoder,
    compute_null2: bool,
) -> Option<(Domain, Option<Vec<f32>>)> {
    // Null2 correction
    let null2 = compute_null2
        .then(|| crate::hmmer_core::null2::null2_by_expectation(profile, posterior_matrix));

    // Optimal accuracy alignment + traceback
    let oa = oa_decoder.decode(profile, posterior_matrix).ok()?;
    let trace_domain = oa.trace.compute_domains().into_iter().next()?;

    // Create domain (domain_correction set by caller)
    let mut dom = Domain {
        envelope_start: i,
        envelope_end: j,
        envelope_score: forward_score,
        domain_correction: 0.0,
        optimal_accuracy_score: oa.score,
        ..Domain::default()
    };

    dom.alignment_start = i + trace_domain.sequence_start as usize - 1;
    dom.alignment_end = i + trace_domain.sequence_end as usize - 1;
    dom.hmm_from = trace_domain.hmm_start as usize;
    dom.hmm_to = trace_domain.hmm_end as usize;

    Some((dom, null2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = DomainConfig::default();
        assert_eq!(config.nsamples, 200);
        assert!((config.region_threshold - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_workspace_create() {
        let ws = DomainWorkspace::new();
        assert!(ws.model_occupancy.is_empty());
    }

    #[test]
    fn test_workspace_grow_to() {
        let mut ws = DomainWorkspace::new();
        ws.grow_to(1000);
        assert!(ws.model_occupancy.len() > 1000);
    }

    #[test]
    fn c_terminal_region_closes_from_final_exit_posterior() {
        let config = DomainConfig::default();
        let mut ws = DomainWorkspace::new();
        ws.grow_to(4);
        ws.model_occupancy[1..=4].copy_from_slice(&[0.30, 0.50, 0.50, 0.80]);
        ws.exit_totals[1..=4].copy_from_slice(&[0.0, 0.0, 0.0, 1.0]);

        assert_eq!(find_regions(&ws, &config, 4), [(1, 4)]);
    }

    #[test]
    fn open_terminal_region_is_not_synthetically_flushed() {
        let config = DomainConfig::default();
        let mut ws = DomainWorkspace::new();
        ws.grow_to(4);
        ws.model_occupancy[1..=4].copy_from_slice(&[0.30, 0.50, 0.50, 0.80]);

        // This deliberately inconsistent synthetic posterior has no final exit
        // probability. HMMER leaves it open instead of inventing an envelope.
        assert!(find_regions(&ws, &config, 4).is_empty());
    }

    #[test]
    fn multidomain_region_uses_expected_exit_then_begin_count() {
        let config = DomainConfig::default();
        let mut ws = DomainWorkspace::new();
        ws.grow_to(4);
        ws.exit_totals[1..=4].copy_from_slice(&[0.0, 0.3, 0.3, 0.3]);
        ws.begin_totals[1..=4].copy_from_slice(&[0.0, 0.0, 0.3, 0.3]);

        assert!(is_multidomain_region(&ws, &config, 1, 4));

        ws.begin_totals.fill(0.0);
        assert!(!is_multidomain_region(&ws, &config, 1, 4));
    }
}
