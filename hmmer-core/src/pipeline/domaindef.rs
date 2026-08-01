// p7_domaindef.rs - Domain definition (identification of domains in a sequence)
//
// Port of src/p7_domaindef.c

use crate::domain::Domain;
use crate::dynamic_programming::optimal_accuracy::OaDecoder;
use crate::dynamic_programming::simd::fwd_bck::OddsSegmentBuf;
use crate::dynamic_programming::simd::oprofile::OptimizedProfile;
use crate::profile::Profile;
use crate::score_matrix::ScoreMatrix;
use crate::trace::Trace;

/// Pooled buffers for per-domain rescoring, reused across domains to avoid repeated allocation.
#[derive(Debug)]
pub struct RescoringBuffers {
    posterior_matrix: ScoreMatrix,
    oa_decoder: OaDecoder,
}

impl RescoringBuffers {
    pub fn new(num_nodes: usize) -> Self {
        RescoringBuffers {
            posterior_matrix: ScoreMatrix::new(num_nodes, 1).unwrap(),
            oa_decoder: OaDecoder::new(num_nodes),
        }
    }
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
    pub do_reseeding: bool,
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
    pub length: usize,
    pub null2_scores: Vec<f32>,
    pub trace: Trace,
}

/// Backward-compatible alias for `DomainWorkspace`.
pub type DomainDef = DomainWorkspace;

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

    pub fn reuse(&mut self) {
        self.length = 0;
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
        seg_buf: &mut OddsSegmentBuf,
    ) -> DomainResult {
        self.null2_scores.fill(0.0);

        let mut result = DomainResult {
            domains: Vec::new(),
            num_expected: self.begin_totals[seq_n],
            num_regions: 0,
            num_clustered: 0,
            num_overlaps: 0,
            num_envelopes: 0,
        };

        let mut region_start: Option<usize> = None;
        let mut triggered = false;

        for j in 1..=seq_n {
            if !triggered {
                if self.model_occupancy[j] - (self.begin_totals[j] - self.begin_totals[j - 1])
                    < config.cluster_threshold
                    || region_start.is_none()
                {
                    region_start = Some(j);
                }
                if self.model_occupancy[j] >= config.region_threshold {
                    triggered = true;
                }
            } else if self.model_occupancy[j] - (self.exit_totals[j] - self.exit_totals[j - 1])
                < config.cluster_threshold
            {
                let region_i = region_start.expect("triggered implies region_start is set");
                let region_j = j;
                result.num_regions += 1;

                if is_multidomain_region(self, config, region_i, region_j) {
                    result.num_clustered += 1;
                }

                // Save profile + OM state, reconfig for isolated domain
                let save_l = profile.target_length;
                let save_nj = profile.expected_j_uses;
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
                    seg_buf,
                );

                // Restore profile + OM state
                if save_nj > 0.0 {
                    profile.reconfig_multihit(save_l);
                    om.reconfig_multihit(save_l);
                } else {
                    profile.reconfigure_length(save_l);
                    om.reconfigure_length(save_l);
                }

                if let Some((mut dom, null2)) = domain {
                    // Accumulate null2 scores using the returned null2 vector
                    let mut domcorrection = 0.0f32;
                    for pos in region_i..=region_j {
                        if pos <= seq_n {
                            let x = digital_sequence[pos - 1] as usize;
                            if x < null2.len() {
                                let null2_log_correction = null2[x].ln();
                                if pos < self.null2_scores.len() {
                                    self.null2_scores[pos] += null2_log_correction;
                                }
                                domcorrection += null2_log_correction;
                            }
                        }
                    }
                    dom.domain_correction = domcorrection;
                    result.domains.push(dom);
                }

                result.num_envelopes += 1;
                region_start = None;
                triggered = false;
            }
        }

        result
    }
}

/// Output of domain definition by posterior heuristics.
#[derive(Debug, Default)]
pub struct DomainResult {
    pub domains: Vec<Domain>,
    pub num_expected: f32,
    pub num_regions: usize,
    pub num_clustered: usize,
    pub num_overlaps: usize,
    pub num_envelopes: usize,
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
/// Returns the scored Domain and null2 vector, or None if backward decode fails.
fn rescore_isolated_domain(
    digital_sequence: &[u8],
    profile: &Profile,
    i: usize,
    j: usize,
    buf: &mut RescoringBuffers,
    om: &OptimizedProfile,
    seg_buf: &mut OddsSegmentBuf,
) -> Option<(Domain, Vec<f32>)> {
    use crate::dynamic_programming::simd::fwd_bck;

    let domain_length = j - i + 1;
    let dsq_sub = &digital_sequence[i - 1..j];

    let (checkpoints, simd_data) = fwd_bck::forward_checkpointed_simd(dsq_sub, domain_length, om);
    let forward_score = checkpoints.overall_score;

    if crate::forward_backward::backward_decode_prob_space(
        dsq_sub,
        profile,
        om,
        &checkpoints,
        &simd_data,
        &mut buf.posterior_matrix,
        None,
        seg_buf,
    )
    .is_err()
    {
        return None;
    }

    Some(finish_domain_scoring(
        profile,
        i,
        j,
        forward_score,
        &buf.posterior_matrix,
        &mut buf.oa_decoder,
    ))
}

/// Shared tail of domain scoring: null2, optimal accuracy, traceback, domain creation.
///
/// Returns the scored Domain along with the null2 odds ratios. The `domain_correction`
/// field is left at 0.0; the caller computes and sets it from the null2 vector.
fn finish_domain_scoring(
    profile: &Profile,
    i: usize,
    j: usize,
    forward_score: f32,
    posterior_matrix: &ScoreMatrix,
    oa_decoder: &mut OaDecoder,
) -> (Domain, Vec<f32>) {
    // Null2 correction
    let null2 = crate::null2::null2_by_expectation(profile, posterior_matrix);

    // Optimal accuracy alignment + traceback
    let oa = oa_decoder.decode(profile, posterior_matrix);

    // Create domain (domain_correction set by caller)
    let mut dom = Domain {
        envelope_start: i,
        envelope_end: j,
        envelope_score: forward_score,
        domain_correction: 0.0,
        optimal_accuracy_score: oa.as_ref().map_or(0.0, |r| r.score),
        ..Domain::default()
    };

    if let Ok(result) = oa {
        let domains = result.trace.compute_domains();
        if let Some(d) = domains.first() {
            dom.alignment_start = i + d.sequence_start as usize - 1;
            dom.alignment_end = i + d.sequence_end as usize - 1;
            dom.hmm_from = d.hmm_start as usize;
            dom.hmm_to = d.hmm_end as usize;
        } else {
            dom.alignment_start = i;
            dom.alignment_end = j;
        }
        dom.trace = Some(result.trace);
    } else {
        dom.alignment_start = i;
        dom.alignment_end = j;
    }

    (dom, null2)
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
        assert_eq!(ws.length, 0);
    }

    #[test]
    fn test_workspace_grow_to() {
        let mut ws = DomainWorkspace::new();
        ws.grow_to(1000);
        assert!(ws.model_occupancy.len() > 1000);
    }
}
