// p7_pipeline.rs - The accelerated comparison pipeline
//
// Port of src/p7_pipeline.c

use crate::background::BackgroundModel;
use crate::config::*;
use crate::constants::pipeline::search::{
    DEFAULT_FORWARD_THRESHOLD, DEFAULT_MSV_THRESHOLD, DEFAULT_SEG_BUF_STRIPES,
    DEFAULT_VITERBI_THRESHOLD, LOG2, MIN_PVALUE_CLAMP, SCORE_LENGTH_PRIOR_OFFSET,
};
use crate::domaindef::{DomainConfig, DomainWorkspace, RescoringBuffers};
use crate::dynamic_programming::simd::f32_msv;
use crate::dynamic_programming::simd::f32_viterbi;
use crate::dynamic_programming::simd::forward_filter as simd_fwd;
use crate::dynamic_programming::simd::fwd_bck as simd_fwd_bck;
use crate::dynamic_programming::simd::oprofile::OptimizedProfile;
use crate::errors::HmmerError;
use crate::evalues;
use crate::forward_backward;
use crate::hit::Hit;
use crate::logsum;
use crate::profile::Profile;
use crate::score_matrix::*;
use crate::sequence::DigitalSequence;
use crate::tophits::TopHits;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::Arc;

/// Reporting and inclusion thresholds.
#[derive(Debug, Clone)]
pub struct Thresholds {
    // Reporting thresholds
    pub by_e: bool,
    pub evalue: f64,
    pub bitscore: f64,
    pub domain_by_evalue: bool,
    pub domain_evalue: f64,
    pub domain_bitscore: f64,
    pub use_bit_cutoffs: bool,

    // Inclusion thresholds
    pub inclusion_by_evalue: bool,
    pub inclusion_evalue: f64,
    pub inclusion_bitscore: f64,
    pub inclusion_domain_by_evalue: bool,
    pub inclusion_domain_evalue: f64,
    pub inclusion_domain_bitscore: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            by_e: true,
            evalue: 10.0,
            bitscore: 0.0,
            domain_by_evalue: true,
            domain_evalue: 10.0,
            domain_bitscore: 0.0,
            use_bit_cutoffs: false,
            inclusion_by_evalue: true,
            inclusion_evalue: 0.01,
            inclusion_bitscore: 0.0,
            inclusion_domain_by_evalue: true,
            inclusion_domain_evalue: 0.01,
            inclusion_domain_bitscore: 0.0,
        }
    }
}

/// Pipeline accounting counters.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub num_models: u64,
    pub num_sequences: u64,
    pub num_residues: u64,
    pub num_nodes: u64,
    pub sequences_past_msv: u64,
    pub sequences_past_bias: u64,
    pub sequences_past_viterbi: u64,
    pub sequences_past_forward: u64,
    pub sequences_output: u64,
    pub residues_past_msv: u64,
    pub residues_past_bias: u64,
    pub residues_past_viterbi: u64,
    pub residues_past_forward: u64,
    pub residues_output: u64,
}

impl PipelineStats {
    /// Merge accounting from another stats instance.
    pub fn merge(&mut self, other: &PipelineStats) {
        self.num_models += other.num_models;
        self.num_sequences += other.num_sequences;
        self.num_residues += other.num_residues;
        self.num_nodes += other.num_nodes;
        self.sequences_past_msv += other.sequences_past_msv;
        self.sequences_past_bias += other.sequences_past_bias;
        self.sequences_past_viterbi += other.sequences_past_viterbi;
        self.sequences_past_forward += other.sequences_past_forward;
        self.sequences_output += other.sequences_output;
        self.residues_past_msv += other.residues_past_msv;
        self.residues_past_bias += other.residues_past_bias;
        self.residues_past_viterbi += other.residues_past_viterbi;
        self.residues_past_forward += other.residues_past_forward;
        self.residues_output += other.residues_output;
    }

    /// Reset per-sequence counters.
    pub fn reuse(&mut self) {
        self.sequences_past_msv = 0;
        self.sequences_past_bias = 0;
        self.sequences_past_viterbi = 0;
        self.sequences_past_forward = 0;
        self.sequences_output = 0;
    }
}

/// Mergeable execution metrics for a search worker.
#[derive(Debug, Clone, Default)]
pub struct SearchMetrics {
    pub stats: PipelineStats,
}

impl SearchMetrics {
    pub fn merge(&mut self, other: &SearchMetrics) {
        self.stats.merge(&other.stats);
    }
}

/// Filter configuration for the search executor.
#[derive(Debug, Clone)]
pub struct FilterPolicy {
    pub max_mode: bool,
    pub msv_threshold: f64,
    pub viterbi_threshold: f64,
    pub forward_threshold: f64,
}

impl Default for FilterPolicy {
    fn default() -> Self {
        Self {
            max_mode: false,
            msv_threshold: DEFAULT_MSV_THRESHOLD,
            viterbi_threshold: DEFAULT_VITERBI_THRESHOLD,
            forward_threshold: DEFAULT_FORWARD_THRESHOLD,
        }
    }
}

/// Immutable compiled query state shared by workers.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    profile_template: Profile,
    background_template: BackgroundModel,
    optimized_template: OptimizedProfile,
}

impl SearchQuery {
    /// Build a reusable search query from a configured profile and background model.
    pub fn from_configured_profile(
        profile: Profile,
        background: BackgroundModel,
    ) -> Result<Self, HmmerError> {
        let optimized_template = OptimizedProfile::from_profile(&profile);
        Ok(Self {
            profile_template: profile,
            background_template: background,
            optimized_template,
        })
    }

    pub fn profile(&self) -> &Profile {
        &self.profile_template
    }

    pub fn background(&self) -> &BackgroundModel {
        &self.background_template
    }
}

#[derive(Debug)]
struct SearchPlanInner {
    query: SearchQuery,
    filters: FilterPolicy,
    domain_config: DomainConfig,
}

/// Immutable execution plan shared across one or more workers.
#[derive(Debug, Clone)]
pub struct SearchPlan {
    inner: Arc<SearchPlanInner>,
}

impl SearchPlan {
    pub fn builder(query: SearchQuery) -> SearchPlanBuilder {
        SearchPlanBuilder {
            query,
            filters: FilterPolicy::default(),
            domain_config: DomainConfig::default(),
        }
    }

    pub fn spawn_worker(&self, hints: CapacityHints) -> Result<SearchWorker, HmmerError> {
        SearchWorker::from_plan(Arc::clone(&self.inner), hints)
    }
}

/// Builder for a reusable search plan.
#[derive(Debug, Clone)]
pub struct SearchPlanBuilder {
    query: SearchQuery,
    filters: FilterPolicy,
    domain_config: DomainConfig,
}

impl SearchPlanBuilder {
    pub fn filters(mut self, filters: FilterPolicy) -> Self {
        self.filters = filters;
        self
    }

    pub fn domain_config(mut self, domain_config: DomainConfig) -> Self {
        self.domain_config = domain_config;
        self
    }

    pub fn build(self) -> SearchPlan {
        SearchPlan {
            inner: Arc::new(SearchPlanInner {
                query: self.query,
                filters: self.filters,
                domain_config: self.domain_config,
            }),
        }
    }
}

/// Capacity hint used when allocating a fresh worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityHints {
    pub target_length: usize,
}

impl Default for CapacityHints {
    fn default() -> Self {
        Self { target_length: 400 }
    }
}

/// The reason a sequence stopped before becoming a hit candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterReason {
    Msv,
    Bias,
    Viterbi,
    Forward,
    DecodeFailed,
    NoDomains,
}

/// Lightweight per-sequence execution trace for debugging and tests.
#[derive(Debug, Clone, Default)]
pub struct SearchTrace {
    pub sequence_length: usize,
    pub null_score: Option<f32>,
    pub msv_raw_score: Option<f32>,
    pub viterbi_raw_score: Option<f32>,
    pub forward_raw_score: Option<f32>,
    pub domain_count: usize,
    pub filter_reason: Option<FilterReason>,
}

impl SearchTrace {
    fn new(sequence_length: usize) -> Self {
        Self {
            sequence_length,
            ..Self::default()
        }
    }

    fn into_filtered_report(mut self, reason: FilterReason) -> SearchReport {
        self.filter_reason = Some(reason);
        SearchReport {
            outcome: SearchOutcome::Filtered(reason),
            trace: self,
        }
    }
}

/// Result of searching one sequence with a worker.
#[derive(Debug)]
pub struct SearchReport {
    pub outcome: SearchOutcome,
    pub trace: SearchTrace,
}

/// Sequence-level outcome from the public worker API.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum SearchOutcome {
    Filtered(FilterReason),
    Hit(Hit),
}

#[derive(Debug, Clone)]
struct QueryState {
    profile: Profile,
    background: BackgroundModel,
}

struct ScoreHitRequest<'a> {
    profile: &'a Profile,
    background: &'a BackgroundModel,
    sequence: &'a DigitalSequence,
    domain_result: &'a mut crate::domaindef::DomainResult,
    null_score: f32,
    forward_raw_score: f32,
    sequence_length: usize,
}

/// Reusable per-thread search executor.
#[derive(Debug)]
pub struct SearchWorker {
    plan: Arc<SearchPlanInner>,
    engine: SearchEngine,
    state: QueryState,
}

impl SearchWorker {
    fn from_plan(plan: Arc<SearchPlanInner>, hints: CapacityHints) -> Result<Self, HmmerError> {
        let mut engine =
            SearchEngine::new(plan.query.profile_template.num_nodes, hints.target_length);
        engine.do_max = plan.filters.max_mode;
        engine.msv_threshold = plan.filters.msv_threshold;
        engine.viterbi_threshold = plan.filters.viterbi_threshold;
        engine.forward_threshold = plan.filters.forward_threshold;
        engine.thresholds.by_e = false;
        engine.domain_config = plan.domain_config.clone();
        engine.oprofile = Some(plan.query.optimized_template.clone());

        let mut profile = plan.query.profile_template.clone();
        let mut background = plan.query.background_template.clone();
        crate::modelconfig::reconfigure_length(&mut profile, hints.target_length);
        background.set_length(hints.target_length);

        Ok(Self {
            plan,
            engine,
            state: QueryState {
                profile,
                background,
            },
        })
    }

    /// Search a single target sequence and return either a candidate hit or the
    /// stage that filtered it out.
    pub fn search(&mut self, sequence: &DigitalSequence) -> Result<SearchReport, HmmerError> {
        let prepared = TargetRun::<Prepared>::prepare(self, sequence)?;
        let passed = match prepared.run_filters() {
            ControlFlow::Break(report) => return Ok(report),
            ControlFlow::Continue(passed) => passed,
        };

        let decoded = match passed.decode()? {
            ControlFlow::Break(report) => return Ok(report),
            ControlFlow::Continue(decoded) => decoded,
        };

        Ok(decoded.score()?.finish())
    }

    pub fn metrics(&self) -> SearchMetrics {
        SearchMetrics {
            stats: self.engine.stats.clone(),
        }
    }

    pub fn into_metrics(self) -> SearchMetrics {
        SearchMetrics {
            stats: self.engine.stats,
        }
    }

    pub fn reset_metrics(&mut self) {
        self.engine.stats = PipelineStats::default();
    }

    pub fn plan(&self) -> SearchPlan {
        SearchPlan {
            inner: Arc::clone(&self.plan),
        }
    }
}

#[derive(Debug)]
struct Prepared;

#[derive(Debug)]
struct PassedFilters;

#[derive(Debug)]
struct Decoded;

#[derive(Debug)]
struct Scored;

#[derive(Debug)]
struct TargetRun<'w, 's, Phase> {
    worker: &'w mut SearchWorker,
    sequence: &'s DigitalSequence,
    sequence_length: usize,
    null_score: f32,
    forward_raw_score: f32,
    domain_result: Option<crate::domaindef::DomainResult>,
    hit: Option<Hit>,
    trace: SearchTrace,
    _phase: PhantomData<Phase>,
}

impl<'w, 's> TargetRun<'w, 's, Prepared> {
    fn prepare(
        worker: &'w mut SearchWorker,
        sequence: &'s DigitalSequence,
    ) -> Result<Self, HmmerError> {
        let sequence_length = sequence.len();
        worker.engine.stats.num_sequences += 1;
        worker.engine.stats.num_residues += sequence_length as u64;

        let null_score = worker.engine.prepare_target(
            &mut worker.state.profile,
            &mut worker.state.background,
            &sequence.residues,
            sequence_length,
        );

        let mut trace = SearchTrace::new(sequence_length);
        trace.null_score = Some(null_score);

        Ok(Self {
            worker,
            sequence,
            sequence_length,
            null_score,
            forward_raw_score: 0.0,
            domain_result: None,
            hit: None,
            trace,
            _phase: PhantomData,
        })
    }

    fn run_filters(mut self) -> ControlFlow<SearchReport, TargetRun<'w, 's, PassedFilters>> {
        let om = self.worker.engine.oprofile.as_ref().unwrap();

        let msv_raw_score =
            f32_msv::msv_filter_f32(&self.sequence.residues, self.sequence_length, om);

        self.trace.msv_raw_score = Some(msv_raw_score);

        let seq_score_msv = (msv_raw_score - self.null_score) / LOG2;
        let p_msv = SearchEngine::msv_pvalue(&self.worker.state.profile, seq_score_msv);
        if !self.worker.engine.do_max && p_msv > self.worker.engine.msv_threshold {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
        }
        self.worker.engine.stats.sequences_past_msv += 1;
        self.worker.engine.stats.residues_past_msv += self.sequence_length as u64;

        let seq_score_bias = (msv_raw_score - self.null_score) / LOG2;
        let p_bias = SearchEngine::msv_pvalue(&self.worker.state.profile, seq_score_bias);
        if !self.worker.engine.do_max && p_bias > self.worker.engine.msv_threshold {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
        }
        self.worker.engine.stats.sequences_past_bias += 1;
        self.worker.engine.stats.residues_past_bias += self.sequence_length as u64;

        let Some(viterbi_raw_score) = self.worker.engine.run_viterbi_filter(
            &self.worker.state.profile,
            &self.sequence.residues,
            self.sequence_length,
            self.null_score,
        ) else {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Viterbi));
        };
        self.trace.viterbi_raw_score = Some(viterbi_raw_score);

        let Some(forward_raw_score) = self.worker.engine.run_forward_filter(
            &self.worker.state.profile,
            &self.sequence.residues,
            self.sequence_length,
            self.null_score,
        ) else {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Forward));
        };
        self.trace.forward_raw_score = Some(forward_raw_score);

        ControlFlow::Continue(TargetRun {
            worker: self.worker,
            sequence: self.sequence,
            sequence_length: self.sequence_length,
            null_score: self.null_score,
            forward_raw_score,
            domain_result: None,
            hit: None,
            trace: self.trace,
            _phase: PhantomData,
        })
    }
}

impl<'w, 's> TargetRun<'w, 's, PassedFilters> {
    fn decode(self) -> Result<ControlFlow<SearchReport, TargetRun<'w, 's, Decoded>>, HmmerError> {
        let Some((forward_raw_score, domain_result)) = self
            .worker
            .engine
            .run_checkpointed_forward_backward_and_domains(
                &mut self.worker.state.profile,
                &self.sequence.residues,
                self.sequence_length,
            )?
        else {
            return Ok(ControlFlow::Break(
                self.trace.into_filtered_report(FilterReason::DecodeFailed),
            ));
        };

        if domain_result.domains.is_empty() {
            return Ok(ControlFlow::Break(
                self.trace.into_filtered_report(FilterReason::NoDomains),
            ));
        }

        let mut trace = self.trace;
        trace.forward_raw_score = Some(forward_raw_score);
        trace.domain_count = domain_result.domains.len();

        Ok(ControlFlow::Continue(TargetRun {
            worker: self.worker,
            sequence: self.sequence,
            sequence_length: self.sequence_length,
            null_score: self.null_score,
            forward_raw_score,
            domain_result: Some(domain_result),
            hit: None,
            trace,
            _phase: PhantomData,
        }))
    }
}

impl<'w, 's> TargetRun<'w, 's, Decoded> {
    fn score(mut self) -> Result<TargetRun<'w, 's, Scored>, HmmerError> {
        let mut hits = TopHits::new();
        let mut domain_result = self
            .domain_result
            .take()
            .expect("decoded phase must have a domain result");

        let added = self.worker.engine.score_and_add_hit(
            &mut hits,
            ScoreHitRequest {
                profile: &self.worker.state.profile,
                background: &self.worker.state.background,
                sequence: self.sequence,
                domain_result: &mut domain_result,
                null_score: self.null_score,
                forward_raw_score: self.forward_raw_score,
                sequence_length: self.sequence_length,
            },
        );
        if !added || hits.is_empty() {
            return Err(HmmerError::Internal(
                "decoded search hit unexpectedly vanished during scoring".into(),
            ));
        }

        let hit = hits
            .hits
            .into_iter()
            .next()
            .expect("score_and_add_hit should have produced one hit");

        Ok(TargetRun {
            worker: self.worker,
            sequence: self.sequence,
            sequence_length: self.sequence_length,
            null_score: self.null_score,
            forward_raw_score: self.forward_raw_score,
            domain_result: None,
            hit: Some(hit),
            trace: self.trace,
            _phase: PhantomData,
        })
    }
}

impl<'w, 's> TargetRun<'w, 's, Scored> {
    fn finish(self) -> SearchReport {
        SearchReport {
            outcome: SearchOutcome::Hit(self.hit.expect("scored phase must contain a hit")),
            trace: self.trace,
        }
    }
}

/// The accelerated comparison pipeline.
#[derive(Debug)]
struct SearchEngine {
    // Reporting and inclusion thresholds
    thresholds: Thresholds,

    // Search space
    search_space: f64,

    // Filter thresholds
    do_max: bool,
    msv_threshold: f64,     // MSV filter threshold (default 0.02)
    viterbi_threshold: f64, // Viterbi filter threshold (default 1e-3)
    forward_threshold: f64, // Forward filter threshold (default 1e-5)

    // Accounting
    stats: PipelineStats,

    // Reusable DP matrices (allocated once, grown as needed)
    posterior_matrix: ScoreMatrix,

    // Pooled buffers for per-domain rescoring (reused across sequences)
    rescore_buf: RescoringBuffers,

    // Domain definition configuration (immutable after construction)
    domain_config: DomainConfig,

    // Reusable domain definition workspace (pooled across sequences)
    domain_workspace: DomainWorkspace,

    // SIMD-optimized profile (built lazily from the generic profile)
    oprofile: Option<OptimizedProfile>,

    // Reusable buffer for odds-space segment replay
    seg_buf: simd_fwd_bck::OddsSegmentBuf,
}

impl SearchEngine {
    /// Create a new pipeline with default parameters.
    fn new(m: usize, sequence_length: usize) -> Self {
        SearchEngine {
            thresholds: Thresholds::default(),
            search_space: 0.0,
            do_max: false,
            msv_threshold: DEFAULT_MSV_THRESHOLD,
            viterbi_threshold: DEFAULT_VITERBI_THRESHOLD,
            forward_threshold: DEFAULT_FORWARD_THRESHOLD,
            stats: PipelineStats::default(),
            posterior_matrix: ScoreMatrix::new(m, sequence_length).unwrap(),
            rescore_buf: RescoringBuffers::new(m),
            domain_config: DomainConfig::default(),
            domain_workspace: DomainWorkspace::new(),
            oprofile: None,
            seg_buf: simd_fwd_bck::OddsSegmentBuf::new(DEFAULT_SEG_BUF_STRIPES, m.div_ceil(4)),
        }
    }

    #[inline]
    fn msv_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        if profile.ev_params[EvParam::MsvMu.idx()] != EV_PARAM_UNSET {
            evalues::msv_score_to_pvalue(
                bitscore,
                profile.ev_params[EvParam::MsvMu.idx()],
                profile.ev_params[EvParam::MsvLambda.idx()],
            )
        } else {
            0.0
        }
    }

    #[inline]
    fn viterbi_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        if profile.ev_params[EvParam::ViterbiMu.idx()] != EV_PARAM_UNSET {
            evalues::viterbi_score_to_pvalue(
                bitscore,
                profile.ev_params[EvParam::ViterbiMu.idx()],
                profile.ev_params[EvParam::ViterbiLambda.idx()],
            )
        } else {
            0.0
        }
    }

    #[inline]
    fn forward_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        if profile.ev_params[EvParam::ForwardTau.idx()] != EV_PARAM_UNSET {
            evalues::forward_score_to_pvalue(
                bitscore,
                profile.ev_params[EvParam::ForwardTau.idx()],
                profile.ev_params[EvParam::ForwardLambda.idx()],
            )
        } else {
            0.0
        }
    }

    fn score_and_add_hit(&mut self, top_hits: &mut TopHits, request: ScoreHitRequest<'_>) -> bool {
        let ScoreHitRequest {
            profile,
            background,
            sequence,
            domain_result,
            null_score,
            forward_raw_score,
            sequence_length,
        } = request;
        let num_domains = domain_result.domains.len();

        // ---- Per-domain scoring (matches C p7_pipeline.c) ----
        for d in 0..num_domains {
            let dom = &mut domain_result.domains[d];
            let domain_length = dom.envelope_end - dom.envelope_start + 1;

            // Per-domain null2 bias: log(1 + omega * exp(domcorrection))
            dom.domain_bias = logsum::flogsum(0.0, (background.omega).ln() + dom.domain_correction);

            // Domain bitscore: envsc + flanking_null - null_score - dombias, all in nats, then / LOG2
            let flanking_null = (sequence_length as f32 - domain_length as f32)
                * (sequence_length as f32 / (sequence_length as f32 + SCORE_LENGTH_PRIOR_OFFSET))
                    .ln();
            dom.bitscore =
                (dom.envelope_score + flanking_null - null_score - dom.domain_bias) / LOG2;

            // Domain E-value
            dom.log_pvalue = if profile.ev_params[EvParam::ForwardTau.idx()] != EV_PARAM_UNSET {
                let pv = evalues::forward_score_to_pvalue(
                    dom.bitscore,
                    profile.ev_params[EvParam::ForwardTau.idx()],
                    profile.ev_params[EvParam::ForwardLambda.idx()],
                );
                pv.max(MIN_PVALUE_CLAMP).ln()
            } else {
                -(dom.bitscore as f64)
            };
        }

        // ---- Sequence-level scoring ----
        // pre_score: Forward score without null2 correction
        let pre_score = (forward_raw_score - null_score) / LOG2;

        // seqbias: computed from domain_def->n2sc (per-position null2 log scores)
        let seqbias_sum: f32 = self.domain_workspace.null2_scores[1..=sequence_length]
            .iter()
            .sum();
        let seqbias = logsum::flogsum(0.0, (background.omega).ln() + seqbias_sum);

        // Corrected sequence score
        let mut seq_score = (forward_raw_score - null_score - seqbias) / LOG2;

        // sum_score reconstruction: only count positive-scoring domains
        let mut sum_raw = 0.0f32;
        let mut sum_correction = 0.0f32;
        let mut sum_ld = 0usize;
        for dom in &domain_result.domains {
            if dom.envelope_score - dom.domain_correction > 0.0 {
                sum_raw += dom.envelope_score;
                sum_ld += dom.envelope_end - dom.envelope_start + 1;
                sum_correction += dom.domain_correction;
            }
        }
        let sum_seqbias = logsum::flogsum(0.0, (background.omega).ln() + sum_correction);
        sum_raw += (sequence_length as f32 - sum_ld as f32)
            * (sequence_length as f32 / (sequence_length as f32 + SCORE_LENGTH_PRIOR_OFFSET)).ln();
        let pre2_score = (sum_raw - null_score) / LOG2;
        let sum_score = (sum_raw - null_score - sum_seqbias) / LOG2;

        // Use reconstruction score if better
        let mut final_pre_score = pre_score;
        if sum_ld > 0 && sum_score > seq_score {
            seq_score = sum_score;
            final_pre_score = pre2_score;
        }

        // E-value computation
        let z = if self.search_space > 0.0 {
            self.search_space
        } else {
            self.stats.num_sequences.max(1) as f64
        };

        let log_pvalue = if profile.ev_params[EvParam::ForwardTau.idx()] != EV_PARAM_UNSET {
            let pv = evalues::forward_score_to_pvalue(
                seq_score,
                profile.ev_params[EvParam::ForwardTau.idx()],
                profile.ev_params[EvParam::ForwardLambda.idx()],
            );
            pv.max(MIN_PVALUE_CLAMP).ln()
        } else {
            -(seq_score as f64)
        };

        // Reporting threshold check
        let evalue = (log_pvalue + z.ln()).exp();
        if self.thresholds.by_e && evalue > self.thresholds.evalue {
            return false;
        }

        // ---- Create hit ----
        let hit = top_hits.create_next_hit();
        hit.name = sequence.name.clone();
        hit.accession = sequence.accession.clone();
        hit.description = sequence.description.clone();
        hit.score = seq_score;
        hit.pre_score = final_pre_score;
        hit.sum_score = sum_score;
        hit.log_pvalue = log_pvalue;
        hit.pre_log_pvalue = log_pvalue;
        hit.sum_log_pvalue = log_pvalue;
        hit.sortkey = -log_pvalue;
        hit.num_expected = domain_result.num_expected;
        hit.num_regions = domain_result.num_regions;
        hit.num_clustered = domain_result.num_clustered;
        hit.num_overlaps = domain_result.num_overlaps;
        hit.num_envelopes = domain_result.num_envelopes;
        hit.num_domains = num_domains;
        hit.seqidx = if sequence.database_index >= 0 {
            Some(sequence.database_index as usize)
        } else {
            None
        };

        // Copy domain data
        let mut best_score = f32::NEG_INFINITY;
        let mut best_d = None;
        for (d, dom) in domain_result.domains.drain(..).enumerate() {
            if dom.bitscore > best_score {
                best_score = dom.bitscore;
                best_d = Some(d);
            }
            hit.domains.push(dom);
        }
        hit.best_domain = best_d;

        self.stats.sequences_output += 1;
        self.stats.residues_output += sequence_length as u64;
        true
    }

    fn prepare_target(
        &mut self,
        profile: &mut Profile,
        background: &mut BackgroundModel,
        digital_sequence: &[u8],
        sequence_length: usize,
    ) -> f32 {
        // Set the target length model on both background and profile.
        background.set_length(sequence_length);
        crate::modelconfig::reconfigure_length(profile, sequence_length);

        // Build or update the SIMD-optimized profile (lazy, reused across sequences).
        let om = self
            .oprofile
            .get_or_insert_with(|| OptimizedProfile::from_profile(profile));
        om.reconfigure_length(sequence_length);

        background.null_one(digital_sequence, sequence_length)
    }

    fn run_viterbi_filter(
        &mut self,
        profile: &Profile,
        digital_sequence: &[u8],
        sequence_length: usize,
        null_score: f32,
    ) -> Option<f32> {
        let om = self.oprofile.as_ref().unwrap();
        let viterbi_raw_score =
            f32_viterbi::viterbi_filter_f32(digital_sequence, sequence_length, om);

        let seq_score_vit = (viterbi_raw_score - null_score) / LOG2;
        let p_vit = Self::viterbi_pvalue(profile, seq_score_vit);
        if !self.do_max && p_vit > self.viterbi_threshold {
            return None;
        }

        self.stats.sequences_past_viterbi += 1;
        self.stats.residues_past_viterbi += sequence_length as u64;
        Some(viterbi_raw_score)
    }

    fn run_forward_filter(
        &mut self,
        profile: &Profile,
        digital_sequence: &[u8],
        sequence_length: usize,
        null_score: f32,
    ) -> Option<f32> {
        let om = self.oprofile.as_ref().unwrap();
        let fwd_result = simd_fwd::forward_filter(digital_sequence, sequence_length, om);

        let seq_score_fwd = (fwd_result.score - null_score) / LOG2;
        let p_fwd = Self::forward_pvalue(profile, seq_score_fwd);
        if !self.do_max && p_fwd > self.forward_threshold {
            return None;
        }

        self.stats.sequences_past_forward += 1;
        self.stats.residues_past_forward += sequence_length as u64;
        Some(fwd_result.score)
    }

    fn run_checkpointed_forward_backward_and_domains(
        &mut self,
        profile: &mut Profile,
        digital_sequence: &[u8],
        sequence_length: usize,
    ) -> Result<Option<(f32, crate::domaindef::DomainResult)>, HmmerError> {
        self.domain_workspace.reuse();
        self.domain_workspace.grow_to(sequence_length);

        let om = self.oprofile.as_ref().unwrap();
        let (checkpoints, simd_cp_data) =
            simd_fwd_bck::forward_checkpointed_simd(digital_sequence, sequence_length, om);
        let forward_raw_score = checkpoints.overall_score;

        self.posterior_matrix
            .resize(profile.num_nodes, sequence_length)?;
        let om = self.oprofile.as_ref().unwrap();
        if forward_backward::backward_decode_prob_space(
            digital_sequence,
            profile,
            om,
            &checkpoints,
            &simd_cp_data,
            &mut self.posterior_matrix,
            Some(&mut self.domain_workspace),
            &mut self.seg_buf,
        )
        .is_err()
        {
            return Ok(None);
        }

        let mut om = self.oprofile.take().unwrap();
        let mut seg_buf = std::mem::take(&mut self.seg_buf);
        let domain_result = self.domain_workspace.by_posterior_heuristics(
            &self.domain_config,
            digital_sequence,
            sequence_length,
            profile,
            &mut self.rescore_buf,
            &mut om,
            &mut seg_buf,
        );
        self.oprofile = Some(om);
        self.seg_buf = seg_buf;

        Ok(Some((forward_raw_score, domain_result)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::Alphabet;
    use crate::rng::XorShift64;
    use crate::test_helpers::{random_digital_seq, setup_profile};

    #[test]
    fn test_filter_policy_defaults() {
        let filters = FilterPolicy::default();
        assert!((filters.msv_threshold - 0.02).abs() < 1e-10);
        assert!((filters.viterbi_threshold - 1e-3).abs() < 1e-10);
        assert!((filters.forward_threshold - 1e-5).abs() < 1e-10);
        assert!(!filters.max_mode);
    }

    #[test]
    fn test_search_plan_spawns_worker() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(7);
        let (_, bg, gm) = setup_profile(&mut rng, 8, 32, &abc, crate::config::SearchMode::Local);

        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let plan = SearchPlan::builder(query).build();
        let worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        assert_eq!(worker.state.profile.num_nodes, 8);
        assert_eq!(worker.state.profile.target_length, 32);
    }

    #[test]
    fn test_search_worker_returns_report() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(11);
        let (_, bg, gm) = setup_profile(&mut rng, 8, 32, &abc, crate::config::SearchMode::Local);
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "random-seq".to_string(),
            accession: None,
            description: None,
            residues,
            source_length: 32,
            database_index: -1,
        };

        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let plan = SearchPlan::builder(query).build();
        let mut worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        let report = worker.search(&sequence).unwrap();
        assert_eq!(report.trace.sequence_length, 32);
        match report.outcome {
            SearchOutcome::Filtered(_) | SearchOutcome::Hit(_) => {}
        }
    }
}
