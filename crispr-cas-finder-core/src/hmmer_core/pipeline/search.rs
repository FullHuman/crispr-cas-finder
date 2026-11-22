// p7_pipeline.rs - The accelerated comparison pipeline
//
// Port of src/p7_pipeline.c

use crate::hmmer_core::background::BackgroundModel;
use crate::hmmer_core::config::*;
use crate::hmmer_core::constants::pipeline::search::{
    DEFAULT_FORWARD_THRESHOLD, DEFAULT_MSV_THRESHOLD, DEFAULT_VITERBI_THRESHOLD, LOG2,
    MIN_PVALUE_CLAMP, SCORE_LENGTH_PRIOR_OFFSET,
};
use crate::hmmer_core::domaindef::{DomainConfig, DomainWorkspace, RescoringBuffers};
use crate::hmmer_core::dynamic_programming::odds_decode::{self, OddsTables};
use crate::hmmer_core::dynamic_programming::simd::forward_filter as simd_fwd;
use crate::hmmer_core::dynamic_programming::simd::msv_filter;
use crate::hmmer_core::dynamic_programming::simd::oprofile::OptimizedProfile;
use crate::hmmer_core::dynamic_programming::simd::viterbi_filter;
use crate::hmmer_core::errors::HmmerError;
use crate::hmmer_core::evalues;
use crate::hmmer_core::hit::Hit;
use crate::hmmer_core::logsum;
use crate::hmmer_core::profile::Profile;
use crate::hmmer_core::score_matrix::*;
use crate::hmmer_core::sequence::DigitalSequence;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::sync::Arc;

/// Pipeline accounting counters.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub num_sequences: u64,
    pub sequences_past_msv: u64,
    pub sequences_past_bias: u64,
    pub sequences_past_viterbi: u64,
    pub sequences_past_forward: u64,
    pub sequences_output: u64,
}

/// Execution metrics of a search worker.
#[derive(Debug, Clone, Default)]
pub struct SearchMetrics {
    pub stats: PipelineStats,
}

/// Filter configuration for the search executor.
///
/// Every filter is decided on the score HMMER's own filter returns, in
/// bytes for the MSV and bias filters and in 16-bit words for the Viterbi
/// filter, with HMMER's tests: the same sequences pass each stage as in
/// `hmmsearch`.
#[derive(Debug, Clone)]
pub struct FilterPolicy {
    /// `hmmsearch --max`: no MSV, bias or Viterbi filter runs and the
    /// Forward score is not tested, so every sequence is decoded.
    pub max_mode: bool,
    pub bias_filter: bool,
    pub msv_threshold: f64,
    pub viterbi_threshold: f64,
    pub forward_threshold: f64,
}

impl Default for FilterPolicy {
    fn default() -> Self {
        Self {
            max_mode: false,
            bias_filter: true,
            msv_threshold: DEFAULT_MSV_THRESHOLD,
            viterbi_threshold: DEFAULT_VITERBI_THRESHOLD,
            forward_threshold: DEFAULT_FORWARD_THRESHOLD,
        }
    }
}

impl FilterPolicy {
    /// Validate filter P-value thresholds before constructing a search plan.
    pub fn validate(&self) -> Result<(), HmmerError> {
        for (name, threshold) in [
            ("MSV", self.msv_threshold),
            ("Viterbi", self.viterbi_threshold),
            ("Forward", self.forward_threshold),
        ] {
            if !threshold.is_finite() || !(0.0 < threshold && threshold <= 1.0) {
                return Err(HmmerError::InvalidArgument(format!(
                    "{name} filter threshold must be finite and in (0, 1], got {threshold}"
                )));
            }
        }
        if self.msv_threshold < self.viterbi_threshold
            || self.viterbi_threshold < self.forward_threshold
        {
            return Err(HmmerError::InvalidArgument(format!(
                "filter thresholds must satisfy MSV >= Viterbi >= Forward, got {} >= {} >= {}",
                self.msv_threshold, self.viterbi_threshold, self.forward_threshold
            )));
        }
        Ok(())
    }
}

/// Immutable compiled query state shared by workers.
#[derive(Debug, Clone)]
pub struct SearchQuery {
    profile_template: Profile,
    background_template: BackgroundModel,
    optimized_template: OptimizedProfile,
}

fn validate_profile_calibration(profile: &Profile) -> Result<(), HmmerError> {
    let parameters = [
        (EvParam::MsvMu, "MSV mu", false),
        (EvParam::MsvLambda, "MSV lambda", true),
        (EvParam::ViterbiMu, "Viterbi mu", false),
        (EvParam::ViterbiLambda, "Viterbi lambda", true),
        (EvParam::ForwardTau, "Forward tau", false),
        (EvParam::ForwardLambda, "Forward lambda", true),
    ];

    for (parameter, name, must_be_positive) in parameters {
        let value = profile.ev_params[parameter.idx()];
        if value == EV_PARAM_UNSET {
            return Err(HmmerError::InvalidArgument(format!(
                "search profile is missing the {name} calibration parameter"
            )));
        }
        if !value.is_finite() || (must_be_positive && value <= 0.0) {
            let requirement = if must_be_positive {
                "finite and greater than zero"
            } else {
                "finite"
            };
            return Err(HmmerError::InvalidArgument(format!(
                "search profile {name} calibration parameter must be {requirement}, got {value}"
            )));
        }
    }

    Ok(())
}

impl SearchQuery {
    /// Build a reusable search query from a configured profile and background model.
    pub fn from_configured_profile(
        profile: Profile,
        mut background: BackgroundModel,
    ) -> Result<Self, HmmerError> {
        if profile.num_nodes == 0 {
            return Err(HmmerError::InvalidArgument(
                "search profile must contain at least one model node".into(),
            ));
        }
        validate_profile_calibration(&profile)?;
        let canonical_size = profile.alphabet.canonical_size;
        background.set_filter(
            profile.num_nodes,
            &profile.model_composition[..canonical_size],
        )?;
        let optimized_template = OptimizedProfile::from_profile(&profile);
        Ok(Self {
            profile_template: profile,
            background_template: background,
            optimized_template,
        })
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

    pub fn build(self) -> Result<SearchPlan, HmmerError> {
        self.filters.validate()?;
        Ok(SearchPlan {
            inner: Arc::new(SearchPlanInner {
                query: self.query,
                filters: self.filters,
                domain_config: self.domain_config,
            }),
        })
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
    pub null_score: Option<f32>,
    pub bias_filter_score: Option<f32>,
    /// What HMMER's MSV filter returns, the score the MSV and bias filters
    /// are decided on: plus infinity when its bytes overflowed. `None` with
    /// `FilterPolicy::max_mode`.
    pub msv_filter_score: Option<f32>,
    /// What HMMER's Viterbi filter returns, the score the Viterbi filter is
    /// decided on: plus infinity when its words overflowed. `None` with
    /// `FilterPolicy::max_mode`, and when the filter did not run: the
    /// sequence stopped earlier, or its MSV score already met the Viterbi
    /// threshold.
    pub viterbi_filter_score: Option<f32>,
    pub forward_raw_score: Option<f32>,
    pub domain_count: usize,
    pub filter_reason: Option<FilterReason>,
}

impl SearchTrace {
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
    domain_result: &'a mut crate::hmmer_core::domaindef::DomainResult,
    null_score: f32,
    forward_raw_score: f32,
    sequence_length: usize,
}

/// Reusable per-thread search executor.
#[derive(Debug)]
pub struct SearchWorker {
    engine: SearchEngine,
    state: QueryState,
}

impl SearchWorker {
    fn from_plan(plan: Arc<SearchPlanInner>, hints: CapacityHints) -> Result<Self, HmmerError> {
        let mut engine =
            SearchEngine::new(plan.query.profile_template.num_nodes, hints.target_length);
        engine.do_max = plan.filters.max_mode;
        engine.do_biasfilter = plan.filters.bias_filter && !plan.filters.max_mode;
        engine.msv_threshold = plan.filters.msv_threshold;
        engine.viterbi_threshold = plan.filters.viterbi_threshold;
        engine.forward_threshold = plan.filters.forward_threshold;
        engine.domain_config = plan.domain_config.clone();
        engine.odds_tables = OddsTables::new(&plan.query.profile_template);
        engine.oprofile = Some(plan.query.optimized_template.clone());

        let mut profile = plan.query.profile_template.clone();
        let mut background = plan.query.background_template.clone();
        crate::hmmer_core::modelconfig::reconfigure_length(&mut profile, hints.target_length);
        background.set_length(hints.target_length);

        Ok(Self {
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

        Ok(decoded.score().finish())
    }

    pub fn into_metrics(self) -> SearchMetrics {
        SearchMetrics {
            stats: self.engine.stats,
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
    domain_result: Option<crate::hmmer_core::domaindef::DomainResult>,
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

        let null_score = worker.engine.prepare_target(
            &mut worker.state.profile,
            &mut worker.state.background,
            &sequence.residues,
            sequence_length,
        );

        let trace = SearchTrace {
            null_score: Some(null_score),
            ..SearchTrace::default()
        };

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

    /// The filter cascade of `p7_Pipeline()`, each stage decided on the
    /// number HMMER decides it on: the scores its MSV and Viterbi filters
    /// return (msv_filter.rs, viterbi_filter.rs), the null and bias filter
    /// scores in its arithmetic, and its P-values, compared as it compares
    /// them. The same sequences then pass each stage as in `hmmsearch`.
    ///
    /// An overflowed filter score is plus infinity, whose P-value is zero:
    /// the sequence passes. A sequence whose MSV score, against the bias
    /// filter score, already meets the Viterbi threshold skips the Viterbi
    /// filter, as in HMMER.
    ///
    /// With `FilterPolicy::max_mode`, `hmmsearch --max`, the Forward pass is
    /// all that runs, and its score is not tested: every sequence counts as
    /// past each stage and goes on to be decoded. HMMER gets there with the
    /// bias filter off and its three thresholds at one, which no P-value
    /// exceeds. Here the MSV filter is not computed at all and its P-value
    /// stands at zero, which passes the MSV test and, as in HMMER, skips
    /// the Viterbi filter.
    fn run_filters(mut self) -> ControlFlow<SearchReport, TargetRun<'w, 's, PassedFilters>> {
        let engine = &mut self.worker.engine;
        let ev_params = &self.worker.state.profile.ev_params;
        let residues = &self.sequence.residues;
        let length = self.sequence_length;
        let om = engine.oprofile.as_ref().unwrap();

        let msv_score = if engine.do_max {
            None
        } else {
            Some(msv_filter::msv_filter_with(
                residues,
                length,
                om,
                &mut engine.msv_segment_row,
                &mut engine.msv_filter_row,
            ))
        };
        self.trace.msv_filter_score = msv_score;
        let msv_pvalue = |reference: f32| {
            msv_score.map_or(0.0, |score| {
                evalues::gumbel_survival(
                    evalues::filter_bit_score(score, reference),
                    ev_params[EvParam::MsvMu.idx()],
                    ev_params[EvParam::MsvLambda.idx()],
                )
            })
        };
        let mut pvalue = msv_pvalue(self.null_score);
        if pvalue > engine.msv_threshold {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
        }
        engine.stats.sequences_past_msv += 1;

        let filter_score = if engine.do_biasfilter {
            let score = self.worker.state.background.filter_score(residues, length);
            self.trace.bias_filter_score = Some(score);
            pvalue = msv_pvalue(score);
            if pvalue > engine.msv_threshold {
                return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
            }
            score
        } else {
            self.null_score
        };
        engine.stats.sequences_past_bias += 1;

        if pvalue > engine.viterbi_threshold {
            let viterbi_score = viterbi_filter::viterbi_filter_with(
                residues,
                length,
                om,
                &mut engine.viterbi_filter_rows,
            );
            self.trace.viterbi_filter_score = Some(viterbi_score);
            let pvalue = evalues::gumbel_survival(
                evalues::filter_bit_score(viterbi_score, filter_score),
                ev_params[EvParam::ViterbiMu.idx()],
                ev_params[EvParam::ViterbiLambda.idx()],
            );
            if pvalue > engine.viterbi_threshold {
                return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Viterbi));
            }
        }
        engine.stats.sequences_past_viterbi += 1;

        let forward_raw_score = simd_fwd::forward_filter(residues, length, om);
        self.trace.forward_raw_score = Some(forward_raw_score);
        let pvalue = evalues::exponential_survival(
            evalues::filter_bit_score(forward_raw_score, filter_score),
            ev_params[EvParam::ForwardTau.idx()],
            ev_params[EvParam::ForwardLambda.idx()],
        );
        if !engine.do_max && pvalue > engine.forward_threshold {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Forward));
        }
        engine.stats.sequences_past_forward += 1;

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
        let Some((forward_raw_score, domain_result)) =
            self.worker.engine.run_forward_backward_and_domains(
                &mut self.worker.state.profile,
                &self.sequence.residues,
                self.sequence_length,
                self.forward_raw_score,
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
    fn score(mut self) -> TargetRun<'w, 's, Scored> {
        let mut domain_result = self
            .domain_result
            .take()
            .expect("decoded phase must have a domain result");

        let hit = self.worker.engine.score_hit(ScoreHitRequest {
            profile: &self.worker.state.profile,
            background: &self.worker.state.background,
            sequence: self.sequence,
            domain_result: &mut domain_result,
            null_score: self.null_score,
            forward_raw_score: self.forward_raw_score,
            sequence_length: self.sequence_length,
        });

        TargetRun {
            worker: self.worker,
            sequence: self.sequence,
            sequence_length: self.sequence_length,
            null_score: self.null_score,
            forward_raw_score: self.forward_raw_score,
            domain_result: None,
            hit: Some(hit),
            trace: self.trace,
            _phase: PhantomData,
        }
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
    // Filter thresholds
    do_max: bool,
    do_biasfilter: bool,
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

    // Emission and transition odds for probability-space decoding
    odds_tables: OddsTables,

    // DP rows of HMMER's own MSV and Viterbi filters, reused across
    // sequences: the single-segment pass of the MSV filter, its full
    // recurrence, and the Viterbi filter
    msv_segment_row: Vec<i8>,
    msv_filter_row: Vec<u8>,
    viterbi_filter_rows: viterbi_filter::Rows,

    // The null score of the last target length, which depends on nothing
    // else
    null_score_of_length: Option<(usize, f32)>,
}

impl SearchEngine {
    /// Create a new pipeline with default parameters.
    fn new(m: usize, sequence_length: usize) -> Self {
        SearchEngine {
            do_max: false,
            do_biasfilter: true,
            msv_threshold: DEFAULT_MSV_THRESHOLD,
            viterbi_threshold: DEFAULT_VITERBI_THRESHOLD,
            forward_threshold: DEFAULT_FORWARD_THRESHOLD,
            stats: PipelineStats::default(),
            posterior_matrix: ScoreMatrix::new(m, sequence_length).unwrap(),
            rescore_buf: RescoringBuffers::new(m),
            domain_config: DomainConfig::default(),
            domain_workspace: DomainWorkspace::new(),
            oprofile: None,
            odds_tables: OddsTables::default(),
            msv_segment_row: Vec::new(),
            msv_filter_row: Vec::new(),
            viterbi_filter_rows: viterbi_filter::Rows::default(),
            null_score_of_length: None,
        }
    }

    #[inline]
    fn forward_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        evalues::forward_score_to_pvalue(
            bitscore,
            profile.ev_params[EvParam::ForwardTau.idx()],
            profile.ev_params[EvParam::ForwardLambda.idx()],
        )
    }

    fn score_hit(&mut self, request: ScoreHitRequest<'_>) -> Hit {
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
            debug_assert!(dom.envelope_end >= dom.envelope_start);
            let domain_length = dom.envelope_end - dom.envelope_start + 1;
            debug_assert!(domain_length <= sequence_length);

            // Per-domain null2 bias: log(1 + omega * exp(domcorrection))
            dom.domain_bias = logsum::flogsum(0.0, (background.omega).ln() + dom.domain_correction);

            // HMMER's target-length prior charges the envelope for the null
            // score of its flanking residues; the +3 is p7_pipeline's prior offset.
            let flanking_length = sequence_length.saturating_sub(domain_length);
            let flanking_null = flanking_length as f32
                * (sequence_length as f32 / (sequence_length as f32 + SCORE_LENGTH_PRIOR_OFFSET))
                    .ln();
            dom.bitscore =
                (dom.envelope_score + flanking_null - null_score - dom.domain_bias) / LOG2;

            // Domain E-value
            let pv = Self::forward_pvalue(profile, dom.bitscore);
            dom.log_pvalue = pv.max(MIN_PVALUE_CLAMP).ln();
        }

        // ---- Sequence-level scoring ----
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
        debug_assert!(sum_ld <= sequence_length);
        sum_raw += sequence_length.saturating_sub(sum_ld) as f32
            * (sequence_length as f32 / (sequence_length as f32 + SCORE_LENGTH_PRIOR_OFFSET)).ln();
        let sum_score = (sum_raw - null_score - sum_seqbias) / LOG2;

        // Use reconstruction score if better
        if sum_ld > 0 && sum_score > seq_score {
            seq_score = sum_score;
        }

        let log_pvalue = Self::forward_pvalue(profile, seq_score)
            .max(MIN_PVALUE_CLAMP)
            .ln();

        self.stats.sequences_output += 1;
        Hit {
            name: sequence.name.clone(),
            score: seq_score,
            log_pvalue,
            num_regions: domain_result.num_regions,
            num_clustered: domain_result.num_clustered,
            num_overlaps: domain_result.num_overlaps,
            num_envelopes: domain_result.num_envelopes,
            domains: std::mem::take(&mut domain_result.domains),
        }
    }

    fn prepare_target(
        &mut self,
        profile: &mut Profile,
        background: &mut BackgroundModel,
        digital_sequence: &[u8],
        sequence_length: usize,
    ) -> f32 {
        // Set the target length model on both background and profile. Targets
        // come sorted by length, so consecutive ones often share it; domain
        // rescoring restores the multihit length model it found, so a profile
        // configured for this length stays configured.
        background.set_length(sequence_length);
        if profile.target_length != sequence_length {
            crate::hmmer_core::modelconfig::reconfigure_length(profile, sequence_length);
        }

        // Build or update the SIMD-optimized profile (lazy, reused across sequences).
        let om = self
            .oprofile
            .get_or_insert_with(|| OptimizedProfile::from_profile(profile));
        if om.l != sequence_length {
            om.reconfigure_length(sequence_length);
        }

        match self.null_score_of_length {
            Some((length, score)) if length == sequence_length => score,
            _ => {
                let score = background.null_one(digital_sequence, sequence_length);
                self.null_score_of_length = Some((sequence_length, score));
                score
            }
        }
    }

    fn run_forward_backward_and_domains(
        &mut self,
        profile: &mut Profile,
        digital_sequence: &[u8],
        sequence_length: usize,
        forward_raw_score: f32,
    ) -> Result<Option<(f32, crate::hmmer_core::domaindef::DomainResult)>, HmmerError> {
        self.domain_workspace.grow_to(sequence_length);

        self.posterior_matrix
            .resize(profile.num_nodes, sequence_length)?;
        let decoded = odds_decode::decode_odds_space(
            digital_sequence,
            profile,
            &self.odds_tables,
            forward_raw_score,
            &mut self.posterior_matrix,
            Some(&mut self.domain_workspace),
        );
        if decoded.is_err() {
            return Ok(None);
        }

        let mut om = self.oprofile.take().unwrap();
        let domain_result = self.domain_workspace.by_posterior_heuristics(
            &self.domain_config,
            digital_sequence,
            sequence_length,
            profile,
            &mut self.rescore_buf,
            &mut om,
            &self.odds_tables,
        );
        self.oprofile = Some(om);

        Ok(Some((forward_raw_score, domain_result)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hmmer_core::alphabet::Alphabet;
    use crate::hmmer_core::hmm::Hmm;
    use crate::hmmer_core::rng::XorShift64;
    use crate::hmmer_core::test_helpers::{random_digital_seq, setup_profile};

    fn calibrate(profile: &mut Profile) {
        profile.ev_params = [0.0, 0.7, 0.0, 0.7, 0.0, 0.7];
    }

    /// The most likely residue of every match state of `hmm`, as a digital
    /// sequence: a target the profile scores well whatever the filters.
    fn consensus_sequence(hmm: &Hmm, alphabet: &Alphabet) -> DigitalSequence {
        let residues: Vec<u8> = (1..=hmm.num_nodes)
            .map(|node| {
                let emissions = &hmm.match_emissions(node)[..alphabet.canonical_size];
                emissions
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(code, _)| code as u8)
                    .unwrap()
            })
            .collect();
        DigitalSequence {
            name: "consensus".to_string(),
            residues,
        }
    }

    /// A sampled profile whose match states nearly always follow one
    /// another. The consensus of a plain sample pays as much in its random
    /// transitions as it gains in emissions and fails the Viterbi filter.
    fn conserved_profile(
        rng: &mut XorShift64,
        model_length: usize,
        target_length: usize,
        alphabet: &Alphabet,
    ) -> (Hmm, BackgroundModel, Profile) {
        let mut hmm = crate::hmmer_core::test_helpers::hmm_sample(rng, model_length, alphabet);
        for node in 0..model_length {
            let transitions = hmm.transitions_mut(node);
            transitions[HTransition::MatchToMatch as usize] = 0.98;
            transitions[HTransition::MatchToInsert as usize] = 0.01;
            transitions[HTransition::MatchToDelete as usize] = 0.01;
        }
        let mut background = BackgroundModel::new(alphabet);
        background.set_length(target_length);
        let mut profile = Profile::new(hmm.num_nodes, alphabet);
        crate::hmmer_core::modelconfig::profile_config(
            &hmm,
            &background,
            &mut profile,
            target_length,
        );
        (hmm, background, profile)
    }

    fn search_with(
        query: &SearchQuery,
        filters: FilterPolicy,
        sequence: &DigitalSequence,
    ) -> SearchReport {
        let plan = SearchPlan::builder(query.clone())
            .filters(filters)
            .build()
            .unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints {
                target_length: sequence.residues.len(),
            })
            .unwrap();
        worker.search(sequence).unwrap()
    }

    #[test]
    fn test_filter_policy_defaults() {
        let filters = FilterPolicy::default();
        assert!((filters.msv_threshold - 0.02).abs() < 1e-10);
        assert!((filters.viterbi_threshold - 1e-3).abs() < 1e-10);
        assert!((filters.forward_threshold - 1e-5).abs() < 1e-10);
        assert!(!filters.max_mode);
        assert!(filters.bias_filter);
    }

    #[test]
    fn test_filter_policy_rejects_invalid_thresholds() {
        for filters in [
            FilterPolicy {
                msv_threshold: 0.0,
                ..FilterPolicy::default()
            },
            FilterPolicy {
                viterbi_threshold: f64::NAN,
                ..FilterPolicy::default()
            },
            FilterPolicy {
                forward_threshold: 1.1,
                ..FilterPolicy::default()
            },
            FilterPolicy {
                msv_threshold: 1e-4,
                viterbi_threshold: 1e-3,
                ..FilterPolicy::default()
            },
        ] {
            assert!(filters.validate().is_err());
        }
    }

    #[test]
    fn test_search_query_rejects_uncalibrated_profile() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(5);
        let (_, bg, gm) = setup_profile(&mut rng, 8, 32, &abc);

        let error = SearchQuery::from_configured_profile(gm, bg).unwrap_err();
        assert!(error.to_string().contains("calibration parameter"));
    }

    #[test]
    fn test_search_plan_spawns_worker() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(7);
        let (_, bg, mut gm) = setup_profile(&mut rng, 8, 32, &abc);
        calibrate(&mut gm);

        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let plan = SearchPlan::builder(query).build().unwrap();
        let worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        assert_eq!(worker.state.profile.num_nodes, 8);
        assert_eq!(worker.state.profile.target_length, 32);
    }

    #[test]
    fn test_search_plan_rejects_invalid_filter_policy() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(9);
        let (_, bg, mut gm) = setup_profile(&mut rng, 8, 32, &abc);
        calibrate(&mut gm);
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let filters = FilterPolicy {
            forward_threshold: 2.0,
            ..FilterPolicy::default()
        };

        assert!(SearchPlan::builder(query).filters(filters).build().is_err());
    }

    #[test]
    fn test_search_worker_returns_report() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(11);
        let (_, bg, mut gm) = setup_profile(&mut rng, 8, 32, &abc);
        calibrate(&mut gm);
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "random-seq".to_string(),
            residues,
        };

        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let plan = SearchPlan::builder(query).build().unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        let report = worker.search(&sequence).unwrap();
        assert!(report.trace.null_score.is_some());
        // Without `max_mode` every sequence is scored with HMMER's MSV filter.
        assert!(report.trace.msv_filter_score.is_some());
        match report.outcome {
            SearchOutcome::Filtered(_) | SearchOutcome::Hit(_) => {}
        }
    }

    #[test]
    fn test_max_mode_computes_no_filter_before_the_forward_pass() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(13);
        let (_, bg, mut gm) = setup_profile(&mut rng, 8, 32, &abc);
        calibrate(&mut gm);
        // No MSV score reaches a P-value under the threshold.
        gm.ev_params[EvParam::MsvMu.idx()] = 1_000_000.0;
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "max-mode-seq".to_string(),
            residues,
        };
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();

        let filtered = search_with(&query, FilterPolicy::default(), &sequence);
        assert_eq!(filtered.trace.filter_reason, Some(FilterReason::Msv));

        let filters = FilterPolicy {
            max_mode: true,
            ..FilterPolicy::default()
        };
        let plan = SearchPlan::builder(query).filters(filters).build().unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();
        let report = worker.search(&sequence).unwrap();

        assert!(report.trace.msv_filter_score.is_none());
        assert!(report.trace.bias_filter_score.is_none());
        assert!(report.trace.viterbi_filter_score.is_none());
        assert!(report.trace.forward_raw_score.is_some());
        assert!(!matches!(
            report.trace.filter_reason,
            Some(
                FilterReason::Msv
                    | FilterReason::Bias
                    | FilterReason::Viterbi
                    | FilterReason::Forward
            )
        ));
        // The sequence counts as past every stage.
        let stats = worker.into_metrics().stats;
        assert_eq!(stats.num_sequences, 1);
        assert_eq!(stats.sequences_past_msv, 1);
        assert_eq!(stats.sequences_past_bias, 1);
        assert_eq!(stats.sequences_past_viterbi, 1);
        assert_eq!(stats.sequences_past_forward, 1);
    }

    #[test]
    fn test_msv_threshold_filters_a_sequence_end_to_end() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(17);
        let (_, bg, mut gm) = setup_profile(&mut rng, 8, 32, &abc);
        calibrate(&mut gm);
        gm.ev_params[EvParam::MsvMu.idx()] = 1_000_000.0;
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "filtered-seq".to_string(),
            residues,
        };
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let plan = SearchPlan::builder(query).build().unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        let report = worker.search(&sequence).unwrap();

        assert!(matches!(
            report.outcome,
            SearchOutcome::Filtered(FilterReason::Msv)
        ));
        assert_eq!(report.trace.filter_reason, Some(FilterReason::Msv));
        assert!(report.trace.msv_filter_score.is_some());
        assert!(report.trace.viterbi_filter_score.is_none());
        assert!(report.trace.forward_raw_score.is_none());
    }

    #[test]
    fn test_max_mode_reports_the_hit_the_default_filters_let_through() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(19);
        let (hmm, bg, mut gm) = conserved_profile(&mut rng, 40, 40, &abc);
        calibrate(&mut gm);
        let sequence = consensus_sequence(&hmm, &abc);
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();

        let filtered = search_with(&query, FilterPolicy::default(), &sequence);
        let unfiltered = search_with(
            &query,
            FilterPolicy {
                max_mode: true,
                ..FilterPolicy::default()
            },
            &sequence,
        );

        // The default policy decides on the scores of HMMER's filters; max
        // mode computes none of them.
        assert!(filtered.trace.msv_filter_score.is_some());
        assert!(filtered.trace.bias_filter_score.is_some());
        assert!(unfiltered.trace.msv_filter_score.is_none());
        assert!(unfiltered.trace.bias_filter_score.is_none());
        assert!(unfiltered.trace.viterbi_filter_score.is_none());

        // The filters decide which sequences are decoded, not what is
        // reported for one.
        let (SearchOutcome::Hit(filtered_hit), SearchOutcome::Hit(unfiltered_hit)) =
            (filtered.outcome, unfiltered.outcome)
        else {
            panic!("the consensus of the profile is a hit with either policy");
        };
        assert_eq!(filtered_hit.score, unfiltered_hit.score);
        assert_eq!(filtered_hit.domains.len(), unfiltered_hit.domains.len());
        for (a, b) in filtered_hit.domains.iter().zip(&unfiltered_hit.domains) {
            assert_eq!(
                (a.envelope_start, a.envelope_end, a.bitscore),
                (b.envelope_start, b.envelope_end, b.bitscore)
            );
        }
    }
}
