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
use crate::hmmer_core::dynamic_programming::simd::f32_msv;
use crate::hmmer_core::dynamic_programming::simd::f32_viterbi;
use crate::hmmer_core::dynamic_programming::simd::forward_filter as simd_fwd;
use crate::hmmer_core::dynamic_programming::simd::i8_msv;
use crate::hmmer_core::dynamic_programming::simd::i16_msv;
use crate::hmmer_core::dynamic_programming::simd::i16_viterbi;
use crate::hmmer_core::dynamic_programming::simd::int_msv;
use crate::hmmer_core::dynamic_programming::simd::oprofile::OptimizedProfile;
use crate::hmmer_core::errors::HmmerError;
use crate::hmmer_core::evalues;
use crate::hmmer_core::forward_backward;
use crate::hmmer_core::hit::Hit;
use crate::hmmer_core::logsum;
use crate::hmmer_core::profile::Profile;
use crate::hmmer_core::score_matrix::*;
use crate::hmmer_core::sequence::DigitalSequence;
use crate::hmmer_core::tophits::TopHits;
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

    /// Reset all accounting counters.
    pub fn reuse(&mut self) {
        *self = Self::default();
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
    pub bias_filter: bool,
    pub msv_threshold: f64,
    pub viterbi_threshold: f64,
    pub forward_threshold: f64,
    /// Decide the MSV and bias filters with the exact single-precision MSV
    /// score and the Viterbi filter with the exact Viterbi score. Otherwise
    /// the MSV filters use an 8-bit lower bound on the MSV score, within
    /// about half a nat of it, and the Viterbi filter its 16-bit upper
    /// bound: a little less sensitive at the MSV stage, a little more at the
    /// Viterbi stage, and much faster.
    pub exact: bool,
}

impl Default for FilterPolicy {
    fn default() -> Self {
        Self {
            max_mode: false,
            bias_filter: true,
            exact: false,
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
#[cfg_attr(feature = "hmmer-serde", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "hmmer-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct SearchTrace {
    pub sequence_length: usize,
    pub null_score: Option<f32>,
    pub bias_filter_score: Option<f32>,
    /// Exact MSV score. `None` when the integer bounds alone settled the
    /// MSV and bias filters; see `msv_score_bound` and `msv_score_lower_bound`.
    pub msv_raw_score: Option<f32>,
    /// Upper bound on the MSV score from the integer pre-filters, when they
    /// ran: infinite when they could not bound it.
    pub msv_score_bound: Option<f32>,
    /// Lower bound on the MSV score from the integer pre-filters, when known.
    pub msv_score_lower_bound: Option<f32>,
    /// Viterbi filter score: exact with `FilterPolicy::exact`, otherwise the
    /// 16-bit upper bound that decided the filter, or the exact score when
    /// that bound saturated and could not decide.
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
    domain_result: &'a mut crate::hmmer_core::domaindef::DomainResult,
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
        engine.exact = plan.filters.exact;
        engine.do_biasfilter = plan.filters.bias_filter && !plan.filters.max_mode;
        engine.msv_threshold = plan.filters.msv_threshold;
        engine.viterbi_threshold = plan.filters.viterbi_threshold;
        engine.forward_threshold = plan.filters.forward_threshold;
        engine.thresholds.by_e = false;
        engine.domain_config = plan.domain_config.clone();
        engine.domain_config.exact_decode = plan.filters.exact;
        engine.odds_tables = OddsTables::new(&plan.query.profile_template);
        engine.oprofile = Some(plan.query.optimized_template.clone());

        let mut profile = plan.query.profile_template.clone();
        let mut background = plan.query.background_template.clone();
        crate::hmmer_core::modelconfig::reconfigure_length(&mut profile, hints.target_length);
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

    /// The P-value of an MSV score `raw`, in nats, against `reference`: the
    /// null score for the MSV filter, the bias score for the bias filter.
    fn msv_pvalue_of(&self, raw: f32, reference: f32) -> f64 {
        SearchEngine::msv_pvalue(&self.worker.state.profile, (raw - reference) / LOG2)
    }

    /// Whether an upper bound on the MSV score already misses the MSV
    /// threshold. An infinite bound means no bound is known.
    fn msv_bound_fails(&self, bound: f32) -> bool {
        bound.is_finite()
            && self.msv_pvalue_of(bound, self.null_score) > self.worker.engine.msv_threshold
    }

    /// The exact MSV score, computed on first use.
    fn exact_msv_score(&mut self) -> f32 {
        if let Some(score) = self.trace.msv_raw_score {
            return score;
        }
        let om = self.worker.engine.oprofile.as_ref().unwrap();
        let score = f32_msv::msv_filter_f32_with(
            &self.sequence.residues,
            self.sequence_length,
            om,
            &mut self.worker.engine.msv_row,
        );
        self.trace.msv_raw_score = Some(score);
        score
    }

    /// The MSV and bias filters decided by the 8-bit lower bound on the MSV
    /// score, which stands in for the score itself: a sequence passes when
    /// the bound passes. Then the Viterbi and forward filters as usual.
    fn run_fast_msv_filters(
        mut self,
    ) -> ControlFlow<SearchReport, TargetRun<'w, 's, PassedFilters>> {
        let msv_threshold = self.worker.engine.msv_threshold;
        let om = self.worker.engine.oprofile.as_ref().unwrap();
        let lower_bound = i8_msv::msv_lower_bound_with(
            &self.sequence.residues,
            self.sequence_length,
            om,
            &mut self.worker.engine.msv_bound_row8,
        );
        self.trace.msv_score_lower_bound = lower_bound;
        let Some(msv_score) = lower_bound else {
            // No cell rose above the begin state: far below any threshold.
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
        };
        // A saturated score passes both filters, as in HMMER: the sequence
        // scores higher than the cells can count.
        if msv_score.is_finite() && self.msv_pvalue_of(msv_score, self.null_score) > msv_threshold {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
        }
        self.worker.engine.stats.sequences_past_msv += 1;
        self.worker.engine.stats.residues_past_msv += self.sequence_length as u64;

        let filter_score = if self.worker.engine.do_biasfilter {
            let score = self
                .worker
                .state
                .background
                .filter_score(&self.sequence.residues, self.sequence_length);
            self.trace.bias_filter_score = Some(score);
            if msv_score.is_finite() && self.msv_pvalue_of(msv_score, score) > msv_threshold {
                return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
            }
            score
        } else {
            self.null_score
        };
        if !filter_score.is_finite() {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
        }
        self.worker.engine.stats.sequences_past_bias += 1;
        self.worker.engine.stats.residues_past_bias += self.sequence_length as u64;
        self.run_late_filters(filter_score)
    }

    /// The Viterbi and forward filters.
    fn run_late_filters(
        mut self,
        filter_score: f32,
    ) -> ControlFlow<SearchReport, TargetRun<'w, 's, PassedFilters>> {
        let Some(viterbi_raw_score) = self.worker.engine.run_viterbi_filter(
            &self.worker.state.profile,
            &self.sequence.residues,
            self.sequence_length,
            filter_score,
        ) else {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Viterbi));
        };
        self.trace.viterbi_raw_score = Some(viterbi_raw_score);

        let Some(forward_raw_score) = self.worker.engine.run_forward_filter(
            &self.worker.state.profile,
            &self.sequence.residues,
            self.sequence_length,
            filter_score,
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

    fn run_filters(mut self) -> ControlFlow<SearchReport, TargetRun<'w, 's, PassedFilters>> {
        let do_max = self.worker.engine.do_max;
        let msv_threshold = self.worker.engine.msv_threshold;

        // Cheap bounds first, the coarsest one leading: a sequence whose upper
        // bound fails the MSV threshold fails the exact filter too, and one
        // whose lower bound passes it passes the exact filter too. The exact
        // score is only computed when the bounds leave a decision open.
        let mut lower_bound = None;
        let mut upper_bound = f32::INFINITY;
        // A windowed upper bound carries the coarse bounds of the blocks it
        // skipped; the whole-sequence bound is tighter when it matters.
        let mut upper_bound_is_windowed = false;
        if !do_max && !self.worker.engine.exact {
            return self.run_fast_msv_filters();
        }
        if !do_max {
            let om = self.worker.engine.oprofile.as_ref().unwrap();
            let bound = i8_msv::msv_upper_bound_with(
                &self.sequence.residues,
                self.sequence_length,
                om,
                &mut self.worker.engine.msv_bound_row8,
                Some(&mut self.worker.engine.msv_blocks8),
            );
            if self.msv_bound_fails(bound) {
                self.trace.msv_score_bound = Some(bound);
                return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
            }
            // The finer bound only needs the blocks whose coarse bound passes,
            // unless the coarse pass saturated or the sequence has more blocks
            // than the selection can name.
            let blocks = &self.worker.engine.msv_blocks8;
            let mut selected = 0u64;
            if !blocks.saturated && blocks.bounds.len() <= u64::BITS as usize {
                for (block, &block_bound) in blocks.bounds.iter().enumerate() {
                    if !self.msv_bound_fails(block_bound) {
                        selected |= 1 << block;
                    }
                }
            }
            let bound = if selected == 0 {
                i16_msv::msv_upper_bound_with(
                    &self.sequence.residues,
                    self.sequence_length,
                    om,
                    &mut self.worker.engine.msv_bound_row16,
                )
            } else {
                upper_bound_is_windowed = true;
                i16_msv::msv_upper_bound_within(
                    &self.sequence.residues,
                    self.sequence_length,
                    om,
                    &mut self.worker.engine.msv_bound_row16,
                    &self.worker.engine.msv_blocks8,
                    selected,
                )
            };
            if self.msv_bound_fails(bound) {
                self.trace.msv_score_bound = Some(bound);
                return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
            }
            upper_bound = bound;
            // A saturated lower bound says nothing exact; the exact score then
            // decides.
            lower_bound = i16_msv::msv_lower_bound_with(
                &self.sequence.residues,
                self.sequence_length,
                om,
                &mut self.worker.engine.msv_bound_row16,
            )
            .filter(|bound| bound.is_finite());
            self.trace.msv_score_bound = Some(upper_bound);
            self.trace.msv_score_lower_bound = lower_bound;
        }

        let null_score = self.null_score;
        let passes_msv = !do_max
            && lower_bound
                .is_some_and(|lower| self.msv_pvalue_of(lower, null_score) <= msv_threshold);
        if !passes_msv {
            let msv_raw_score = self.exact_msv_score();
            let p_msv = self.msv_pvalue_of(msv_raw_score, null_score);
            if !do_max && p_msv > msv_threshold {
                return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Msv));
            }
        }
        self.worker.engine.stats.sequences_past_msv += 1;
        self.worker.engine.stats.residues_past_msv += self.sequence_length as u64;

        let filter_score = if self.worker.engine.do_biasfilter {
            let score = self
                .worker
                .state
                .background
                .filter_score(&self.sequence.residues, self.sequence_length);
            self.trace.bias_filter_score = Some(score);

            let passes_bias =
                lower_bound.is_some_and(|lower| self.msv_pvalue_of(lower, score) <= msv_threshold);
            if !passes_bias {
                let mut fails_bias = upper_bound.is_finite()
                    && self.msv_pvalue_of(upper_bound, score) > msv_threshold;
                if !fails_bias && upper_bound_is_windowed {
                    let om = self.worker.engine.oprofile.as_ref().unwrap();
                    upper_bound = i16_msv::msv_upper_bound_with(
                        &self.sequence.residues,
                        self.sequence_length,
                        om,
                        &mut self.worker.engine.msv_bound_row16,
                    );
                    self.trace.msv_score_bound = Some(upper_bound);
                    fails_bias = upper_bound.is_finite()
                        && self.msv_pvalue_of(upper_bound, score) > msv_threshold;
                }
                if fails_bias {
                    return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
                }
                let msv_raw_score = self.exact_msv_score();
                if self.msv_pvalue_of(msv_raw_score, score) > msv_threshold {
                    return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
                }
            }
            score
        } else {
            self.null_score
        };
        if !filter_score.is_finite() {
            return ControlFlow::Break(self.trace.into_filtered_report(FilterReason::Bias));
        }
        self.worker.engine.stats.sequences_past_bias += 1;
        self.worker.engine.stats.residues_past_bias += self.sequence_length as u64;
        self.run_late_filters(filter_score)
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
    exact: bool,
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

    // DP rows of the integer over-estimates, reused across sequences
    msv_bound_row8: Vec<i8_msv::I8x16>,
    msv_blocks8: int_msv::Blocks<i8, { i8_msv::LANES }>,
    msv_bound_row16: Vec<i16_msv::I16x8>,
    viterbi_bound_rows: i16_viterbi::Rows,

    // DP rows of the exact MSV and Viterbi filters, reused across sequences
    msv_row: Vec<f32_msv::F32x4>,
    viterbi_rows: f32_viterbi::Rows,
}

impl SearchEngine {
    /// Create a new pipeline with default parameters.
    fn new(m: usize, sequence_length: usize) -> Self {
        SearchEngine {
            thresholds: Thresholds::default(),
            search_space: 0.0,
            do_max: false,
            exact: false,
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
            msv_bound_row8: Vec::new(),
            msv_blocks8: int_msv::Blocks::default(),
            msv_bound_row16: Vec::new(),
            viterbi_bound_rows: i16_viterbi::Rows::default(),
            msv_row: Vec::new(),
            viterbi_rows: f32_viterbi::Rows::default(),
        }
    }

    #[inline]
    fn msv_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        evalues::msv_score_to_pvalue(
            bitscore,
            profile.ev_params[EvParam::MsvMu.idx()],
            profile.ev_params[EvParam::MsvLambda.idx()],
        )
    }

    #[inline]
    fn viterbi_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        evalues::viterbi_score_to_pvalue(
            bitscore,
            profile.ev_params[EvParam::ViterbiMu.idx()],
            profile.ev_params[EvParam::ViterbiLambda.idx()],
        )
    }

    #[inline]
    fn forward_pvalue(profile: &Profile, bitscore: f32) -> f64 {
        evalues::forward_score_to_pvalue(
            bitscore,
            profile.ev_params[EvParam::ForwardTau.idx()],
            profile.ev_params[EvParam::ForwardLambda.idx()],
        )
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
        debug_assert!(sum_ld <= sequence_length);
        sum_raw += sequence_length.saturating_sub(sum_ld) as f32
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

        let log_pvalue = Self::forward_pvalue(profile, seq_score)
            .max(MIN_PVALUE_CLAMP)
            .ln();

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
        // Cheap over-estimate first: a sequence whose upper bound fails the
        // Viterbi threshold fails the exact filter too.
        if !self.do_max {
            let bound = i16_viterbi::viterbi_upper_bound_with(
                digital_sequence,
                sequence_length,
                om,
                &mut self.viterbi_bound_rows,
            );
            if bound.is_finite() {
                let p_bound = Self::viterbi_pvalue(profile, (bound - null_score) / LOG2);
                if p_bound > self.viterbi_threshold {
                    return None;
                }
                if !self.exact {
                    // The bound stands in for the score: the sequence passes.
                    self.stats.sequences_past_viterbi += 1;
                    self.stats.residues_past_viterbi += sequence_length as u64;
                    return Some(bound);
                }
            }
        }
        let viterbi_raw_score = f32_viterbi::viterbi_filter_f32_with(
            digital_sequence,
            sequence_length,
            om,
            &mut self.viterbi_rows,
        );

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

    fn run_forward_backward_and_domains(
        &mut self,
        profile: &mut Profile,
        digital_sequence: &[u8],
        sequence_length: usize,
        forward_raw_score: f32,
    ) -> Result<Option<(f32, crate::hmmer_core::domaindef::DomainResult)>, HmmerError> {
        self.domain_workspace.reuse();
        self.domain_workspace.grow_to(sequence_length);

        self.posterior_matrix
            .resize(profile.num_nodes, sequence_length)?;
        let decoded = if self.domain_config.exact_decode {
            forward_backward::backward_decode_log_space(
                digital_sequence,
                profile,
                forward_raw_score,
                &mut self.posterior_matrix,
                Some(&mut self.domain_workspace),
            )
        } else {
            odds_decode::decode_odds_space(
                digital_sequence,
                profile,
                &self.odds_tables,
                forward_raw_score,
                &mut self.posterior_matrix,
                Some(&mut self.domain_workspace),
            )
        };
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
            accession: None,
            description: None,
            source_length: residues.len() as i64,
            residues,
            database_index: -1,
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
            crate::hmmer_core::config::SearchMode::Local,
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
        assert!(!filters.exact);
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
    fn test_pipeline_stats_reuse_resets_every_counter() {
        let mut stats = PipelineStats {
            num_models: 1,
            num_sequences: 2,
            num_residues: 3,
            num_nodes: 4,
            sequences_past_msv: 5,
            sequences_past_bias: 6,
            sequences_past_viterbi: 7,
            sequences_past_forward: 8,
            sequences_output: 9,
            residues_past_msv: 10,
            residues_past_bias: 11,
            residues_past_viterbi: 12,
            residues_past_forward: 13,
            residues_output: 14,
        };

        stats.reuse();

        assert_eq!(stats, PipelineStats::default());
    }

    #[test]
    fn test_search_query_rejects_uncalibrated_profile() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(5);
        let (_, bg, gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );

        let error = SearchQuery::from_configured_profile(gm, bg).unwrap_err();
        assert!(error.to_string().contains("calibration parameter"));
    }

    #[test]
    fn test_search_plan_spawns_worker() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(7);
        let (_, bg, mut gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );
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
        let (_, bg, mut gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );
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
        let (_, bg, mut gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );
        calibrate(&mut gm);
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
        let plan = SearchPlan::builder(query).build().unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        let report = worker.search(&sequence).unwrap();
        assert_eq!(report.trace.sequence_length, 32);
        assert!(report.trace.null_score.is_some());
        assert!(
            report.trace.msv_raw_score.is_some()
                || report.trace.msv_score_bound.is_some()
                || report.trace.msv_score_lower_bound.is_some()
                || report.trace.filter_reason == Some(FilterReason::Msv)
        );
        match report.outcome {
            SearchOutcome::Filtered(_) | SearchOutcome::Hit(_) => {}
        }
    }

    #[test]
    fn test_max_mode_bypasses_filter_cascade_and_bias_filter() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(13);
        let (_, bg, mut gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );
        calibrate(&mut gm);
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "max-mode-seq".to_string(),
            accession: None,
            description: None,
            residues,
            source_length: 32,
            database_index: -1,
        };
        let filters = FilterPolicy {
            max_mode: true,
            ..FilterPolicy::default()
        };
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();
        let plan = SearchPlan::builder(query).filters(filters).build().unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints { target_length: 32 })
            .unwrap();

        let report = worker.search(&sequence).unwrap();

        assert!(report.trace.msv_raw_score.is_some());
        assert!(report.trace.bias_filter_score.is_none());
        assert!(report.trace.viterbi_raw_score.is_some());
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
    }

    #[test]
    fn test_msv_threshold_filters_a_sequence_end_to_end() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(17);
        let (_, bg, mut gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );
        calibrate(&mut gm);
        gm.ev_params[EvParam::MsvMu.idx()] = 1_000_000.0;
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "filtered-seq".to_string(),
            accession: None,
            description: None,
            residues,
            source_length: 32,
            database_index: -1,
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
        assert!(report.trace.viterbi_raw_score.is_none());
        assert!(report.trace.forward_raw_score.is_none());
    }

    #[test]
    fn test_exact_and_default_filters_agree_on_a_hit() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(19);
        let (hmm, bg, mut gm) = conserved_profile(&mut rng, 40, 40, &abc);
        calibrate(&mut gm);
        let sequence = consensus_sequence(&hmm, &abc);
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();

        let fast = search_with(&query, FilterPolicy::default(), &sequence);
        let exact = search_with(
            &query,
            FilterPolicy {
                exact: true,
                ..FilterPolicy::default()
            },
            &sequence,
        );

        // The default policy decides the MSV filters with the 8-bit lower
        // bound and never computes the exact MSV score; the exact policy
        // records the upper bound it starts from.
        assert!(fast.trace.msv_score_lower_bound.is_some());
        assert!(fast.trace.msv_score_bound.is_none());
        assert!(fast.trace.msv_raw_score.is_none());
        assert!(exact.trace.msv_score_bound.is_some());

        let (SearchOutcome::Hit(fast_hit), SearchOutcome::Hit(exact_hit)) =
            (fast.outcome, exact.outcome)
        else {
            panic!("the consensus of the profile is a hit with either policy");
        };
        assert_eq!(fast_hit.domains.len(), exact_hit.domains.len());
        for (a, b) in fast_hit.domains.iter().zip(&exact_hit.domains) {
            assert_eq!(
                (a.envelope_start, a.envelope_end),
                (b.envelope_start, b.envelope_end)
            );
        }
        let tolerance = 0.01 * exact_hit.score.abs().max(1.0);
        assert!(
            (fast_hit.score - exact_hit.score).abs() <= tolerance,
            "default {} exact {}",
            fast_hit.score,
            exact_hit.score
        );
    }

    #[test]
    fn test_exact_filters_drop_a_sequence_on_an_upper_bound() {
        let abc = Alphabet::amino();
        let mut rng = XorShift64::new(23);
        let (_, bg, mut gm) = setup_profile(
            &mut rng,
            8,
            32,
            &abc,
            crate::hmmer_core::config::SearchMode::Local,
        );
        calibrate(&mut gm);
        gm.ev_params[EvParam::MsvMu.idx()] = 1_000_000.0;
        let residues =
            random_digital_seq(&mut rng, &bg.residue_frequencies, abc.canonical_size, 32);
        let sequence = DigitalSequence {
            name: "filtered-exactly".to_string(),
            accession: None,
            description: None,
            residues,
            source_length: 32,
            database_index: -1,
        };
        let query = SearchQuery::from_configured_profile(gm, bg).unwrap();

        let report = search_with(
            &query,
            FilterPolicy {
                exact: true,
                ..FilterPolicy::default()
            },
            &sequence,
        );

        assert!(matches!(
            report.outcome,
            SearchOutcome::Filtered(FilterReason::Msv)
        ));
        // An upper bound settled it, which the default policy never computes.
        assert!(report.trace.msv_score_bound.is_some());
        assert!(report.trace.viterbi_raw_score.is_none());
    }
}
