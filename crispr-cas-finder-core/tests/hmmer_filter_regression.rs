//! Filter outcomes of HMMER 3.4 `hmmsearch`, frozen for sequences that sit
//! near its thresholds.
//!
//! The targets are proteins of E. coli K-12 MG1655 (the genome bundled with
//! the command-line tool), the queries profiles of the bundled CasFinder
//! 2.0.3 set. For each pair, `hmmsearch` (HMMER 3.4, Homebrew arm64) on the
//! sequence alone gave the stage it stops at, and its thresholds gave the
//! scores: its "Passed MSV filter", "Passed bias filter" and "Passed Vit
//! filter" counts flip when `--F1` or `--F2` goes from just above to just
//! below (one part in 10^9) the P-value of the score frozen here. A byte of
//! the MSV filter moves that P-value by a quarter, a word of the Viterbi
//! filter by a thousandth and the last bit of the bias filter score by a
//! few parts in 10^7, so the flips pin every one of them.
use crispr_cas_finder_core::hmmer_core::{
    alphabet::Alphabet,
    background::BackgroundModel,
    config::EvParam,
    evalues::{filter_bit_score, gumbel_survival},
    modelconfig,
    pipeline::search::{
        CapacityHints, FilterPolicy, FilterReason, SearchOutcome, SearchPlan, SearchQuery,
    },
    profile::Profile,
    sequence::DigitalSequence,
    simd::{msv_filter::msv_filter, oprofile::OptimizedProfile, viterbi_filter::viterbi_filter},
};
use crispr_cas_finder_core::hmmer_io::HmmFile;

/// Where `hmmsearch` stops a sequence: the filter that drops it, or a
/// reported hit with its bit score.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome {
    Stopped(FilterReason),
    Hit(f32),
}

struct Case {
    profile: &'static str,
    /// Gene number in the proteome of NC_000913.3, for the record.
    gene: u32,
    sequence: &'static [u8],
    outcome: Outcome,
    /// What `p7_MSVFilter()` returns, in nats.
    msv: f32,
    /// `p7_bg_NullOne()`.
    null: f32,
    /// `p7_bg_FilterScore()`.
    bias: f32,
    /// What `p7_ViterbiFilter()` returns, in nats.
    viterbi: f32,
}

const GENE_253: &[u8] = b"MSWRPFLYLIINPHPLLSARRGHLRLVQG";
const GENE_1248: &[u8] = b"MPSGNQEPRRDPELKRKAWLAVFLGSALFWVVVALLIWKVWG";
const GENE_1541: &[u8] = b"MSVTIQGNTSTVISNNSAPEGTSEIAKITRQIQVLTEKLGKISSEEGMTTQQKKEMAALVQKQIESLWAQLEQLLRQQAEKKNEDATVQPDKKEEKKDDTNTAGTIDIYV";
const GENE_1547: &[u8] = b"MDFDTIMEKAYEEYFEGLAEGEEALSFSEFKQALSSSAKSNG";
const GENE_1829: &[u8] = b"MTISALKEIEMQFCQILRLCAVLMTKADKKVVIFFHNMISVRFLPNGKR";
const GENE_2705: &[u8] = b"MSMLVVVTENVPPRLRGRLAIWLLEVRAGVYVGDVSAKIREMIWEQIAGLAEEGNVVMAWATNTETGFEFQTFGLNRRTPVDLDGLRLVSFLPV";
const GENE_2974: &[u8] = b"MLKPDNLPVTFGKNDVEIIARETLYRGFFSLDLYRFRHRLFNGQMSHEVRREIFERGHAAVLLPFDPVRDEVVLIEQIRIAAYDTSETPWLLEMVAGMIEEGESVEDVARREAIEEAGLIVKRTKPVLSFLASPGGTSERSSIMVGEVDATTASGIHGLADENEDIRVHVVSREQAYQWVEEGKIDNAASVIALQWLQLHHQALKNEWA";
const GENE_3848: &[u8] =
    b"MSLEVFEKLEAKVQQAIDTITLLQMEIEELKEKNNSLSQEVQNAQHQREELERENNHLKEQQNGWQERLQALLGRMEEV";
const GENE_4258: &[u8] = b"MEFHENRAKAPFIGLVQLWQAVRRWRRQMQTRRVLQQMSDERLKDIGLRREDVE";

/// The P-values are those of the frozen scores with the profiles' own
/// statistics; the MSV and bias thresholds are 0.02, the Viterbi threshold
/// 0.001.
const CASES: &[Case] = &[
    // Past the MSV filter by a hair (P = 0.0195), past the bias filter,
    // dropped by the Viterbi filter.
    Case {
        profile: "Cmr6_0_IIIC.hmm",
        gene: 253,
        sequence: GENE_253,
        outcome: Outcome::Stopped(FilterReason::Viterbi),
        msv: -8.083079,
        null: -4.384342,
        bias: -4.837689,
        viterbi: -8.019772,
    },
    // The same sequence and the same byte score with another profile: short
    // of the MSV filter by a hair (P = 0.0205).
    Case {
        profile: "Csf1_1_IV.hmm",
        gene: 253,
        sequence: GENE_253,
        outcome: Outcome::Stopped(FilterReason::Msv),
        msv: -8.083079,
        null: -4.384342,
        bias: -4.577695,
        viterbi: -8.023931,
    },
    // Past the MSV filter (P = 0.0117), short of the bias filter by a hair
    // (P = 0.0226).
    Case {
        profile: "Cas8a1_1_IB.hmm",
        gene: 1547,
        sequence: GENE_1547,
        outcome: Outcome::Stopped(FilterReason::Bias),
        msv: -8.545177,
        null: -4.749481,
        bias: -4.086351,
        viterbi: -9.256346,
    },
    // A strong MSV score (P = 0.00097) in a biased sequence: the bias
    // filter drops it (P = 0.045).
    Case {
        profile: "Cas10_0_IIIC.hmm",
        gene: 3848,
        sequence: GENE_3848,
        outcome: Outcome::Stopped(FilterReason::Bias),
        msv: -7.158883,
        null: -5.3757505,
        bias: -1.5267072,
        viterbi: -8.649149,
    },
    // Past the bias filter by a hair (P = 0.0185), dropped by the Viterbi
    // filter.
    Case {
        profile: "Cas12b_0_VB.hmm",
        gene: 4258,
        sequence: GENE_4258,
        outcome: Outcome::Stopped(FilterReason::Viterbi),
        msv: -7.8520303,
        null: -4.9981866,
        bias: -2.802795,
        viterbi: -9.693029,
    },
    // Past the Viterbi filter by a hair (P = 0.00091), dropped by the
    // Forward filter.
    Case {
        profile: "Cmr6_1_IIIB.hmm",
        gene: 253,
        sequence: GENE_253,
        outcome: Outcome::Stopped(FilterReason::Forward),
        msv: -5.5415397,
        null: -4.384342,
        bias: -4.727146,
        viterbi: -5.8585386,
    },
    // Short of the Viterbi filter by a hair (P = 0.00115).
    Case {
        profile: "Cas10_1_IIIB.hmm",
        gene: 1248,
        sequence: GENE_1248,
        outcome: Outcome::Stopped(FilterReason::Viterbi),
        msv: -5.0794415,
        null: -4.749481,
        bias: -4.002096,
        viterbi: -4.6261234,
    },
    // The MSV score meets the Viterbi threshold already (P = 0.00039 against
    // the bias filter score), so HMMER does not run the Viterbi filter,
    // which would drop the sequence (P = 0.00109): the Forward filter does.
    Case {
        profile: "Cas8a1_1_IA.hmm",
        gene: 1829,
        sequence: GENE_1829,
        outcome: Outcome::Stopped(FilterReason::Forward),
        msv: -4.6173434,
        null: -4.9019556,
        bias: -4.6100526,
        viterbi: -6.1011405,
    },
    // Both filters overflow: a hit, Cas2 itself.
    Case {
        profile: "Cas2_0_IE.hmm",
        gene: 2705,
        sequence: GENE_2705,
        outcome: Outcome::Hit(143.9),
        msv: f32::INFINITY,
        null: -5.548595,
        bias: -4.8096046,
        viterbi: f32::INFINITY,
    },
    // A reported hit that the Viterbi filter would have dropped (P = 0.0099)
    // had the MSV score (P = 0.00074) not made HMMER skip it.
    Case {
        profile: "Cmr5_1_IIIC.hmm",
        gene: 1541,
        sequence: GENE_1541,
        outcome: Outcome::Hit(7.4),
        msv: -2.0758038,
        null: -5.7050123,
        bias: -1.8265195,
        viterbi: -5.1265755,
    },
    // An MSV score above the base of its bytes: a segment that pays for the
    // J state, scored by the full recurrence and not the single-segment one.
    Case {
        profile: "Cas2_0_IE.hmm",
        gene: 2974,
        sequence: GENE_2974,
        outcome: Outcome::Hit(14.5),
        msv: 0.0036377907,
        null: -6.3447227,
        bias: -6.0189686,
        viterbi: -1.864625,
    },
];

/// The bundled profile `file`, configured for targets of `length` residues,
/// with its background model.
fn bundled_profile(file: &str, length: usize) -> (Alphabet, BackgroundModel, Profile) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../crispr-cas-finder-cli/data/CasFinder-2.0.3/CASprofiles-2.0.3")
        .join(file);
    let mut hmm_file = HmmFile::open(path.to_str().unwrap(), None).unwrap();
    let (alphabet, hmm) = hmm_file.read().unwrap();
    let background = BackgroundModel::new(&Alphabet::amino());
    let mut profile = Profile::new(hmm.num_nodes, &alphabet);
    modelconfig::profile_config(&hmm, &background, &mut profile, length);
    (alphabet, background, profile)
}

/// The four scores the filter cascade compares are HMMER's: the byte score
/// of its MSV filter, its null score, its bias filter score and the word
/// score of its Viterbi filter.
#[test]
fn filter_scores_are_those_of_hmmer_3_4() {
    for case in CASES {
        let name = format!("{} on gene {}", case.profile, case.gene);
        let length = case.sequence.len();
        let (alphabet, mut background, profile) = bundled_profile(case.profile, length);
        let target = DigitalSequence::from_bytes("target", case.sequence, &alphabet);
        let om = OptimizedProfile::from_profile(&profile);
        background
            .set_filter(
                profile.num_nodes,
                &profile.model_composition[..alphabet.canonical_size],
            )
            .unwrap();
        background.set_length(length);

        let msv = msv_filter(&target.residues, length, &om);
        assert_eq!(
            msv.to_bits(),
            case.msv.to_bits(),
            "{name}: MSV filter {msv}"
        );
        let null = background.null_one(&target.residues, length);
        assert_eq!(null.to_bits(), case.null.to_bits(), "{name}: null {null}");
        let viterbi = viterbi_filter(&target.residues, length, &om);
        assert_eq!(
            viterbi.to_bits(),
            case.viterbi.to_bits(),
            "{name}: Viterbi filter {viterbi}"
        );
        // The bias filter score is a sum over the sequence in single
        // precision, of terms that go through the C library's `logf` and,
        // when the profile is read, `expf`. Those of macOS are not correctly
        // rounded, so the last bits are those of `hmmsearch` on the machine
        // the values were frozen on, an Apple silicon Mac, and may differ
        // elsewhere.
        let bias = background.filter_score(&target.residues, length);
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert_eq!(bias.to_bits(), case.bias.to_bits(), "{name}: bias {bias}");
        } else {
            assert!((bias - case.bias).abs() < 2e-5, "{name}: bias {bias}");
        }
    }
}

/// Every sequence stops where `hmmsearch` stops it, on the scores above, and
/// the Viterbi filter runs when HMMER runs it.
#[test]
fn sequences_near_the_thresholds_stop_where_hmmer_3_4_stops_them() {
    for case in CASES {
        let name = format!("{} on gene {}", case.profile, case.gene);
        let length = case.sequence.len();
        let (alphabet, background, profile) = bundled_profile(case.profile, length);
        let ev_params = profile.ev_params;
        let target = DigitalSequence::from_bytes("target", case.sequence, &alphabet);
        let query = SearchQuery::from_configured_profile(profile, background).unwrap();
        let plan = SearchPlan::builder(query)
            .filters(FilterPolicy::default())
            .build()
            .unwrap();
        let mut worker = plan
            .spawn_worker(CapacityHints {
                target_length: length,
            })
            .unwrap();
        let report = worker.search(&target).unwrap();

        let trace = &report.trace;
        assert_eq!(
            trace.msv_filter_score.map(f32::to_bits),
            Some(case.msv.to_bits()),
            "{name}"
        );
        assert_eq!(
            trace.null_score.map(f32::to_bits),
            Some(case.null.to_bits()),
            "{name}"
        );
        let past_msv = case.outcome != Outcome::Stopped(FilterReason::Msv);
        assert_eq!(trace.bias_filter_score.is_some(), past_msv, "{name}");
        // HMMER runs its Viterbi filter on what passes the bias filter with
        // an MSV P-value above the Viterbi threshold.
        let msv_pvalue = gumbel_survival(
            filter_bit_score(case.msv, case.bias),
            ev_params[EvParam::MsvMu.idx()],
            ev_params[EvParam::MsvLambda.idx()],
        );
        let past_bias = past_msv && case.outcome != Outcome::Stopped(FilterReason::Bias);
        let runs_viterbi = past_bias && msv_pvalue > 1e-3;
        assert_eq!(
            trace.viterbi_filter_score.map(f32::to_bits),
            runs_viterbi.then_some(case.viterbi.to_bits()),
            "{name}: MSV P-value {msv_pvalue}"
        );

        let stats = worker.into_metrics().stats;
        let passed = [
            stats.sequences_past_msv,
            stats.sequences_past_bias,
            stats.sequences_past_viterbi,
            stats.sequences_past_forward,
        ];
        match (case.outcome, &report.outcome) {
            (Outcome::Stopped(expected), SearchOutcome::Filtered(reason)) => {
                assert_eq!(*reason, expected, "{name}");
                let stages = match expected {
                    FilterReason::Msv => 0,
                    FilterReason::Bias => 1,
                    FilterReason::Viterbi => 2,
                    _ => 3,
                };
                let mut counts = [0; 4];
                counts[..stages].fill(1);
                assert_eq!(passed, counts, "{name}");
            }
            (Outcome::Hit(score), SearchOutcome::Hit(hit)) => {
                // `hmmsearch` prints a tenth of a bit.
                assert!((hit.score - score).abs() < 0.06, "{name}: {}", hit.score);
                assert_eq!(passed, [1; 4], "{name}");
            }
            (expected, got) => panic!("{name}: expected {expected:?}, got {got:?}"),
        }
    }
    // The table holds both sides of the rule that skips the Viterbi filter.
    let stopped = |reason| {
        CASES
            .iter()
            .filter(|case| case.outcome == Outcome::Stopped(reason))
            .count()
    };
    assert!(stopped(FilterReason::Msv) >= 1 && stopped(FilterReason::Bias) >= 2);
    assert!(stopped(FilterReason::Viterbi) >= 3 && stopped(FilterReason::Forward) >= 2);
}
