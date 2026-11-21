// evalues.rs - E-value calibration of profile HMMs
//
// Port of src/evalues.c

/// Convert a Forward score to a P-value.
pub fn forward_score_to_pvalue(score: f32, tau: f32, lambda: f32) -> f64 {
    if score <= tau {
        return 1.0;
    }
    // P(S >= x) = exp(-lambda * (x - tau))
    let x = -(lambda as f64) * (score as f64 - tau as f64);
    x.exp()
}

/// Convert a Viterbi or MSV score to a P-value. The filters do not use it
/// (see [`gumbel_survival`]); the tests compare the two.
pub fn viterbi_score_to_pvalue(score: f32, mu: f32, lambda: f32) -> f64 {
    // Gumbel: P(S >= x) = 1 - exp(-exp(-lambda(x-mu)))
    let x = -(lambda as f64) * (score as f64 - mu as f64);
    -(-x.exp()).exp_m1()
}

/// The bit score a filter of `p7_Pipeline()` tests: `(score - null) /
/// eslCONST_LOG2`, the difference in single precision, the division in
/// double precision, the quotient stored in single precision.
#[inline]
pub fn filter_bit_score(score: f32, null: f32) -> f32 {
    (f64::from(score - null) / std::f64::consts::LN_2) as f32
}

/// `esl_gumbel_surv()`: the P-value HMMER's MSV, bias and Viterbi filters
/// compare with their thresholds, operation for operation. Mathematically
/// the same as [`viterbi_score_to_pvalue`]; the last bits can differ.
#[inline]
pub fn gumbel_survival(score: f32, mu: f32, lambda: f32) -> f64 {
    let y = f64::from(lambda) * (f64::from(score) - f64::from(mu));
    let ey = -(-y).exp();
    // Easel's `eslSMALLX1`: 1 - e^x is -x for a small x.
    if ey.abs() < 5e-9 { -ey } else { 1.0 - ey.exp() }
}

/// `esl_exp_surv()`: the P-value HMMER's Forward filter compares with its
/// threshold.
#[inline]
pub fn exponential_survival(score: f32, tau: f32, lambda: f32) -> f64 {
    let (score, tau) = (f64::from(score), f64::from(tau));
    if score < tau {
        return 1.0;
    }
    (-f64::from(lambda) * (score - tau)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_survival_functions_agree_with_the_reporting_ones() {
        for score in [-30.0f32, -8.5, -1.0, 0.0, 3.25, 12.0, 40.0, 400.0] {
            let (mu, lambda) = (-9.3f32, 0.71f32);
            // Easel subtracts from one where the reporting function uses
            // `exp_m1`: the two agree to the last bits of a double near one.
            let filter = gumbel_survival(score, mu, lambda);
            let reported = viterbi_score_to_pvalue(score, mu, lambda);
            assert!((filter - reported).abs() <= 3e-16 + 1e-12 * reported);
            let filter = exponential_survival(score, mu, lambda);
            let reported = forward_score_to_pvalue(score, mu, lambda);
            assert!((filter - reported).abs() <= 1e-12 * reported);
        }
        // An overflowed filter score passes every threshold; a score of
        // minus infinity none.
        assert_eq!(gumbel_survival(f32::INFINITY, -9.3, 0.71), 0.0);
        assert_eq!(gumbel_survival(f32::NEG_INFINITY, -9.3, 0.71), 1.0);
        assert_eq!(filter_bit_score(f32::INFINITY, -7.5), f32::INFINITY);
        assert_eq!(
            filter_bit_score(-3.0, -7.5),
            (4.5 / std::f64::consts::LN_2) as f32
        );
    }

    #[test]
    fn exponential_survival_is_bounded() {
        assert_eq!(forward_score_to_pvalue(-100.0, 0.0, 1.0), 1.0);
        assert_eq!(forward_score_to_pvalue(0.0, 0.0, 1.0), 1.0);
        assert_eq!(forward_score_to_pvalue(f32::INFINITY, 0.0, 1.0), 0.0);
        assert!((forward_score_to_pvalue(2.0, 0.0, 1.0) - (-2.0f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn gumbel_survival_preserves_small_probabilities() {
        let p = viterbi_score_to_pvalue(100.0, 0.0, 1.0);
        assert!(p > 0.0);
        assert!((p / (-100.0f64).exp() - 1.0).abs() < 1e-12);
        assert_eq!(viterbi_score_to_pvalue(-1000.0, 0.0, 1.0), 1.0);
        assert_eq!(viterbi_score_to_pvalue(f32::INFINITY, 0.0, 1.0), 0.0);
    }
}
