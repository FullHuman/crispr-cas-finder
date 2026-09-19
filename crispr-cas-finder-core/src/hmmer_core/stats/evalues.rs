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

/// Convert a Viterbi score to a P-value.
pub fn viterbi_score_to_pvalue(score: f32, mu: f32, lambda: f32) -> f64 {
    // Gumbel: P(S >= x) = 1 - exp(-exp(-lambda(x-mu)))
    let x = -(lambda as f64) * (score as f64 - mu as f64);
    -(-x.exp()).exp_m1()
}

/// Convert an MSV score to a P-value.
pub fn msv_score_to_pvalue(score: f32, mu: f32, lambda: f32) -> f64 {
    // Same Gumbel distribution
    viterbi_score_to_pvalue(score, mu, lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

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
