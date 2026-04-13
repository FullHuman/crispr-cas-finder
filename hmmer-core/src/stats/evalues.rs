// evalues.rs - E-value calibration of profile HMMs
//
// Port of src/evalues.c

/// Convert a Forward score to a P-value.
pub fn forward_score_to_pvalue(score: f32, tau: f32, lambda: f32) -> f64 {
    // P(S >= x) = exp(-lambda * (x - tau))
    let x = -(lambda as f64) * ((score - tau) as f64);
    x.exp()
}

/// Convert a Viterbi score to a P-value.
pub fn viterbi_score_to_pvalue(score: f32, mu: f32, lambda: f32) -> f64 {
    // Gumbel: P(S >= x) = 1 - exp(-exp(-lambda(x-mu)))
    let x = -(lambda as f64) * ((score - mu) as f64);
    1.0 - (-x.exp()).exp()
}

/// Convert an MSV score to a P-value.
pub fn msv_score_to_pvalue(score: f32, mu: f32, lambda: f32) -> f64 {
    // Same Gumbel distribution
    viterbi_score_to_pvalue(score, mu, lambda)
}
