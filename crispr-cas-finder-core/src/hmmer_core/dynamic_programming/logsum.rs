// logsum.rs - p7_FLogsum() function used in the Forward() algorithm.
//
// Port of src/logsum.c
//
// Internally, HMMER3 profile scores are in nats: floating point
// log-odds probabilities, with the log odds taken relative to
// background residue frequencies, and the log to the base e.
//
// The Forward algorithm needs to calculate sums of probabilities.
// Given two log probabilities A and B, we need to compute log(e^A + e^B).
//
// We use a table-driven approximation for speed. The lookup table is
// generated at compile time by build.rs.

use crate::hmmer_core::constants::dynamic_programming::logsum::LOGSUM_SCALE;

include!(concat!(env!("OUT_DIR"), "/flogsum_table.rs"));

/// Approximate log(e^a + e^b).
///
/// Returns a fast table-driven approximation to log(e^a + e^b).
///
/// Either a or b (or both) may be -infinity,
/// but neither may be +infinity or NaN.
///
/// This function is a critical optimization target, because
/// it's in the inner loop of generic Forward() algorithms.
#[inline]
pub fn flogsum(a: f32, b: f32) -> f32 {
    let max = a.max(b);
    let diff = (a - b).abs();
    if diff >= 15.7 {
        max
    } else {
        max + FLOGSUM_LOOKUP[(diff * LOGSUM_SCALE) as usize]
    }
}

// ---------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flogsum_specials() {
        // log(e^0 + e^-inf) = log(1 + 0) = 0
        assert_eq!(flogsum(0.0, f32::NEG_INFINITY), 0.0);
        // log(e^-inf + e^0) = 0
        assert_eq!(flogsum(f32::NEG_INFINITY, 0.0), 0.0);
        // log(e^-inf + e^-inf) = -inf
        assert_eq!(
            flogsum(f32::NEG_INFINITY, f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn test_flogsum_accuracy() {
        let max_val = 20.0_f32;
        let n = 10000;
        let mut max_err = 0.0_f32;
        let mut avg_err = 0.0_f32;

        // Use a simple deterministic "random" sequence for reproducibility
        let mut state = 42u64;
        let mut rand_f = || -> f32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64) / (u64::MAX as f64) - 0.5) as f32 * max_val * 2.0
        };

        for _ in 0..n {
            let a = rand_f();
            let b = rand_f();

            let exact = ((a as f64).exp() + (b as f64).exp()).ln() as f32;
            let result = flogsum(a, b);
            let err = (exact - result).abs() / max_val;

            avg_err += err;
            if err > max_err {
                max_err = err;
            }
        }
        avg_err /= n as f32;

        assert!(max_err < 0.0001, "maximum error of {} is too high", max_err);
        assert!(avg_err < 0.0001, "average error of {} is too high", avg_err);
    }
}
