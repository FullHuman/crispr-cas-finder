// rng.rs — Random number generator
//
// Replaces EslRandomness with a named xorshift64 implementation.

use std::fmt;

/// A simple xorshift64 random number generator.
///
/// Deterministic and reproducible given the same seed.
/// Used for HMM sampling, e-value calibration, stochastic traceback, etc.
pub struct XorShift64 {
    seed: u64,
    state: u64,
}

impl XorShift64 {
    /// Create a new RNG with the given seed.
    pub fn new(seed: u64) -> Self {
        let mut rng = XorShift64 { seed, state: seed };
        // Warm up the generator
        for _ in 0..10 {
            rng.next_u64();
        }
        rng
    }

    /// Generate next random u64 via xorshift64.
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Return a random float in [0, 1).
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Return a random integer in [0, n).
    pub fn roll(&mut self, n: usize) -> usize {
        (self.random() * n as f64) as usize
    }

    /// Get the original seed.
    pub fn get_seed(&self) -> u64 {
        self.seed
    }
}

impl fmt::Debug for XorShift64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("XorShift64")
            .field("seed", &self.seed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut rng1 = XorShift64::new(42);
        let mut rng2 = XorShift64::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.random(), rng2.random());
        }
    }

    #[test]
    fn test_range() {
        let mut rng = XorShift64::new(1234);
        for _ in 0..1000 {
            let v = rng.random();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn test_roll() {
        let mut rng = XorShift64::new(99);
        for _ in 0..1000 {
            let v = rng.roll(10);
            assert!(v < 10);
        }
    }

    #[test]
    fn test_seed() {
        let rng = XorShift64::new(42);
        assert_eq!(rng.get_seed(), 42);
    }
}
