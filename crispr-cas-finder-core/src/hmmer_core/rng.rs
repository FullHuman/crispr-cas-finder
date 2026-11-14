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
        // Zero is an absorbing state for xorshift. Keep the caller's seed for
        // reporting, but map it to a fixed nonzero state for reproducibility.
        let state = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        let mut rng = XorShift64 { seed, state };
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
        assert!(n > 0, "roll requires a nonempty range");
        ((self.random() * n as f64) as usize).min(n - 1)
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

/// Easel's "fast" generator, the one HMMER's search pipeline samples
/// stochastic tracebacks with (`esl_randomness_CreateFast()` in
/// `esl_random.c`, Easel 0.49): a linear congruential generator on 32 bits,
/// `x <- 69069 x + 1`, whose first state is the seed dispersed by
/// `esl_mix3()`.
///
/// It reproduces that generator bit for bit; the clustering of sampled
/// domain envelopes only gives HMMER's envelopes if it draws HMMER's numbers.
#[derive(Debug, Clone)]
pub struct EaselFastRng {
    seed: u32,
    state: u32,
}

impl EaselFastRng {
    /// HMMER's default `--seed`.
    pub const DEFAULT_SEED: u32 = 42;

    /// A generator seeded as `esl_randomness_CreateFast(seed)` seeds it.
    ///
    /// Easel replaces a seed of 0 with one taken from the clock and the
    /// process number. Neither exists on every target of this crate and the
    /// search has to be reproducible, so 0 stands for the default seed here.
    pub fn new(seed: u32) -> Self {
        let mut rng = EaselFastRng { seed: 0, state: 0 };
        rng.reseed(seed);
        rng
    }

    /// `esl_randomness_Init()`: restart the sequence of `seed`.
    pub fn reseed(&mut self, seed: u32) {
        let seed = if seed == 0 { Self::DEFAULT_SEED } else { seed };
        self.seed = seed;
        self.state = mix3(seed, 87_654_321, 12_345_678);
        if self.state == 0 {
            self.state = 42;
        }
    }

    /// The seed of the current sequence.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    /// The next 32 bits (`knuth()` in `esl_random.c`).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(69_069).wrapping_add(1);
        self.state
    }

    /// `esl_random()`: a uniform deviate in `[0, 1)`.
    #[inline]
    pub fn random(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }

    /// `esl_rnd_FChoose()`: an index drawn from the probabilities `p`.
    ///
    /// Always draws one number. `None` when no index is reached, which takes
    /// a sum that is not a number or not positive; Easel aborts there.
    #[inline]
    pub fn choose(&mut self, p: &[f32]) -> Option<usize> {
        let roll = self.random();
        let mut norm = 0.0f64;
        for &value in p {
            norm += f64::from(value);
        }
        let mut sum = 0.0f64;
        for (index, &value) in p.iter().enumerate() {
            sum += f64::from(value);
            if roll < sum / norm {
                return Some(index);
            }
        }
        None
    }
}

/// Bob Jenkins' three-word mix, as `esl_mix3()` in `easel.c`.
fn mix3(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    a = a.wrapping_sub(b).wrapping_sub(c) ^ (c >> 13);
    b = b.wrapping_sub(c).wrapping_sub(a) ^ (a << 8);
    c = c.wrapping_sub(a).wrapping_sub(b) ^ (b >> 13);
    a = a.wrapping_sub(b).wrapping_sub(c) ^ (c >> 12);
    b = b.wrapping_sub(c).wrapping_sub(a) ^ (a << 16);
    c = c.wrapping_sub(a).wrapping_sub(b) ^ (b >> 5);
    a = a.wrapping_sub(b).wrapping_sub(c) ^ (c >> 3);
    b = b.wrapping_sub(c).wrapping_sub(a) ^ (a << 10);
    c = c.wrapping_sub(a).wrapping_sub(b) ^ (b >> 15);
    c
}

/// `esl_vec_FNorm()`: scale `values` to sum to one, with Easel's compensated
/// sum; a vector summing to zero becomes uniform.
#[inline]
pub fn normalize_probabilities(values: &mut [f32]) {
    let mut sum = 0.0f32;
    let mut compensation = 0.0f32;
    for &value in values.iter() {
        let y = value - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    if sum != 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    } else {
        let uniform = (1.0f64 / values.len() as f64) as f32;
        values.fill(uniform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first numbers of seed 42, from a separate transcription of
    /// `esl_mix3()` and `knuth()`. The regression test against `hmmsearch`
    /// envelopes (`hmmer-core/tests`) is what ties them to HMMER itself.
    #[test]
    fn easel_fast_generator_sequence_of_seed_42() {
        let mut rng = EaselFastRng::new(42);
        assert_eq!(rng.seed(), 42);
        let drawn: Vec<u32> = (0..5).map(|_| rng.next_u32()).collect();
        assert_eq!(
            drawn,
            [
                432_788_820,
                3_613_595_717,
                2_598_039_618,
                264_748_763,
                2_256_532_576
            ]
        );
        rng.reseed(42);
        assert_eq!(rng.random(), 432_788_820.0 / 4_294_967_296.0);
        assert_eq!(EaselFastRng::new(0).next_u32(), 432_788_820);
    }

    #[test]
    fn choose_follows_the_cumulative_distribution() {
        let mut rng = EaselFastRng::new(42);
        // Rolls of seed 42: 0.1008, 0.8414, 0.6049, 0.0616.
        let p = [0.25f32, 0.25, 0.25, 0.25];
        let picks: Vec<_> = (0..4).map(|_| rng.choose(&p).unwrap()).collect();
        assert_eq!(picks, [0, 3, 2, 0]);
        assert_eq!(rng.choose(&[0.0, 0.0]), None);
    }

    #[test]
    fn normalize_handles_zero_vectors() {
        let mut values = [1.0f32, 3.0];
        normalize_probabilities(&mut values);
        assert_eq!(values, [0.25, 0.75]);
        let mut zeros = [0.0f32; 4];
        normalize_probabilities(&mut zeros);
        assert_eq!(zeros, [0.25; 4]);
    }

    #[test]
    fn zero_seed_is_reproducible_and_nonconstant() {
        let mut first = XorShift64::new(0);
        let mut second = XorShift64::new(0);
        assert_eq!(first.get_seed(), 0);
        let values: Vec<_> = (0..100).map(|_| first.random()).collect();
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
        assert!(values.windows(2).any(|v| v[0] != v[1]));
        assert_eq!(
            values,
            (0..100).map(|_| second.random()).collect::<Vec<_>>()
        );
    }

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
