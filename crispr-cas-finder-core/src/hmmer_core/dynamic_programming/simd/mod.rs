// Vectorised kernels of the HMM search.
//
// The kernels carry `#[inline(never)]` so that each keeps its own symbol
// under fat LTO: a sampling profiler then attributes time to the kernel
// rather than to the search pipeline it is called from.

pub mod f32_msv;
pub mod f32_viterbi;
pub mod forward_filter;
pub mod fwd_bck;
pub mod i16_msv;
pub mod i16_viterbi;
pub mod i8_msv;
pub mod int_msv;
pub mod oprofile;
