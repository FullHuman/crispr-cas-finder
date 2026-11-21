// Vectorised kernels of the HMM search.
//
// The kernels carry `#[inline(never)]` so that each keeps its own symbol
// under fat LTO: a sampling profiler then attributes time to the kernel
// rather than to the search pipeline it is called from.

pub mod forward_filter;
pub mod int_msv;
pub mod msv_filter;
pub mod oprofile;
pub mod striped_forward;
pub mod viterbi_filter;
