//! Optimal accuracy (OA) decoding and traceback.
//!
//! Port of HMMER's `src/generic_optacc.c`.
//!
//! # Algorithm overview
//!
//! Given a posterior decoding matrix (produced by the Forward/Backward algorithm),
//! optimal accuracy alignment finds the state path through the HMM that maximises
//! the *expected number of correctly decoded residues*. The algorithm proceeds in
//! two passes:
//!
//! 1. **OA fill** ([`optimal_accuracy`]) — a DP recurrence analogous to Viterbi,
//!    but accumulating posterior probabilities rather than log-odds scores. Each
//!    cell stores the maximum expected accuracy on any path ending there.
//!
//! 2. **OA traceback** ([`oa_trace`]) — a greedy max-posterior walk backward
//!    through the filled OA matrix, emitting the optimal alignment path.
//!
//! ## Transition deltas
//!
//! Instead of adding log-transition scores (as Viterbi does), the OA recurrence
//! multiplies accumulated OA values by a *transition delta*: `1.0` for a finite
//! (allowed) transition and [`f32::MIN_POSITIVE`] for a `-∞` (forbidden) one.
//! Using `MIN_POSITIVE` rather than `0.0` prevents `NaN` from `0.0 × -∞` while
//! still making forbidden paths effectively unreachable.

use crate::config::{PROFILE_NUM_TRANSITIONS, PTsc, TraceStateType};
use crate::errors::HmmerError;
use crate::profile::Profile;
use crate::score_matrix::{BEGIN_STATE, C_STATE, EXIT_STATE, J_STATE, N_STATE, ScoreMatrix};
use crate::trace::Trace;

// ── Transition delta helpers ─────────────────────────────────────────────────

/// Convert a raw transition log-score to an OA transition delta.
///
/// Returns `1.0` for any finite score (the transition is allowed and the OA
/// value propagates unchanged) or [`f32::MIN_POSITIVE`] for `-∞` (the
/// transition is forbidden; a tiny positive value rather than `0.0` prevents
/// `NaN` from `0.0 × -∞`).
#[inline]
fn to_delta(score: f32) -> f32 {
    if score == f32::NEG_INFINITY {
        f32::MIN_POSITIVE
    } else {
        1.0
    }
}

/// Look up the OA transition delta for main-state transition `s` at node `k`.
///
/// `k` is a 0-based node index; the flat table is laid out as
/// `transition_scores[k * PROFILE_NUM_TRANSITIONS + s]`.
#[inline]
fn tscdelta(transition_scores: &[f32], k: usize, s: usize) -> f32 {
    to_delta(transition_scores[k * PROFILE_NUM_TRANSITIONS + s])
}

/// Pre-computed OA transition deltas for the special-state transitions.
///
/// These are loop-invariant over sequence position and computed once before the
/// main DP loop to avoid repeated branching inside it.
struct SpecialTransitionDeltas {
    n_loop: f32,
    n_move: f32,
    j_loop: f32,
    j_move: f32,
    e_loop: f32,
    e_move: f32,
    c_loop: f32,
}

impl SpecialTransitionDeltas {
    /// Build the delta table from the profile's special-state log-scores.
    fn from_profile(profile: &Profile) -> Self {
        let s = &profile.special_scores;
        Self {
            n_loop: to_delta(s.n_loop),
            n_move: to_delta(s.n_move),
            j_loop: to_delta(s.j_loop),
            j_move: to_delta(s.j_move),
            e_loop: to_delta(s.e_loop),
            e_move: to_delta(s.e_move),
            c_loop: to_delta(s.c_loop),
        }
    }
}

/// Pre-compute OA transition deltas for all main-state transitions.
///
/// Returns a flat table of length `num_nodes × PROFILE_NUM_TRANSITIONS` where
/// each entry is either `1.0` (allowed) or [`f32::MIN_POSITIVE`] (forbidden).
/// Materialising the table eliminates per-cell branching in the hot inner loop.
fn build_transition_delta_table(transition_scores: &[f32], num_nodes: usize) -> Vec<f32> {
    let len = num_nodes * PROFILE_NUM_TRANSITIONS;
    transition_scores[..len]
        .iter()
        .map(|&s| to_delta(s))
        .collect()
}

// The main-state recurrences are fused into the row loop below. Besides reducing
// indexing overhead, this lets adjacent cells reuse values from both DP rows.

// ── Per-row fill helpers ─────────────────────────────────────────────────────

/// Fill all main-model nodes (k = 1 …= M) for sequence row `i`.
///
/// Interior nodes `1 ..< M` receive match, insert, and delete scores. The final
/// node `M` has no insert state (fixed to `-∞`). In local-alignment mode every
/// match cell feeds the exit-state accumulator; in glocal mode only the final
/// node M does (both its match and delete scores contribute unconditionally).
///
/// Returns `x_e`, the maximum OA score reaching the E (exit) state at row `i`.
fn fill_main_nodes(
    i: usize,
    dp: &mut ScoreMatrix,
    pp: &ScoreMatrix,
    td: &[f32],
    prev_b: f32,
    exit_gate: f32,
    num_nodes: usize,
) -> f32 {
    let mut x_e = f32::NEG_INFINITY;

    // Values at node k-1 are reused by the next match recurrence. The current
    // row's values also directly feed the left-to-right delete recurrence.
    let mut prev_match = f32::NEG_INFINITY;
    let mut prev_insert = f32::NEG_INFINITY;
    let mut prev_delete = f32::NEG_INFINITY;
    let mut left_match = f32::NEG_INFINITY;
    let mut left_delete = f32::NEG_INFINITY;

    for k in 1..num_nodes {
        let base = (k - 1) * PROFILE_NUM_TRANSITIONS;

        let mm = pp.match_score(i, k)
            + f32::max(
                f32::max(
                    td[base + PTsc::MatchToMatch as usize] * prev_match,
                    td[base + PTsc::InsertToMatch as usize] * prev_insert,
                ),
                f32::max(
                    td[base + PTsc::DeleteToMatch as usize] * prev_delete,
                    td[base + PTsc::BeginToMatch as usize] * prev_b,
                ),
            );
        dp.set_match_score(i, k, mm);
        x_e = f32::max(x_e, exit_gate * mm);

        // These values become the k-1 predecessors on the next iteration.
        let next_prev_match = dp.match_score(i - 1, k);
        let next_prev_insert = dp.insert_score(i - 1, k);
        let next_prev_delete = dp.delete_score(i - 1, k);

        let insert_base = base + PROFILE_NUM_TRANSITIONS;
        let im = pp.insert_score(i, k)
            + f32::max(
                td[insert_base + PTsc::MatchToInsert as usize] * next_prev_match,
                td[insert_base + PTsc::InsertToInsert as usize] * next_prev_insert,
            );
        dp.set_insert_score(i, k, im);

        let dm = f32::max(
            td[base + PTsc::MatchToDelete as usize] * left_match,
            td[base + PTsc::DeleteToDelete as usize] * left_delete,
        );
        dp.set_delete_score(i, k, dm);

        prev_match = next_prev_match;
        prev_insert = next_prev_insert;
        prev_delete = next_prev_delete;
        left_match = mm;
        left_delete = dm;
    }

    // Final node k = M has no insert state.
    let base = (num_nodes - 1) * PROFILE_NUM_TRANSITIONS;
    let mm = pp.match_score(i, num_nodes)
        + f32::max(
            f32::max(
                td[base + PTsc::MatchToMatch as usize] * prev_match,
                td[base + PTsc::InsertToMatch as usize] * prev_insert,
            ),
            f32::max(
                td[base + PTsc::DeleteToMatch as usize] * prev_delete,
                td[base + PTsc::BeginToMatch as usize] * prev_b,
            ),
        );
    dp.set_match_score(i, num_nodes, mm);

    let dm = f32::max(
        td[base + PTsc::MatchToDelete as usize] * left_match,
        td[base + PTsc::DeleteToDelete as usize] * left_delete,
    );
    dp.set_delete_score(i, num_nodes, dm);
    dp.set_insert_score(i, num_nodes, f32::NEG_INFINITY);

    x_e = f32::max(x_e, f32::max(mm, dm));
    x_e
}

/// Fill the special states (E, J, C, N, B) for sequence row `i`.
///
/// The special-state recurrences are:
///
/// ```text
/// E(i)  = x_e  (passed in from fill_main_nodes)
/// J(i)  = max(δ(J_loop) · (J(i-1) + pp_J(i)),  δ(E_loop) · E(i))
/// C(i)  = max(δ(C_loop) · (C(i-1) + pp_C(i)),  δ(E_move) · E(i))
/// N(i)  = δ(N_loop) · (N(i-1) + pp_N(i))
/// B(i)  = max(δ(N_move) · N(i),  δ(J_move) · J(i))
/// ```
///
/// States are written in a specific dependency order: E → J → C → N → B.
fn fill_special_states(
    i: usize,
    x_e: f32,
    dp: &mut ScoreMatrix,
    pp: &ScoreMatrix,
    td: &SpecialTransitionDeltas,
) {
    dp.special[[i, EXIT_STATE]] = x_e;

    // J: re-enter (self-loop) accumulating posterior, or transition from E.
    let j_prev = dp.special[[i - 1, J_STATE]];
    dp.special[[i, J_STATE]] = f32::max(
        td.j_loop * (j_prev + pp.special_score(i, J_STATE)),
        td.e_loop * x_e,
    );

    // C: C-terminal tail self-loop, or move from E.
    let c_prev = dp.special[[i - 1, C_STATE]];
    dp.special[[i, C_STATE]] = f32::max(
        td.c_loop * (c_prev + pp.special_score(i, C_STATE)),
        td.e_move * x_e,
    );

    // N: N-terminal self-loop only (N cannot be entered from other states mid-sequence).
    let n_prev = dp.special[[i - 1, N_STATE]];
    dp.special[[i, N_STATE]] = td.n_loop * (n_prev + pp.special_score(i, N_STATE));

    // B: begin a domain from N (first domain) or re-enter from J (subsequent domains).
    let n_cur = dp.special[[i, N_STATE]];
    let j_cur = dp.special[[i, J_STATE]];
    dp.special[[i, BEGIN_STATE]] = f32::max(td.n_move * n_cur, td.j_move * j_cur);
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Result of an optimal-accuracy decode: the OA score and the traceback.
#[derive(Debug)]
pub struct OaResult {
    pub score: f32,
    pub trace: Trace,
}

/// Optimal-accuracy decoder with a reusable DP matrix.
///
/// Owns a pooled [`ScoreMatrix`] that is resized (never shrunk) across calls,
/// eliminating repeated allocation when rescoring many domains.
#[derive(Debug)]
pub struct OaDecoder {
    dp: ScoreMatrix,
}

impl OaDecoder {
    /// Create a new decoder pre-allocated for models up to `num_nodes` nodes.
    pub fn new(num_nodes: usize) -> Self {
        OaDecoder {
            dp: ScoreMatrix::new(num_nodes, 1).unwrap(),
        }
    }

    /// Run optimal-accuracy fill + traceback, returning the OA score and trace.
    pub fn decode(
        &mut self,
        profile: &Profile,
        posterior: &ScoreMatrix,
    ) -> Result<OaResult, HmmerError> {
        let score = optimal_accuracy_fill(profile, posterior, &mut self.dp)?;
        let trace = oa_trace(profile, posterior, &self.dp)?;
        Ok(OaResult { score, trace })
    }
}

/// Optimal accuracy decoding — fill step.
///
/// Given a posterior decoding matrix `posterior` (output of the Forward/Backward
/// algorithm), fills `dp` with the OA DP matrix and returns the expected number
/// of correctly decoded residue positions (the OA score).
fn optimal_accuracy_fill(
    profile: &Profile,
    posterior: &ScoreMatrix,
    dp: &mut ScoreMatrix,
) -> Result<f32, HmmerError> {
    let seq_len = posterior.sequence_length;
    let num_nodes = profile.num_nodes;

    // In local mode every match node can exit the model; in glocal mode only node
    // M can. The exit_gate encodes this: 1.0 passes the value through, while the
    // final-node case is handled unconditionally in fill_main_nodes regardless.
    let exit_gate: f32 = if profile.mode.is_local() { 1.0 } else { 0.0 };

    dp.resize(num_nodes, seq_len)?;

    // Precompute transition deltas once — avoids per-cell branching in the hot loop.
    let td = build_transition_delta_table(profile.transition_scores_raw(), num_nodes);
    let special_td = SpecialTransitionDeltas::from_profile(profile);

    // Row 0 initialisation: N(0) = B(0) = 0.0; all other states unreachable.
    dp.set_special_row(
        0,
        [
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
        ],
    );
    dp.fill_main_row(0, f32::NEG_INFINITY);

    for i in 1..=seq_len {
        // Boundary node k=0: no residues can be emitted from here.
        dp.set_match_score(i, 0, f32::NEG_INFINITY);
        dp.set_insert_score(i, 0, f32::NEG_INFINITY);
        dp.set_delete_score(i, 0, f32::NEG_INFINITY);

        let prev_b = dp.special_score(i - 1, BEGIN_STATE);
        let x_e = fill_main_nodes(i, dp, posterior, &td, prev_b, exit_gate, num_nodes);
        fill_special_states(i, x_e, dp, posterior, &special_td);
    }

    // The OA score is the maximum expected accuracy arriving at the C-terminal state.
    Ok(dp.special_score(seq_len, C_STATE))
}

/// Optimal accuracy traceback.
///
/// Traces back through the OA DP matrix `dp` (filled by the fill step)
/// to find the state path maximising the expected number of correctly decoded
/// residues. Returns the trace in forward (5'→3') order.
fn oa_trace(
    profile: &Profile,
    posterior: &ScoreMatrix,
    dp: &ScoreMatrix,
) -> Result<Trace, HmmerError> {
    let seq_len = dp.sequence_length;
    let num_nodes = profile.num_nodes;
    let tsc = profile.transition_scores_raw();

    let mut seq_pos = seq_len as i32;
    let mut k: i32 = 0;

    let mut trace = Trace::with_pp();

    // Seed the trace at the terminal end; traceback proceeds right-to-left.
    trace.append_with_pp(TraceStateType::Terminate, 0, seq_pos, 0.0);
    trace.append_with_pp(TraceStateType::CTerminal, 0, seq_pos, 0.0);

    let mut sprv = TraceStateType::CTerminal;

    while sprv != TraceStateType::Start {
        let scur =
            dispatch_traceback_state(profile, posterior, dp, tsc, sprv, &mut seq_pos, &mut k)?;

        let pp = get_postprob(posterior, scur, sprv, k as usize, seq_pos as usize);
        trace.append_with_pp(scur, k, seq_pos, pp);

        // Self-loops on N, J, and C each consume one sequence position.
        if (scur == TraceStateType::NTerminal
            || scur == TraceStateType::Jump
            || scur == TraceStateType::CTerminal)
            && scur == sprv
        {
            seq_pos -= 1;
        }

        sprv = scur;
    }

    trace.model_length = num_nodes;
    trace.sequence_length = seq_len;
    trace.reverse();
    Ok(trace)
}

fn dispatch_traceback_state(
    profile: &Profile,
    posterior: &ScoreMatrix,
    dp: &ScoreMatrix,
    tsc: &[f32],
    sprv: TraceStateType,
    seq_pos: &mut i32,
    k: &mut i32,
) -> Result<TraceStateType, HmmerError> {
    let i = *seq_pos as usize;
    let ku = *k as usize;

    // Determine the predecessor state that led into `sprv`, and advance the
    // sequence/node cursors to reflect the transition we are tracing back.
    match sprv {
        TraceStateType::Match => Ok(traceback_from_match(tsc, dp, i, ku, seq_pos, k)),
        TraceStateType::Delete => Ok(traceback_from_delete(tsc, dp, i, ku, k)),
        TraceStateType::Insert => Ok(traceback_from_insert(tsc, dp, i, ku, seq_pos)),
        // N with i=0 means we have reached the true start of the sequence.
        TraceStateType::NTerminal => Ok(if i == 0 {
            TraceStateType::Start
        } else {
            TraceStateType::NTerminal
        }),
        TraceStateType::CTerminal => Ok(select_c(profile, posterior, dp, i)),
        TraceStateType::Jump => Ok(select_j(profile, posterior, dp, i)),
        TraceStateType::End => {
            let (s, kk) = select_e(profile, dp, i);
            *k = kk;
            Ok(s)
        }
        TraceStateType::Begin => Ok(select_b(profile, dp, i)),
        _ => Err(HmmerError::Internal(
            "OA traceback reached unexpected state".into(),
        )),
    }
}

fn traceback_from_match(
    tsc: &[f32],
    dp: &ScoreMatrix,
    i: usize,
    ku: usize,
    seq_pos: &mut i32,
    k: &mut i32,
) -> TraceStateType {
    let s = select_m(tsc, dp, i, ku);
    *k -= 1;
    *seq_pos -= 1;
    s
}

fn traceback_from_delete(
    tsc: &[f32],
    dp: &ScoreMatrix,
    i: usize,
    ku: usize,
    k: &mut i32,
) -> TraceStateType {
    let s = select_d(tsc, dp, i, ku);
    *k -= 1;
    s
}

fn traceback_from_insert(
    tsc: &[f32],
    dp: &ScoreMatrix,
    i: usize,
    ku: usize,
    seq_pos: &mut i32,
) -> TraceStateType {
    let s = select_i(tsc, dp, i, ku);
    *seq_pos -= 1;
    s
}

// ── Posterior probability lookup ─────────────────────────────────────────────

/// Return the posterior probability to annotate a newly appended trace cell.
///
/// Emitting states (M, I) carry residue-level posterior probabilities.
/// Self-loop transitions on N, J, and C carry position-level posteriors.
/// All other states (non-emitting, one-time transitions) get `0.0`.
fn get_postprob(
    pp: &ScoreMatrix,
    scur: TraceStateType,
    sprv: TraceStateType,
    k: usize,
    i: usize,
) -> f32 {
    match scur {
        TraceStateType::Match => pp.match_score(i, k),
        TraceStateType::Insert => pp.insert_score(i, k),
        TraceStateType::NTerminal if sprv == scur => pp.special_score(i, N_STATE),
        TraceStateType::CTerminal if sprv == scur => pp.special_score(i, C_STATE),
        TraceStateType::Jump if sprv == scur => pp.special_score(i, J_STATE),
        _ => 0.0,
    }
}

// ── Traceback state-selection functions ──────────────────────────────────────

/// Select the predecessor of a **match** state at `(i, k)`.
///
/// Evaluates four candidate predecessors — M(i-1,k-1), I(i-1,k-1), D(i-1,k-1),
/// and B(i-1) — weighted by their respective transition deltas, and returns the
/// state with the highest OA value.
fn select_m(tsc: &[f32], dp: &ScoreMatrix, i: usize, k: usize) -> TraceStateType {
    let scores = [
        tscdelta(tsc, k - 1, PTsc::MatchToMatch as usize) * dp.match_score(i - 1, k - 1),
        tscdelta(tsc, k - 1, PTsc::InsertToMatch as usize) * dp.insert_score(i - 1, k - 1),
        tscdelta(tsc, k - 1, PTsc::DeleteToMatch as usize) * dp.delete_score(i - 1, k - 1),
        tscdelta(tsc, k - 1, PTsc::BeginToMatch as usize) * dp.special_score(i - 1, BEGIN_STATE),
    ];
    const STATES: [TraceStateType; 4] = [
        TraceStateType::Match,
        TraceStateType::Insert,
        TraceStateType::Delete,
        TraceStateType::Begin,
    ];
    STATES[argmax(&scores)]
}

/// Select the predecessor of a **delete** state at `(i, k)`.
///
/// Delete states are non-emitting, so candidates are M(i,k-1) and D(i,k-1)
/// read from the current row.
fn select_d(tsc: &[f32], dp: &ScoreMatrix, i: usize, k: usize) -> TraceStateType {
    let p_match = tscdelta(tsc, k - 1, PTsc::MatchToDelete as usize) * dp.match_score(i, k - 1);
    let p_delete = tscdelta(tsc, k - 1, PTsc::DeleteToDelete as usize) * dp.delete_score(i, k - 1);
    if p_match >= p_delete {
        TraceStateType::Match
    } else {
        TraceStateType::Delete
    }
}

/// Select the predecessor of an **insert** state at `(i, k)`.
///
/// Candidates are M(i-1,k) and I(i-1,k).
fn select_i(tsc: &[f32], dp: &ScoreMatrix, i: usize, k: usize) -> TraceStateType {
    let p_match = tscdelta(tsc, k, PTsc::MatchToInsert as usize) * dp.match_score(i - 1, k);
    let p_insert = tscdelta(tsc, k, PTsc::InsertToInsert as usize) * dp.insert_score(i - 1, k);
    if p_match >= p_insert {
        TraceStateType::Match
    } else {
        TraceStateType::Insert
    }
}

/// Select the predecessor of the **C** (C-terminal) state at position `i`.
///
/// Either continues the C self-loop (accumulating posterior probability) or
/// transitions from the E state (end of last domain).
fn select_c(profile: &Profile, pp: &ScoreMatrix, dp: &ScoreMatrix, i: usize) -> TraceStateType {
    let p_loop = to_delta(profile.special_scores.c_loop)
        * (dp.special_score(i - 1, C_STATE) + pp.special_score(i, C_STATE));
    let p_exit = to_delta(profile.special_scores.e_move) * dp.special_score(i, EXIT_STATE);
    if p_loop > p_exit {
        TraceStateType::CTerminal
    } else {
        TraceStateType::End
    }
}

/// Select the predecessor of the **J** (inter-domain jump) state at position `i`.
///
/// Either continues the J self-loop (accumulating posterior) or transitions from
/// E after the end of a domain.
fn select_j(profile: &Profile, pp: &ScoreMatrix, dp: &ScoreMatrix, i: usize) -> TraceStateType {
    let p_loop = to_delta(profile.special_scores.j_loop)
        * (dp.special_score(i - 1, J_STATE) + pp.special_score(i, J_STATE));
    let p_exit = to_delta(profile.special_scores.e_loop) * dp.special_score(i, EXIT_STATE);
    if p_loop > p_exit {
        TraceStateType::Jump
    } else {
        TraceStateType::End
    }
}

/// Select the predecessor of the **E** (exit/end) state, and its source node.
///
/// In glocal mode only node M can exit. In local mode every node can, so we
/// scan all nodes for the highest OA score.
///
/// Returns `(predecessor_state, node_index)`.
fn select_e(profile: &Profile, dp: &ScoreMatrix, i: usize) -> (TraceStateType, i32) {
    let m = profile.num_nodes;

    if !profile.mode.is_local() {
        // Glocal: the exit must come from match or delete at the final node M.
        let state = if dp.match_score(i, m) >= dp.delete_score(i, m) {
            TraceStateType::Match
        } else {
            TraceStateType::Delete
        };
        return (state, m as i32);
    }

    // Local: scan all nodes for the best OA score (match takes priority on ties).
    let mut best = f32::NEG_INFINITY;
    let mut best_state = TraceStateType::Match;
    let mut best_k: i32 = 1;

    for k in 1..=m {
        let ms = dp.match_score(i, k);
        let ds = dp.delete_score(i, k);
        if ms >= best {
            best = ms;
            best_state = TraceStateType::Match;
            best_k = k as i32;
        }
        if ds > best {
            best = ds;
            best_state = TraceStateType::Delete;
            best_k = k as i32;
        }
    }

    (best_state, best_k)
}

/// Select the predecessor of the **B** (begin) state at position `i`.
///
/// Either comes from N (entering the first domain) or J (re-entering after an
/// inter-domain region).
fn select_b(profile: &Profile, dp: &ScoreMatrix, i: usize) -> TraceStateType {
    let p_n = to_delta(profile.special_scores.n_move) * dp.special_score(i, N_STATE);
    let p_j = to_delta(profile.special_scores.j_move) * dp.special_score(i, J_STATE);
    if p_n > p_j {
        TraceStateType::NTerminal
    } else {
        TraceStateType::Jump
    }
}

// ── Utility ──────────────────────────────────────────────────────────────────

/// Return the index of the largest element in `v`.
///
/// Ties are broken in favour of the earlier (lower-index) element, consistent
/// with the predecessor-selection convention used throughout OA traceback.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}
