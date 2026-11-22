pub use crate::hmmer_core::constants::*;

// ---------------------------------------------------------------
// E-value parameter indices
// ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvParam {
    MsvMu = 0,
    MsvLambda = 1,
    ViterbiMu = 2,
    ViterbiLambda = 3,
    ForwardTau = 4,
    ForwardLambda = 5,
}

impl EvParam {
    pub const fn idx(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------
// HMM transitions
// ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HTransition {
    MatchToMatch = 0,
    MatchToInsert = 1,
    MatchToDelete = 2,
    InsertToMatch = 3,
    InsertToInsert = 4,
    DeleteToMatch = 5,
    DeleteToDelete = 6,
}

impl HTransition {
    pub const fn idx(self) -> usize {
        self as usize
    }
}

// ---------------------------------------------------------------
// State types for traces
// ---------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceStateType {
    Match = 1,
    Delete = 2,
    Insert = 3,
    Start = 4,
    NTerminal = 5,
    Begin = 6,
    End = 7,
    CTerminal = 8,
    Terminate = 9,
    Jump = 10,
}

impl TraceStateType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Match => "M",
            Self::Delete => "D",
            Self::Insert => "I",
            Self::Start => "S",
            Self::NTerminal => "N",
            Self::Begin => "B",
            Self::End => "E",
            Self::CTerminal => "C",
            Self::Terminate => "T",
            Self::Jump => "J",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PTsc {
    MatchToMatch = 0,
    InsertToMatch = 1,
    DeleteToMatch = 2,
    BeginToMatch = 3,
    MatchToDelete = 4,
    DeleteToDelete = 5,
    MatchToInsert = 6,
    InsertToInsert = 7,
}

impl PTsc {
    pub const fn idx(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PRsc {
    MatchScore = 0,
    InsertScore = 1,
}

impl PRsc {
    pub const fn idx(self) -> usize {
        self as usize
    }
}
