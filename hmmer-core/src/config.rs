pub use crate::constants::*;

/// Search mode for profile configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Multihit local: "fs" mode
    Local = 1,
    /// Multihit glocal: "ls" mode
    Glocal = 2,
    /// Unihit local: "sw" mode
    UniLocal = 3,
    /// Unihit glocal: "s" mode
    UniGlocal = 4,
}

impl SearchMode {
    pub fn is_local(self) -> bool {
        matches!(self, Self::Local | Self::UniLocal)
    }

    pub fn is_multihit(self) -> bool {
        matches!(self, Self::Local | Self::Glocal)
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cutoff {
    GatheringSequence = 0,
    GatheringDomain = 1,
    TrustedSequence = 2,
    TrustedDomain = 3,
    NoiseSequence = 4,
    NoiseDomain = 5,
}

impl Cutoff {
    pub const fn idx(self) -> usize {
        self as usize
    }
}

/// Which strand(s) should be searched
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strands {
    TopOnly = 0,
    BottomOnly = 1,
    Both = 2,
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
    Bogus = 0,
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
    Missing = 11, // missing data: for local entry/exits
}

impl TraceStateType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bogus => "?",
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
            Self::Missing => "X",
        }
    }

    pub const fn idx(self) -> usize {
        self as usize
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
