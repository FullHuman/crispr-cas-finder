"""Type stubs for crispr_cas_finder."""

from __future__ import annotations

__version__: str

class Repeat:
    """A single direct repeat within a CRISPR array.

    All attributes are read-only.
    """

    start: int
    """1-based start position on the source sequence."""

    end: int
    """1-based end position on the source sequence (inclusive)."""

    sequence: str
    """DNA sequence of the direct repeat."""

    def __len__(self) -> int:
        """Return the length of the repeat sequence."""
        ...

    def __repr__(self) -> str: ...

class Spacer:
    """A single spacer between two direct repeats.

    All attributes are read-only.
    """

    start: int
    """1-based start position on the source sequence."""

    end: int
    """1-based end position on the source sequence (inclusive)."""

    sequence: str
    """DNA sequence of the spacer."""

    def __len__(self) -> int:
        """Return the length of the spacer sequence."""
        ...

    def __repr__(self) -> str: ...

class CrisprArray:
    """A detected CRISPR array with its direct repeats, spacers, and metadata.

    All attributes are read-only.
    """

    seq_id: str
    """Identifier of the source sequence (FASTA header, without the ``>``)."""

    start: int
    """1-based start position of the first direct repeat."""

    end: int
    """1-based end position of the last direct repeat (inclusive)."""

    consensus_repeat: str
    """Consensus direct repeat sequence chosen for this array."""

    repeats: list[Repeat]
    """All direct repeats, in array order."""

    spacers: list[Spacer]
    """All spacers, in array order (always one fewer than ``repeats``)."""

    evidence_level: int
    """Confidence level from 1 (low) to 4 (high)."""

    orientation: str
    """Strand orientation: ``"+"`` (forward), ``"-"`` (reverse), ``"."`` (unknown)."""

    repeat_id: str
    """Canonical repeat ID from the CRISPRdb database, e.g. ``"R10"``."""

    crispr_direction: str
    """Direction from the CRISPRDirection database (``"+"``, ``"-"``, or ``"ND"``)."""

    @property
    def spacer_count(self) -> int:
        """Number of spacers in this array."""
        ...

    @property
    def repeat_count(self) -> int:
        """Number of direct repeats in this array."""
        ...

    def __len__(self) -> int:
        """Array length in base pairs (``end - start + 1``)."""
        ...

    def __repr__(self) -> str: ...

def find_crispr_arrays(
    fasta_content: str,
    *,
    min_repeat_length: int = 23,
    max_repeat_length: int = 55,
    min_spacer_length: int = 25,
    max_spacer_length: int = 60,
    no_mismatch: bool = False,
    min_spacer_count: int = 1,
    min_evidence_level: int = 1,
) -> list[CrisprArray]:
    """Find CRISPR arrays in a FASTA-formatted string.

    Parameters
    ----------
    fasta_content:
        Multi-sequence FASTA text to scan for CRISPR arrays.
    min_repeat_length:
        Minimum direct repeat length in bp. Default: 23.
    max_repeat_length:
        Maximum direct repeat length in bp. Default: 55.
    min_spacer_length:
        Minimum spacer length in bp. Default: 25.
    max_spacer_length:
        Maximum spacer length in bp. Default: 60.
    no_mismatch:
        If ``True``, require repeats to be identical (zero mismatches).
        Default: ``False``.
    min_spacer_count:
        Minimum number of spacers an array must contain. Default: 1.
    min_evidence_level:
        Discard arrays with an evidence level below this value (1–4).
        Default: 1.

    Returns
    -------
    list[CrisprArray]
        All CRISPR arrays found, in sequence / position order.

    Raises
    ------
    ValueError
        If the FASTA content is malformed or detection fails.

    Examples
    --------
    >>> arrays = find_crispr_arrays(open("genome.fasta").read())
    >>> for array in arrays:
    ...     print(array.seq_id, array.start, array.end, array.evidence_level)
    """
    ...

def find_crispr_arrays_in_file(
    path: str,
    *,
    min_repeat_length: int = 23,
    max_repeat_length: int = 55,
    min_spacer_length: int = 25,
    max_spacer_length: int = 60,
    no_mismatch: bool = False,
    min_spacer_count: int = 1,
    min_evidence_level: int = 1,
) -> list[CrisprArray]:
    """Find CRISPR arrays in a FASTA file.

    A convenience wrapper around :func:`find_crispr_arrays` that reads
    the file for you.

    Parameters
    ----------
    path:
        Path to the input FASTA file.
    min_repeat_length:
        Minimum direct repeat length in bp. Default: 23.
    max_repeat_length:
        Maximum direct repeat length in bp. Default: 55.
    min_spacer_length:
        Minimum spacer length in bp. Default: 25.
    max_spacer_length:
        Maximum spacer length in bp. Default: 60.
    no_mismatch:
        If ``True``, require repeats to be identical. Default: ``False``.
    min_spacer_count:
        Minimum spacer count per array. Default: 1.
    min_evidence_level:
        Minimum evidence level (1–4). Default: 1.

    Returns
    -------
    list[CrisprArray]
        All detected CRISPR arrays.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IOError
        If the file cannot be read.
    ValueError
        If the file is not valid FASTA or detection fails.

    Examples
    --------
    >>> arrays = find_crispr_arrays_in_file("genome.fasta")
    >>> print(f"Found {len(arrays)} CRISPR arrays")
    """
    ...
