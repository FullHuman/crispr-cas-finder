"""
crispr_cas_finder
~~~~~~~~~~~~~~~~~

Python bindings for the CRISPR-Cas detection engine written in Rust.

Typical usage::

    from crispr_cas_finder import find_crispr_arrays, find_crispr_arrays_in_file

    # Scan a FASTA string directly
    with open("genome.fasta") as fh:
        arrays = find_crispr_arrays(fh.read())

    # Or use the convenience file helper
    arrays = find_crispr_arrays_in_file("genome.fasta")

    for array in arrays:
        print(f"{array.seq_id}  {array.start}–{array.end}  "
              f"({array.spacer_count} spacers, evidence level {array.evidence_level})")
        for spacer in array.spacers:
            print(f"  spacer: {spacer.sequence}")

    # Full report, optionally with Cas systems (same structure as report.json)
    report = analyze(fasta_text, cas=True,
                     cas_models_dir="CasFinder-2.0.3/DEF-SubTyping-2.0.3",
                     cas_profiles_dir="CasFinder-2.0.3/CASprofiles-2.0.3")
    print(report["summary"])
"""

from .crispr_cas_finder import (
    CrisprArray,
    Repeat,
    Spacer,
    __version__,
    analyze,
    find_crispr_arrays,
    find_crispr_arrays_in_file,
)

__all__ = [
    "__version__",
    "CrisprArray",
    "Repeat",
    "Spacer",
    "analyze",
    "find_crispr_arrays",
    "find_crispr_arrays_in_file",
]
