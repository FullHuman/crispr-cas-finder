#!/usr/bin/env python3
"""Compare the arrays and Cas systems reported by both tools for each genome.

Reads the outputs kept by bench.py (result.json from CRISPRCasFinder,
report.json from crispr-cas-finder) and writes one row per array and per Cas
system to CSV, plus per-genome totals. Two arrays, or two Cas systems, are
paired when they lie on the same sequence and overlap by at least half of the
shorter one; each is paired at most once, largest overlap first.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENT)[::-1]


def load_upstream(path):
    """Read the outputs of CRISPRCasFinder for one genome.

    Arrays come from result.json; Cas systems from Cas_REPORT.tsv, because
    result.json loses every system of a sequence when the script loses one.
    Returns (arrays, systems, sequence lengths, valid, dropped). `valid` is
    False when result.json was not parseable as written. `dropped` counts, per
    sequence, the systems of MacSyFinder's best solution that the Perl script
    printed as an empty "### System:" header: it indexes systems 1..n, but
    MacSyFinder numbers them among all candidates, so a best solution holding
    systems 1 and 3 loses the second.
    """
    arrays, systems, lengths, names = [], [], {}, {}
    text = path.read_text()
    repaired = re.sub(r'"Cas":\[\s*\}', '"Cas":[]\n}', text)
    for sequence in json.loads(repaired)["Sequences"]:
        seq_id = sequence["Version"]
        names[sequence["Id"]] = seq_id
        lengths[seq_id] = sequence["Length"]
        for array in sequence["Crisprs"]:
            arrays.append({
                "seq": seq_id, "start": array["Start"], "end": array["End"],
                "level": array["Evidence_Level"], "repeat": array["DR_Consensus"].upper(),
                "spacers": [(r["Start"], r["End"], r["Sequence"].upper())
                            for r in array["Regions"] if r["Type"] == "Spacer"],
            })

    dropped = defaultdict(int)
    report = path.with_name("Cas_REPORT.tsv")
    current = system = None
    for line in report.read_text().splitlines() if report.exists() else []:
        header = re.match(r"(\S+) \( .* \)\s*$", line)
        fields = line.split("\t")
        if header:
            current = names.get(header.group(1), header.group(1))
        elif line.startswith("### System:"):
            name = line.removeprefix("### System:").strip()
            if name:
                kind = re.sub(r"^.*?_CAS_", "", name)
                system = {"seq": current, "type": re.sub(r"_\d+$", "", kind), "genes": []}
                systems.append(system)
            else:
                system = None
                dropped[current] += 1
        elif system is not None and len(fields) >= 8 and fields[5].isdigit():
            system["genes"].append((int(fields[5]), int(fields[6]), fields[1]))
    for system in systems:
        system["genes"].sort()
        system["start"] = min(gene[0] for gene in system["genes"])
        system["end"] = max(gene[1] for gene in system["genes"])
    return arrays, systems, lengths, repaired == text, dropped


def load_rust(path):
    """Return (arrays, systems, sequence lengths) from a crispr-cas-finder report.json."""
    report = json.loads(path.read_text())
    lengths = {s["id"]: s["length"] for s in report["sequences"]}
    arrays = [{
        "seq": a["seq_id"], "start": a["start"], "end": a["end"],
        "level": a["evidence_level"], "repeat": a["consensus_repeat"].upper(),
        "spacers": [(s["start"], s["end"], s["sequence"].upper()) for s in a["spacers"]],
    } for a in report["crisprs"]]
    systems = [{
        "seq": c["seq_id"], "start": c["start"], "end": c["end"],
        "type": re.sub(r"^.*?/CAS_", "", c["system"]),
        "genes": sorted((g["start"], g["end"], g["profile"]) for g in c["genes"]),
    } for c in report.get("cas_clusters", [])]
    return arrays, systems, lengths


def pair(left, right):
    """Pair intervals one to one by overlap; return (pairs, unpaired left, unpaired right)."""
    candidates = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if a["seq"] != b["seq"]:
                continue
            overlap = min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1
            shorter = min(a["end"] - a["start"], b["end"] - b["start"]) + 1
            if overlap * 2 >= shorter:
                candidates.append((overlap, i, j))
    used_left, used_right, pairs = set(), set(), []
    for _, i, j in sorted(candidates, reverse=True):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            pairs.append((left[i], right[j]))
    return (
        pairs,
        [a for i, a in enumerate(left) if i not in used_left],
        [b for j, b in enumerate(right) if j not in used_right],
    )


def identical_spacers(a, b):
    """Number of spacers of `a` whose sequence is in `b` on either strand."""
    pool = {s for _, _, s in b["spacers"]} | {reverse_complement(s) for _, _, s in b["spacers"]}
    return sum(1 for _, _, s in a["spacers"] if s in pool)


def overlapping_spacers(a, b):
    """Number of spacers of `a` covered at 80% or more by a spacer of `b`.

    The two tools can place a repeat boundary a base or two apart, which moves
    every spacer edge of the array without changing which spacers it holds.
    """
    count = 0
    for start, end, _ in a["spacers"]:
        for other_start, other_end, _ in b["spacers"]:
            overlap = min(end, other_end) - max(start, other_start) + 1
            if overlap * 5 >= 4 * max(end - start + 1, other_end - other_start + 1):
                count += 1
                break
    return count


def same_repeat(a, b):
    return a["repeat"] == b["repeat"] or a["repeat"] == reverse_complement(b["repeat"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", type=Path, help="results directory written by bench.py")
    parser.add_argument("--upstream", default="upstream", help="configuration holding result.json")
    parser.add_argument("--rust", default="rust-all", help="configuration holding report.json")
    args = parser.parse_args()

    array_rows, system_rows, genome_rows = [], [], []
    for genome_dir in sorted((args.results / "raw").iterdir()):
        upstream_json = genome_dir / args.upstream / "result.json"
        rust_json = genome_dir / args.rust / "report.json"
        if not (upstream_json.exists() and rust_json.exists()):
            continue
        genome = genome_dir.name
        up_arrays, up_systems, lengths, valid_json, dropped = load_upstream(upstream_json)
        rs_arrays, rs_systems, rs_lengths = load_rust(rust_json)
        lengths = {**rs_lengths, **lengths}

        pairs, only_up, only_rs = pair(up_arrays, rs_arrays)
        for up, rs in pairs:
            array_rows.append({
                "genome": genome, "seq": up["seq"], "status": "both",
                "seq_length": lengths.get(up["seq"], ""),
                "upstream_start": up["start"], "upstream_end": up["end"],
                "rust_start": rs["start"], "rust_end": rs["end"],
                "upstream_level": up["level"], "rust_level": rs["level"],
                "upstream_spacers": len(up["spacers"]), "rust_spacers": len(rs["spacers"]),
                "identical_spacers": identical_spacers(up, rs),
                "overlapping_spacers": overlapping_spacers(up, rs),
                "same_coordinates": up["start"] == rs["start"] and up["end"] == rs["end"],
                "boundary_shift": max(abs(up["start"] - rs["start"]), abs(up["end"] - rs["end"])),
                "same_repeat": same_repeat(up, rs),
            })
        for status, group in (("upstream_only", only_up), ("rust_only", only_rs)):
            side = "upstream" if status == "upstream_only" else "rust"
            for array in group:
                array_rows.append({
                    "genome": genome, "seq": array["seq"], "status": status,
                    "seq_length": lengths.get(array["seq"], ""),
                    f"{side}_start": array["start"], f"{side}_end": array["end"],
                    f"{side}_level": array["level"], f"{side}_spacers": len(array["spacers"]),
                })

        system_pairs, systems_only_up, systems_only_rs = pair(up_systems, rs_systems)
        for up, rs in system_pairs:
            system_rows.append({
                "genome": genome, "seq": up["seq"], "status": "both",
                "seq_length": lengths.get(up["seq"], ""),
                "upstream_type": up["type"], "rust_type": rs["type"],
                "upstream_start": up["start"], "upstream_end": up["end"],
                "rust_start": rs["start"], "rust_end": rs["end"],
                "upstream_genes": len(up["genes"]), "rust_genes": len(rs["genes"]),
                "same_type": up["type"] == rs["type"],
                "same_gene_coordinates": [g[:2] for g in up["genes"]] == [g[:2] for g in rs["genes"]],
                "same_genes": up["genes"] == rs["genes"],
            })
        for status, group in (("upstream_only", systems_only_up), ("rust_only", systems_only_rs)):
            side = "upstream" if status == "upstream_only" else "rust"
            for system in group:
                system_rows.append({
                    "genome": genome, "seq": system["seq"], "status": status,
                    "seq_length": lengths.get(system["seq"], ""),
                    f"{side}_type": system["type"], f"{side}_start": system["start"],
                    f"{side}_end": system["end"], f"{side}_genes": len(system["genes"]),
                    "dropped_by_upstream": status == "rust_only" and dropped[system["seq"]] > 0,
                })

        levels = defaultdict(int)
        for array in up_arrays:
            levels[f"upstream_level{array['level']}"] += 1
        for array in rs_arrays:
            levels[f"rust_level{array['level']}"] += 1
        genome_rows.append({
            "genome": genome,
            "upstream_arrays": len(up_arrays), "rust_arrays": len(rs_arrays),
            "arrays_both": len(pairs), "arrays_upstream_only": len(only_up),
            "arrays_rust_only": len(only_rs),
            **{f"{tool}_level{n}": levels[f"{tool}_level{n}"]
               for tool in ("upstream", "rust") for n in (1, 2, 3, 4)},
            "upstream_cas": len(up_systems), "rust_cas": len(rs_systems),
            "cas_both": len(system_pairs), "cas_upstream_only": len(systems_only_up),
            "cas_rust_only": len(systems_only_rs),
            "upstream_dropped_cas": sum(dropped.values()),
            "upstream_json_valid": valid_json,
        })

    def write(name, rows, fields):
        with open(args.results / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

    write("arrays.csv", array_rows, [
        "genome", "seq", "seq_length", "status", "upstream_start", "upstream_end",
        "rust_start", "rust_end", "upstream_level", "rust_level", "upstream_spacers",
        "rust_spacers", "identical_spacers", "overlapping_spacers", "same_coordinates",
        "boundary_shift", "same_repeat",
    ])
    write("cas_systems.csv", system_rows, [
        "genome", "seq", "seq_length", "status", "upstream_type", "rust_type",
        "upstream_start", "upstream_end", "rust_start", "rust_end",
        "upstream_genes", "rust_genes", "same_type", "same_gene_coordinates", "same_genes",
        "dropped_by_upstream",
    ])
    if genome_rows:
        write("genomes.csv", genome_rows, list(genome_rows[0]))
    print(f"{len(genome_rows)} genomes, {len(array_rows)} array rows, {len(system_rows)} Cas rows")


if __name__ == "__main__":
    main()
