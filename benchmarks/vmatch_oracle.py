#!/usr/bin/env python3
"""Compare the repeats found by this project with those Vmatch reports.

For every sequence of each FASTA file, runs mkvtree and vmatch with the
arguments CRISPRCasFinder uses and compares the repeats, as (left length,
left start, right length, right start), with the output of the
`repeat_matches` example. Needs the tools set up for bench.py.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "vendor/upstream-native"
VMATCH = ["-l", "23", "25", "60", "-e", "1", "-s", "leftseq", "-evalue", "1", "-absolute",
          "-nodist", "-noevalue", "-noscore", "-noidentity", "-sort", "ia", "-best", "1000000",
          "-selfun", str(NATIVE / "sel392v2.so"), "55"]


def sequences(path):
    name, chunks = None, []
    for line in open(path):
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(chunks)
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def vmatch_repeats(sequence, work):
    fasta = work / "in.fasta"
    fasta.write_text(f">s\n{sequence}\n")
    env = {"PATH": f"{NATIVE / 'bin'}:/usr/bin:/bin", "LC_ALL": "C"}
    subprocess.run(["mkvtree", "-db", "in.fasta", "-dna", "-pl", "-lcp", "-suf", "-tis", "-ois",
                    "-bwt", "-bck", "-sti1"], cwd=work, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    output = subprocess.run(["vmatch", *VMATCH, "in.fasta"], cwd=work, env=env, check=True,
                            capture_output=True, text=True).stdout
    repeats = set()
    for line in output.splitlines():
        if line.startswith(">"):
            fields = line[1:].split()
            repeats.add((int(fields[0]), int(fields[1]), int(fields[3]), int(fields[4])))
    return repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("genomes", nargs="+", type=Path)
    parser.add_argument("--example", type=Path,
                        default=ROOT / "target/release/examples/repeat_matches")
    parser.add_argument("--show", type=int, default=0, help="differences to print per genome")
    args = parser.parse_args()

    totals = [0, 0, 0]
    for genome in args.genomes:
        ours = {}
        output = subprocess.run([str(args.example), str(genome)], check=True,
                                capture_output=True, text=True).stdout
        for line in output.splitlines():
            number, *fields = map(int, line.split("\t"))
            ours.setdefault(number, set()).add(tuple(fields))
        both = only_vmatch = only_ours = 0
        shown = 0
        for number, (name, sequence) in enumerate(sequences(genome)):
            if len(sequence) < 2:
                continue
            with tempfile.TemporaryDirectory() as work:
                theirs = vmatch_repeats(sequence, Path(work))
            mine = ours.get(number, set())
            both += len(theirs & mine)
            only_vmatch += len(theirs - mine)
            only_ours += len(mine - theirs)
            for tag, repeats in (("vmatch only", theirs - mine), ("ours only", mine - theirs)):
                for repeat in sorted(repeats, key=lambda r: r[1]):
                    if shown < args.show:
                        print(f"  {name} {tag}: {repeat}")
                        shown += 1
        print(f"{genome.stem}\tboth {both}\tvmatch only {only_vmatch}\tours only {only_ours}",
              flush=True)
        for index, count in enumerate((both, only_vmatch, only_ours)):
            totals[index] += count
    print(f"total\tboth {totals[0]}\tvmatch only {totals[1]}\tours only {totals[2]}")
    return 0 if totals[1] == totals[2] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
