#!/usr/bin/env python3
"""Turn the CSV files written by bench.py and compare.py into a Markdown summary.

Timings are the median over repetitions. Ratios are original / Rust, so a value
above 1 means the Rust implementation was faster or used less memory.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def geometric_mean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def seconds(value):
    if value is None:
        return ""
    if value >= 120:
        return f"{value / 60:.1f} min"
    return f"{value:.1f} s" if value >= 10 else f"{value:.2f} s"


def ratio(a, b):
    return f"{a / b:.1f}x" if a and b else ""


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


class Timings:
    """Median wall time, CPU time and peak memory per (genome, configuration)."""

    def __init__(self, rows):
        self.wall = defaultdict(list)
        self.cpu = defaultdict(list)
        self.rss = defaultdict(list)
        self.meta = {}
        for row in rows:
            if row["exit_code"] != "0":
                continue
            key = (row["genome"], row["config"])
            self.wall[key].append(float(row["wall_s"]))
            self.cpu[key].append(float(row["user_s"]) + float(row["sys_s"]))
            self.rss[key].append(float(row["max_rss_mb"]))
            self.meta[row["genome"]] = (int(row["sequences"]), int(row["bp"]))

    def median(self, store, genome, config):
        values = store.get((genome, config))
        return statistics.median(values) if values else None


def timing_section(t, title, original, rust_one, rust_all, original_parallel=None):
    """Per-genome table for one pair of workloads, plus aggregate ratios."""
    genomes = [g for g in t.meta
               if t.median(t.wall, g, original) and t.median(t.wall, g, rust_all)]
    if not genomes:
        return ""
    genomes.sort(key=lambda g: (t.meta[g][0], g))

    header = ["genome", "sequences", "Mb", "original"]
    if original_parallel:
        header.append("original, all cores")
    header += ["Rust, 1 thread", "Rust, all cores", "speed-up", "original RSS", "Rust RSS"]
    body, speedups, single, parallel, memory = [], [], [], [], []
    for genome in genomes:
        o = t.median(t.wall, genome, original)
        r1 = t.median(t.wall, genome, rust_one)
        ra = t.median(t.wall, genome, rust_all)
        op = t.median(t.wall, genome, original_parallel) if original_parallel else None
        o_rss = t.median(t.rss, genome, original)
        r_rss = t.median(t.rss, genome, rust_all)
        speedups.append(o / ra)
        if r1:
            single.append(o / r1)
        if op:
            parallel.append(op / ra)
        memory.append(o_rss / r_rss)
        row = [genome, t.meta[genome][0], f"{t.meta[genome][1] / 1e6:.2f}", seconds(o)]
        if original_parallel:
            row.append(seconds(op))
        row += [seconds(r1), seconds(ra), ratio(o, ra), f"{o_rss:.0f} MB", f"{r_rss:.0f} MB"]
        body.append(row)

    lines = [f"## {title}", "", f"{len(genomes)} genomes. Geometric means of original / Rust:", ""]
    lines.append(f"- default settings, wall time: **{geometric_mean(speedups):.1f}x** "
                 f"(range {min(speedups):.1f}x to {max(speedups):.1f}x)")
    if single:
        lines.append(f"- Rust held to one thread: **{geometric_mean(single):.2f}x** "
                     f"(range {min(single):.2f}x to {max(single):.2f}x)")
    if parallel:
        lines.append(f"- both using every core: **{geometric_mean(parallel):.1f}x** "
                     f"(range {min(parallel):.1f}x to {max(parallel):.1f}x, "
                     f"{len(parallel)} genomes)")
    lines.append(f"- peak memory of the largest process: **{geometric_mean(memory):.1f}x**")
    total_o = sum(t.median(t.wall, g, original) for g in genomes)
    total_r = sum(t.median(t.wall, g, rust_all) for g in genomes)
    cpu_o = sum(t.median(t.cpu, g, original) for g in genomes)
    cpu_r = sum(t.median(t.cpu, g, rust_all) for g in genomes)
    lines.append(f"- whole set, one genome after another: {seconds(total_o)} against "
                 f"{seconds(total_r)} of wall time, {seconds(cpu_o)} against {seconds(cpu_r)} "
                 "of CPU time")
    lines += ["", table(header, body), ""]
    return "\n".join(lines)


def scaling_section(t):
    """Wall time against thread count for genomes timed at three or more counts."""
    lines = []
    for genome in sorted(t.meta):
        counts = sorted(int(config.removeprefix("rust-t")) for g, config in t.wall
                        if g == genome and config.startswith("rust-t"))
        if len(counts) < 3:
            continue
        base = t.median(t.wall, genome, f"rust-t{counts[0]}")
        body = []
        for n in counts:
            wall = t.median(t.wall, genome, f"rust-t{n}")
            body.append([n, seconds(wall), f"{base / wall:.1f}x",
                         seconds(t.median(t.cpu, genome, f"rust-t{n}"))])
        lines += [f"### {genome}", "",
                  table(["threads", "wall time", "against 1 thread", "CPU time"], body), ""]
    return "\n".join(["## Rust thread scaling (arrays and Cas systems)", "", *lines]) if lines else ""


def spread_section(t):
    """Largest run-to-run spread among measurements repeated three times or more."""
    spreads = [(max(v) - min(v)) / statistics.median(v) for v in t.wall.values()
               if len(v) >= 3 and statistics.median(v) >= 1]
    if not spreads:
        return ""
    return (f"## Repeatability\n\n{len(spreads)} measurements of one second or more were "
            f"repeated three times or more. The spread between the fastest and slowest "
            f"repetition is {100 * statistics.median(spreads):.1f}% of the median at the "
            f"median and {100 * max(spreads):.1f}% at worst.\n")


def browser_section(rows):
    if not rows:
        return ""
    runs = defaultdict(list)
    for row in rows:
        runs[(row["browser"], row["cas"], row["worker"])].append(float(row["seconds"]))
    body = [[browser, "arrays and Cas systems" if cas == "true" else "arrays only",
             "all" if worker == "threaded" else "1",
             seconds(statistics.median(values)), len(values)]
            for (browser, cas, worker), values in sorted(runs.items())]
    return "\n".join(["## Browser build and native build (E. coli K-12, bundled CasFinder 2.0.3)", "",
                      "The browser rows are the WebAssembly build in headless Chromium, timed "
                      "from worker start to result. The native rows are the CLI with its "
                      "embedded models, timed from start to exit.", "",
                      table(["build", "analysis", "threads", "median", "runs"], body), ""])


def array_section(rows):
    if not rows:
        return ""
    lines = ["## CRISPR arrays", ""]
    body = []
    for level in ("1", "2", "3", "4"):
        up = [r for r in rows if r["upstream_level"] == level]
        both = [r for r in up if r["status"] == "both"]
        same_coordinates = [r for r in both if r["same_coordinates"] == "True"]
        near = [r for r in both if int(r["boundary_shift"]) <= 5]
        same_spacers = [r for r in both
                        if r["upstream_spacers"] == r["rust_spacers"] == r["overlapping_spacers"]]
        same_level = [r for r in both if r["rust_level"] == level]
        rust_only = [r for r in rows if r["status"] == "rust_only" and r["rust_level"] == level]
        body.append([
            level, len(up), len(both),
            f"{100 * len(both) / len(up):.0f}%" if up else "",
            len(same_coordinates), len(near), len(same_spacers), len(same_level), len(rust_only),
        ])
    lines.append(table(
        ["evidence level (original)", "arrays in original", "also in Rust", "recovered",
         "same coordinates", "within 5 bp", "same spacer set", "same level",
         "Rust only (its level)"],
        body,
    ))
    up_spacers = sum(int(r["upstream_spacers"]) for r in rows if r["upstream_spacers"])
    identical = sum(int(r["identical_spacers"]) for r in rows if r["identical_spacers"])
    overlapping = sum(int(r["overlapping_spacers"]) for r in rows if r["overlapping_spacers"])
    rust_spacers = sum(int(r["rust_spacers"]) for r in rows if r["rust_spacers"])
    lines += ["", f"Spacers: {up_spacers} in the original output, {rust_spacers} in the Rust "
                  f"output. Of the original's, {overlapping} "
                  f"({100 * overlapping / up_spacers:.1f}%) are reported by Rust at the same "
                  f"place, {identical} ({100 * identical / up_spacers:.1f}%) with the identical "
                  "sequence; the rest of the shared ones differ by a base or two at an edge "
                  "because the repeat boundary is placed differently.", ""]
    return "\n".join(lines)


def cas_section(rows, genomes):
    if not rows:
        return ""
    both = [r for r in rows if r["status"] == "both"]
    only_up = [r for r in rows if r["status"] == "upstream_only"]
    only_rs = [r for r in rows if r["status"] == "rust_only"]
    dropped = [r for r in only_rs if r["dropped_by_upstream"] == "True"]
    short = [r for r in only_rs if r["dropped_by_upstream"] != "True"
             and r["seq_length"] and int(r["seq_length"]) < 100_000]
    other = len(only_rs) - len(dropped) - len(short)
    lines = ["## Cas systems", ""]
    lines.append(table(
        ["", "count"],
        [
            ["in both", len(both)],
            ["... same subtype", sum(r["same_type"] == "True" for r in both)],
            ["... same genes at the same coordinates",
             sum(r["same_gene_coordinates"] == "True" for r in both)],
            ["... and every gene matched to the same profile",
             sum(r["same_genes"] == "True" for r in both)],
            ["original only", len(only_up)],
            ["Rust only", len(only_rs)],
            ["... found by MacSyFinder inside the original, then lost by its Perl script",
             len(dropped)],
            ["... on sequences under 100 kb, where the original reports none", len(short)],
            ["... other", other],
        ],
    ))
    invalid = [g["genome"] for g in genomes if g["upstream_json_valid"] == "False"]
    lines += ["", f"The original wrote a result.json that is not valid JSON for {len(invalid)} "
                  f"of {len(genomes)} genomes."]
    types = defaultdict(int)
    for r in both:
        types[r["upstream_type"]] += 1
    lines += ["", "Subtypes in both: " + ", ".join(f"{t} ({n})" for t, n in sorted(types.items())), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("timing", type=Path, help="results directory of the sequential timing runs")
    parser.add_argument("--outputs", type=Path,
                        help="results directory compare.py was run on (default: the timing one)")
    parser.add_argument("--out", type=Path, help="file to write (default: summary.md in timing)")
    args = parser.parse_args()
    outputs = args.outputs or args.timing

    t = Timings(read(args.timing / "timings.csv"))
    sections = [
        timing_section(t, "Arrays and Cas systems", "upstream", "rust-t1", "rust-all",
                       "upstream-cpuM"),
        timing_section(t, "Arrays only", "upstream-arrays", "rust-arrays-t1", "rust-arrays"),
        scaling_section(t),
        spread_section(t),
        browser_section(read(args.timing / "browser_timings.csv")),
        array_section(read(outputs / "arrays.csv")),
        cas_section(read(outputs / "cas_systems.csv"), read(outputs / "genomes.csv")),
    ]
    text = "\n".join(s for s in sections if s)
    (args.out or args.timing / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
