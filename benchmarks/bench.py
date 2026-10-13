#!/usr/bin/env python3
"""Time crispr-cas-finder against the original CRISPRCasFinder on a set of genomes.

Each (genome, configuration, repetition) is one process tree, measured with
wait4: wall time, user and system CPU time of the tree, and the largest
resident set size of any single process in it. Rows are appended to timings.csv
so an interrupted run can be resumed. The outputs of the first repetition are
kept for compare.py.

Both tools read the same CasFinder 3.1.0 models. See README.md for how the
original pipeline was set up.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "vendor/CRISPRCasFinder"
NATIVE = ROOT / "vendor/upstream-native"
MODELS = ROOT / "vendor/casfinder3/CasFinder"
RUST = Path(os.environ.get("CCF_BIN", ROOT / "target/release/crispr-cas-finder"))
CONDA_ENV = Path(os.environ.get("CCF_CONDA_ENV", Path.home() / "miniconda3/envs/ccf_orig"))
NCPU = os.cpu_count() or 1

FIELDS = [
    "genome", "config", "rep", "wall_s", "user_s", "sys_s", "max_rss_mb",
    "exit_code", "sequences", "bp",
]


def upstream_command(fasta, extra):
    return [
        "perl", str(UPSTREAM / "CRISPRCasFinder.pl"),
        "-in", fasta.name,
        "-so", str(NATIVE / "sel392v2.so"),
        "-drpt", str(UPSTREAM / "supplementary_files/repeatDirection.tsv"),
        "-rpts", str(UPSTREAM / "supplementary_files/Repeat_List.csv"),
        "-out", "res",
        *extra,
    ]


def rust_command(fasta, cas, threads):
    command = [str(RUST), "-i", fasta.name, "--outdir", "res", "-q"]
    if cas:
        command += ["--cas", "--cas-models-dir", str(MODELS), "--workers", str(threads)]
    return command


def plan(config):
    """Return (tool, argv builder, environment overrides) for a configuration name."""
    if config == "upstream":
        return "upstream", lambda f: upstream_command(f, ["-cas"]), {}
    if config == "upstream-cpuM":
        return "upstream", lambda f: upstream_command(f, ["-cas", "-cpuM", str(NCPU)]), {}
    if config == "upstream-arrays":
        return "upstream", lambda f: upstream_command(f, []), {}
    if config == "rust-all":
        return "rust", lambda f: rust_command(f, True, 0), {}
    if config == "rust-arrays":
        return "rust", lambda f: rust_command(f, False, 0), {}
    if config.startswith("rust-arrays-t"):
        threads = int(config.removeprefix("rust-arrays-t"))
        return "rust", lambda f: rust_command(f, False, threads), {"RAYON_NUM_THREADS": str(threads)}
    if config.startswith("rust-t"):
        # --workers only bounds the HMM search; the rayon pool does the rest.
        threads = int(config.removeprefix("rust-t"))
        return "rust", lambda f: rust_command(f, True, threads), {"RAYON_NUM_THREADS": str(threads)}
    sys.exit(f"unknown configuration: {config}")


def environment(tool, overrides):
    env = {"HOME": os.environ["HOME"], "LC_ALL": "C", "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
    if tool == "upstream":
        env["PATH"] = f"{NATIVE / 'bin'}:{CONDA_ENV / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin"
    else:
        env["PATH"] = "/usr/bin:/bin"
    env.update(overrides)
    return env


def timed(argv, cwd, env):
    with open(cwd / "stdout.log", "wb") as out, open(cwd / "stderr.log", "wb") as err:
        start = time.perf_counter()
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=out, stderr=err)
        _, status, usage = os.wait4(process.pid, 0)
        wall = time.perf_counter() - start
    process.returncode = os.waitstatus_to_exitcode(status)
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux.
    rss_mb = usage.ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    return wall, usage.ru_utime, usage.ru_stime, rss_mb, process.returncode


def fasta_size(path):
    sequences = bp = 0
    with open(path, "rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                sequences += 1
            else:
                bp += len(line.strip())
    return sequences, bp


def keep_outputs(tool, work, destination, failed):
    destination.mkdir(parents=True, exist_ok=True)
    result = work / "res"
    if tool == "upstream":
        wanted = [result / "result.json", *sorted((result / "TSV").glob("*.tsv"))]
    else:
        wanted = [result / "report.json", *sorted(result.glob("*.tsv"))]
    for path in wanted:
        if path.exists():
            shutil.copy(path, destination / path.name)
    if failed:
        # The original prints several megabytes per draft assembly; only keep
        # the logs when they are needed to see what went wrong.
        for log in ("stdout.log", "stderr.log"):
            shutil.copy(work / log, destination / log)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("genomes", nargs="+", type=Path, help="FASTA files")
    parser.add_argument("--configs", required=True, help="comma-separated configuration names")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="results directory")
    parser.add_argument("--work", type=Path, required=True, help="scratch directory")
    parser.add_argument("--jobs", type=int, default=1,
                        help="runs to execute at once; anything above 1 is for collecting "
                             "outputs only, the timings are then not comparable")
    args = parser.parse_args()

    configs = args.configs.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    # Runs made several at a time compete for cores; keep them out of timings.csv.
    timings = args.out / ("timings.csv" if args.jobs == 1 else "runs_parallel.csv")
    done = set()
    if timings.exists():
        with open(timings, newline="") as handle:
            done = {(row["genome"], row["config"], int(row["rep"])) for row in csv.DictReader(handle)}
    else:
        with open(timings, "w", newline="") as handle:
            csv.DictWriter(handle, FIELDS).writeheader()

    tasks = []
    for genome in args.genomes:
        sequences, bp = fasta_size(genome)
        for rep in range(1, args.reps + 1):
            for config in configs:
                if (genome.stem, config, rep) not in done:
                    tasks.append((genome, sequences, bp, rep, config))

    lock = threading.Lock()

    def run(task):
        genome, sequences, bp, rep, config = task
        name = genome.stem
        tool, build, overrides = plan(config)
        work = args.work / f"{name}.{config}.{rep}.{os.getpid()}"
        work.mkdir()
        fasta = work / genome.name
        shutil.copy(genome, fasta)
        wall, user, system, rss, code = timed(build(fasta), work, environment(tool, overrides))
        with lock:
            with open(timings, "a", newline="") as handle:
                csv.DictWriter(handle, FIELDS).writerow({
                    "genome": name, "config": config, "rep": rep,
                    "wall_s": f"{wall:.3f}", "user_s": f"{user:.3f}", "sys_s": f"{system:.3f}",
                    "max_rss_mb": f"{rss:.1f}", "exit_code": code,
                    "sequences": sequences, "bp": bp,
                })
            print(f"{name}\t{config}\trep {rep}\t{wall:.2f}s\t{rss:.0f} MB\texit {code}", flush=True)
        if rep == 1:
            keep_outputs(tool, work, args.out / "raw" / name / config, code != 0)
        shutil.rmtree(work)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, tasks))


if __name__ == "__main__":
    main()
