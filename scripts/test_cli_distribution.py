#!/usr/bin/env python3
"""Run a relocated CLI executable with no adjacent model files."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="Executable, or a release .tar.gz/.zip archive")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="crispr-cli-install-") as directory:
        directory = Path(directory)
        if args.binary.name.endswith((".tar.gz", ".zip")):
            from check_release_archives import check_archive
            check_archive(args.binary)
            unpacked = directory / "unpacked"
            if args.binary.suffix == ".zip":
                with zipfile.ZipFile(args.binary) as archive:
                    archive.extractall(unpacked)
            else:
                with tarfile.open(args.binary) as archive:
                    archive.extractall(unpacked, filter="data")
            candidates = [p for p in unpacked.rglob("crispr-cas-finder*")
                          if p.is_file() and p.name in ("crispr-cas-finder", "crispr-cas-finder.exe")]
            binary, = candidates
        else:
            binary = directory / args.binary.name
            shutil.copy2(args.binary, binary)
        shutil.copy2(ROOT / "crispr-cas-finder-cli/data/ecoli.fasta", directory / "ecoli.fasta")
        subprocess.run([str(binary), "--in", "ecoli.fasta", "--cas", "--workers", "2", "--outdir", "results"],
                       cwd=directory, check=True, timeout=900)
        report_path = directory / "results/report.json"
        report = json.loads(report_path.read_text())
        assert len(report["crisprs"]) == 5, "E. coli CRISPR regression"
        assert len(report["cas_clusters"]) == 1, "E. coli Cas regression"
        if args.report:
            shutil.copy2(report_path, args.report)
    print("PASS: relocated CLI with embedded models: five arrays, one Cas system")


if __name__ == "__main__":
    main()
