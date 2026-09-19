#!/usr/bin/env python3
"""Verify the public crates together in a fresh staging registry, then keep the archives."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

from check_release_archives import check_archive

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    # Cargo caches unpacked registry sources by registry URL/name/version. Reusing
    # target/package/tmp-registry for another unpublished build of the same
    # version can resolve the previous core sources while checking the new CLI.
    # A fresh target directory gives each verification a distinct staging URL.
    with tempfile.TemporaryDirectory(prefix="crispr-package-") as directory:
        command = ["cargo", "package", "--locked", "--target-dir", directory,
                   "-p", "crispr-cas-finder-core", "-p", "crispr-cas-finder-cli"]
        if args.allow_dirty:
            command.append("--allow-dirty")
        subprocess.run(command, cwd=ROOT, check=True)
        archives = list((Path(directory) / "package").glob("*.crate"))
        assert len(archives) == 2, "Expected the core and CLI package archives"
        output = ROOT / "target/package"
        output.mkdir(parents=True, exist_ok=True)
        for archive in archives:
            check_archive(archive)
            shutil.copy2(archive, output / archive.name)


if __name__ == "__main__":
    main()
