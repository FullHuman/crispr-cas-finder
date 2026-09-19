#!/usr/bin/env python3
"""Build an sdist outside the checkout and test its wheel in a fresh environment."""
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from check_release_archives import check_archive
from test_python_wheel import install_and_test

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="crispr-python-dist-") as directory:
        temporary = Path(directory)
        distributions = temporary / "dist"
        subprocess.run(["maturin", "sdist", "--manifest-path", str(ROOT / "crispr-cas-finder-python/Cargo.toml"),
                        "--out", str(distributions)], check=True, cwd=ROOT)
        sdist, = distributions.glob("*.tar.gz")
        check_archive(sdist)
        with tarfile.open(sdist) as archive:
            archive.extractall(temporary / "source", filter="data")
        source, = (temporary / "source").iterdir()
        # The archive itself must select nightly without a CI/developer override.
        environment = os.environ.copy()
        environment.pop("RUSTUP_TOOLCHAIN", None)
        subprocess.run(["maturin", "build", "--locked", "--release", "--out", str(distributions),
                        "--interpreter", sys.executable], cwd=source, env=environment, check=True)
        install_and_test(distributions)
    print("PASS: Python source archive, toolchain pin, wheel metadata, and installed API")


if __name__ == "__main__":
    main()
