#!/usr/bin/env python3
"""Install and test the wheel for the current interpreter in a clean environment."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv

from check_release_archives import check_archive

ROOT = Path(__file__).resolve().parents[1]


def install_and_test(distributions):
    wheels = list(distributions.glob("*.whl"))
    assert wheels, f"No wheels in {distributions}"
    for wheel in wheels:
        check_archive(wheel)
    with tempfile.TemporaryDirectory(prefix="crispr-wheel-test-") as directory:
        temporary = Path(directory)
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(temporary / "venv")
        python = temporary / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--only-binary=:all:",
                        "--find-links", str(distributions.resolve()), "crispr-cas-finder"], check=True)
        shutil.copytree(ROOT / "crispr-cas-finder-python/tests", temporary / "tests")
        subprocess.run([str(python), "-m", "unittest", "discover", "-s", "tests", "-v"],
                       cwd=temporary, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", type=Path)
    install_and_test(parser.parse_args().distributions)
