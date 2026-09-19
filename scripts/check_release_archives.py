#!/usr/bin/env python3
"""Check that release archives actually contain their license and notice texts."""
import argparse
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check_archive(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            contents = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    else:
        with tarfile.open(path) as archive:
            contents = {member.name: archive.extractfile(member).read()
                        for member in archive.getmembers() if member.isfile()}
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        expected = (ROOT / name).read_bytes()
        assert any(Path(member).name == name and body == expected for member, body in contents.items()), \
            f"{path}: missing or stale {name}"
    if path.name.endswith(".tar.gz") and any(Path(member).name == "pyproject.toml" for member in contents):
        # pip invokes the build backend at the source-distribution root.
        prefix = next(iter(contents)).split("/")[0]
        assert contents.get(prefix + "/rust-toolchain.toml") == (ROOT / "rust-toolchain.toml").read_bytes(), \
            f"{path}: missing root toolchain pin"
    print(f"PASS: {path.name}: release metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    for path in parser.parse_args().archives:
        check_archive(path)
