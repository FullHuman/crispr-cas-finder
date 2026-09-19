#!/usr/bin/env python3
"""Check workspace, Python manifest, and Python package version consistency."""
import argparse
from pathlib import Path
import tomllib


def check_versions(root: Path, expected: str | None = None) -> str:
    workspace = tomllib.loads((root / "Cargo.toml").read_text())["workspace"]
    version = expected or workspace["package"]["version"]
    manifests = [root / member / "Cargo.toml" for member in workspace["members"]]
    manifests.append(root / "crispr-cas-finder-python/Cargo.toml")
    versions = {"workspace": workspace["package"]["version"]}
    for path in manifests:
        package = tomllib.loads(path.read_text())["package"]
        value = package["version"]
        versions[package["name"]] = workspace["package"]["version"] if isinstance(value, dict) else value
    project = tomllib.loads((root / "crispr-cas-finder-python/pyproject.toml").read_text())["project"]
    versions["Python project"] = project["version"]
    mismatches = [f"{name}: {value} (expected {version})" for name, value in versions.items() if value != version]
    if mismatches:
        raise ValueError("Version mismatch:\n" + "\n".join(mismatches))
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected")
    args = parser.parse_args()
    try:
        print(check_versions(Path(__file__).resolve().parent.parent, args.expected))
    except ValueError as error:
        parser.exit(1, f"{error}\n")
