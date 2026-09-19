#!/usr/bin/env python3
"""Copy canonical release notices/toolchain into each independently shipped package."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    destinations = [ROOT / name for name in (
        "crispr-cas-finder-core", "crispr-cas-finder-cli",
        "crispr-cas-finder-python", "crispr-cas-finder-wasm/www",
    )]
    copies = [(ROOT / name, directory / name)
              for directory in destinations
              for name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "LIMITATIONS.md")]
    copies.append((ROOT / "rust-toolchain.toml", ROOT / "crispr-cas-finder-python/rust-toolchain.toml"))
    stale = []
    for source, target in copies:
        data = source.read_bytes()
        if not target.exists() or target.read_bytes() != data:
            if args.check:
                stale.append(str(target.relative_to(ROOT)))
            else:
                target.write_bytes(data)
    if stale:
        parser.exit(1, "Run python3 scripts/sync_release_metadata.py; stale files:\n" + "\n".join(stale) + "\n")
    print("Release metadata is synchronized")


if __name__ == "__main__":
    main()
