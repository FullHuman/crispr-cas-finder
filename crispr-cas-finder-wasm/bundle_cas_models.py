#!/usr/bin/env python3
"""Bundle the repository's CAS profiles and definitions for the browser app."""
import argparse
import gzip
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAS_DIR = SCRIPT_DIR.parent / "crispr-cas-finder-cli/data/CasFinder-2.0.3"


def build_bundle(cas_dir: Path) -> dict:
    models = [
        {"name": path.stem, "family": "CASFinder", "content": path.read_text(encoding="utf-8")}
        for path in sorted((cas_dir / "DEF-SubTyping-2.0.3").glob("*.xml"))
    ]
    profiles = [
        {"name": path.stem, "data": path.read_text(encoding="utf-8")}
        for path in sorted((cas_dir / "CASprofiles-2.0.3").glob("*.hmm"))
    ]
    if not models or not profiles:
        raise ValueError(f"No CAS models or HMM profiles found under {cas_dir}")
    return {"models": models, "profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cas-dir", type=Path, default=DEFAULT_CAS_DIR)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "www/cas-models.json")
    args = parser.parse_args()
    try:
        bundle = build_bundle(args.cas_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    # Validate all sources before touching an existing bundle.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(bundle).encode("utf-8")
    args.output.write_bytes(encoded)
    args.output.with_suffix(args.output.suffix + ".gz").write_bytes(gzip.compress(encoded, mtime=0))
    print(f"Created {args.output}: {len(bundle['models'])} models, "
          f"{len(bundle['profiles'])} profiles, {args.output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
