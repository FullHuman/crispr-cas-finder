#!/usr/bin/env python3
"""Download the reference genomes listed in panel.tsv from NCBI as FASTA.

One file per row, holding every accession of the row. Existing files are kept.
"""

import argparse
import csv
import time
import urllib.parse
import urllib.request
from pathlib import Path

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch(accessions):
    query = urllib.parse.urlencode({
        "db": "nuccore", "id": accessions, "rettype": "fasta", "retmode": "text",
    })
    with urllib.request.urlopen(f"{EFETCH}?{query}", timeout=300) as response:
        return response.read().decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--panel", type=Path, default=Path(__file__).with_name("panel.tsv"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with open(args.panel, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            target = args.outdir / f"{row['name']}.fasta"
            if target.exists():
                continue
            text = fetch(row["accessions"])
            wanted = row["accessions"].count(",") + 1
            headers = [line for line in text.splitlines() if line.startswith(">")]
            if len(headers) != wanted:
                raise SystemExit(f"{row['name']}: expected {wanted} sequences, got {len(headers)}")
            target.write_text(text)
            for header in headers:
                print(f"{row['name']}\t{header[1:]}", flush=True)
            time.sleep(0.5)  # NCBI allows three requests per second without a key


if __name__ == "__main__":
    main()
