#!/usr/bin/env python3
"""Recover and verify the complete table bundle embedded in the executed notebook."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zlib


def find_bundle(notebook: dict) -> dict:
    matches = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            metadata = output.get("metadata", {})
            bundle = metadata.get("aistats_probe_transfer_v3")
            if bundle is not None:
                matches.append(bundle)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one embedded result bundle; found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Executed research notebook")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/recovered_tables"),
        help="Directory for recovered CSV files (default: results/recovered_tables)",
    )
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    bundle = find_bundle(notebook)
    payload = zlib.decompress(base64.b64decode(bundle["tables_zlib_base64"], validate=True))
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != bundle.get("tables_sha256"):
        raise ValueError(
            "Result bundle checksum mismatch: "
            f"expected {bundle.get('tables_sha256')}, got {actual_digest}"
        )

    tables = json.loads(payload)
    expected_rows = bundle.get("table_rows", {})
    if set(tables) != set(expected_rows):
        raise ValueError("Embedded table names do not match the recorded row-count manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for name, csv_text in sorted(tables.items()):
        with io.StringIO(csv_text, newline="") as stream:
            row_count = max(0, sum(1 for _ in csv.reader(stream)) - 1)  # exclude header
        if row_count != int(expected_rows[name]):
            raise ValueError(
                f"Row-count mismatch for {name}: expected {expected_rows[name]}, got {row_count}"
            )
        target = args.output_dir / f"{name}.csv"
        target.write_text(csv_text, encoding="utf-8", newline="")
        total_rows += row_count

    print(f"Verified SHA-256: {actual_digest}")
    print(f"Recovered {len(tables)} tables and {total_rows:,} data rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zlib.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
