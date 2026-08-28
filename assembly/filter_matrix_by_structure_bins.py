from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Set, Tuple

def load_structure_starts(structure_path: str) -> Tuple[str, Set[int]]:
    starts: Set[int] = set()
    chrom = None
    with open(structure_path, "r", newline="") as f:
        header = f.readline()
        if not header:
            raise ValueError(f"{structure_path} was empty")
        f.seek(0)

        delimiter = "," if ("," in header and "\t" not in header) else "\t"
        r = csv.DictReader(f, delimiter=delimiter)

        fieldnames = [h.strip() for h in (r.fieldnames or [])]
        by_lower = {h.lower(): h for h in fieldnames}

        chr_col = by_lower.get("chr") or by_lower.get("chrom") or by_lower.get("chromosome")
        start_col = by_lower.get("start") or by_lower.get("start_bp")

        missing = []
        if chr_col is None:
            missing.append("chr/chrom/chromosome")
        if start_col is None:
            missing.append("start/start_bp")
        if missing:
            raise ValueError(f"{structure_path} missing columns: {missing}; got {fieldnames}")

        for row in r:
            if chrom is None:
                chrom = row[chr_col]
            starts.add(int(float(row[start_col])))
    if chrom is None:
        raise ValueError(f"{structure_path} had no rows")
    return chrom, starts

def is_finite_number(s: str) -> bool:
    try:
        v = float(s)
    except Exception:
        return False
    return math.isfinite(v)

def main() -> None:
    ap = argparse.ArgumentParser(description="Filter 3-col matrix by structure start bins")
    ap.add_argument("--matrix", required=True, help="Input 3-col matrix (bp1 bp2 IF)")
    ap.add_argument("--structure", required=True, help="Structure file with start bins")
    ap.add_argument(
        "--out",
        required=True,
        help="Output filtered matrix path (3-col, whitespace-delimited)",
    )
    ap.add_argument(
        "--stats",
        default=None,
        help="Optional stats output path (default: <out>.stats.txt)",
    )
    ap.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Process at most N input lines (0 means all). Useful for quick tests.",
    )

    args = ap.parse_args()

    chrom, starts = load_structure_starts(args.structure)

    in_lines = 0
    kept = 0
    dropped_nan = 0
    dropped_nonpos = 0
    dropped_not_in_struct = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    with open(args.matrix, "r") as fin, open(args.out, "w") as fout:
        for line in fin:
            in_lines += 1
            if args.max_lines and in_lines > args.max_lines:
                break

            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            a_s, b_s, if_s = parts[0], parts[1], parts[2]
            if not is_finite_number(if_s):
                dropped_nan += 1
                continue
            IF = float(if_s)
            if IF <= 0.0:
                dropped_nonpos += 1
                continue
            bp1 = int(float(a_s))
            bp2 = int(float(b_s))
            if bp1 not in starts or bp2 not in starts:
                dropped_not_in_struct += 1
                continue

            fout.write(f"{bp1}\t{bp2}\t{IF}\n")
            kept += 1

    stats_path = args.stats or f"{args.out}.stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"chrom\t{chrom}\n")
        f.write(f"structure_bins\t{len(starts)}\n")
        f.write(f"input_lines_seen\t{in_lines}\n")
        f.write(f"kept_rows\t{kept}\n")
        f.write(f"dropped_nan\t{dropped_nan}\n")
        f.write(f"dropped_nonpos\t{dropped_nonpos}\n")
        f.write(f"dropped_not_in_structure\t{dropped_not_in_struct}\n")

    print(f"{chrom}: kept {kept} rows from {in_lines} lines")
    print(f"Wrote: {args.out}")
    print(f"Wrote: {stats_path}")

if __name__ == "__main__":
    main()
