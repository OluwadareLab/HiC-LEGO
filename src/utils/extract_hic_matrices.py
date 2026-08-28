from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from resolution_utils import resolve_resolution

BACKBONE_RESOLUTION_BP = 1_000_000
DEFAULT_NORM = "KR"
DEFAULT_DATA_TYPE = "observed"


def _normalize_chr_label(chr_name: str) -> str:
    chr_name = chr_name.strip()
    if not chr_name:
        raise ValueError("--chr must be a non-empty string")
    return chr_name if chr_name.lower().startswith("chr") else f"chr{chr_name}"


def _chrom_candidates(chr_label: str) -> List[str]:
    label = _normalize_chr_label(chr_label)
    bare = label[3:] if label.lower().startswith("chr") else label
    candidates: List[str] = []
    for name in (bare, label, bare.upper(), label.upper(), f"chr{bare.upper()}"):
        if name and name not in candidates:
            candidates.append(name)
    return candidates


def _resolve_hic_chrom(hic_path: Path, chr_label: str) -> str:
    
    import hicstraw

    candidates = _chrom_candidates(chr_label)
    hic = hicstraw.HiCFile(str(hic_path))

    available = []
    for chrom in hic.getChromosomes():
        name = getattr(chrom, "name", None)
        if name is None or str(name).lower() in ("all", ""):
            continue
        available.append(str(name))

    available_lower = {n.lower(): n for n in available}
    for cand in candidates:
        if cand.lower() in available_lower:
            resolved = available_lower[cand.lower()]
            print(f"  chromosome in .hic: {resolved!r} (from --chr {chr_label})")
            return resolved

    raise ValueError(
        f"Chromosome {chr_label!r} not found in {hic_path}. "
        f"Tried {candidates}. Available: {available}"
    )


def _ensure_resolutions(hic_path: Path, needed: Sequence[int]) -> None:
    import hicstraw

    hic = hicstraw.HiCFile(str(hic_path))
    available = list(hic.getResolutions())
    missing = [r for r in needed if r not in available]
    if missing:
        raise ValueError(
            f"Resolution(s) {missing} not present in {hic_path}. "
            f"Available BP resolutions: {available}"
        )
    print(f"  resolutions OK: {list(needed)} (file has {len(available)} levels)")


def _write_contacts_3col(records: Iterable, out_path: Path) -> Tuple[int, int]:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    with out_path.open("w") as out:
        for rec in records:
            try:
                x = int(rec.binX)
                y = int(rec.binY)
                c = float(rec.counts)
            except (AttributeError, TypeError, ValueError):
                n_skipped += 1
                continue
            if not math.isfinite(c):
                n_skipped += 1
                continue
            out.write(f"{x}\t{y}\t{c}\n")
            n_written += 1
    return n_written, n_skipped


def extract_contacts(
    hic_path: Path,
    chrom: str,
    resolution: int,
    out_path: Path,
    normalization: str = DEFAULT_NORM,
    data_type: str = DEFAULT_DATA_TYPE,
) -> Path:
    
    import hicstraw

    print(
        f"\nExtracting {data_type}/{normalization} "
        f"chr={chrom!r} res={resolution} bp -> {out_path}"
    )
    sys.stdout.flush()

    records = hicstraw.straw(
        data_type,
        normalization,
        str(hic_path),
        chrom,
        chrom,
        "BP",
        int(resolution),
    )
    n_written, n_skipped = _write_contacts_3col(records, out_path)
    if n_written == 0:
        raise RuntimeError(
            f"No finite contacts written for {chrom!r} at {resolution} bp "
            f"(skipped={n_skipped}). Check that KR exists for this resolution."
        )
    print(f"  wrote {n_written:,} contacts ({n_skipped:,} skipped) -> {out_path}")
    return out_path


def extract_fine_and_backbone(
    hic_path: Path,
    chr_label: str,
    fine_res_bp: int,
    fine_res_label: str,
    output_dir: Path,
    normalization: str = DEFAULT_NORM,
    data_type: str = DEFAULT_DATA_TYPE,
    skip_fine: bool = False,
    skip_1mb: bool = False,
) -> Tuple[Path, Path]:
    chr_name = _normalize_chr_label(chr_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    fine_out = output_dir / f"{chr_name}_{fine_res_label}.txt"
    backbone_out = output_dir / f"{chr_name}_1mb.txt"

    print("=" * 70)
    print("extract_hic_matrices")
    print(f"  hic_file   : {hic_path}")
    print(f"  chromosome : {chr_name}")
    print(f"  fine res   : {fine_res_label} ({fine_res_bp} bp)")
    print(f"  backbone   : 1mb ({BACKBONE_RESOLUTION_BP} bp)")
    print(f"  norm/type  : {normalization} / {data_type}")
    print(f"  output_dir : {output_dir}")
    print("=" * 70)

    if not hic_path.is_file():
        raise FileNotFoundError(f".hic file not found: {hic_path}")

    needed = []
    if not (skip_fine and fine_out.is_file()):
        needed.append(fine_res_bp)
    if not (skip_1mb and backbone_out.is_file()):
        needed.append(BACKBONE_RESOLUTION_BP)
    if needed:
        _ensure_resolutions(hic_path, needed)

    chrom = _resolve_hic_chrom(hic_path, chr_name)

    if skip_fine and fine_out.is_file():
        print(f"\nSkipping fine matrix; using existing {fine_out}")
    else:
        extract_contacts(
            hic_path, chrom, fine_res_bp, fine_out, normalization, data_type
        )

    if skip_1mb and backbone_out.is_file():
        print(f"\nSkipping 1 Mb matrix; using existing {backbone_out}")
    else:
        extract_contacts(
            hic_path,
            chrom,
            BACKBONE_RESOLUTION_BP,
            backbone_out,
            normalization,
            data_type,
        )

    print("\nExtraction complete.")
    print(f"  fine     : {fine_out}")
    print(f"  backbone : {backbone_out}")
    return fine_out, backbone_out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract fine-resolution and 1 Mb KR-normalized contact lists from a .hic "
            "file (via hic-straw) for use as HiC-LEGO inputs."
        )
    )
    p.add_argument("--hic-file", required=True, help="Path to .hic file (local or URL).")
    p.add_argument("--chr", required=True, help="Chromosome label (e.g. chr22 or 22).")
    p.add_argument(
        "--res",
        default=None,
        help="Fine resolution label or bp (e.g. 5kb, 5000). Default: 5kb.",
    )
    p.add_argument(
        "--res-val",
        type=int,
        default=None,
        help="Optional fine resolution in bp (backward compatible).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for {chr}_{res}.txt and {chr}_1mb.txt.",
    )
    p.add_argument(
        "--normalization",
        default=DEFAULT_NORM,
        help=f"Normalization present in the .hic (default: {DEFAULT_NORM}).",
    )
    p.add_argument(
        "--data-type",
        default=DEFAULT_DATA_TYPE,
        choices=["observed", "oe"],
        help=f"Matrix data type (default: {DEFAULT_DATA_TYPE}).",
    )
    p.add_argument(
        "--skip-fine",
        action="store_true",
        help="Reuse existing fine-resolution matrix if present.",
    )
    p.add_argument(
        "--skip-1mb",
        action="store_true",
        help="Reuse existing 1 Mb matrix if present.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    try:
        res_label, res_bp = resolve_resolution(args.res, args.res_val)
    except ValueError as exc:
        print(f"ERROR: Invalid resolution: {exc}", file=sys.stderr)
        return 1

    hic_path = Path(args.hic_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        extract_fine_and_backbone(
            hic_path=hic_path,
            chr_label=args.chr,
            fine_res_bp=res_bp,
            fine_res_label=res_label,
            output_dir=output_dir,
            normalization=args.normalization,
            data_type=args.data_type,
            skip_fine=args.skip_fine,
            skip_1mb=args.skip_1mb,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
