from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set

from resolution_utils import resolve_resolution

def _normalize_chr(chr_name: str) -> str:
    chr_name = chr_name.strip()
    if not chr_name:
        raise ValueError("--chr must be a non-empty string")
    return chr_name if chr_name.lower().startswith("chr") else f"chr{chr_name}"

def _run(cmd: list, cwd: Path, step: str) -> None:
    print(f"\n[{step}] cwd={cwd}")
    print(f"[{step}] {' '.join(str(c) for c in cmd)}")
    sys.stdout.flush()
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise RuntimeError(f"{step} failed with exit code {result.returncode}")

def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        print(f"  already in place: {dst}")
        return
    shutil.copy2(src, dst)
    print(f"  staged: {dst}")

def _collect_hic_bin_starts(hic_matrix: Path) -> Set[int]:
    coords: Set[int] = set()
    n_lines = 0
    n_3col = 0

    with hic_matrix.open("r") as f:
        for line in f:
            n_lines += 1
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            if len(parts) > 3:
                raise ValueError(
                    f"Hi-C file looks like a dense NxN matrix (row has {len(parts)} columns). "
                    f"Coordinate mapping requires a 3-column genomic contact list "
                    f"(bin1_start\\tbin2_start\\tcontact): {hic_matrix}"
                )

            try:
                c1 = int(float(parts[0]))
                c2 = int(float(parts[1]))
            except ValueError:
                continue

            coords.add(c1)
            coords.add(c2)
            if len(parts) == 3:
                n_3col += 1

    if not coords:
        raise ValueError(f"No genomic coordinates found in Hi-C matrix: {hic_matrix}")

    print(
        f"  scanned {n_lines} lines from {hic_matrix.name} "
        f"({n_3col} 3-col contact rows, {len(coords)} unique bin starts)"
    )
    return coords

def _snap_down(coord: int, resolution: int) -> int:
    return (coord // resolution) * resolution

def build_coordinate_mapping(
    hic_matrix: Path,
    out_mapping: Path,
    resolution: int,
) -> Path:
    if resolution <= 0:
        raise ValueError(f"resolution must be a positive integer (bp); got {resolution}")

    observed = _collect_hic_bin_starts(hic_matrix)
    raw_min = min(observed)
    raw_max = max(observed)

    start = _snap_down(raw_min, resolution)
    end = _snap_down(raw_max, resolution)
    if end < start:
        raise ValueError(
            f"Invalid genomic span after snapping to {resolution} bp: "
            f"start={start}, end={end} (raw min/max={raw_min}/{raw_max})"
        )

    n_off = sum(1 for c in observed if c % resolution != 0)
    if n_off:
        print(
            f"  warning: {n_off} observed coordinates are not multiples of "
            f"{resolution} bp; grid is snapped to resolution.",
            file=sys.stderr,
        )

    bins: List[int] = list(range(start, end + 1, resolution))
    if not bins:
        raise ValueError(f"No bins produced for span [{start}, {end}] at {resolution} bp")

    observed_on_grid = {_snap_down(c, resolution) for c in observed}
    grid_set = set(bins)
    n_missing_in_contacts = len(grid_set - observed_on_grid)

    out_mapping.parent.mkdir(parents=True, exist_ok=True)
    with out_mapping.open("w") as out:
        for i, coord in enumerate(bins):
            out.write(f"{coord}\t{i}\n")

    print(f"  resolution     : {resolution} bp")
    print(f"  observed span  : {raw_min} .. {raw_max}")
    print(f"  grid span      : {start} .. {end}")
    print(f"  unique bins    : {len(bins)} (indices 0 .. {len(bins) - 1})")
    print(f"  grid bins with no contacts in Hi-C: {n_missing_in_contacts}")
    print(f"  wrote mapping  : {out_mapping}")
    return out_mapping

def generate_optimal_domains(
    repo_root: Path,
    domains_dir: Path,
    chr_name: str,
    output_file: Path,
) -> Path:
    script = repo_root / "src" / "utils" / "optimal_domain.py"
    if not script.is_file():
        raise FileNotFoundError(f"optimal_domain.py not found: {script}")
    if not domains_dir.is_dir():
        raise FileNotFoundError(f"Domains directory not found: {domains_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--input-dir",
        str(domains_dir),
        "--chr",
        chr_name,
        "--output-file",
        str(output_file),
    ]
    _run(cmd, cwd=repo_root, step="optimal domains")
    if not output_file.is_file():
        raise FileNotFoundError(f"Expected optimal domains file was not written: {output_file}")
    return output_file

def generate_1mb_structure(
    repo_root: Path,
    hic_1mb: Path,
    output_pdb: Path,
) -> Path:
    src_dir = repo_root / "src"
    main_py = src_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"main.py not found: {main_py}")
    if not hic_1mb.is_file():
        raise FileNotFoundError(f"1 Mb Hi-C matrix not found: {hic_1mb}")

    stem = hic_1mb.stem
    native_pdb = src_dir / "Outputs" / f"{stem}_structure.pdb"

    hic_abs = hic_1mb.resolve()
    cmd = [sys.executable, "-u", "-m", "main", str(hic_abs)]
    _run(cmd, cwd=src_dir, step="1 Mb structure (Phase 2)")

    if not native_pdb.is_file():
        raise FileNotFoundError(f"Expected 1 Mb PDB was not written: {native_pdb}")

    _copy_file(native_pdb, output_pdb)
    return output_pdb

def stage_preprocessing(
    repo_root: Path,
    experiment: str,
    chr_name: str,
    res: str,
    suffix: str,
    hic_fine: Path,
    mapping: Path,
    optimal_domains: Path,
    backbone_pdb: Path,
) -> Path:
    run_id = f"{chr_name}_{res}_{suffix}"
    prep_dir = repo_root / "src" / "preprocessing" / experiment / run_id
    prep_dir.mkdir(parents=True, exist_ok=True)

    _copy_file(hic_fine, prep_dir / f"{chr_name}_{res}.txt")
    _copy_file(mapping, prep_dir / f"{chr_name}_{res}_coordinate_mapping.txt")
    _copy_file(optimal_domains, prep_dir / f"{chr_name}_optimal_domains_{suffix}.txt")
    _copy_file(backbone_pdb, prep_dir / f"{chr_name}_1mb_structure.pdb")

    print(f"\nPreprocessing staged under: {prep_dir}")
    return prep_dir

def _resolve_path(repo_root: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p

def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate optimal domains, 1 Mb backbone, and coordinate mapping; "
            "stage them for run_all.py under src/preprocessing/<experiment>/<chr>_<res>_<suffix>/. "
            "Prefer explicit --domains-dir / --hic-matrix / --hic-1mb paths; "
            "--input-dir remains as optional sugar for the legacy folder layout."
        )
    )
    p.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Optional legacy chromosome input folder containing domains/, "
            "<chr>_<res>.txt, and <chr>_1mb.txt (e.g. inputs/gm12878_chr22). "
            "Used to fill any of --domains-dir / --hic-matrix / --hic-1mb that are omitted."
        ),
    )
    p.add_argument(
        "--domains-dir",
        default=None,
        help="Directory of domain-list files (any filenames). Overrides --input-dir/domains.",
    )
    p.add_argument(
        "--hic-matrix",
        default=None,
        help="Fine-resolution Hi-C contact list (any filename). Overrides --input-dir/<chr>_<res>.txt.",
    )
    p.add_argument(
        "--hic-1mb",
        default=None,
        help="1 Mb Hi-C contact list (any filename). Overrides --input-dir/<chr>_1mb.txt.",
    )
    p.add_argument("--experiment", required=True, help="Experiment/cell-line name used by run_all.py")
    p.add_argument("--chr", default="chr22", help="Chromosome label used for staged naming (e.g. chr22)")
    p.add_argument(
        "--res",
        default=None,
        help=(
            "Resolution as a label or base pairs (e.g. 5kb, 10kb, 1mb, or 5000). "
            "Default: 5kb. The numeric bp value is derived automatically."
        ),
    )
    p.add_argument(
        "--res-val",
        type=int,
        default=None,
        help=(
            "Optional resolution in bp (e.g. 5000). Usually omit this and pass --res only; "
            "kept for backward compatibility."
        ),
    )
    p.add_argument("--suffix", default="h2", help="Run suffix (e.g. h2)")
    p.add_argument(
        "--repo-root",
        default=None,
        help="HiCLego_final repo root (default: parent of src/, inferred from this script).",
    )
    p.add_argument(
        "--skip-optimal-domains",
        action="store_true",
        help=(
            "Reuse existing optimal-domains file if present in the staging dir "
            "(or legacy --input-dir location)."
        ),
    )
    p.add_argument(
        "--skip-1mb-structure",
        action="store_true",
        help=(
            "Reuse existing 1 Mb PDB if present in the staging dir "
            "(or legacy --input-dir location)."
        ),
    )
    p.add_argument(
        "--skip-mapping",
        action="store_true",
        help=(
            "Reuse existing coordinate mapping if present in the staging dir "
            "(or legacy --input-dir location)."
        ),
    )
    p.add_argument(
        "--mapping-file",
        default=None,
        help=(
            "Optional pre-built mapping to copy instead of generating from the Hi-C file. "
            "By default a unique fixed-resolution index is built from --hic-matrix."
        ),
    )
    return p.parse_args(argv)

def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else script_path.parents[2]

    chr_name = _normalize_chr(args.chr)
    try:
        res, res_val = resolve_resolution(args.res, args.res_val)
    except ValueError as exc:
        raise ValueError(f"Invalid resolution: {exc}") from exc
    suffix = args.suffix
    experiment = args.experiment.strip()
    if not experiment:
        raise ValueError("--experiment cannot be empty")

    input_dir: Optional[Path] = None
    if args.input_dir:
        input_dir = _resolve_path(repo_root, args.input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir not found or not a directory: {input_dir}")

    if args.domains_dir:
        domains_dir = _resolve_path(repo_root, args.domains_dir)
    elif input_dir is not None:
        domains_dir = input_dir / "domains"
    else:
        domains_dir = None

    if args.hic_matrix:
        hic_fine = _resolve_path(repo_root, args.hic_matrix)
    elif input_dir is not None:
        hic_fine = input_dir / f"{chr_name}_{res}.txt"
    else:
        hic_fine = None

    if args.hic_1mb:
        hic_1mb = _resolve_path(repo_root, args.hic_1mb)
    elif input_dir is not None:
        hic_1mb = input_dir / f"{chr_name}_1mb.txt"
    else:
        hic_1mb = None

    missing = []
    if domains_dir is None:
        missing.append("--domains-dir (or --input-dir)")
    if hic_fine is None:
        missing.append("--hic-matrix (or --input-dir)")
    if hic_1mb is None:
        missing.append("--hic-1mb (or --input-dir)")
    if missing:
        raise ValueError(
            "Missing required inputs: "
            + ", ".join(missing)
            + ". Pass explicit paths and/or --input-dir."
        )

    run_id = f"{chr_name}_{res}_{suffix}"
    prep_dir = repo_root / "src" / "preprocessing" / experiment / run_id
    prep_dir.mkdir(parents=True, exist_ok=True)

    optimal_out = prep_dir / f"{chr_name}_optimal_domains_{suffix}.txt"
    pdb_out = prep_dir / f"{chr_name}_1mb_structure.pdb"
    mapping_out = prep_dir / f"{chr_name}_{res}_coordinate_mapping.txt"

    legacy_optimal = (
        input_dir / f"{chr_name}_optimal_domains_{suffix}.txt" if input_dir is not None else None
    )
    legacy_pdb = input_dir / f"{chr_name}_1mb_structure.pdb" if input_dir is not None else None
    legacy_mapping = (
        input_dir / f"{chr_name}_{res}_coordinate_mapping.txt" if input_dir is not None else None
    )

    print("=" * 70)
    print("prepare_run_inputs")
    print(f"  repo_root   : {repo_root}")
    print(f"  input_dir   : {input_dir if input_dir is not None else '(none)'}")
    print(f"  domains_dir : {domains_dir}")
    print(f"  hic_matrix  : {hic_fine}")
    print(f"  hic_1mb     : {hic_1mb}")
    print(f"  experiment  : {experiment}")
    print(f"  run_id      : {run_id}")
    print(f"  resolution  : {res} ({res_val} bp)")
    print(f"  staging     : {prep_dir}")
    print("=" * 70)

    if not domains_dir.is_dir():
        raise FileNotFoundError(f"Domains directory not found: {domains_dir}")
    if not hic_fine.is_file():
        raise FileNotFoundError(f"Fine-resolution Hi-C matrix not found: {hic_fine}")
    if not hic_1mb.is_file():
        raise FileNotFoundError(f"1 Mb Hi-C matrix not found: {hic_1mb}")

    cached_optimal = _first_existing(optimal_out, legacy_optimal) if args.skip_optimal_domains else None
    if cached_optimal is not None:
        print(f"\nSkipping optimal-domain generation; using {cached_optimal}")
        if cached_optimal.resolve() != optimal_out.resolve():
            _copy_file(cached_optimal, optimal_out)
    else:
        generate_optimal_domains(repo_root, domains_dir, chr_name, optimal_out)

    cached_pdb = _first_existing(pdb_out, legacy_pdb) if args.skip_1mb_structure else None
    if cached_pdb is not None:
        print(f"\nSkipping 1 Mb structure generation; using {cached_pdb}")
        if cached_pdb.resolve() != pdb_out.resolve():
            _copy_file(cached_pdb, pdb_out)
    else:
        generate_1mb_structure(repo_root, hic_1mb, pdb_out)

    if args.mapping_file:
        mapping_src = _resolve_path(repo_root, args.mapping_file)
        if not mapping_src.is_file():
            raise FileNotFoundError(f"Mapping file not found: {mapping_src}")
        print(f"\nUsing provided mapping file: {mapping_src}")
        _copy_file(mapping_src, mapping_out)
    else:
        cached_mapping = _first_existing(mapping_out, legacy_mapping) if args.skip_mapping else None
        if cached_mapping is not None:
            print(f"\nSkipping mapping generation; using {cached_mapping}")
            if cached_mapping.resolve() != mapping_out.resolve():
                _copy_file(cached_mapping, mapping_out)
        else:
            print(f"\nBuilding unique-index coordinate mapping from {hic_fine} ...")
            build_coordinate_mapping(hic_fine, mapping_out, resolution=res_val)

    prep_dir = stage_preprocessing(
        repo_root=repo_root,
        experiment=experiment,
        chr_name=chr_name,
        res=res,
        suffix=suffix,
        hic_fine=hic_fine,
        mapping=mapping_out,
        optimal_domains=optimal_out,
        backbone_pdb=pdb_out,
    )

    print("\nReady for run_all.py Steps 1–3.")
    print(f"  preprocessing: {prep_dir}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
