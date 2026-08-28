from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_sample_id(sample_id: str) -> tuple[str, int, str]:
    parts = sample_id.split("_")
    if len(parts) < 2:
        raise ValueError(
            f"run-name must be like chr11_5kb_mysample (got {sample_id!r})"
        )
    chr_part = parts[0]
    if not re.match(r"chr\d+", chr_part, re.I):
        raise ValueError(
            f"run-name must start with chrN (e.g. chr11), got {chr_part!r}"
        )
    res_str = "5kb"
    for p in parts[1:]:
        if p.lower() == "5kb":
            res_str = "5kb"
            break
        if p.lower() == "10kb":
            res_str = "10kb"
            break
    res_bp = 5000 if res_str.lower() == "5kb" else 10000
    return chr_part, res_bp, res_str


def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def stage_inputs(
    base_dir: Path,
    run_name: str,
    hic_matrix: Path,
    domain_list: Path,
    backbone: Path,
    mapping_file: Path,
    resolution: int,
    use_symlinks: bool,
) -> tuple[Path, Path, Path, Path]:
    chr_part, res_bp, res_str = parse_sample_id(run_name)
    prep_dir = base_dir / "HiCGAT" / "preprocessing" / run_name
    ensure_dir(prep_dir)

    hic_name = f"{chr_part}_{res_str}.txt"
    backbone_name = f"{chr_part}_1mb_structure.pdb"
    domain_staged = prep_dir / "domain_list.txt"
    mapping_staged = prep_dir / "coordinate_mapping.txt"
    hic_staged = prep_dir / hic_name
    backbone_staged = prep_dir / backbone_name

    def put(src: Path, dst: Path) -> None:
        src = src.resolve()
        if src == dst.resolve():
            print(f"  Staged (unchanged): {dst.relative_to(base_dir)}")
            return
        if use_symlinks:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        else:
            shutil.copy2(src, dst)
        print(f"  Staged: {dst.relative_to(base_dir)}")

    put(hic_matrix, hic_staged)
    put(domain_list, domain_staged)
    put(backbone, backbone_staged)
    put(mapping_file, mapping_staged)

    return hic_staged, domain_staged, backbone_staged, mapping_staged


def run_stage1_docker(
    base_dir: Path,
    run_name: str,
    resolution: int,
    bin_size: int,
    threads: int,
    docker_image: str,
    domain_staged: Path,
    hic_staged: Path,
    mapping_staged: Path,
) -> bool:
    base_dir = base_dir.resolve()
    workspace = Path("/workspace")
    hicgat = workspace / "HiCGAT"
    domain_in = workspace / domain_staged.relative_to(base_dir)
    hic_in = workspace / hic_staged.relative_to(base_dir)
    mapping_in = workspace / mapping_staged.relative_to(base_dir)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{base_dir}:{workspace}",
        docker_image,
        "python",
        "-u",
        str(hicgat / "run_pipeline.py"),
        "--config",
        run_name,
        "--domain-list",
        str(domain_in),
        "--hic-matrix",
        str(hic_in),
        "--mapping-file",
        str(mapping_in),
        "--run-name",
        run_name,
        "--resolution",
        str(resolution),
        "--bin-size",
        str(bin_size),
        "--threads",
        str(threads),
    ]
    print("\n" + "=" * 70)
    print("Stage 1: HiCGAT pipeline (inside Docker)")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    print()
    r = subprocess.run(cmd, cwd=str(base_dir))
    return r.returncode == 0


def run_stage2_host(
    base_dir: Path,
    run_name: str,
    threads: int,
) -> bool:
    script = base_dir / "run_two_phase_pipeline.py"
    if not script.is_file():
        print(f"Stage 2 script not found: {script}", file=sys.stderr)
        return False
    cmd = [
        sys.executable,
        "-u",
        str(script),
        run_name,
        "--base-dir",
        str(base_dir),
        "--num-threads",
        str(threads),
    ]
    print("\n" + "=" * 70)
    print("Stage 2: MB generation + global assembly (host)")
    print("=" * 70)
    print("Command:", " ".join(cmd))
    print()
    r = subprocess.run(cmd, cwd=str(base_dir))
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full two-stage pipeline: Stage 1 in Docker (HiCGAT), Stage 2 on host (assembly).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--run-name",
        required=True,
        help="Sample/run ID (e.g. chr11_5kb_h2_mixed). Must start with chrN and contain 5kb or 10kb.",
    )
    ap.add_argument(
        "--hic-matrix",
        type=Path,
        default=None,
        help="Path to Hi-C matrix file (e.g. .txt). Required unless --stage2-only.",
    )
    ap.add_argument(
        "--domain-list",
        type=Path,
        default=None,
        help="Path to domain list file. Required unless --stage2-only.",
    )
    ap.add_argument(
        "--backbone",
        type=Path,
        default=None,
        help="Path to 1 Mb backbone structure file (e.g. .pdb). Required unless --stage2-only.",
    )
    ap.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help="Path to coordinate mapping file. Required unless --stage2-only.",
    )
    ap.add_argument(
        "--resolution",
        type=int,
        default=5000,
        help="Resolution in base pairs (default: 5000).",
    )
    ap.add_argument(
        "--bin-size",
        type=int,
        default=None,
        help="Bin size for domain structures (default: same as --resolution).",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=10,
        help="Number of threads for Stage 1 Phase 3 and Stage 2 (default: 10).",
    )
    ap.add_argument(
        "--docker-image",
        default="oluwadarelab/hicgnn:latest",
        help="Docker image for Stage 1 (default: oluwadarelab/hicgnn:latest).",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Project root (default: directory containing this script).",
    )
    ap.add_argument(
        "--link-inputs",
        action="store_true",
        help="Symlink inputs instead of copying (faster for large files; paths must be accessible from container).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only stage inputs and print commands; do not run Docker or Stage 2.",
    )
    ap.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 (Docker); do not run Stage 2.",
    )
    ap.add_argument(
        "--stage2-only",
        action="store_true",
        help="Run only Stage 2 (host). Inputs must already be staged and Stage 1 outputs must exist.",
    )
    args = ap.parse_args()

    base_dir = (args.base_dir or Path(__file__).resolve().parent).resolve()
    run_name = args.run_name.strip()
    bin_size = args.bin_size if args.bin_size is not None else args.resolution

    try:
        chr_part, res_bp, res_str = parse_sample_id(run_name)
    except ValueError as e:
        print(f"Invalid --run-name: {e}", file=sys.stderr)
        sys.exit(1)

    if args.stage2_only:
        if not (base_dir / "HiCGAT").is_dir():
            print(f"HiCGAT not found at {base_dir / 'HiCGAT'}. Set --base-dir?", file=sys.stderr)
            sys.exit(1)
        ok = run_stage2_host(base_dir=base_dir, run_name=run_name, threads=args.threads)
        sys.exit(0 if ok else 1)

    if None in (args.hic_matrix, args.domain_list, args.backbone, args.mapping_file):
        print("For full pipeline or --stage1-only, all of --hic-matrix, --domain-list, --backbone, --mapping-file are required.", file=sys.stderr)
        sys.exit(1)
    hic_matrix = args.hic_matrix.expanduser().resolve()
    domain_list = args.domain_list.expanduser().resolve()
    backbone = args.backbone.expanduser().resolve()
    mapping_file = args.mapping_file.expanduser().resolve()

    missing = []
    if not hic_matrix.is_file():
        missing.append(f"--hic-matrix: {hic_matrix}")
    if not domain_list.is_file():
        missing.append(f"--domain-list: {domain_list}")
    if not backbone.is_file():
        missing.append(f"--backbone: {backbone}")
    if not mapping_file.is_file():
        missing.append(f"--mapping-file: {mapping_file}")
    if missing:
        print("Missing or invalid input files:", file=sys.stderr)
        for m in missing:
            print("  ", m, file=sys.stderr)
        sys.exit(1)

    hicgat = base_dir / "HiCGAT"
    if not hicgat.is_dir():
        print(f"HiCGAT not found at {hicgat}. Run this script from the Hiclego_full project root.", file=sys.stderr)
        sys.exit(1)
    if not (hicgat / "run_pipeline.py").is_file():
        print(f"HiCGAT/run_pipeline.py not found.", file=sys.stderr)
        sys.exit(1)

    print("Staging inputs into HiCGAT/preprocessing/{} ...".format(run_name))
    hic_staged, domain_staged, backbone_staged, mapping_staged = stage_inputs(
        base_dir=base_dir,
        run_name=run_name,
        hic_matrix=hic_matrix,
        domain_list=domain_list,
        backbone=backbone,
        mapping_file=mapping_file,
        resolution=args.resolution,
        use_symlinks=args.link_inputs,
    )

    if args.dry_run:
        print("\n[DRY RUN] Would run Stage 1 (Docker):")
        print("  docker run --rm -v {}:/workspace {} python -u /workspace/HiCGAT/run_pipeline.py ...".format(
            base_dir, args.docker_image
        ))
        print("\n[DRY RUN] Would run Stage 2 (host):")
        print("  python run_two_phase_pipeline.py {} --base-dir {} --num-threads {}".format(
            run_name, base_dir, args.threads
        ))
        print("\nRemove --dry-run to execute.")
        return

    if args.stage2_only:
        ok = run_stage2_host(base_dir=base_dir, run_name=run_name, threads=args.threads)
        if not ok:
            sys.exit(1)
        print("\nStage 2 completed. Final output:", base_dir / "assembly" / "global_assembly" / "outputs" / run_name)
        return

    ok = run_stage1_docker(
        base_dir=base_dir,
        run_name=run_name,
        resolution=args.resolution,
        bin_size=bin_size,
        threads=args.threads,
        docker_image=args.docker_image,
        domain_staged=domain_staged,
        hic_staged=hic_staged,
        mapping_staged=mapping_staged,
    )
    if not ok:
        print("Stage 1 (Docker) failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    if not args.stage1_only:
        ok = run_stage2_host(base_dir=base_dir, run_name=run_name, threads=args.threads)
        if not ok:
            print("Stage 2 failed. Exiting.", file=sys.stderr)
            sys.exit(1)
    else:
        print("\nStage 1 only. Run Stage 2 manually: python run_two_phase_pipeline.py {} --base-dir {} --num-threads {}".format(
            run_name, base_dir, args.threads
        ))

    out_dir = base_dir / "assembly" / "global_assembly" / "outputs" / run_name
    print("\n" + "#" * 70)
    print("# Full pipeline completed successfully.")
    print("# Final output directory:", out_dir)
    print("#" * 70)


if __name__ == "__main__":
    main()
