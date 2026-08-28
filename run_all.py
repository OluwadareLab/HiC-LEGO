import os
import subprocess
import argparse
import sys
import time
import re
from datetime import datetime, timedelta

_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)
from resolution_utils import resolve_resolution


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        ensure_dir(parent)


def run_command(cmd, log_path, working_dir, step_name):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏩ Starting {step_name}...")
    start_time = time.time()

    ensure_parent_dir(log_path)
    
    with open(log_path, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f" STARTING {step_name}\n")
        f.write(f" TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" DIR:  {working_dir}\n")
        f.write(f" CMD:  {cmd}\n")
        f.write("="*80 + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd, 
            shell=True, 
            cwd=working_dir, 
            stdout=f, 
            stderr=subprocess.STDOUT,
            text=True
        )
        process.wait()
        
    duration = time.time() - start_time
    
    if process.returncode != 0:
        print(f"❌ Error in {step_name}. Check the unified log: {log_path}")
        sys.exit(1)
        
    print(f"✅ {step_name} completed in {timedelta(seconds=int(duration))}")
    return duration

def main():
    parser = argparse.ArgumentParser(
        description=(
            "HiCGAT Master Pipeline (Unified Log). "
            "Pass --hic-file to extract fine + 1 Mb KR matrices from a .hic via hic-straw; "
            "or pass --hic-matrix / --hic-1mb (or --input-dir) for prebuilt contact lists. "
            "Domains still come from --domains-dir or --input-dir."
        )
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment/cell line name, e.g., hmec, imr90")
    parser.add_argument("--chr", type=str, default="chr22", help="Chromosome label for staged naming (e.g., chr22)")
    parser.add_argument(
        "--res",
        type=str,
        default=None,
        help=(
            "Resolution as a label or base pairs (e.g. 5kb, 10kb, 1mb, or 5000). "
            "Default: 5kb. The numeric bp value is derived automatically."
        ),
    )
    parser.add_argument(
        "--res-val",
        type=int,
        default=None,
        help=(
            "Optional resolution in bp (e.g. 5000). Usually omit this and pass --res only; "
            "kept for backward compatibility."
        ),
    )
    parser.add_argument("--suffix", type=str, default="h2", help="Run suffix (e.g., h2)")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Optional legacy chromosome input folder (e.g. inputs/gm12878_chr22) containing "
            "domains/, <chr>_<res>.txt, and <chr>_1mb.txt. Used to fill any omitted explicit "
            "path flags for Step 0."
        ),
    )
    parser.add_argument(
        "--domains-dir",
        type=str,
        default=None,
        help="Directory of domain-list files (any filenames). Overrides --input-dir/domains.",
    )
    parser.add_argument(
        "--hic-file",
        type=str,
        default=None,
        help=(
            "Path to a .hic file. When set, Step 0a extracts KR-normalized fine-resolution "
            "and 1 Mb contact lists via hic-straw (writes under src/preprocessing/...)."
        ),
    )
    parser.add_argument(
        "--hic-matrix",
        type=str,
        default=None,
        help="Fine-resolution Hi-C contact list (any filename). Overrides --input-dir/<chr>_<res>.txt.",
    )
    parser.add_argument(
        "--hic-1mb",
        type=str,
        default=None,
        help="1 Mb Hi-C contact list (any filename). Overrides --input-dir/<chr>_1mb.txt.",
    )
    parser.add_argument(
        "--skip-hic-extract",
        action="store_true",
        help="With --hic-file: reuse existing extracted fine/1 Mb matrices if present.",
    )
    parser.add_argument(
        "--skip-optimal-domains",
        action="store_true",
        help="With Step 0: reuse existing optimal domains file if present.",
    )
    parser.add_argument(
        "--skip-1mb-structure",
        action="store_true",
        help="With Step 0: reuse existing 1 Mb PDB if present.",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="With Step 0: reuse existing coordinate mapping if present.",
    )
    args = parser.parse_args()

    experiment = args.experiment.strip()
    if not experiment:
        print("❌ --experiment cannot be empty or whitespace.")
        sys.exit(1)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment):
        print("❌ Invalid --experiment. Use only letters, numbers, underscore, or dash.")
        sys.exit(1)

    chr_name = args.chr.strip()
    if chr_name and not chr_name.lower().startswith("chr"):
        chr_name = f"chr{chr_name}"

    try:
        res_label, res_val = resolve_resolution(args.res, args.res_val)
    except ValueError as exc:
        print(f"❌ Invalid resolution: {exc}")
        sys.exit(1)

    hic_file = args.hic_file.strip() if args.hic_file else None
    hic_matrix_cli = args.hic_matrix
    hic_1mb_cli = args.hic_1mb

    run_step0 = bool(
        args.input_dir or args.domains_dir or hic_matrix_cli or hic_1mb_cli or hic_file
    )
    if run_step0:
        has_domains = bool(args.domains_dir or args.input_dir)
        has_hic = bool(hic_matrix_cli or args.input_dir or hic_file)
        has_1mb = bool(hic_1mb_cli or args.input_dir or hic_file)
        missing = []
        if not has_domains:
            missing.append("--domains-dir (or --input-dir)")
        if not has_hic:
            missing.append("--hic-matrix, --hic-file, or --input-dir")
        if not has_1mb:
            missing.append("--hic-1mb, --hic-file, or --input-dir")
        if missing:
            print("❌ Step 0 is incomplete. Missing: " + ", ".join(missing))
            sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = f"{chr_name}_{res_label}_{args.suffix}"
    run_group = os.path.join(experiment, run_id)
    unified_log = os.path.join(base_dir, "logs", experiment, f"{run_id}_full_pipeline.log")
    extract_out_dir = os.path.join(base_dir, "src", "preprocessing", experiment, run_id)

    ensure_dir(os.path.join(base_dir, "logs", experiment))
    ensure_dir(os.path.join(base_dir, "assembly", "mb_generation", "outputs", experiment))
    ensure_dir(os.path.join(base_dir, "assembly", "global_assembly", "outputs", experiment))
    ensure_dir(os.path.join(base_dir, "src", "Outputs"))
    ensure_dir(os.path.join(base_dir, "src", "preprocessing", experiment))

    if os.path.exists(unified_log):
        os.remove(unified_log)
    
    src_dir = os.path.join(base_dir, "src")
    mb_dir = os.path.join(base_dir, "assembly", "mb_generation")
    global_dir = os.path.join(base_dir, "assembly", "global_assembly")
    
    script_extract = os.path.join(src_dir, "utils", "extract_hic_matrices.py")
    script0 = os.path.join(src_dir, "utils", "prepare_run_inputs.py")
    script1 = os.path.join(src_dir, "run_pipeline.py")
    script2 = os.path.join(mb_dir, "generate_mb_structures.py")
    script3 = os.path.join(global_dir, "global_assembly_pipeline.py")

    global_start = time.time()
    step_times = {}

    print(f"🚀 Initializing Master Pipeline for experiment: {experiment}")
    print(f"🔖 Run ID: {run_id}")
    print(f"🔬 Resolution: {res_label} ({res_val} bp)")
    print(f"📝 Logging all outputs to: {unified_log}")
    if args.input_dir:
        print(f"📥 Input dir: {args.input_dir}")
    if args.domains_dir:
        print(f"📥 Domains dir: {args.domains_dir}")
    if hic_file:
        print(f"📥 .hic file: {hic_file}")
    if hic_matrix_cli:
        print(f"📥 Hi-C matrix: {hic_matrix_cli}")
    if hic_1mb_cli:
        print(f"📥 1 Mb Hi-C: {hic_1mb_cli}")
    print(f"📁 MB outputs: assembly/mb_generation/outputs/{experiment}/{run_id}_fixed_optimaldomains/")
    print(f"📁 Global outputs: assembly/global_assembly/outputs/{experiment}/output_{run_id}_global/")
    print("-" * 60)

    if hic_file:
        if not os.path.isfile(script_extract):
            print(f"❌ Missing extract script: {script_extract}")
            sys.exit(1)
        hic_path = hic_file if os.path.isabs(hic_file) else os.path.join(base_dir, hic_file)
        if not os.path.isfile(hic_path):
            print(f"❌ .hic file not found: {hic_path}")
            sys.exit(1)
        ensure_dir(extract_out_dir)
        cmd_extract = (
            f"python -u {script_extract} "
            f"--hic-file {hic_path} "
            f"--chr {chr_name} "
            f"--res {res_label} "
            f"--output-dir {extract_out_dir}"
        )
        if args.skip_hic_extract:
            cmd_extract += " --skip-fine --skip-1mb"
        step_times['Step 0a: Extract matrices from .hic'] = run_command(
            cmd_extract, unified_log, base_dir, "Step 0a"
        )
        if not hic_matrix_cli:
            hic_matrix_cli = os.path.join(extract_out_dir, f"{chr_name}_{res_label}.txt")
        if not hic_1mb_cli:
            hic_1mb_cli = os.path.join(extract_out_dir, f"{chr_name}_1mb.txt")

    if run_step0:
        if not os.path.isfile(script0):
            print(f"❌ Missing prepare script: {script0}")
            sys.exit(1)
        cmd0 = (
            f"python -u {script0} "
            f"--experiment {experiment} "
            f"--chr {chr_name} "
            f"--res {res_label} "
            f"--suffix {args.suffix} "
            f"--repo-root {base_dir}"
        )
        if args.input_dir:
            cmd0 += f" --input-dir {args.input_dir}"
        if args.domains_dir:
            cmd0 += f" --domains-dir {args.domains_dir}"
        if hic_matrix_cli:
            cmd0 += f" --hic-matrix {hic_matrix_cli}"
        if hic_1mb_cli:
            cmd0 += f" --hic-1mb {hic_1mb_cli}"
        if args.skip_optimal_domains:
            cmd0 += " --skip-optimal-domains"
        if args.skip_1mb_structure:
            cmd0 += " --skip-1mb-structure"
        if args.skip_mapping:
            cmd0 += " --skip-mapping"
        step_times['Step 0: Prepare inputs (domains + 1Mb + mapping)'] = run_command(
            cmd0, unified_log, base_dir, "Step 0"
        )

    hic_matrix = f"preprocessing/{run_group}/{chr_name}_{res_label}.txt"
    domain_list = f"preprocessing/{run_group}/{chr_name}_optimal_domains_{args.suffix}.txt"
    mapping_file = f"preprocessing/{run_group}/{chr_name}_{res_label}_coordinate_mapping.txt"

    cmd1 = (f"python -u {script1} --config {run_group} --domain-list {domain_list} "
            f"--hic-matrix {hic_matrix} --mapping-file {mapping_file} --run-name {run_id} "
            f"--resolution {res_val} --bin-size {res_val} --threads 1")
    
    step_times['Step 1: Domain Preprocessing'] = run_command(
        cmd1, unified_log, src_dir, "Step 1"
    )

    domain_folder = os.path.join(src_dir, f"intra_domains/outputs/{run_group}")
    bed_file = os.path.join(src_dir, f"preprocessing/{run_group}/output/{run_id}/{run_id}_filtered_domains.txt")
    hic_abs = os.path.join(src_dir, hic_matrix)
    mb_output_name = os.path.join("outputs", experiment, f"{run_id}_fixed_optimaldomains")
    coords_mapping = os.path.join(src_dir, f"preprocessing/{run_group}/{chr_name}_{res_label}_coordinate_mapping.txt")
    
    cmd2 = (f"python -u {script2} --domain-folder {domain_folder} --bed {bed_file} --hic {hic_abs} "
            f"--fine-res {res_val} --enable-refine --refine-lam-contact 0.0 "
            f"--refine-lam-smooth 0.002 --refine-lam-chain 0.01 --refine-max-iter 150 "
            f"--refine-revert-drop 0.10 --mb-workers 4 --coordinate-mapping {coords_mapping} --output-dir {mb_output_name}")
    
    step_times['Step 2: Micro-Block Generation'] = run_command(
        cmd2, unified_log, mb_dir, "Step 2"
    )

    mb_final_path = os.path.join(mb_dir, mb_output_name)
    backbone_pdb = os.path.join(src_dir, f"preprocessing/{run_group}/{chr_name}_1mb_structure.pdb")
    
    cmd3 = (f"python -u {script3} --mb-dir {mb_final_path} --backbone {backbone_pdb} --hic {hic_abs} "
        f"--chr {chr_name} --resolution {res_val} --coordinate-mapping {coords_mapping} --num-threads 1 --output-dir outputs/{experiment}/output_{run_id}_global "
            f"--stage-g-max-sweeps 5 --stage-g-convergence-threshold 1e-6 --stage-g-mu-translation 0.05 "
            f"--stage-g-lambda-inter 0.5 --stage-g-huber-delta 1.0 --stage-g-max-rotation-step 0.6 "
            f"--stage-g-max-translation-step 2.0 --stage-g-per-structure-maxiter 10")
    
    step_times['Step 3: Global Assembly Pipeline'] = run_command(
        cmd3, unified_log, global_dir, "Step 3"
    )

    global_duration = time.time() - global_start
    summary = "\n" + "="*60 + "\n"
    summary += f"🏁 PIPELINE SUMMARY FOR {run_id}\n"
    summary += "="*60 + "\n"
    for step, duration in step_times.items():
        summary += f"{step:<35} : {timedelta(seconds=int(duration))}\n"
    summary += "-" * 60 + "\n"
    summary += f"{'TOTAL EXECUTION TIME':<35} : {timedelta(seconds=int(global_duration))}\n"
    summary += "="*60 + "\n"
    
    print(summary)
    
    with open(unified_log, "a") as f:
        f.write(summary)

if __name__ == "__main__":
    main()
