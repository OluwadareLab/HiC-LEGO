import json
from typing import List, Set, Dict, Optional, Iterable, Tuple
import argparse
from pathlib import Path
import pandas as pd
import math
import csv
import re

def read_domain_file(file_path: str) -> Set[str]:
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def _is_candidate_domain_file(fp: Path) -> bool:
    if not fp.is_file():
        return False
    if fp.name.startswith("."):
        return False

    s = str(fp).lower()
    excluded_markers = [
        "intermediate_topdom_inputs",
        "_coordinate_mapping",
        "_nxn",
        "_topdom.txt",
    ]
    if any(marker in s for marker in excluded_markers):
        return False

    return True

def compute_scores(input_files: List[str]) -> Dict[str, float]:
    from itertools import combinations

    def get_MoC_from_domains(file1, file2):
        tads = read_(file1)
        true_tads = read_(file2)

        MoC = 0
        avg_MoC = []
        for _, check_row in true_tads.iterrows():
            true_start = int(check_row[0])
            true_end = int(check_row[1])
            for _, row in tads.iterrows():
                ref_start = int(row[0])
                ref_end = int(row[1])
                if true_start < ref_end and true_end > ref_start:
                    if true_end <= ref_end and true_start >= ref_start:
                        avg_MoC.append(math.pow(true_end - true_start, 2) / (
                            (true_end - true_start) * (ref_end - ref_start)))
                    elif true_end <= ref_end and true_start <= ref_start:
                        avg_MoC.append(math.pow(true_end - ref_start, 2) / (
                            (true_end - true_start) * (ref_end - ref_start)))
                    elif true_end >= ref_end and true_start >= ref_start:
                        avg_MoC.append(math.pow(ref_end - true_start, 2) / (
                            (true_end - true_start) * (ref_end - ref_start)))
                    else:
                        avg_MoC.append(math.pow(ref_end - ref_start, 2) / (
                            (true_end - true_start) * (ref_end - ref_start)))
                else:
                    avg_MoC.append(0)
        if len(avg_MoC) == 1 and avg_MoC[0] > 0:
            MoC = avg_MoC[0]
        elif sum(avg_MoC) <= 0:
            MoC = .000001
        else:
            MoC = sum(avg_MoC) / (math.sqrt(len(avg_MoC)) - 1)
        return round(MoC, 2)

    scores = {}
    for i, j in combinations(range(len(input_files)), 2):
        file1, file2 = input_files[i], input_files[j]
        scores[f"O{i+1}{j+1}"] = get_MoC_from_domains(file1, file2)

    print("Pairwise MoC Scores:")
    for key, score in scores.items():
        print(f"{key}: {score:.2f}")
    return scores

def select_highest_overlap(scores: Dict[str, float], input_files: List[str]) -> (str, str):
    highest_score_key = max(scores, key=scores.get)
    file1_index, file2_index = map(int, highest_score_key[1:])
    return input_files[file1_index - 1], input_files[file2_index - 1]

def write_single_domain_list_passthrough(input_file: str, output_file: str, chr_name: str) -> None:
    tads = read_(input_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for _, row in tads.iterrows():
            f.write(f"{chr_name}\t{int(row['start'])}\t{int(row['end'])}\n")

def recompute_and_write_optimal(file1: str, file2: str, output_file: str, chr_name: str):
    tads1 = read_(file1)
    tads2 = read_(file2)
    avg_scores = []
    optimal_domains = []

    def calculate_moc(domain1, tads2):
        domain1_start = int(domain1[0])
        domain1_end = int(domain1[1])
        moc_scores = []

        for _, domain2 in tads2.iterrows():
            domain2_start = int(domain2[0])
            domain2_end = int(domain2[1])

            if domain1_start < domain2_end and domain1_end > domain2_start:
                if domain1_end <= domain2_end and domain1_start >= domain2_start:
                    moc_scores.append(math.pow(domain1_end - domain1_start, 2) / (
                        (domain1_end - domain1_start) * (domain2_end - domain2_start)))
                elif domain1_end <= domain2_end and domain1_start <= domain2_start:
                    moc_scores.append(math.pow(domain1_end - domain2_start, 2) / (
                        (domain1_end - domain1_start) * (domain2_end - domain2_start)))
                elif domain1_end >= domain2_end and domain1_start >= domain2_start:
                    moc_scores.append(math.pow(domain2_end - domain1_start, 2) / (
                        (domain1_end - domain1_start) * (domain2_end - domain2_start)))
                else:
                    moc_scores.append(math.pow(domain2_end - domain2_start, 2) / (
                        (domain1_end - domain1_start) * (domain2_end - domain2_start)))
            else:
                moc_scores.append(0)

        if len(moc_scores) == 1 and moc_scores[0] > 0:
            return moc_scores[0]
        elif sum(moc_scores) <= 0:
            return 0
        else:
            return sum(moc_scores) / (math.sqrt(len(moc_scores)) - 1)

    for _, domain1 in tads1.iterrows():
        avg_score = calculate_moc(domain1, tads2)
        avg_scores.append(avg_score)
        if avg_score > 0:
            optimal_domains.append(f"{int(domain1[0])}\t{int(domain1[1])}")

    unique_optimal_domains = sorted(
        set(optimal_domains),
        key=lambda d: int(d.split("\t")[0])
    )

    with open(output_file, 'w') as f:
        for domain in unique_optimal_domains:
            domain_start, domain_end = domain.split("\t")
            f.write(f"{chr_name}\t{domain_start}\t{domain_end}\n")

def read_(file_path):
    moc_data = []
    with open(file_path, newline='') as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            row = re.split(r"\s+", line)

            if len(row) >= 3 and row[0].lower().startswith("chr"):
                start, end = row[1], row[2]
            elif len(row) >= 2:
                start, end = row[0], row[1]
            else:
                continue

            if not re.fullmatch(r"-?\d+", start) or not re.fullmatch(r"-?\d+", end):
                continue

            s, e = int(start), int(end)
            if s >= e:
                continue
            moc_data.append([s, e])

    if not moc_data:
        raise ValueError(f"No valid domain intervals parsed from file: {file_path}")

    moc_df = pd.DataFrame(moc_data, columns=["start", "end"])
    moc_df = moc_df.apply(pd.to_numeric, errors='coerce')
    return moc_df

def _normalize_chr(chr_name: str) -> str:
    chr_name = chr_name.strip()
    if not chr_name:
        raise ValueError("--chr must be a non-empty string")
    return chr_name if chr_name.lower().startswith("chr") else f"chr{chr_name}"

def _discover_input_files(input_dir: str, chr_name: str, pattern: Optional[str] = None) -> List[str]:
    p = Path(input_dir).expanduser()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {input_dir}")

    search_pattern = pattern if pattern else "*"
    files = [
        fp
        for fp in p.rglob(search_pattern)
        if _is_candidate_domain_file(fp)
    ]

    files = sorted(files, key=lambda x: str(x))
    return [str(fp) for fp in files]

def _collect_v2_experiment_files(
    input_root: str,
    chr_name: str,
    experiment_id: str,
    callers: Iterable[str] = ("topdom", "spectraltad", "hickey"),
    pattern: Optional[str] = None,
) -> List[str]:
    root = Path(input_root).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {input_root}")

    search_pattern = pattern if pattern else "*"

    files: List[Path] = []
    for caller in callers:
        base = root / caller / experiment_id
        if not base.exists() or not base.is_dir():
            continue
        for fp in base.rglob(search_pattern):
            if not _is_candidate_domain_file(fp):
                continue
            files.append(fp)

    files = sorted(set(files), key=lambda x: str(x))
    return [str(fp) for fp in files]

def _discover_v2_experiments(input_root: str, callers: Iterable[str] = ("topdom", "spectraltad", "hickey")) -> List[str]:

    root = Path(input_root).expanduser()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input root not found or not a directory: {input_root}")

    exps: Set[str] = set()
    for caller in callers:
        d = root / caller
        if not d.exists() or not d.is_dir():
            continue
        for child in d.iterdir():
            if child.is_dir():
                exps.add(child.name)

    return sorted(exps)

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def _best_pair_score(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        raise ValueError("scores is empty")
    best_key, best_score = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return best_key, float(best_score)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare all domain-list files found under the input directory (any filenames), "
            "pick the best-overlap pair by pairwise MoC when two or more are present, "
            "then emit optimal domains. A single file is written through as-is."
        )
    )
    parser.add_argument(
        "input_dir_positional",
        nargs="?",
        default=None,
        metavar="INPUT_DIR",
        help="Directory of domain files (same as --input-dir).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help=(
            "Directory containing domain files to compare (recursive search; all non-artifact "
            "files are used regardless of filename). Required for a single run. No default path is assumed."
        ),
    )
    parser.add_argument(
        "--chr",
        required=True,
        help="Chromosome label to write in the output (e.g., chr23 or 23).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Optional explicit output path. If omitted, uses <output-dir>/<chr>_optimal_domains_h2.txt."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory when --output-file is not provided. "
            "If omitted, uses the parent of --input-dir (e.g. .../gm12878_chr22/domains -> .../gm12878_chr22/)."
        ),
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Optional recursive glob pattern to pre-filter files inside --input-dir (e.g., '*.txt'). "
            "If omitted, all non-artifact files under the directory are considered."
        ),
    )

    parser.add_argument(
        "--batch-by-experiment",
        action="store_true",
        help=(
            "Process a v2 output-root (e.g., Integrated/outputs_chr13_20kb_v2) by batching over experiment subfolders "
            "and writing one optimal-domain output per experiment."
        ),
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help=(
            "Root directory in the v2 output layout (contains topdom/, spectraltad/, hickey/). "
            "If provided, you likely want to also pass --batch-by-experiment."
        ),
    )
    parser.add_argument(
        "--callers",
        default="topdom,spectraltad,hickey",
        help="Comma-separated caller subfolders to consider under --input-root (default: topdom,spectraltad,hickey).",
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help=(
            "Optional comma-separated list of experiment IDs to run (matches subfolder names under caller dirs). "
            "If omitted, all experiments found under --input-root are processed."
        ),
    )
    args = parser.parse_args()

    chr_name = _normalize_chr(args.chr)

    input_dir = args.input_dir or args.input_dir_positional

    callers = [c.strip() for c in str(args.callers).split(",") if c.strip()]
    if not callers:
        raise ValueError("--callers must contain at least one caller directory name")

    if args.batch_by_experiment:
        input_root = args.input_root or input_dir
        if not input_root:
            raise ValueError(
                "Batch mode requires --input-root or --input-dir / INPUT_DIR "
                "(path to the v2 output root containing caller subfolders)."
            )
        if args.output_dir:
            out_root = Path(args.output_dir).expanduser().resolve()
        else:
            out_root = Path(input_root).expanduser().resolve().parent
        out_root.mkdir(parents=True, exist_ok=True)

        if args.experiments:
            experiments = [e.strip() for e in str(args.experiments).split(",") if e.strip()]
        else:
            experiments = _discover_v2_experiments(input_root, callers)

        if not experiments:
            raise ValueError(f"No experiment subfolders found under v2 root: {input_root}")

        summary_rows: List[Dict[str, object]] = []
        failures: List[str] = []

        for exp in experiments:
            exp_out_dir = out_root / exp
            exp_out_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(exp_out_dir / f"{chr_name}_optimal_domains_h2.txt")

            try:
                input_files = _collect_v2_experiment_files(input_root, chr_name, exp, callers=callers, pattern=args.pattern)
                if len(input_files) == 0:
                    raise ValueError(
                        f"No domain files found for {chr_name} under {input_root} for experiment {exp}."
                    )

                print("\n============================================================")
                print(f"Experiment: {exp}")
                print("Input files:")
                for fp in input_files:
                    print(f"- {fp}")

                if len(input_files) == 1:
                    only_file = input_files[0]
                    print(
                        f"Only one domain list found; writing it through as optimal domains: {only_file}"
                    )
                    write_single_domain_list_passthrough(only_file, output_file, chr_name)
                    _write_text(
                        exp_out_dir / "selected_pair.txt",
                        "\n".join(
                            [
                                f"experiment\t{exp}",
                                f"chromosome\t{chr_name}",
                                f"best_key\tsingle_file_passthrough",
                                f"best_score\t",
                                f"file1\t{only_file}",
                                f"file2\t",
                            ]
                        )
                        + "\n",
                    )
                    summary_rows.append(
                        {
                            "experiment": exp,
                            "chrom": chr_name,
                            "n_inputs": 1,
                            "best_key": "single_file_passthrough",
                            "best_score": "",
                            "file1": only_file,
                            "file2": "",
                            "output_file": output_file,
                        }
                    )
                    continue

                print("Calculating pairwise MoC scores...")
                scores = compute_scores(input_files)

                print("Selecting files with the highest pairwise MoC score...")
                file1, file2 = select_highest_overlap(scores, input_files)
                best_key, best_score = _best_pair_score(scores)
                print(f"Selected files: {file1}, {file2} (best={best_key} score={best_score:.2f})")

                (exp_out_dir / "pairwise_moc.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
                _write_text(
                    exp_out_dir / "selected_pair.txt",
                    "\n".join(
                        [
                            f"experiment\t{exp}",
                            f"chromosome\t{chr_name}",
                            f"best_key\t{best_key}",
                            f"best_score\t{best_score:.6f}",
                            f"file1\t{file1}",
                            f"file2\t{file2}",
                        ]
                    )
                    + "\n",
                )

                print(f"Recomputing MoC scores and writing optimal domains to: {output_file}")
                recompute_and_write_optimal(file1, file2, output_file, chr_name)

                summary_rows.append(
                    {
                        "experiment": exp,
                        "chrom": chr_name,
                        "n_inputs": len(input_files),
                        "best_key": best_key,
                        "best_score": best_score,
                        "file1": file1,
                        "file2": file2,
                        "output_file": output_file,
                    }
                )

            except Exception as exc:
                msg = f"[{exp}] {exc}"
                print(f"ERROR: {msg}")
                failures.append(msg)
                continue

        summary_csv = out_root / f"{chr_name}_optimal_domains_summary.csv"
        if summary_rows:
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "experiment",
                        "chrom",
                        "n_inputs",
                        "best_key",
                        "best_score",
                        "file1",
                        "file2",
                        "output_file",
                    ],
                )
                w.writeheader()
                for r in summary_rows:
                    w.writerow(r)

        if failures:
            _write_text(out_root / "failures.txt", "\n".join(failures) + "\n")

        print("\n==================== Batch complete ====================")
        print(f"Experiments processed: {len(experiments)}")
        print(f"Successful: {len(summary_rows)}")
        print(f"Failed: {len(failures)}")
        print(f"Summary CSV: {summary_csv}")
        if failures:
            print(f"Failures log: {out_root / 'failures.txt'}")
        return

    if not input_dir:
        raise ValueError(
            "Provide the domain input directory as INPUT_DIR or --input-dir "
            "(e.g. inputs/gm12878_chr22/domains)."
        )

    input_files = _discover_input_files(input_dir, chr_name, args.pattern)
    if len(input_files) == 0:
        raise ValueError(
            f"No domain files found for {chr_name} under {input_dir}."
        )

    if args.output_file:
        output_file = args.output_file
    else:
        if args.output_dir:
            out_dir = Path(args.output_dir).expanduser()
        else:

            out_dir = Path(input_dir).expanduser().resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(out_dir / f"{chr_name}_optimal_domains_h2.txt")

    print("Input files:")
    for fp in input_files:
        print(f"- {fp}")

    if len(input_files) == 1:
        only_file = input_files[0]
        print(
            f"Only one domain list found; writing it through as optimal domains: {only_file}"
        )
        print(f"Writing optimal domains to: {output_file}")
        write_single_domain_list_passthrough(only_file, output_file, chr_name)
        return

    print("Calculating pairwise MoC scores...")
    scores = compute_scores(input_files)

    print("Selecting files with the highest pairwise MoC score...")
    file1, file2 = select_highest_overlap(scores, input_files)
    print(f"Selected files: {file1}, {file2}")

    print(f"Recomputing MoC scores and writing optimal domains to: {output_file}")
    recompute_and_write_optimal(file1, file2, output_file, chr_name)

if __name__ == "__main__":
    main()
