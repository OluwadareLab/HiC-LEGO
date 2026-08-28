import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

def read_coordinate_mapping(mapping_file: str) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            coord, bin_index = parts[0], parts[1]
            mapping[int(coord)] = int(bin_index)
    return mapping

def map_domain_to_bins(
    domain_start: int,
    domain_end: int,
    coordinate_mapping: Dict[int, int],
    resolution: Optional[int] = None,
) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:

    if resolution is not None and resolution > 0:
        bin_mappings = [
            (coord, bin_index)
            for coord, bin_index in coordinate_mapping.items()
            if coord < domain_end and (coord + resolution) > domain_start
        ]
    else:
        bin_mappings = [
            (coord, bin_index)
            for coord, bin_index in coordinate_mapping.items()
            if domain_start <= coord < domain_end
        ]

    if not bin_mappings:
        print(f"Warning: No bins found for domain {domain_start}-{domain_end}", file=sys.stderr)
        return None, []

    bin_mappings.sort(key=lambda x: x[1])
    bins = [b for (_, b) in bin_mappings]
    return (min(bins), max(bins)), bin_mappings

def calculate_domain_interaction(domain1_bins: Optional[Tuple[int, int]], domain2_bins: Optional[Tuple[int, int]], hic_matrix: np.ndarray) -> float:
    if domain1_bins is None or domain2_bins is None:
        return 0.0

    start_bin1, end_bin1_inclusive = domain1_bins
    start_bin2, end_bin2_inclusive = domain2_bins

    s1 = max(0, start_bin1)
    e1 = min(hic_matrix.shape[0], end_bin1_inclusive + 1)
    s2 = max(0, start_bin2)
    e2 = min(hic_matrix.shape[1], end_bin2_inclusive + 1)

    if s1 >= e1 or s2 >= e2:
        return 0.0

    interaction_sum = hic_matrix[s1:e1, s2:e2].sum()
    return float(interaction_sum)

def read_domain_list(domain_list_file: str) -> List[Tuple[str, int, int]]:
    domain_list: List[Tuple[str, int, int]] = []
    with open(domain_list_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], parts[1], parts[2]
            domain_list.append((chrom, int(start), int(end)))
    return domain_list

def convert_3col_to_nxn(hic_matrix_file: str, coordinate_mapping: Optional[Dict[int, int]] = None) -> np.ndarray:

    coord_to_bin: Optional[Dict[int, int]] = None
    if coordinate_mapping is not None:
        coord_to_bin = {coord: bin_idx for coord, bin_idx in coordinate_mapping.items()}
        max_bin_from_mapping = max(coordinate_mapping.values()) if coordinate_mapping else -1
    else:
        max_bin_from_mapping = -1

    data = []
    max_bin = -1

    with open(hic_matrix_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                coord1 = int(float(parts[0]))
                coord2 = int(float(parts[1]))
                interaction = float(parts[2])
                if not np.isfinite(interaction):
                    continue

                if coord_to_bin is not None:
                    if coord1 not in coord_to_bin or coord2 not in coord_to_bin:
                        continue
                    bin1 = coord_to_bin[coord1]
                    bin2 = coord_to_bin[coord2]
                else:
                    bin1 = coord1
                    bin2 = coord2

                data.append((bin1, bin2, interaction))
                max_bin = max(max_bin, bin1, bin2)
            except (ValueError, IndexError, KeyError):
                continue

    matrix_size = max(max_bin, max_bin_from_mapping) + 1

    if matrix_size <= 0:
        raise ValueError(f"No valid data found in {hic_matrix_file} or empty mapping")

    matrix = np.zeros((matrix_size, matrix_size), dtype=float)

    for bin1, bin2, interaction in data:
        matrix[bin1, bin2] += interaction
        matrix[bin2, bin1] += interaction

    return matrix

def read_hic_matrix(hic_matrix_file: str, coordinate_mapping: Optional[Dict[int, int]] = None) -> np.ndarray:

    with open(hic_matrix_file, "r") as f:
        first_line = f.readline().strip()
        if not first_line:
            raise ValueError(f"Empty file: {hic_matrix_file}")

        first_cols = len(first_line.split())

        if first_cols == 3:
            next_lines = [f.readline().strip() for _ in range(min(5, 10))]
            all_3col = all(len(line.split()) == 3 for line in next_lines if line)

            if all_3col:
                print(f"Detected 3-column format. Converting to NxN matrix...", file=sys.stderr)
                return convert_3col_to_nxn(hic_matrix_file, coordinate_mapping)

    try:
        matrix = np.loadtxt(hic_matrix_file)

        if matrix.ndim == 2:
            if matrix.shape[0] == matrix.shape[1]:
                print(f"Detected NxN format. Matrix size: {matrix.shape[0]}x{matrix.shape[1]}", file=sys.stderr)
                return matrix
            else:
                if matrix.shape[1] == 3:
                    print(f"Detected 3-column format (from shape {matrix.shape}). Converting to NxN matrix...", file=sys.stderr)
                    return convert_3col_to_nxn(hic_matrix_file, coordinate_mapping)
                else:
                    raise ValueError(f"Matrix is not square: {matrix.shape}. Expected NxN or 3-column format.")
        elif matrix.ndim == 1:
            print(f"Detected 3-column format. Converting to NxN matrix...", file=sys.stderr)
            return convert_3col_to_nxn(hic_matrix_file, coordinate_mapping)
        else:
            raise ValueError(f"Unexpected matrix dimensions: {matrix.ndim}. Expected 2D NxN matrix.")
    except (ValueError, UnicodeDecodeError) as e:
        try:
            print(f"Failed to read as NxN, trying 3-column format...", file=sys.stderr)
            return convert_3col_to_nxn(hic_matrix_file, coordinate_mapping)
        except Exception as e2:
            raise ValueError(
                f"Failed to read Hi-C matrix from {hic_matrix_file}. "
                f"Expected either NxN tab-delimited matrix or 3-column format (coord1\\tcoord2\\tinteraction). "
                f"Error: {e2}"
            )

def filter_domains_by_mapping_start(
    domain_list: List[Tuple[str, int, int]],
    coordinate_mapping: Dict[int, int],
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:

    coord_set = set(coordinate_mapping.keys())
    kept: List[Tuple[str, int, int]] = []
    removed: List[Tuple[str, int, int]] = []
    for d in domain_list:
        if d[1] in coord_set:
            kept.append(d)
        else:
            removed.append(d)
    return kept, removed

def ensure_output_directory(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def write_domain_list(domain_list_file: str, domain_list: List[Tuple[str, int, int]]) -> None:
    ensure_output_directory(domain_list_file)
    with open(domain_list_file, "w") as f:
        for chrom, start, end in domain_list:
            f.write(f"{chrom}\t{start}\t{end}\n")

def filter_domains_by_zero_interaction(
    domain_list: List[Tuple[str, int, int]],
    hic_matrix: np.ndarray,
    coordinate_mapping: Dict[int, int],
    resolution: Optional[int] = None,
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    slices: List[Optional[Tuple[int, int]]] = []
    for chrom, start, end in domain_list:
        bins, _ = map_domain_to_bins(start, end, coordinate_mapping, resolution)
        if bins is None:
            slices.append(None)
            continue
        b0, b1_inclusive = bins
        s = max(0, b0)
        e = min(hic_matrix.shape[0], b1_inclusive + 1)
        if s >= e:
            slices.append(None)
        else:
            slices.append((s, e))

    totals = np.zeros(len(domain_list), dtype=float)
    for i, si in enumerate(slices):
        if si is None:
            continue
        s1, e1 = si
        for j, sj in enumerate(slices):
            if sj is None:
                continue
            s2, e2 = sj
            totals[i] += float(hic_matrix[s1:e1, s2:e2].sum())

    kept: List[Tuple[str, int, int]] = []
    removed: List[Tuple[str, int, int]] = []
    for d, total in zip(domain_list, totals):
        if total == 0.0:
            removed.append(d)
        else:
            kept.append(d)

    return kept, removed

def filter_domains_combined(
    domain_list: List[Tuple[str, int, int]],
    hic_matrix: np.ndarray,
    coordinate_mapping: Dict[int, int],
    resolution: Optional[int] = None,
) -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]], Dict[str, int]]:

    coord_set = set(coordinate_mapping.keys())
    filtered_by_mapping: List[Tuple[str, int, int]] = []
    for d in domain_list:
        if d[1] in coord_set:
            filtered_by_mapping.append(d)

    unmapped_count = len(domain_list) - len(filtered_by_mapping)

    if not filtered_by_mapping:
        return [], domain_list, {"unmapped_start": unmapped_count, "zero_interactions": 0, "duplicate_start": 0}

    slices: List[Optional[Tuple[int, int]]] = []
    for chrom, start, end in filtered_by_mapping:
        bins, _ = map_domain_to_bins(start, end, coordinate_mapping, resolution)
        if bins is None:
            slices.append(None)
            continue
        b0, b1_inclusive = bins
        s = max(0, b0)
        e = min(hic_matrix.shape[0], b1_inclusive + 1)
        if s >= e:
            slices.append(None)
        else:
            slices.append((s, e))

    totals = np.zeros(len(filtered_by_mapping), dtype=float)
    for i, si in enumerate(slices):
        if si is None:
            continue
        s1, e1 = si
        for j, sj in enumerate(slices):
            if sj is None:
                continue
            s2, e2 = sj
            totals[i] += float(hic_matrix[s1:e1, s2:e2].sum())

    filtered_by_interaction: List[Tuple[str, int, int]] = []
    interaction_totals: List[float] = []
    for d, total in zip(filtered_by_mapping, totals):
        if total > 0.0:
            filtered_by_interaction.append(d)
            interaction_totals.append(total)

    zero_count = len(filtered_by_mapping) - len(filtered_by_interaction)

    if zero_count > 0:
        n_logged = 0
        for i, (d, t) in enumerate(zip(filtered_by_mapping, totals)):
            if t == 0.0 and n_logged < 5:
                sl = slices[i]
                if sl is not None:
                    print(
                        f"  Zero-total domain example: {d[0]} {d[1]}-{d[2]} -> bin range [{sl[0]}, {sl[1]})",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"  Zero-total domain (no bins): {d[0]} {d[1]}-{d[2]}",
                        file=sys.stderr,
                    )
                n_logged += 1

    if not filtered_by_interaction:
        return [], domain_list, {"unmapped_start": unmapped_count, "zero_interactions": zero_count, "duplicate_start": 0}

    unique_domains: Dict[int, Tuple[Tuple[str, int, int], float]] = {}
    for d, total in zip(filtered_by_interaction, interaction_totals):
        start_coord = d[1]
        if start_coord not in unique_domains:
            unique_domains[start_coord] = (d, total)
        else:
            _, existing_total = unique_domains[start_coord]
            if total > existing_total:
                unique_domains[start_coord] = (d, total)

    kept = [d for d, _ in unique_domains.values()]
    duplicate_count = len(filtered_by_interaction) - len(kept)

    kept_set = set(kept)
    removed = [d for d in domain_list if d not in kept_set]

    breakdown = {"unmapped_start": unmapped_count, "zero_interactions": zero_count, "duplicate_start": duplicate_count}
    return kept, removed, breakdown

def write_domain_interactions_raw_and_3col(
    raw_output_file: str,
    col3_output_file: str,
    domain_list: List[Tuple[str, int, int]],
    hic_matrix: np.ndarray,
    coordinate_mapping: Dict[int, int],
    resolution: Optional[int] = None,
):
    ensure_output_directory(raw_output_file)
    ensure_output_directory(col3_output_file)
    with open(raw_output_file, "w") as raw_f, open(col3_output_file, "w") as col3_f:
        for i, domain1 in enumerate(domain_list):
            domain1_bins, _ = map_domain_to_bins(domain1[1], domain1[2], coordinate_mapping, resolution)
            for j, domain2 in enumerate(domain_list):
                domain2_bins, _ = map_domain_to_bins(domain2[1], domain2[2], coordinate_mapping, resolution)
                interaction_strength = calculate_domain_interaction(domain1_bins, domain2_bins, hic_matrix)
                raw_f.write(f"{domain1[0]}\t{domain1[1]}\t{domain1[2]}\t{domain2[1]}\t{domain2[2]}\t{interaction_strength}\n")
                col3_f.write(f"{domain1[1]}\t{domain2[1]}\t{interaction_strength}\n")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute domain-domain interactions and produce HiC-GNN 3-column file")
    p.add_argument("--domain-list", required=True, help="Path to domain list file (chrom\tstart\tend per line)")
    p.add_argument("--hic-matrix", required=True, help="Path to Hi-C matrix. Supports both NxN (tab-delimited) and 3-column (bin1\\tbin2\\tinteraction) formats. Format is auto-detected.")
    p.add_argument("--mapping", required=True, help="Path to coordinate->bin mapping file (coord\tindex)")
    p.add_argument("--out-raw", required=True, help="Output path for raw domain interactions (full 6-col)")
    p.add_argument("--out-3col", required=True, help="Output path for reformatted 3-column HiC-GNN file")
    p.add_argument("--resolution", type=int, default=10000, help="Bin size (bp). Used to include bins that overlap domains (bin at coord covers [coord, coord+resolution)).")
    p.add_argument(
        "--out-filtered-domain-list",
        default=None,
        help="If set, write a filtered domain list containing only domains whose start coordinate exists in the mapping.",
    )
    p.add_argument(
        "--filter-domains-by-mapping-start",
        action="store_true",
        help="Filter domains (and interactions) to only those whose start coordinate exists in mapping.",
    )
    p.add_argument(
        "--out-filtered-domain-list-by-zero-interaction",
        default=None,
        help=(
            "If set, write an additional filtered domain list that removes domains whose total"
            " interaction with all domains sums to 0.0 (matches downstream deletion of all-zero"
            " rows/cols when converting the 3-col list to an adjacency matrix)."
        ),
    )
    p.add_argument(
        "--out-filtered-domain-list-combined",
        default=None,
        help=(
            "If set, write a filtered domain list using combined filtering: (1) start coordinate in mapping, "
            "(2) non-zero interactions, and (3) unique start coordinates (deduplicates by keeping domain "
            "with highest interaction total). This ensures unique indexing for downstream pipelines."
        ),
    )
    p.add_argument("--debug", action="store_true", help="Print debug info for first 10 domains")
    return p.parse_args()

def log_debug_info(
    domain_list: List[Tuple[str, int, int]],
    coordinate_mapping: Dict[int, int],
    hic_matrix: np.ndarray,
    resolution: Optional[int] = None,
) -> None:
    print("=== Debug Info ===")
    print("First 10 Domain-to-Bin Mappings:")
    for i, domain in enumerate(domain_list[:10]):
        domain_bins, bin_mappings = map_domain_to_bins(domain[1], domain[2], coordinate_mapping, resolution)
        print(f"Domain {i + 1}: {domain[1]}-{domain[2]} -> Bins {domain_bins}")
        print(f"    Mapped Bins: {bin_mappings}")

    print("\nFirst 10 Interaction Calculations:")
    for i, domain1 in enumerate(domain_list[:10]):
        domain1_bins, _ = map_domain_to_bins(domain1[1], domain1[2], coordinate_mapping, resolution)
        for j, domain2 in enumerate(domain_list[:10]):
            domain2_bins, _ = map_domain_to_bins(domain2[1], domain2[2], coordinate_mapping, resolution)
            interaction_strength = calculate_domain_interaction(domain1_bins, domain2_bins, hic_matrix)
            print(f"Interaction (Domain {i + 1} <-> Domain {j + 1}): {interaction_strength}")

def main() -> None:
    args = parse_args()

    domain_list = read_domain_list(args.domain_list)
    coordinate_mapping = read_coordinate_mapping(args.mapping)
    hic_matrix = read_hic_matrix(args.hic_matrix, coordinate_mapping)

    if args.out_filtered_domain_list_combined:
        domain_list_combined, removed_combined, breakdown = filter_domains_combined(
            domain_list, hic_matrix, coordinate_mapping, resolution=args.resolution
        )
        if removed_combined:
            print(
                f"Combined filter: Removed {len(removed_combined)} domains "
                f"(unmapped start coords, zero interactions, or duplicate start coords).",
                file=sys.stderr,
            )
            print(
                f"  Breakdown: {breakdown['unmapped_start']} unmapped start coords, "
                f"{breakdown['zero_interactions']} zero interactions, "
                f"{breakdown['duplicate_start']} duplicate start coords.",
                file=sys.stderr,
            )
        write_domain_list(args.out_filtered_domain_list_combined, domain_list_combined)
        domain_list = domain_list_combined
    else:
        removed: List[Tuple[str, int, int]] = []
        if args.filter_domains_by_mapping_start or args.out_filtered_domain_list:
            domain_list, removed = filter_domains_by_mapping_start(domain_list, coordinate_mapping)
            if removed:
                print(
                    f"Filtered out {len(removed)} domains whose start coord was not present in mapping.",
                    file=sys.stderr,
                )

        if args.out_filtered_domain_list:
            write_domain_list(args.out_filtered_domain_list, domain_list)

        if args.out_filtered_domain_list_by_zero_interaction:
            domain_list_zero_filtered, removed_zero = filter_domains_by_zero_interaction(
                domain_list, hic_matrix, coordinate_mapping, resolution=args.resolution
            )
            if removed_zero:
                print(
                    f"Filtered out {len(removed_zero)} domains with 0.0 total domain-domain interaction.",
                    file=sys.stderr,
                )
            write_domain_list(args.out_filtered_domain_list_by_zero_interaction, domain_list_zero_filtered)

    if args.debug:
        log_debug_info(domain_list, coordinate_mapping, hic_matrix, resolution=args.resolution)

    write_domain_interactions_raw_and_3col(
        args.out_raw, args.out_3col, domain_list, hic_matrix, coordinate_mapping, resolution=args.resolution
    )

if __name__ == "__main__":
    main()
