import numpy as np
import os
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import torch
from models import UniversalHiCGNN
from torch.nn import MSELoss
from torch.optim import Adam
import torch.optim as optim
import torch.nn as nn
from torch import cdist
import shutil
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import utils
import copy
import argparse
from multiprocessing import Pool
from functools import partial
from pathlib import Path

_QUIET_LOG = True

def _log(*args, **kwargs):
    flush = kwargs.pop("flush", True)
    force = kwargs.pop("force", False)
    kwargs.pop("sep", None)
    kwargs.pop("end", None)
    kwargs.pop("file", None)

    msg = " ".join(str(a) for a in args)
    if force or not _QUIET_LOG:
        print(msg, flush=flush)
        return

    m = msg.strip()
    if not m:
        return

    mu = m.upper()
    if "[ERROR]" in mu or mu.startswith("ERROR"):
        print(m, flush=flush)
        return
    if "[WARN" in mu or "[WARNING]" in mu or mu.startswith("WARNING") or "WARNING:" in mu:
        print(m, flush=flush)
        return

    step_prefixes = (
        "STEP ",
        "INTRA-DOMAIN STRUCTURE GENERATION",
        "DOMAIN STRUCTURES COMPLETE",
    )
    mu_lstrip = m.lstrip().upper()
    if any(mu_lstrip.startswith(pref) for pref in step_prefixes):
        if set(m.replace(" ", "")) <= {"=", "-", "*"}:
            return
        print(m, flush=flush)

def train_and_save_model(
    normed_untrained,
    aligned_embeddings,
    domain_folder,
    name_trained,
    pretrained_model_path,
    lr=0.001,
    thresh=1e-4,
    conversions=[0.1, 0.1, 2]
):

    temp_spear = []
    temp_models = []
    temp_mse = []
    model_list = []

    if len(conversions) == 3:
        low, step, high = conversions
        conversions = list(np.arange(low, high + step, step))
        conversions = [round(c, 10) for c in conversions]
        _log(f'Generated conversion factors: {conversions}')
    elif len(conversions) == 1:
        conversions = [conversions[0]]
        _log(f'Using single conversion factor: {conversions}')
    else:
        raise ValueError('Invalid conversion input. Provide either [conversion] or [low, step, high].')

    data_untrained_fit = utils.load_input(normed_untrained, aligned_embeddings)

    for conversion in conversions:
        _log(f'\nTraining model using conversion value {conversion}.')

        model = UniversalHiCGNN()
        model.load_state_dict(torch.load(pretrained_model_path))
        model = model.to(data_untrained_fit.x.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        oldloss = float('inf')
        lossdiff = float('inf')

        truth = utils.cont2dist(data_untrained_fit.y, conversion).float().to(data_untrained_fit.x.device)

        idx = torch.triu_indices(truth.size(0), truth.size(1), offset=1)

        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()

            coords = model.get_structure(data_untrained_fit.x.float(), data_untrained_fit.edge_index)

            dist_out = torch.cdist(coords, coords)

            dist_pred = dist_out[idx[0], idx[1]]
            dist_truth_subset = truth[idx[0], idx[1]].detach()

            loss = criterion(dist_pred, dist_truth_subset)

            lossdiff = abs(oldloss - loss.item())
            oldloss = loss.item()

            loss.backward()
            optimizer.step()

            _log(f'Loss: {loss.item():.6f}', end='\r')

        _log(f'\nFinished training for conversion factor {conversion}. Final Loss: {loss.item():.6f}')

        model.eval()
        with torch.no_grad():
            coords = model.get_structure(data_untrained_fit.x.float(), data_untrained_fit.edge_index)
            dist_out = torch.cdist(coords, coords)
            dist_pred = dist_out[idx[0], idx[1]].cpu().numpy()
            dist_truth = truth[idx[0], idx[1]].cpu().numpy()
            SpRho = spearmanr(dist_truth, dist_pred)[0]

        _log(f'Spearman Correlation Coefficient (dSCC): {SpRho:.4f}')

        temp_spear.append(SpRho)
        temp_models.append(coords.cpu())
        temp_mse.append(loss.item())
        model_list.append(copy.deepcopy(model))

    best_idx = np.argmax(temp_spear)
    repmod = temp_models[best_idx]
    repspear = temp_spear[best_idx]
    repmse = temp_mse[best_idx]
    repconv = conversions[best_idx]
    repnet = model_list[best_idx]

    _log(f'\nOptimal conversion factor: {repconv}')
    _log(f'Optimal dSCC: {repspear:.4f}')
    _log(f'Final MSE loss: {repmse:.6f}')

    os.makedirs(domain_folder, exist_ok=True)

    log_path = os.path.join(domain_folder, f"{name_trained}_log.txt")
    with open(log_path, 'w') as f:
        f.write(f'Optimal conversion factor: {repconv}\n')
        f.write(f'Optimal dSCC: {repspear}\n')
        f.write(f'Final MSE loss: {repmse}\n')

    pdb_path = os.path.join(domain_folder, f"{name_trained}_structure.pdb")
    utils.WritePDB(repmod.detach().numpy() * 100, pdb_path)
    _log(f'Saved optimal structure to {pdb_path}')

def process_domain_first_loop(args):
    index, domain, hic_dense_npy, mapping_file, output_dir = args
    chrom, domain_start, domain_end = domain
    domain_id = f'domain{index}'
    _log(f'Processing {domain_id}...')
    domain_folder = os.path.join(output_dir, domain_id)
    os.makedirs(domain_folder, exist_ok=True)

    coordinate_mapping = utils.load_coordinate_mapping(mapping_file)
    hic_data = utils.load_dense_hic_memmap(hic_dense_npy)

    nxn_matrix = utils.generate_nxn_matrix(hic_data, coordinate_mapping, domain_start, domain_end)
    _log(f'{domain_id}: N x N matrix generated.')

    mapping_output_file = os.path.join(domain_folder, f'{domain_id}_coordinate_mapping.txt')
    utils.generate_coordinate_mapping(nxn_matrix, coordinate_mapping, domain_start, domain_end, mapping_output_file)

    _log(f'{domain_id}: Coordinate mapping saved to {mapping_output_file}')

    normed_nxn_matrix = utils.normalize_nxn_matrix(nxn_matrix)
    _log(f'{domain_id}: N x N matrix normalized.')

    if np.count_nonzero(normed_nxn_matrix) == 0:
        _log(f'{domain_id}: Warning: Normalized matrix is empty. Skipping generation of N x E matrix and training for this domain.')
        matrix_size = 0
        nxe_matrix = np.empty((0, 512))
        return (index, matrix_size, nxe_matrix, normed_nxn_matrix, domain_id, domain_folder)

    try:
        nxe_matrix = utils.generate_nxe_matrix(normed_nxn_matrix, embedding_size=512, batch_size=128, epochs=10)
        _log(f'{domain_id}: N x E matrix generated.')
    except ValueError as e:
        if "Empty training data" in str(e):
            _log(f'{domain_id}: Warning: Empty training data encountered during N x E matrix generation. Skipping training for this domain.')
            matrix_size = 0
            nxe_matrix = np.empty((0, 512))
            return (index, matrix_size, nxe_matrix, normed_nxn_matrix, domain_id, domain_folder)
        else:
            raise e

    matrix_size = nxe_matrix.shape[0]

    nxe_output_file = os.path.join(domain_folder, f'{domain_id}_nxe.txt')
    np.savetxt(nxe_output_file, nxe_matrix, fmt='%.6f')
    _log(f'{domain_id}: N x E matrix saved to {nxe_output_file}.')

    return (index, matrix_size, nxe_matrix, normed_nxn_matrix, domain_id, domain_folder)

def process_domain_second_loop(args):

    index, domain, matrix_size, nxe_matrix, normed_nxn_matrix, expanded_embeddings, matrix_sizes, output_dir, pretrained_model_path, bin_size = args
    chrom, domain_start, domain_end = domain
    domain_id = f'domain{index}'
    domain_folder = os.path.join(output_dir, domain_id)

    start_idx = sum(matrix_sizes[:index - 1])
    end_idx = start_idx + matrix_sizes[index - 1]
    domain_expanded_embeddings = expanded_embeddings[start_idx:end_idx]

    aligned_embeddings = utils.domain_alignment(domain_expanded_embeddings, nxe_matrix)

    aligned_output_file = os.path.join(domain_folder, f'{domain_id}_aligned_embedding.txt')
    np.savetxt(aligned_output_file, aligned_embeddings, fmt='%.6f')

    name_trained = f'{domain_id}_trained'
    normed_untrained = normed_nxn_matrix
    train_and_save_model(normed_untrained, aligned_embeddings, domain_folder, name_trained, pretrained_model_path)

    try:
        expected_bins = (domain_end - domain_start) // bin_size
        hic_bins_present = int(normed_untrained.shape[0])
        log_path = os.path.join(domain_folder, f"{name_trained}_log.txt")
        pdb_path = os.path.join(domain_folder, f"{name_trained}_structure.pdb")

        atom_count = 0
        if os.path.exists(pdb_path):
            with open(pdb_path, 'r') as pf:
                for line in pf:
                    if line.startswith('ATOM'):
                        atom_count += 1

        with open(log_path, 'a') as lf:
            lf.write("\n" + "=" * 80 + "\n")
            lf.write("DOMAIN INPUT/OUTPUT SUMMARY\n")
            lf.write("=" * 80 + "\n")
            lf.write(f"Domain: {chrom}:{domain_start}-{domain_end}\n")
            lf.write(f"Bin size: {bin_size} bp\n")
            lf.write(f"Expected bins (span-based): {expected_bins}\n")
            lf.write(f"Hi-C bins present (mapping-based): {hic_bins_present}\n")
            lf.write(f"Structure coordinates (atoms): {atom_count}\n")

            if atom_count == hic_bins_present:
                lf.write("Alignment status (Hi-C vs coordinates): MATCH\n")
            else:
                diff = atom_count - hic_bins_present
                status = 'EXPANDED' if diff > 0 else 'TRUNCATED'
                lf.write(f"Alignment status (Hi-C vs coordinates): {status} ({diff:+d})\n")

            if atom_count == expected_bins:
                lf.write("Legacy status (span-based expected vs coordinates): MATCH\n")
            else:
                diff2 = atom_count - expected_bins
                status2 = 'EXPANDED' if diff2 > 0 else 'TRUNCATED'
                lf.write(f"Legacy status (span-based expected vs coordinates): {status2} ({diff2:+d})\n")
    except Exception as e:
        _log(f"Warning: Failed to append domain summary for {domain_id}: {e}")

    expanded_output_file = os.path.join(domain_folder, f'{domain_id}_expanded.txt')
    np.savetxt(expanded_output_file, domain_expanded_embeddings, fmt='%.6f')

def main():
    parser = argparse.ArgumentParser(description='3D reconstruction pipeline for bioinformatics domains')
    parser.add_argument('--hic_file', type=str, required=True,
                        help='Path to Hi-C data file')
    parser.add_argument('--domain_file', type=str, required=True,
                        help='Path to domain file')
    parser.add_argument('--mapping_file', type=str, required=True,
                        help='Path to coordinate mapping file')
    parser.add_argument('--embedding_file', type=str, required=True,
                        help='Path to embedding file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help='Path to pretrained model weights file')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads/processes to use (default: 1)')
    parser.add_argument('--bin_size', type=int, default=5000,
                        help='Bin size in base pairs (default: 5000)')
    parser.add_argument(
        '--hic_dense_cache',
        type=str,
        default=None,
        help=(
            'Optional path to a cached dense Hi-C matrix saved as .npy. '
            'If not provided, the cache will be created under output_dir. '
            'Used to avoid repeated sparse->dense conversion per-domain.'
        ),
    )
    parser.add_argument(
        '--force_rebuild_hic_cache',
        action='store_true',
        help='Rebuild the dense Hi-C cache even if it already exists.',
    )
    parser.add_argument('--domain_removed_indices', type=str, default='',
                        help='Comma-separated list of domain indices to remove (e.g., "1,2,3")')

    args = parser.parse_args()

    if args.domain_removed_indices:
        domain_removed_indices = [int(x.strip()) for x in args.domain_removed_indices.split(',')]
    else:
        domain_removed_indices = []

    os.makedirs(args.output_dir, exist_ok=True)

    bin_size = args.bin_size

    output_dir = Path(args.output_dir)
    hic_dense_cache = Path(args.hic_dense_cache) if args.hic_dense_cache else (output_dir / 'hic_dense_cache.npy')

    _log("INTRA-DOMAIN STRUCTURE GENERATION")

    _log("STEP 1: Preparing Hi-C cache")
    if (not hic_dense_cache.exists()) or args.force_rebuild_hic_cache:
        _log(f"Preparing dense Hi-C cache at: {hic_dense_cache}")
        coordinate_mapping = utils.load_coordinate_mapping(args.mapping_file)
        dense_hic = utils.load_hic_matrix(args.hic_file, coordinate_mapping=coordinate_mapping)
        hic_dense_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(hic_dense_cache, dense_hic)
        _log(f"Dense Hi-C cache saved: {hic_dense_cache}")
    else:
        _log(f"Using existing dense Hi-C cache: {hic_dense_cache}")

    domains = utils.load_domains(args.domain_file, removed_indices=domain_removed_indices)
    embeddings = utils.load_1xE_embeddings(args.embedding_file)

    first_loop_args = [
        (index, domain, str(hic_dense_cache), args.mapping_file, args.output_dir)
        for index, domain in enumerate(domains, start=1)
    ]

    _log("STEP 2: Processing domains (embeddings)")
    matrix_sizes = [0] * len(domains)
    nxe_matrices = [None] * len(domains)
    normed_nxn_matrices = [None] * len(domains)

    if args.num_threads > 1:
        with Pool(processes=args.num_threads) as pool:
            results = pool.map(process_domain_first_loop, first_loop_args)
    else:
        results = [process_domain_first_loop(arg) for arg in first_loop_args]

    results.sort(key=lambda x: x[0])

    for index, matrix_size, nxe_matrix, normed_nxn_matrix, domain_id, domain_folder in results:
        idx = index - 1
        matrix_sizes[idx] = matrix_size
        nxe_matrices[idx] = nxe_matrix
        normed_nxn_matrices[idx] = normed_nxn_matrix

    expanded_embeddings = utils.expand_embeddings(matrix_sizes, embeddings)

    second_loop_args = [
        (index, domain, matrix_sizes[index - 1], nxe_matrices[index - 1],
         normed_nxn_matrices[index - 1], expanded_embeddings, matrix_sizes,
         args.output_dir, args.pretrained_model_path, bin_size)
        for index, domain in enumerate(domains, start=1)
    ]

    _log("STEP 3: Training domain structures")
    if args.num_threads > 1:
        with Pool(processes=args.num_threads) as pool:
            pool.map(process_domain_second_loop, second_loop_args)
    else:
        for arg in second_loop_args:
            process_domain_second_loop(arg)

    _log("DOMAIN STRUCTURES COMPLETE")

if __name__ == "__main__":
    main()

