import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from scipy.linalg import orthogonal_procrustes
import os
from line import train_line

def generate_nxe_matrix(normed_nxn_matrix, embedding_size=512, batch_size=4096, epochs=50):
    from scipy.sparse import csr_matrix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sparse_matrix = csr_matrix(normed_nxn_matrix)
    return train_line(sparse_matrix, embedding_dim=embedding_size, device=device, epochs=epochs, batch_size=batch_size)

def kr_norm(matrix, max_iter=100, tol=1e-6, epsilon=1e-10, small_constant=1e-10):
    matrix = np.nan_to_num(matrix, nan=0.0)
    n = matrix.shape[0]
    x = np.ones(n)
    matrix += small_constant
    for _ in range(max_iter):
        x_old = x.copy()
        matrix_with_epsilon = matrix + epsilon
        x = 1 / np.dot(matrix_with_epsilon, 1 / np.dot(matrix_with_epsilon, x))
        if np.linalg.norm(x - x_old, 1) < tol:
            break
    d = np.diag(x)
    return np.dot(d, np.dot(matrix, d))

def detect_hic_format(hic_file):
    with open(hic_file, 'r') as f:
        line = f.readline().strip()
        if not line: return 'dense'
        return 'sparse' if len(line.split()) == 3 else 'dense'

def load_hic_matrix_sparse_to_dense(hic_file, coordinate_mapping):
    max_idx = max(coordinate_mapping.values())
    mat = np.zeros((max_idx + 1, max_idx + 1))
    with open(hic_file, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 3: continue
            try:
                i, j, v = coordinate_mapping[int(p[0])], coordinate_mapping[int(p[1])], float(p[2])
                mat[i, j] = mat[j, i] = v
            except: continue
    return mat

def load_hic_matrix(hic_file, coordinate_mapping=None):
    if detect_hic_format(hic_file) == 'sparse':
        return load_hic_matrix_sparse_to_dense(hic_file, coordinate_mapping)
    return np.loadtxt(hic_file)

def load_domains(domain_file, removed_indices=None):
    domains = []
    with open(domain_file, 'r') as f:
        for line in f:
            chrom, start, end = line.strip().split()
            domains.append((chrom, int(start), int(end)))
    if removed_indices:
        domains = [d for idx, d in enumerate(domains) if idx not in removed_indices]
    return domains

def load_coordinate_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            p, b = map(int, line.strip().split())
            mapping[p] = b
    return mapping

def generate_nxn_matrix(hic_data, coordinate_mapping, domain_start, domain_end):
    bin_indices = []
    for pos in range(domain_start, domain_end):
        bin_index = coordinate_mapping.get(pos)
        if bin_index is not None:
            bin_indices.append(bin_index)
    if not bin_indices: return np.zeros((0,0))
    idx = np.array(bin_indices, dtype=int)
    return hic_data[np.ix_(idx, idx)]

def generate_coordinate_mapping(nxn_matrix, global_mapping, domain_start, domain_end, mapping_file):
    domain_mapping = {coord: idx for coord, idx in global_mapping.items() if domain_start <= coord < domain_end}

    sorted_domain_coords = sorted(domain_mapping.keys())

    with open(mapping_file, 'w') as map_file:
        for i, coord in enumerate(sorted_domain_coords):
            map_file.write(f'{coord}\t{global_mapping[coord]}\n')

def load_dense_hic_memmap(dense_npy_path):
    return np.load(dense_npy_path, mmap_mode='r')

def normalize_nxn_matrix(nxn_matrix):
    return kr_norm(nxn_matrix)

def load_1xE_embeddings(embedding_file):
    return np.loadtxt(embedding_file)

def expand_embeddings(matrix_sizes, embeddings):
    expanded = []
    for i, size in enumerate(matrix_sizes):
        expanded.append(np.tile(embeddings[i], (size, 1)))
    return np.vstack(expanded)

def domain_alignment(embeddings1, embeddings2):
    transform = orthogonal_procrustes(embeddings1, embeddings2)[0]
    return np.matmul(embeddings2, transform)

def WritePDB(positions, pdb_file):
    with open(pdb_file, "w") as f:
        f.write("\n")
        for i, pos in enumerate(positions, 1):
            f.write("ATOM  %5d  CA  MET A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n" % (i, i, pos[0], pos[1], pos[2]))
        f.write("END")

def load_input(adj_mat, features):
    np.fill_diagonal(adj_mat, 0)

    truth = torch.tensor(adj_mat, dtype=torch.double)

    graph = nx.from_numpy_array(adj_mat)

    edges = list(graph.edges(data=True))
    if len(edges) > 0:
        edge_index = torch.tensor([[e[0], e[1]] for e in edges], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([e[2].get('weight', 1.0) for e in edges], dtype=torch.float)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.float)

    node_attr = torch.tensor(features, dtype=torch.float)

    data = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=truth)

    return data

def cont2dist(adj, factor):
    adj_stable = torch.clamp(adj, min=1e-12)

    dist = (1/adj_stable)**factor
    dist.fill_diagonal_(0)
    mx = torch.max(torch.nan_to_num(dist, posinf=0))
    dist = torch.nan_to_num(dist, posinf=mx)
    return dist / (mx if mx > 0 else 1.0)
