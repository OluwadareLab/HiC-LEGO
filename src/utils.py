import torch
import numpy as np
import random
import os
import sys
from scipy.sparse import csr_matrix
from line import train_line


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def resolve_torch_device(device="cuda"):
    if isinstance(device, torch.device):
        requested = device
    else:
        requested = torch.device(str(device) if device is not None else "cuda")
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def prepare_tensors(norm_matrix, embeddings, device):
    adj = torch.tensor(norm_matrix, dtype=torch.float32)
    adj = adj + torch.eye(adj.shape[0])
    
    degree = adj.sum(dim=1)
    degree_inv = degree.pow(-1)
    degree_inv[torch.isinf(degree_inv)] = 0
    d_inv = torch.diag(degree_inv)
    
    norm_adj_tensor = torch.matmul(d_inv, adj).to(device)
    feat_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    return feat_tensor, norm_adj_tensor

def contact_to_distance(contacts, alpha):
    c = torch.tensor(contacts, dtype=torch.float32) + 1e-8
    d = (1.0 / c) ** alpha
    d.fill_diagonal_(0)
    d = d / torch.max(torch.nan_to_num(d, posinf=0))
    return d

def write_pdb(positions, pdb_file):
    positions = positions * 100 
    
    with open(pdb_file, "w") as o_file:
        o_file.write("\n")
        bin_num = len(positions)
        for i in range(1, bin_num+1):
            line = "ATOM  %5d  CA  MET A%4d    %8.3f%8.3f%8.3f  1.00  0.00           C\n" % (
                i, i, positions[i-1][0], positions[i-1][1], positions[i-1][2]
            )
            o_file.write(line)
        o_file.write("END")


def generate_nxe_matrix(normed_nxn_matrix, embedding_size=512, batch_size=4096, epochs=50):
    sparse_matrix = csr_matrix(normed_nxn_matrix)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    embeddings = train_line(
        sparse_matrix, 
        embedding_dim=embedding_size, 
        device=device, 
        epochs=epochs, 
        batch_size=batch_size
    )
    return embeddings

def cont2dist(contacts, alpha):
    return contact_to_distance(contacts, alpha)

def WritePDB(positions, pdb_file):
    return write_pdb(positions, pdb_file)

def load_input(norm_matrix, embeddings):
    class DataObj:
        def __init__(self, x, y, edge_index):
            self.x = x
            self.y = y
            self.edge_index = edge_index
            
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_tensor, _ = prepare_tensors(norm_matrix, embeddings, device)
    
    adj_sparse = csr_matrix(norm_matrix)
    rows, cols = adj_sparse.nonzero()
    edge_index = torch.tensor(np.array([rows, cols]), dtype=torch.long).to(device)
    
    y_tensor = torch.tensor(norm_matrix, dtype=torch.float32).to(device)
    
    return DataObj(feat_tensor, y_tensor, edge_index)


def load_coordinate_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[int(parts[0])] = int(parts[1])
    return mapping

def load_dense_hic_memmap(npy_path):
    return np.load(npy_path, mmap_mode='r')

def load_domains(domain_file, removed_indices=None):
    domains = []
    with open(domain_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                domains.append((parts[0], int(parts[1]), int(parts[2])))
    
    if removed_indices:
        domains = [d for i, d in enumerate(domains, 1) if i not in removed_indices]
    return domains

def generate_nxn_matrix(hic_data, mapping, start, end):
    bin_indices = [idx for coord, idx in mapping.items() if start <= coord < end]
    if not bin_indices:
        return np.zeros((1, 1))
    
    s_idx, e_idx = min(bin_indices), max(bin_indices) + 1
    return hic_data[s_idx:e_idx, s_idx:e_idx]

def normalize_nxn_matrix(matrix):
    if np.max(matrix) == 0:
        return matrix
    return matrix / np.max(matrix)

def domain_alignment(expanded_embeddings, local_embeddings):
    return local_embeddings

def expand_embeddings(matrix_sizes, global_embeddings):
    return global_embeddings 

def load_1xE_embeddings(embedding_file):
    return np.loadtxt(embedding_file)
