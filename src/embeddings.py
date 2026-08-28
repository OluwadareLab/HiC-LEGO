import numpy as np
import networkx as nx
import torch
from line import train_line
from utils import set_seed, resolve_torch_device

def generate_embeddings(norm_matrix, dimension=512, device="cuda"):
    set_seed(42)
    device = resolve_torch_device(device)
    print(f" -> Generating Embeddings using LINE (Dim: {dimension}) on {device}...")

    import scipy.sparse as sp

    if not sp.issparse(norm_matrix):
        adj_sparse = sp.coo_matrix(norm_matrix)
    else:
        adj_sparse = norm_matrix.tocoo()

    embeddings = train_line(
        adj_sparse,
        embedding_dim=dimension,
        device=device,
        epochs=100,
        batch_size=4096,
        neg_ratio=5
    )

    return embeddings
