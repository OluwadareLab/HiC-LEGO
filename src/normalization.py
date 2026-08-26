import numpy as np
import scipy.sparse as sp

def kr_normalization(adj_matrix, tol=1e-6, max_iter=100):
    """
    Python implementation of Knight-Ruiz (KR) normalization.
    Replaces the dependency on Rscript.
    """
    # Ensure it's a symmetric matrix
    adj_matrix = np.array(adj_matrix, dtype=float)
    n = adj_matrix.shape[0]
    
    # Handle disconnected nodes (rows/cols with all zeros)
    row_sums = np.sum(adj_matrix, axis=1)
    mask = row_sums > 0
    
    # Submatrix of connected nodes
    A = adj_matrix[np.ix_(mask, mask)]
    
    # KR Algorithm: Balancing the matrix to be doubly stochastic
    x = np.ones((A.shape[0], 1))
    for _ in range(max_iter):
        r = np.dot(A, x)
        x_next = 1.0 / r
        
        # Check convergence
        if np.linalg.norm(x_next - x) < tol:
            x = x_next
            break
        x = x_next
        
    D = np.diag(x.flatten())
    A_norm = np.dot(np.dot(D, A), D)
    
    # Map back to original size
    norm_full = np.zeros_like(adj_matrix)
    norm_full[np.ix_(mask, mask)] = A_norm
    
    return norm_full

def preprocess_hic_file(filepath):
    """Reads HiC file, converts to matrix, and normalizes."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # If 3-column format (COO), convert to Matrix
    if data.shape[1] == 3:
        rows = data[:, 0].astype(int)
        cols = data[:, 1].astype(int)
        vals = data[:, 2]

        # Map indices to 0..N
        unique_nodes = np.unique(np.concatenate([rows, cols]))
        node_map = {node: i for i, node in enumerate(unique_nodes)}

        n = len(unique_nodes)
        mat = np.zeros((n, n))

        for r, c, v in zip(rows, cols, vals):
            if not np.isfinite(v):
                continue
            i, j = node_map[r], node_map[c]
            mat[i, j] = v
            mat[j, i] = v  # Symmetric

        return mat, unique_nodes

    return data, None
