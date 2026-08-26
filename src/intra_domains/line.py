import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- Added this import
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset

class LINEModel(nn.Module):
    """
    PyTorch implementation of LINE (Second Order Proximity).
    Conceptually similar to Skip-gram with Negative Sampling.
    """
    def __init__(self, num_nodes, embedding_dim=128):
        super(LINEModel, self).__init__()
        self.context_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.vertex_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # Initialize weights small and random
        # nn.init.xavier_uniform_(self.context_embeddings.weight)
        # nn.init.xavier_uniform_(self.vertex_embeddings.weight)

        # Initialize with 0 mean and small std dev (Word2Vec style)
        # This helps convergence compared to Xavier uniform for embeddings
        nn.init.normal_(self.context_embeddings.weight, std=0.01)
        nn.init.normal_(self.vertex_embeddings.weight, std=0.01)

    def forward(self, target_nodes, context_nodes, neg_nodes):
        # Target: (batch_size, dim)
        v = self.vertex_embeddings(target_nodes)
        
        # Context (Positive Neighbors): (batch_size, dim)
        u_pos = self.context_embeddings(context_nodes)
        
        # Negative Samples: (batch_size, k, dim)
        u_neg = self.context_embeddings(neg_nodes)
        
        # --- Positive Score (Maximize dot product) ---
        # log sigmoid(v . u_pos)
        pos_score = torch.mul(v, u_pos).sum(dim=1)
        # pos_loss = F.logsigmoid(pos_score).sum()  # <--- FIXED: used F.logsigmoid
        pos_loss = F.logsigmoid(pos_score) 
       
        # --- Negative Score (Minimize dot product -> Maximize negative dot) ---
        # log sigmoid(-v . u_neg)
        v_unsqueezed = v.unsqueeze(1) # (batch, 1, dim)
        neg_score = torch.mul(v_unsqueezed, u_neg).sum(dim=2) # (batch, k)
        # neg_loss = F.logsigmoid(-neg_score).sum() # <--- FIXED: used F.logsigmoid
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1) # Sum over K negatives

        # Total loss (Maximize Likelihood -> Minimize Negative Log Likelihood)
        return -(torch.mean(pos_loss + neg_loss))

def train_line(adjacency_matrix, embedding_dim=512, device="cuda", epochs=50, batch_size=4096, neg_ratio=5):
    """
    Main function to train LINE embeddings.
    """
    # Fall back to CPU when CUDA was requested but is unavailable.
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
    else:
        device_str = str(device) if device is not None else "cuda"
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(device_str)

    num_nodes = adjacency_matrix.shape[0]
    
    # 1. Prepare Data
    # Pre-calculate sampling probability for edges
    rows, cols = adjacency_matrix.nonzero()
    weights = adjacency_matrix.data
    probs = weights / weights.sum()
    
    # 2. Pre-calculate Negative Probabilities (Degree^0.75) ---
    # Sum weights for each node to get 'degree' (strength)
    node_degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    # Avoid zero-degree nodes crashing the power calc
    node_degrees[node_degrees == 0] = 0.001 
    # The Magic Formula from the LINE paper
    neg_probs = np.power(node_degrees, 0.75)
    neg_probs = neg_probs / neg_probs.sum()
    
    model = LINEModel(num_nodes, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # LINE often likes higher LR

    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)
    
    # Pre-convert to tensor for fast sampling on GPU
    try:
        edges_tensor = torch.tensor(np.column_stack((rows, cols)), dtype=torch.long, device=device)
        probs_tensor = torch.tensor(probs, dtype=torch.float, device=device)
        neg_probs_tensor = torch.tensor(neg_probs, dtype=torch.float, device=device) # <--- NEW
    except:
        # Fallback to CPU if VRAM issue (unlikely)
        edges_tensor = torch.tensor(np.column_stack((rows, cols)), dtype=torch.long, device="cpu")
        probs_tensor = torch.tensor(probs, dtype=torch.float, device="cpu")
        neg_probs_tensor = torch.tensor(neg_probs, dtype=torch.float, device="cpu")

    # Quiet: skip "[LINE] Training for N epochs..." console spam

    num_edges = len(rows)
    batches_per_epoch = max(1, num_edges // batch_size)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
            
            # A. Sample Positive Edges (Weighted)
            idx = torch.multinomial(probs_tensor, batch_size, replacement=True)
            batch_edges = edges_tensor[idx]
            
            target_nodes = batch_edges[:, 0].to(device)
            context_nodes = batch_edges[:, 1].to(device)
            
            # B. Sample Negative Edges (Uniform)
            # neg_nodes = torch.randint(0, num_nodes, (batch_size, neg_ratio), device=device)

            # B. Sample Negative Edges (Corrected: Weighted by Degree^0.75)
            # We need (batch_size * neg_ratio) samples
            neg_indices = torch.multinomial(neg_probs_tensor, batch_size * neg_ratio, replacement=True)
            neg_nodes = neg_indices.view(batch_size, neg_ratio) # Reshape to (batch, k)

            # C. Optimize
            loss = model(target_nodes, context_nodes, neg_nodes)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()

    return model.vertex_embeddings.weight.data.cpu().numpy()
