import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GATConv

class UniversalHiCGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GATConv(512, 512)
        self.densea = nn.Linear(512, 256)
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 3) 

    def _ensure_edge_index(self, adj):
        """Converts dense adjacency matrix to sparse edge_index if necessary."""
        if adj.dim() == 2 and adj.size(0) == adj.size(1):
            # Convert dense [N, N] float matrix to [2, E] long indices
            return adj.nonzero().t().contiguous().long()
        return adj.long()

    def forward(self, x, adj):
        edge_index = self._ensure_edge_index(adj)
        
        # 1. GNN Layer
        x = self.conv(x, edge_index)
        x = F.relu(x)
        
        # 2. Deep Bottleneck
        x = F.relu(self.densea(x))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x) 
        
        return torch.cdist(x, x, p=2)

    def get_structure(self, x, adj):
        edge_index = self._ensure_edge_index(adj)
        
        x = F.relu(self.conv(x, edge_index))
        x = F.relu(self.densea(x))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.dense3(x)
