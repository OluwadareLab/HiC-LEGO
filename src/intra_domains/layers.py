import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class GATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, root_weight=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        # Handle tuple input for compatibility
        real_in = in_channels[0] if isinstance(in_channels, tuple) else in_channels
        
        self.lin_l = nn.Linear(real_in, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = nn.Linear(real_in, out_channels, bias=False)

    def forward(self, x, edge_index):
        # 1. Aggregate neighbors: Standard PyG message passing
        # This replaces the need for the SparseTensor matmul
        out = self.propagate(edge_index, x=x)

        # 2. Manual Normalization (Matches your 'adjust_weights' logic)
        # Calculates 1/degree for each node
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        
        # Apply normalization to the aggregated features
        out = out * deg_inv.view(-1, 1)

        # 3. Apply Linear Transformations
        out = self.lin_l(out)
        
        if self.root_weight:
            out += self.lin_r(x)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j):
        return x_j  # Standard summation aggregation