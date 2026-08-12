import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GNNEncoder(nn.Module):
    """
    Graph Attention Network encoder for chiplet netlist graphs.
    Input: node features + edge features
    Output: graph-level embedding
    """

    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=256, n_heads=4, n_layers=3):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim // n_heads
            self.convs.append(
                GATConv(in_dim, out_dim, heads=n_heads, edge_dim=edge_feat_dim, concat=True)
            )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = F.relu(self.node_proj(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr=edge_attr)) + x
        x = self.norm(x)
        x = global_mean_pool(x, batch)
        return F.relu(self.out_proj(x))
