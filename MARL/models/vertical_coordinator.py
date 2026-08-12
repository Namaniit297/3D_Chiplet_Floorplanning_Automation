import torch
import torch.nn as nn
import torch.nn.functional as F


class VerticalCoordinator(nn.Module):
    """
    Attention-based pooling over per-layer embeddings for cross-layer coordination.
    """

    def __init__(self, n_layers, layer_embed_dim, hidden_dim=256, n_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(layer_embed_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pool = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, layer_embeds):
        """
        layer_embeds: (batch, n_layers, layer_embed_dim)
        """
        x = F.relu(self.input_proj(layer_embeds))
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        # Mean pool over layers
        x = x.mean(dim=1)
        return F.relu(self.pool(x))
