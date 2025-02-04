import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Tuple, Optional


class GNN(nn.Module):
    """Graph Neural Network using GraphSAGE convolutions."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        return F.leaky_relu(x)

class xtrimoEmbedding(nn.Module):
    """Learn embeddings for genes using attention mechanism."""
    def __init__(self, b: int, d: int, node2vec: Optional[torch.Tensor] = None):
        """
        Args:
            b: Input dimension (number of genes)
            d: Output embedding dimension
            node2vec: Pre-trained node2vec embeddings (optional)
        """
        super(GeneEmbedding, self).__init__()
        self.w1 = nn.Parameter(torch.randn(1, b))  # 1xb
        self.w2 = nn.Parameter(torch.randn(b, b))  # bxb
        self.alpha = nn.Parameter(torch.randn(b))  # b
        self.T = nn.Parameter(torch.randn(b, d))  # bxd
        self.register_buffer('G', node2vec if node2vec is not None else torch.zeros(b, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention mechanism
        a = F.leaky_relu(x @ self.w1.expand(x.size(1), self.w1.size(1)))  # bx1
        z = a @ self.w2 + self.alpha * a  # bxb

        # Compute attention weights
        gamma = F.softmax(z, dim=1)  # bxb

        # Compute final embeddings
        E = torch.matmul(gamma, self.T)  # bxd
        xtrimo_embeddings = E + self.G  # Add node2vec embeddings

        return xtrimo_embeddings

class MLP(nn.Module):
    """Multi-layer Perceptron for binary classification."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)



class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    def __init__(self, input_dim: int, emb_size: int):
        super().__init__()
        self.query = nn.Linear(input_dim, emb_size)
        self.key = nn.Linear(input_dim, emb_size)
        self.value = nn.Linear(input_dim, emb_size)
        self.linear_o1 = nn.Linear(emb_size, emb_size)
        self.scale = emb_size ** -0.5
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        self.attention_weights = F.softmax(attention_scores, dim=-1)

        return torch.matmul(self.attention_weights, V)

class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout."""
    def __init__(self, input_dim: int, emb_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_size),
            nn.LayerNorm(emb_size),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + residual

class VAE(nn.Module):
    """Variational Autoencoder with self-attention and residual blocks."""
    def __init__(self, input_dim: int, emb_size: int, num_topics: int, block_num: int = 1):
        super().__init__()
        self.attention1 = SelfAttention(input_dim=input_dim, emb_size=emb_size)
        self.residual_block1 = ResidualBlock(input_dim, emb_size)
        self.blocks = nn.ModuleList([
            ResidualBlock(emb_size, emb_size) for _ in range(block_num)
        ])
        self.mu = nn.Linear(emb_size, num_topics, bias=False)
        self.log_sigma = nn.Linear(emb_size, num_topics, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.attention1(x)
        h = self.residual_block1(x, residual)

        for block in self.blocks:
            h = block(h, residual)
            residual = h

        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        kl_theta = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp(), dim=-1).mean()

        return mu, log_sigma, kl_theta

class LDEC(nn.Module):
    """Linear Decoder with optional GNN transformation and parameter sharing."""
    def __init__(self, num_modality: int, emb_size: int, num_topics: int, batch_size: int,
                 shared_module: Optional[nn.Module] = None):
        super().__init__()

        # Create or use shared alpha transformation
        if shared_module is not None:
            self.register_module('alphas', shared_module)
        else:
            alphas = nn.Sequential(
                nn.Linear(emb_size, num_topics, bias=False),
                nn.LeakyReLU(),
                nn.Dropout(p=0.1),
            )
            self.register_module('alphas', alphas)

        # These parameters remain independent for each decoder
        self.rho = nn.Parameter(torch.randn(num_modality, emb_size))
        self.batch_bias = nn.Parameter(torch.randn(batch_size, num_modality))
        self.beta = None

    def forward(self, theta: torch.Tensor, matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        matrix = self.rho if matrix is None else matrix
        beta = F.softmax(self.alphas(matrix), dim=0).transpose(1, 0)
        self.beta = beta
        res = torch.mm(theta, beta)
        return torch.log(res + 1e-6), self.rho if matrix is None else matrix