import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from einops import rearrange
from typing import Tuple

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ClassificationHead(nn.Sequential):
    def __init__(self, output_size, dropout, n_classes):
        super().__init__()

        # Classification head with global average pooling
        self.dropout = dropout
        self.fc = nn.Sequential(
            nn.Linear(70*output_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x_l, x_r):
        x = torch.cat([x_l, x_r], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return out

class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = torch.matmul(adj, output)
        return output

class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def generate_cheby_adj(self, A, K, device):
        support = []
        for i in range(K):
            if i == 0:
                support.append(torch.eye(A.shape[1]))
                temp = torch.eye(A.shape[1])
                support.append(temp)
            elif i == 1:
                support.append(A)
            else:
                temp = torch.matmul(support[-1], A)
                support.append(temp)
        return support

    def forward(self, x, L):
        device = x.device
        adj = self.generate_cheby_adj(L, self.K, device)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i].to(device))
            else:
                result += self.gc1[i](x, adj[i].to(device))
        result = F.relu(result)
        return result


# DSHS
class DSHS(nn.Module):
    def __init__(self, input_size, batch_size, channel, num_out):
        super(DSHS, self).__init__()
        self.batch_size = batch_size
        xdim = [batch_size, channel, input_size]
        self.K = 2
        self.layer1 = Chebynet(xdim, self.K, num_out)
        self.BN_l = nn.BatchNorm1d(input_size)
        self.BN_r = nn.BatchNorm1d(input_size)
        self.A = nn.Parameter(torch.FloatTensor(int(channel / 2), int(channel / 2)))
        nn.init.xavier_normal_(self.A)

    def normalize_A(self, A, symmetry=False):
        A = F.relu(A)
        if symmetry:
            A = A + torch.transpose(A, 0, 1)
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        return L

    def forward(self, x_l, x_r):
        L = self.normalize_A(self.A)
        g_l = self.BN_l(x_l.transpose(1, 2)).transpose(1, 2)
        g_l = self.layer1(g_l, L)
        g_r = self.BN_r(x_r.transpose(1, 2)).transpose(1, 2)
        g_r = self.layer1(g_r, L)
        x_l = g_l
        x_r = g_r
        return x_l, x_r


class BiA(nn.Module):
    def __init__(self, T):
        super(BiA, self).__init__()
        self.T = T
        self.query_proj = nn.Linear(T, T)
        self.key_proj = nn.Linear(T, T)
        self.value_proj = nn.Linear(T, T)
        self.pre_attn_norm = nn.LayerNorm(T)
        self.post_attn_norm = nn.LayerNorm(T)

    def forward(self, x_l, x_r):
        bs, C, T = x_l.shape

        # LayerNorm before attention
        x_l_norm = self.pre_attn_norm(x_l)
        x_r_norm = self.pre_attn_norm(x_r)

        # Project x_l to query and x_r to key
        q = self.query_proj(x_l_norm)
        k = self.key_proj(x_r_norm)

        # Compute attention scores (c, c)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.T)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Project (x_l - x_r) to value
        v = self.value_proj(x_l_norm - x_r_norm)

        # Apply attention to value
        r_l = torch.matmul(attn_scores, v)
        r_r = torch.matmul(attn_scores.transpose(1, 2), v)

        # LayerNorm after attention
        r_l = self.post_attn_norm(r_l)
        r_r = self.post_attn_norm(r_r)

        return r_l, r_r


class CosineRouter(nn.Module):
    def __init__(self, time_dim, channels, expert_dim, top_k=1):
        super().__init__()
        self.c = channels
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.expert_centers = nn.Parameter(torch.randn(channels, expert_dim))
        self.projection = nn.Linear(time_dim * 2, expert_dim)

    def forward(self, x):
        bs, num_channels, _ = x.shape
        x_proj = self.projection(x)

        # Compute cosine similarity
        x_proj = F.normalize(x_proj, p=2, dim=-1)
        expert_centers = F.normalize(self.expert_centers, p=2, dim=-1)
        cosine_sim = torch.matmul(x_proj, expert_centers.T)

        # Select top-k experts for each channel
        topk_probs, topk_indices = torch.topk(cosine_sim, self.top_k, dim=-1)
        topk_probs = F.softmax(topk_probs, dim=-1)

        return topk_probs, topk_indices


# SHMoE
class SHMoE(nn.Module):
    def __init__(self, channels, time_dim, expert_dim=32, top_k=1):
        super().__init__()
        self.c = channels
        self.t = time_dim
        self.top_k = top_k
        self.router = CosineRouter(time_dim, channels, expert_dim, top_k)
        self.experts = nn.ModuleList([
            nn.Linear(time_dim, time_dim)
            for _ in range(channels)
        ])

    def forward(self, x_l, x_r):
        # Concatenate left and right brain signals for processing (bs, 2c, t)
        x = torch.cat([x_l, x_r], dim=2)
        bs, c, t = x.shape
        output_l = torch.zeros_like(x_l)
        output_r = torch.zeros_like(x_r)

        # Get gating weights from Cosine Router
        topk_probs, topk_indices = self.router(x)
        unique_values = torch.unique(topk_indices)

        # Iterate over experts
        for expert_idx in unique_values:
            mask = (topk_indices == expert_idx).any(dim=-1).unsqueeze(-1).float()
            if mask.sum() == 0:  # Skip if no channel selects this expert
                continue

            # Select batch and channel using mask
            selected_signals_l = x_l * mask
            selected_signals_r = x_r * mask

            # Find matching expert probabilities
            selected_probs = topk_probs * mask
            selected_probs = selected_probs.sum(dim=-1, keepdim=True)

            # Process selected signals with expert
            expert_output_l = self.experts[expert_idx](selected_signals_l)
            expert_output_r = self.experts[expert_idx](selected_signals_r)

            # Weighted accumulation
            weighted_output_l = selected_probs * expert_output_l
            weighted_output_r = selected_probs * expert_output_r

            # Accumulate results
            output_l += weighted_output_l
            output_r += weighted_output_r

        # Residual connection
        output_l = output_l + x_l
        output_r = output_r + x_r

        # Return left and right outputs
        return output_l, output_r

class BiDGNBlock(nn.Module):
    def __init__(self, input_size, batch_size, channel, output_size):
        super(BiDGNBlock, self).__init__()
        self.gmoe = SHMoE(channels=channel, time_dim=output_size, top_k=2)
        self.bima = CHA(input_size, output_size, num_heads=1, dropout=0.1)

    def forward(self, x_l, x_r):
        x_l, x_r = self.bima(x_l, x_r)
        x_l, x_r = self.gmoe(x_l, x_r)
        return x_l, x_r

class ShareLink(nn.Module):
    def __init__(self, input_size, batch_size, channel, output_size, dropout, n_classes, num_blocks=2):
        super(ShareLink, self).__init__()
        self.blocks = nn.ModuleList([
            BiDGNBlock(input_size, batch_size, channel, output_size)
            for _ in range(num_blocks)
        ])
        self.classifier = ClassificationHead(output_size, dropout, n_classes)
        self.bidgn = DSHS(input_size, batch_size, channel, output_size)

    def forward(self, x_l, x_r):
        x_l, x_r = self.bidgn(x_l, x_r)
        for block in self.blocks:
            x_l, x_r = block(x_l, x_r)
        out = self.classifier(x_l, x_r)
        return out

# CHA
class CHA(nn.Module):
    def __init__(self, input_size, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.query_proj = nn.Linear(emb_size, emb_size)
        self.key_proj = nn.Linear(emb_size, emb_size)
        self.value_proj = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.pre_attn_norm = nn.LayerNorm(emb_size)
        self.post_attn_norm = nn.LayerNorm(emb_size)
        self.res_connect = nn.Linear(emb_size, emb_size)

    def forward(self, x_l: Tensor, x_r: Tensor) -> Tuple[Tensor, Tensor]:
        # Project x_l to query and x_r to key
        queries = rearrange(self.query_proj(x_l), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.key_proj(x_r), "b n (h d) -> b h n d", h=self.num_heads)

        # Compute attention scores
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        scaling = self.head_dim ** (1 / 2)
        attn_scores = F.softmax(energy / scaling, dim=-1)
        attn_scores = self.att_drop(attn_scores)

        # Project (x_l - x_r) to value
        value_diff = self.value_proj(x_l - x_r)
        values = rearrange(value_diff, "b n (h d) -> b h n d", h=self.num_heads)

        # Apply attention to value
        out_l = torch.einsum('bhal, bhlv -> bhav', attn_scores, values)
        out_r = torch.einsum('bhal, bhlv -> bhav', attn_scores.transpose(2, 3), values)

        # Rearrange and project
        out_l = rearrange(out_l, "b h n d -> b n (h d)")
        out_r = rearrange(out_r, "b h n d -> b n (h d)")

        out_l = self.projection(out_l)
        out_r = self.projection(out_r)

        x_l = self.res_connect(x_l)
        x_r = self.res_connect(x_r)

        out_l = out_l + x_l
        out_r = out_r + x_r

        return out_l, out_r

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

if __name__ == '__main__':
    # Example usage
    input_size = 5  # Example input size
    batch_size = 8  # Example batch size
    channel = 64    # Example channel number
    output_size = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize Block
    model = ShareLink(input_size, batch_size, channel, output_size, 0.1, 4, num_blocks=1)

    # Generate random input data
    x_l = torch.randn(batch_size, int(channel/2), input_size).to(device)
    x_r = torch.randn(batch_size, int(channel/2), input_size).to(device)

    # Forward pass
    out = model(x_l, x_r)

    # Print output
    print("Output shape:", x_l.shape, x_r.shape)
