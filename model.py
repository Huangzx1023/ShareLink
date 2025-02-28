import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential, GCNConv
from einops import rearrange

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

        # global average pooling
        # self.clshead = nn.Sequential(
        #     Reduce('b n e -> b e', reduction='mean'),
        #     nn.LayerNorm(emb_size),
        #     nn.Linear(emb_size, n_classes)
        # )
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
        # x = x_l-x_r
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
        # self.ac = torch.nn.ReLU(inplace=False)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = torch.matmul(adj, output)
        # output = F.relu(output)
        return output

class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()  # https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def generate_cheby_adj(self, A, K, device):
        support = []
        for i in range(K):
            if i == 0:
                support.append(torch.eye(A.shape[1]))  #torch.eye生成单位矩阵
                temp = torch.eye(A.shape[1])
                temp = temp
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


# 左右脑sharing structure DGCNN
class BiDGN(nn.Module):
    def __init__(self, input_size, batch_size, channel, num_out):
        super(BiDGN, self).__init__()
        self.batch_size = batch_size
        xdim = [batch_size, channel, input_size]
        self.K = 2
        self.layer1 = Chebynet(xdim, self.K, num_out)
        self.BN_l = nn.LayerNorm(num_out)  # 对第二维（第一维为batch_size)进行标准化
        self.BN_r = nn.LayerNorm(num_out)
        self.A = nn.Parameter(torch.FloatTensor(int(channel / 2), int(channel / 2)))
        nn.init.xavier_normal_(self.A)

    def normalize_A(self, A, symmetry=False):
        A = F.relu(A)
        if symmetry:
            A = A + torch.transpose(A, 0, 1)  # A+ A的转置
            d = torch.sum(A, 1)  # 对A的第1维度求和
            d = 1 / torch.sqrt(d + 1e-10)  # d的-1/2次方
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        return L

    def forward(self, x_l, x_r):
        # 参数
        L = self.normalize_A(self.A)  # A是自己设置的可训练参数及邻接矩阵
        #
        # # 左脑
        g_l = self.layer1(x_l, L)
        g_l = self.BN_l(g_l)
        #
        # # 右脑   
        g_r = self.layer1(x_r, L)
        g_r = self.BN_l(g_r)


        return g_l, g_r


class BiA(nn.Module):
    def __init__(self, T):
        super(BiA, self).__init__()
        self.T = T

        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(T, T)
        self.key_proj = nn.Linear(T, T)
        self.value_proj = nn.Linear(T, T)

        # LayerNorm layers for pre-attention and post-attention
        self.pre_attn_norm = nn.LayerNorm(T)  # LayerNorm before attention
        self.post_attn_norm = nn.LayerNorm(T)  # LayerNorm after attention

    def forward(self, x_l, x_r):
        bs, C, T = x_l.shape

        # LayerNorm before attention
        x_l_norm = self.pre_attn_norm(x_l)  # (bs, C, T)
        x_r_norm = self.pre_attn_norm(x_r)  # (bs, C, T)

        # Project x_l to query and x_r to key
        q = self.query_proj(x_l_norm)  # (bs, C, T)
        k = self.key_proj(x_r_norm)  # (bs, C, T)

        # Compute attention scores (c, c)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.T)  # (bs, C, C)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # Project (x_l - x_r) to value
        v = self.value_proj(x_l_norm - x_r_norm)  # (bs, C, T)

        # Apply attention to value
        r_l = torch.matmul(attn_scores, v)  # (bs, C, T)
        r_r = torch.matmul(attn_scores.transpose(1, 2), v)  # (bs, C, T)

        # LayerNorm after attention
        r_l = self.post_attn_norm(r_l)  # (bs, C, T)
        r_r = self.post_attn_norm(r_r)  # (bs, C, T)

        return r_l, r_r


class CosineRouter(nn.Module):
    def __init__(self, time_dim, channels, expert_dim, top_k=1):
        super().__init__()
        self.c = channels
        self.expert_dim = expert_dim
        self.top_k = top_k

        # 定义可学习的专家中心向量
        self.expert_centers = nn.Parameter(torch.randn(channels, expert_dim))

        # 定义可学习的投影矩阵，将输入映射到 expert_dim 维度
        self.projection = nn.Linear(time_dim * 2, expert_dim)

    def forward(self, x):
        """
        x: (bs, 2c, 1) - 输入是每个通道的统计特征（时间维度平均）
        返回：
        - topk_probs: (bs, 2c, top_k) - 每个通道的 top-k 专家权重
        - topk_indices: (bs, 2c, top_k) - 每个通道的 top-k 专家索引
        """
        bs, num_channels, _ = x.shape

        # 将输入投影到 expert_dim 维度 (bs, 2c, expert_dim)
        x_proj = self.projection(x)  # (bs, 2c, expert_dim)

        # 计算 Cosine 相似度
        x_proj = F.normalize(x_proj, p=2, dim=-1)  # 归一化投影向量
        expert_centers = F.normalize(self.expert_centers, p=2, dim=-1)  # 归一化专家中心向量
        cosine_sim = torch.matmul(x_proj, expert_centers.T)  # (bs, 2c, 2c)

        # 对每个通道选择 top-k 专家
        topk_probs, topk_indices = torch.topk(cosine_sim, self.top_k, dim=-1)  # (bs, 2c, top_k)
        topk_probs = F.softmax(topk_probs, dim=-1)  # 归一化为概率分布

        return topk_probs, topk_indices


class SGMoEBlock(nn.Module):
    def __init__(self, channels, time_dim, expert_dim=32, top_k=1):
        super().__init__()
        self.c = channels
        self.t = time_dim
        self.top_k = top_k

        # Cosine Router
        self.router = CosineRouter(time_dim, channels, expert_dim, top_k)

        # 定义 2c 个专家（每个通道左右各一个）
        self.experts = nn.ModuleList([
            nn.Linear(time_dim, time_dim)  # 每个专家处理时间维度
            for _ in range(channels)
        ])
        self.lnorm = nn.LayerNorm(time_dim)
        self.rnorm = nn.LayerNorm(time_dim)

    def forward(self, x_l, x_r):
        # 合并左右脑信号便于处理 (bs, 2c, t)
        x = torch.cat([x_l, x_r], dim=2)
        bs, c, t = x.shape
        output_l = torch.zeros_like(x_l)
        output_r = torch.zeros_like(x_r)

        # 通过 Cosine Router 获取门控权重
        topk_probs, topk_indices = self.router(x)  # (bs, 2c, top_k)
        unique_values = torch.unique(topk_indices)

        # 以专家为中心遍历
        for expert_idx in unique_values:
            mask = (topk_indices == expert_idx).any(dim=-1).unsqueeze(-1).float()
            if mask.sum() == 0:  # 如果没有通道选择当前专家，跳过
                continue

            # 使用掩码选择批次和通道
            selected_signals_l = x_l * mask
            selected_signals_r = x_r * mask

            # 找到匹配的专家概率，判断最后一个维度是否全为 0，最后返回最后一个维度的非零值
            selected_probs = topk_probs * mask
            selected_probs = selected_probs.sum(dim=-1, keepdim=True)

            # 对选择的信号进行专家处理
            expert_output_l = self.experts[expert_idx](selected_signals_l)
            expert_output_r = self.experts[expert_idx](selected_signals_r)

            # 加权累加
            weighted_output_l = selected_probs * expert_output_l
            weighted_output_r = selected_probs * expert_output_r

            # 将结果累加到输出中
            output_l += weighted_output_l
            output_r += weighted_output_r

        # 残差连接
        output_l = self.lnorm(output_l) + x_l
        output_r = self.rnorm(output_r) + x_r

        # 拆分左右脑输出
        return output_l, output_r

class BiDGNBlock(nn.Module):
    def __init__(self, input_size, batch_size, channel, output_size):
        super(BiDGNBlock, self).__init__()
        # 初始化三个模块
        # self.bidgn = BiDGN(input_size, batch_size, channel, output_size)
        # self.bia = BiA(output_size)
        self.gmoe = SGMoEBlock(channels=channel, time_dim=output_size, top_k=2)
        self.bima = BiMultiHeadAttention(input_size, output_size, num_heads=1, dropout=0.1)

    def forward(self, x_l, x_r):
        # 前向传播
        # x_l, x_r = self.bidgn(x_l, x_r)  # BiDGN
        # x_l, x_r = self.bia(x_l, x_r)    # BiA
        x_l, x_r = self.bima(x_l, x_r)
        x_l, x_r = self.gmoe(x_l, x_r)   # GMoE
        return x_l, x_r

class GAMoEmotion(nn.Module):
    def __init__(self, input_size, batch_size, channel, output_size, dropout, n_classes):
        super(GAMoEmotion, self).__init__()
        # 初始化三个模块
        self.block1 = BiDGNBlock(input_size, batch_size, channel, output_size)
        self.block2 = BiDGNBlock(input_size, batch_size, channel, output_size)
        self.classifier = ClassificationHead(output_size, dropout, n_classes)

        self.bidgn = BiDGN(input_size, batch_size, channel, output_size)

    def forward(self, x_l, x_r):
        # 前向传播
        # 保存输入特征，用于残差连接
        # residual_l, residual_r = x_l, x_r
        x_l, x_r = self.bidgn(x_l, x_r)  # BiDGN
        # # 第一个 BiDGN 块
        x_l, x_r = self.block1(x_l, x_r)  # BiDGN
        # # 添加残差连接
        # x_l = x_l + residual_l
        # x_r = x_r + residual_r
        #
        # # 保存当前特征，用于下一个残差连接
        # residual_l, residual_r = x_l, x_r
        # #
        # # # 第二个 BiDGN 块
        x_l, x_r = self.block2(x_l, x_r)
        # # # 添加残差连接
        # x_l = x_l + residual_l
        # x_r = x_r + residual_r

        # 分类器（GMoE）
        out = self.classifier(x_l, x_r)  # GMoE

        return out

class BiMultiHeadAttention(nn.Module):
    def __init__(self, input_size, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        # Linear layers for query, key, and value projections
        self.query_proj = nn.Linear(emb_size, emb_size)
        self.key_proj = nn.Linear(emb_size, emb_size)
        self.value_proj = nn.Linear(emb_size, emb_size)

        # Dropout and projection
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

        # LayerNorm layers for pre-attention and post-attention
        self.pre_attn_norm = nn.LayerNorm(emb_size)
        self.lnorm = nn.LayerNorm(emb_size)
        self.rnorm = nn.LayerNorm(emb_size)

        # Linear layers for res connection
        self.res_connect = nn.Linear(emb_size, emb_size)


    def forward(self, x_l: Tensor, x_r: Tensor) -> tuple[Tensor, Tensor]:
        # Project x_l to query and x_r to key
        queries = rearrange(self.query_proj(x_l), "b n (h d) -> b h n d", h=self.num_heads)  # (bs, num_heads, n, head_dim)
        keys = rearrange(self.key_proj(x_r), "b n (h d) -> b h n d", h=self.num_heads)  # (bs, num_heads, n, head_dim)

        # Compute attention scores
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # (bs, num_heads, n, n)
        scaling = self.head_dim ** (1 / 2)
        attn_scores = F.softmax(energy / scaling, dim=-1)  # (bs, num_heads, n, n)
        attn_scores = self.att_drop(attn_scores)

        # Project (x_l - x_r) to value
        value_diff = self.value_proj(x_l - x_r)  # (bs, n, emb_size)
        values = rearrange(value_diff, "b n (h d) -> b h n d", h=self.num_heads)  # (bs, num_heads, n, head_dim)

        # Apply attention to value
        out_l = torch.einsum('bhal, bhlv -> bhav', attn_scores, values)  # (bs, num_heads, n, head_dim)
        out_r = torch.einsum('bhal, bhlv -> bhav', attn_scores.transpose(2, 3), values)  # (bs, num_heads, n, head_dim)

        # Rearrange and project
        out_l = rearrange(out_l, "b h n d -> b n (h d)")  # (bs, n, emb_size)
        out_r = rearrange(out_r, "b h n d -> b n (h d)")  # (bs, n, emb_size)

        out_l = self.projection(out_l)  # (bs, n, emb_size)
        out_r = self.projection(out_r)  # (bs, n, emb_size)
        
        out_l=self.lnorm(out_l)
        out_r=self.rnorm(out_r)

        # x_l = self.res_connect(x_l)
        # x_r = self.res_connect(x_r)

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
    # 使用示例
    input_size = 5  # 假设输入大小
    batch_size = 8    # 假设 batch size
    channel = 64      # 假设通道数
    output_size = 5

    # 初始化 Block
    model = GAMoEmotion(input_size, batch_size, channel, output_size).cuda()

    # 生成随机输入数据
    x_l = torch.randn(batch_size, int(channel/2), input_size).cuda()
    x_r = torch.randn(batch_size, int(channel/2), input_size).cuda()

    # 前向传播
    out = model(x_l, x_r)

    # 打印输出
    print("Output shape:", x_l.shape, x_r.shape)