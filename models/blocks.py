import torch
from torch import nn
import torch.nn.functional as F
import math


class ECALayer(nn.Module):
    """通道注意力模块（ECA-Net改进版）"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2))
        return x * y.expand_as(x)


class ECAResTCNBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0):
        super(ECAResTCNBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=dilation*(3-1)//2, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=dilation*(3-1)//2, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.eca = ECALayer(out_channels)  # 添加通道注意力

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)  # 应用通道注意力
        out += self.shortcut(x)
        return self.dropout(self.elu(out))


def make_layer(block_class, in_channels, out_channels, blocks, dilation, stride, dropout=0):
    layers = [block_class(in_channels, out_channels, stride, dilation, dropout=dropout)]
    for _ in range(1, blocks):
        layers.append(block_class(out_channels, out_channels, stride=1,
                                  dilation=dilation, dropout=dropout))
    return nn.Sequential(*layers)


class MultiHeadAttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        query = self.query.repeat(x.size(1), 1).unsqueeze(0)  # (1, batch, embed_dim)
        attn_out, _ = self.attn(query, x, x)
        return attn_out.squeeze(0)  # (batch, embed_dim)


class SelfAttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.query_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear_query = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, x):
        # x: (seq_len, batch_size, embed_dim)
        x, _ = self.self_attn(x, x, x)
        query = self.linear_query(x.mean(dim=0, keepdim=True))
        x, _ = self.query_attn(query, x, x)
        x = x.squeeze(0)
        # x: (batch_size, embed_dim)
        return x


class GatedStatisticalPool(nn.Module):
    """双向门控统计池化（Bi-directional Gated Statistical Pooling）"""
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean = x.mean(dim=-1)
        std = x.std(dim=-1)
        stats = torch.cat([mean, std], dim=-1)
        gate = self.gate(stats)
        return gate * mean + (1 - gate) * std


class STPyramidPool(nn.Module):
    """时空金字塔池化（Spatial-Temporal Pyramid Pooling）"""
    def __init__(self, embed_dim, levels=None):
        super().__init__()
        if levels is None:
            levels = [1, 3, 5, 7]
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim // len(levels), k, padding=k // 2)
            for k in levels
        ])

    def forward(self, x):
        pooled = [F.adaptive_max_pool1d(conv(x), 1) for conv in self.conv_layers]
        return torch.cat(pooled, dim=1).squeeze(-1)


class Aggregator(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super().__init__()
        self.attention_pool = MultiHeadAttentionPool(embed_dim, num_heads=4)
        self.stat_pool = GatedStatisticalPool(embed_dim)
        self.st_pool = STPyramidPool(embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = self.attention_pool(x.permute(2, 0, 1))
        h2 = self.stat_pool(x)
        h3 = self.st_pool(x)

        fused = self.dropout(torch.cat([h1, h2, h3], dim=-1))
        return self.fusion(fused)
