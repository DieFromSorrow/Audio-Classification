import torch
import torch.nn as nn
from models.blocks import make_layer, ECAResTCNBlock1d, SelfAttentionPool


class ECAResTCN(nn.Module):
    def __init__(self, in_channels, num_classes, layers=None, dilation=None, dropout=0.2):
        super(ECAResTCN, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        if dilation is None:
            dilation = [1, 1, 1, 1]

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64)
        )

        # 使用更高效的残差块结构
        self.res_layers = nn.Sequential(
            make_layer(ECAResTCNBlock1d, 64, 64, layers[0],
                       dilation[0], stride=1, dropout=dropout),
            make_layer(ECAResTCNBlock1d, 64, 128, layers[1],
                       dilation[1], stride=2, dropout=dropout),
            make_layer(ECAResTCNBlock1d, 128, 256, layers[2],
                       dilation[2], stride=2, dropout=dropout),
            make_layer(ECAResTCNBlock1d, 256, 512, layers[3],
                       dilation[3], stride=2, dropout=dropout)
        )

        # 时间注意力模块
        self.temporal_attn = SelfAttentionPool(embed_dim=512, num_heads=4)

        # 自适应池化替代固定尺寸输出
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)
        # x: (batch_size, embed_dim, seq_len)
        # 空间特征聚合
        spatial_feat = self.adaptive_pool(x).squeeze(-1)

        # 时间维度处理
        x = x.permute(2, 0, 1)
        temporal_feat = self.temporal_attn(x)

        # 特征融合
        combined_feat = torch.cat((temporal_feat, spatial_feat), dim=-1)
        return self.fc(combined_feat)
