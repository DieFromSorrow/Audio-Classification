import torch.nn as nn
from models.blocks import MultiHeadAttentionPool, make_layer, ECAResidualBlock1d


class ECAResNetMfccClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, layers=None, dropout=0.2, r: int = 1):
        super(ECAResNetMfccClassifier, self).__init__()

        if layers is None:
            layers = [2, 2, 2, 2]

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64 * r, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64 * r),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 使用更高效的残差块结构
        self.res_layers = nn.Sequential(
            make_layer(ECAResidualBlock1d, 64 * r, 64 * r, layers[0], stride=1),
            nn.Dropout(dropout),
            make_layer(ECAResidualBlock1d, 64 * r, 128 * r, layers[1], stride=2),
            nn.Dropout(dropout),
            make_layer(ECAResidualBlock1d, 128 * r, 256 * r, layers[2], stride=2),
            nn.Dropout(dropout),
            make_layer(ECAResidualBlock1d, 256 * r, 512 * r, layers[3], stride=2),
        )

        self.self_attn = nn.MultiheadAttention(embed_dim=512 * r, num_heads=8)

        # 时间注意力模块
        self.temporal_attn = MultiHeadAttentionPool(embed_dim=512 * r, num_heads=8)

        # 自适应池化替代固定尺寸输出
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(512 * r, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_layers(x)

        # 时间维度处理
        x = x.permute(2, 0, 1)  # [T, B, C]
        x, _ = self.self_attn(x, x, x)

        temporal_feat = self.temporal_attn(x)

        # 空间特征聚合
        spatial_feat = self.adaptive_pool(x.permute(1, 2, 0)).squeeze(-1)

        # 特征融合
        combined_feat = temporal_feat + spatial_feat
        return self.fc(combined_feat)
