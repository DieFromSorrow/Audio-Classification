import torch.nn as nn
from models.blocks import make_layer, ECAResTCNBlock1d


class MiniECAResTCN(nn.Module):
    def __init__(self, in_channels, num_classes, layers=None, dilation=None, dropout=0.2):
        super(MiniECAResTCN, self).__init__()
        if layers is None:
            layers = [1, 1, 1, 1]
        if dilation is None:
            dilation = [1, 2, 4, 8]

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64)
        )

        # 使用更高效的残差块结构
        self.res_layers = nn.Sequential(
            make_layer(ECAResTCNBlock1d, 64, 64, layers[0],
                       dilation[0], stride=1, dropout=dropout),
            make_layer(ECAResTCNBlock1d, 64, 64, layers[1],
                       dilation[1], stride=2, dropout=dropout),
            make_layer(ECAResTCNBlock1d, 64, 64, layers[2],
                       dilation[2], stride=2, dropout=dropout),
            make_layer(ECAResTCNBlock1d, 64, 64, layers[3],
                       dilation[3], stride=2, dropout=dropout)
        )

        # 自适应池化替代固定尺寸输出
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, embed_dim, seq_len)
        x = self.conv1(x)
        x = self.res_layers(x)
        spatial_feat = self.adaptive_pool(x).squeeze(-1)
        return self.fc(spatial_feat)
