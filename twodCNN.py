import torch.nn as nn

class twodCNNModel(nn.Module):
    def __init__(self):
        super(twodCNNModel, self).__init__()
        # CNN处理每个chirp的8x8x2输入
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 处理时序信息
        self.temporal = nn.Sequential(
            nn.Linear(32*4*4*1200, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        # x shape: [batch_size, 1200, 2, 8, 8]
        batch_size = x.size(0)

        # 重塑以处理所有chirps
        x = x.view(batch_size * 1200, 2, 8, 8)

        # CNN特征提取
        x = self.cnn(x)  # [batch_size*1200, 32, 4, 4]

        # 重塑回批次维度
        x = x.view(batch_size, -1)  # [batch_size, 1200*32*4*4]

        # 全连接层得到预测
        x = self.temporal(x)  # [batch_size, 4]

        return x
