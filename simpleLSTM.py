import torch.nn as nn


class ChirpRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 每个chirp的特征提取器
        self.chirp_encoder = nn.Sequential(
            # 输入: (2, 8, 8)
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: (16,4,4)
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 输出: (32,2,2)
        )
        
        # 时间序列聚合器
        self.temporal_aggregator = nn.Sequential(
            nn.Linear(32*2*2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        )
        
        # 回归输出层
        self.regressor = nn.Sequential(
            nn.Linear(64*2, 32),  # 双向LSTM输出拼接
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        # x形状: (batch_size, 1200, 2, 8, 8)
        batch_size, timesteps = x.size(0), x.size(1)
        
        # 处理每个chirp
        x = x.view(batch_size*timesteps, 2, 8, 8)
        chirp_features = self.chirp_encoder(x)  # (batch*1200, 32, 2, 2)
        chirp_features = chirp_features.view(batch_size, timesteps, -1)  # (batch, 1200, 32*2*2)
        
        # 时序聚合
        temporal_out, (h_n, c_n) = self.temporal_aggregator(chirp_features)
        
        # 取最终时间步输出
        last_output = temporal_out[:, -1, :]
        
        return self.regressor(last_output)