import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 4)  # Output 4 values to match the labels

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ChirpRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 每个时间步的特征提取器
        # 输入: (2, 8, 8)
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 输出: (16, 4, 4)
            
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))   # 输出: (32, 2, 2)
        )
        
        # 时间维度的LSTM处理
        self.temporal_lstm = nn.LSTM(
            input_size=32*2*2,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # 维度聚合和分类/回归输出
        self.classifier = nn.Sequential(
            nn.Linear(64*2, 32),  # 双向LSTM输出拼接
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)       # 单个输出值
        )

    def forward(self, x):
        # x形状: (batch_size, 1200, 8, 8, 2)
        batch_size = x.size(0)
        timesteps = x.size(1)
        
        # 重塑并交换维度，使通道维度正确
        x = x.view(batch_size*timesteps, 2, 8, 8)  # 将2作为通道维度
        
        # 应用特征编码器
        features = self.feature_encoder(x)  # (batch*1200, 32, 2, 2)
        features = features.view(batch_size, timesteps, -1)  # (batch, 1200, 32*2*2)
        
        # 应用LSTM进行时序处理
        lstm_out, (h_n, _) = self.temporal_lstm(features)
        
        # 使用最后时间步的隐藏状态
        # 对于双向LSTM，拼接最后一层的两个方向
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 64*2)
        
        # 分类/回归输出
        output = self.classifier(last_hidden).squeeze(-1)  # 移除最后的维度，得到(batch,)
        
        return output