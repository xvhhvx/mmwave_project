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
            nn.Conv2d(2, 8, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 输出: (8, 4, 4)
            
            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))   # 输出: (16, 2, 2)
        )
        
        # 时间维度的LSTM处理
        # 输入: (batch, chirp_len, 16*2*2=64)
        self.temporal_lstm = nn.LSTM(
            input_size=16*2*2,
            hidden_size=32,
            num_layers=1,  # 从双层改到单层LSTM
            batch_first=True,
            dropout=0.0,  # 不进行dropout
            bidirectional=False  # 不使用双向LSTM
        )

        # 维度聚合和分类/回归输出
        # 输入: 32 (LSTM隐藏状态大小)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)       # 单个输出值
        )

        # 添加一个时间维度的平均池化层来减少时间步数
        # 输入: (batch, 64, 1200) -> 输出: (batch, 64, 300)
        self.temporal_pooling = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        # x形状: (batch_size, 1200, 8, 8, 2)
        batch_size = x.size(0)
        timesteps = x.size(1)
        
        # 重塑并交换维度，使通道维度正确
        x = x.view(batch_size*timesteps, 2, 8, 8)  # 将2作为通道维度
        
        # 应用特征编码器
        features = self.feature_encoder(x)  # (batch*1200, 16, 2, 2)
        features = features.view(batch_size, timesteps, -1)  # (batch, 1200, 16*2*2)
        
        # 应用时间维度的池化，减少序列长度
        features_t = features.transpose(1, 2)  # (batch, 64, 1200)
        features_t = self.temporal_pooling(features_t)  # (batch, 64, 300)
        features = features_t.transpose(1, 2)  # (batch, 300, 64)

        # 应用LSTM进行时序处理
        # lstm_out: (batch, 300, 32)
        # h_n: (1, batch, 32)
        lstm_out, (h_n, _) = self.temporal_lstm(features)
        
        # 使用最后时间步的隐藏状态
        last_hidden = h_n[-1]  # (batch, 32)

        # 分类/回归输出
        output = self.classifier(last_hidden).squeeze(-1)  # 移除最后的维度，得到(batch,)
        
        return output