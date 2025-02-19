import torch
import torch.nn as nn

class HeartBeatNet(nn.Module):
    def __init__(self):
        super(HeartBeatNet, self).__init__()
        
        # 上分支卷积块
        self.upper_conv1_1 = nn.Conv1d(3, 3, kernel_size=5, padding=2)
        self.upper_conv1_2 = nn.Conv1d(3, 3, kernel_size=5, padding=2)
        self.upper_conv1_3 = nn.Conv1d(3, 128, kernel_size=1)

        self.upper_conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)

        self.upper_conv3 = nn.Conv1d(128, 3, kernel_size=5, padding=2)

        self.upper_conv4_1 = nn.Conv1d(3, 5, kernel_size=5, padding=2)
        self.upper_conv4_2 = nn.Conv1d(5, 5, kernel_size=5, padding=2)
        
        # 上分支池化层
        self.upper_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upper_pooling2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upper_pooling3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 上分支1x1卷积用于维度匹配
        self.input_proj1 = nn.Conv1d(3, 128, kernel_size=1)
        self.input_proj2 = nn.Conv1d(3, 5, kernel_size=1)
        
        # 下分支池化和卷积层
        self.lower_conv1_1 = nn.Conv1d(5, 5, kernel_size=5, padding=2)
        self.lower_conv1_2 = nn.Conv1d(5, 5, kernel_size=5, padding=2)
        
        self.lower_conv2_1 = nn.Conv1d(5, 5, kernel_size=5, padding=2)
        self.lower_conv2_2 = nn.Conv1d(5, 5, kernel_size=5, padding=2)

        self.lower_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lower_pooling2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lower_pooling3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.input_proj3 = nn.Conv1d(5, 5, kernel_size=1)
        self.input_proj4 = nn.Conv1d(5, 5, kernel_size=1)
        
        # 最终层
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(5 * 937, 2)  # 60000 / 8 = 7500 (3次池化)
        
    def forward(self, x):
        # 第一个跳跃连接 - 输入到upper_conv1输出
        identity1 = self.input_proj1(x)
        x = self.upper_conv1_1(x)
        x = self.upper_conv1_2(x)
        x = self.upper_conv1_3(x)
        x = x + identity1  # 第一个跳跃连接
        x = self.upper_pooling1(x)

        x = self.upper_conv2(x)
        x = self.upper_pooling2(x)

        x = self.upper_conv3(x)
        x = self.upper_pooling3(x)
        identity2 = self.input_proj2(x)

        x = self.upper_conv4_1(x)
        x = self.upper_conv4_2(x)
        x = x + identity2  # 第二个跳跃连接
        

        
        x = self.lower_pooling1(x)
        identity3 = self.input_proj3(x)

        x = self.lower_conv1_1(x)
        x = self.lower_conv1_2(x)
        x = x + identity3  # 第三个跳跃连接
        

        
        x = self.lower_pooling2(x)
        identity4 = self.input_proj4(x)

        x = self.lower_conv2_1(x)
        x = self.lower_conv2_2(x)
        x = x + identity4  # 第四个跳跃连接
        
        x = self.lower_pooling3(x)
        
        # 最终输出
        x = self.flatten(x)
        out = self.fc(x)
        
        return out

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

class onedCNNModel(nn.Module):
    def __init__(self):
        super(onedCNNModel, self).__init__()
        # CNN处理时序数据
        self.cnn = nn.Sequential(
            # 输入维度为[batch_size, 3, 60000]
            nn.Conv1d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)  # 自适应池化到固定长度
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(64 * 100, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出维度改为2
        )

    def forward(self, x):
        # x shape: [batch_size, 60000, 3]
        x = x.transpose(1, 2)  # 转换为[batch_size, 3, 60000]
        
        # CNN特征提取
        x = self.cnn(x)  # [batch_size, 64, 100]
        
        # 展平
        x = x.view(x.size(0), -1)  # [batch_size, 64*100]
        
        # 全连接层得到预测
        x = self.fc(x)  # [batch_size, 2]
        
        return x


class twodCNNModeln(nn.Module):
    def __init__(self):
        super(twodCNNModeln, self).__init__()
        # CNN处理3D时序数据
        self.cnn = nn.Sequential(
            # 输入维度为[batch_size, 3, 3, 60000]
            nn.Conv2d(3, 16, kernel_size=(3,5), stride=(1,2), padding=(1,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
            
            nn.Conv2d(16, 32, kernel_size=(1,5), stride=(1,2), padding=(0,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4)),
            
            nn.Conv2d(32, 64, kernel_size=(1,5), stride=(1,2), padding=(0,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 100))  # 自适应池化到固定长度
        )

        # 全连接层保持不变
        self.fc = nn.Sequential(
            nn.Linear(64 * 100, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出维度为2
        )

    def forward(self, x):
        # x shape: [batch_size, 60000, 3, 3]
        x = x.permute(0, 2, 3, 1)  # 转换为[batch_size, 3, 3, 60000]
        
        # CNN特征提取
        x = self.cnn(x)  # [batch_size, 64, 1, 100]
        
        # 展平
        x = x.view(x.size(0), -1)  # [batch_size, 64*100]
        
        # 全连接层得到预测
        x = self.fc(x)  # [batch_size, 2]
        
        return x

if __name__ == "__main__":
    # 创建模型实例
    model = HeartBeatNet()
    
    # 创建测试输入
    test_input = torch.randn(1, 3, 60000)
    
    # 前向传播
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")