import numpy as np
import matplotlib.pyplot as plt
from readDCA1000 import readDCA1000
from compute_background_and_subtraction import compute_background_and_subtraction as CBAS
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from simpleCNN import ChirpRegressionModel
from toR_hat import toRhat

oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/FMCW radar-based multi-person vital sign monitoring data/2_SymmetricalPosition/1_Radar_Raw_Data/" # 文件夹路径
preProcessData = []

# Load and preprocess data
for i in range(1, 4):
    if i == 1:
        band = '2GHZ'
    elif i == 2:
        band = '2_5GHZ'
    elif i == 3:
        band = '3GHZ'
    for j in range(7, 10):
        folderPath = oriFolderPath + r"position_ (" + str(j) + ")" #访问三个文件夹
        for k in range(1, 7):
            filePath = r'adc_'+ band +'_position'+ str(j) +'_ (' + str(k) +').bin' #访问各个 bin 文件
            binPath = folderPath + '/' + filePath #使用 Windows 电脑可能需要修改为下面这行
            ##binPath = folderPath + '\\' + filePath

            fft2dAll = toRhat(binPath)
            preProcessData.append(fft2dAll) # 54 * 1200 * 8 * 8 * 2

'''
# Load and preprocess data
bin_files = [f for f in os.listdir(oriFolderPath) if f.endswith('.bin')]
for file_name in bin_files:
    file_path = os.path.join(oriFolderPath, file_name)
    
    try:
        # Process the file using toRhat function
        processed_data = toRhat(file_path)
        preProcessData.append(processed_data)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
'''

# Convert preProcessData to numpy array
preProcessData = np.array(preProcessData)

# Preprocess data for CNN
#num_samples = preProcessData.shape[0] * preProcessData.shape[1]  # 54 * 1200
#X = preProcessData.reshape((num_samples, 1, 8, 8))  # Reshape to [batch_size, channels, height, width]
X = preProcessData

# Load validation data
results_target1 = pd.read_excel('results_target1.xlsx')
results_target2 = pd.read_excel('results_target2.xlsx')

rcTarget1 = results_target1['respiratory_count'].values # Respiratory count for target 1
hcTarget1 = results_target1['heartbeat_count'].values # Heartbeat count for target 1
rcTarget2 = results_target2['respiratory_count'].values # Respiratory count for target 2
hcTarget2 = results_target2['heartbeat_count'].values # Heartbeat count for target 2

# Combine validation targets
y = list(zip(rcTarget1, hcTarget1, rcTarget2, hcTarget2)) # Make labels as [(rc1, hc1, rc2, hc2), ...]

# Ensure y has the same number of samples as X
assert len(y) == preProcessData.shape[0], "The number of validation targets must match the number of samples."
y = np.array(y)
# Repeat y to match the number of samples in X
#y = np.repeat(y, preProcessData.shape[1], axis=0)


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_tensor.shape, y_tensor.shape)

# Create DataLoader
class GroupDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # (54, 1200, 2, 8, 8)
        self.labels = labels  # (54, 4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回形状： (1200, 2, 8, 8), (4,)
        return self.data[idx], self.labels[idx]

dataset = GroupDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = ChirpRegressionModel().to(device)
criterion = nn.MSELoss()  # 回归任务使用MSE损失
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
epoch_num = 20

#optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_epoch(model, dataloader, device):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

for epoch in range(epoch_num):
    train_loss = train_epoch(model, train_loader, device)
    scheduler.step(train_loss)
    
    # 打印训练信息
    print(f"Epoch {epoch+1}/"+str(epoch_num))
    print(f"Train Loss: {train_loss:.6f}")
    
    # 每10个epoch保存一次模型
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"regression_model_epoch{epoch+1}.pth")
    print("-"*40)

# Evaluate the model
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

val_loss = evaluate(model, train_loader, device)
print(f"Validation MSE: {val_loss:.6f}")