import numpy as np
import matplotlib.pyplot as plt
from readDCA1000 import readDCA1000
from compute_background_and_subtraction import compute_background_and_subtraction as CBAS
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from simpleCNN import SimpleCNN
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
            preProcessData.append(fft2dAll) # 54 * 1200 * 8 * 8

# Convert preProcessData to numpy array
preProcessData = np.array(preProcessData)

# Preprocess data for CNN
num_samples = preProcessData.shape[0] * preProcessData.shape[1]  # 54 * 1200
X = preProcessData.reshape((num_samples, 1, 8, 8))  # Reshape to [batch_size, channels, height, width]


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

# Repeat y to match the number of samples in X
y = np.repeat(y, preProcessData.shape[1], axis=0)


# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleCNN()
criterion = nn.MSELoss()  # Use nn.BCELoss() for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_loss = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
    test_loss /= len(train_loader)
    print(f'Test loss: {test_loss}')