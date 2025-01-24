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

# Load and preprocess data
for i in range(1, 4):
    if i == 1:
        band = '2GHz'
    elif i == 2:
        band = '2_5GHz'
    elif i == 3:
        band = '3GHz'
    for j in range(7, 10):
        folderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/FMCW radar-based multi-person vital sign monitoring data/2_SymmetricalPosition/1_Radar_Raw_Data/position_ (" + str(j) + ")" #访问三个文件夹
        for k in range(1, 7):
            filePath = r'adc_'+ band +'_position'+ str(j) +'_ (' + str(k) +').bin'
            binPath = folderPath + '/' + filePath
            print(binPath)

binPath = folderPath + '/' + filePath
data = readDCA1000(binPath, 12, 200) # numChirps1200 * num_rx12 * numADCSamples200
data = data[:,np.r_[0:4, 8:12], :] # pick TX1, TX3 only

data_minus = CBAS(data.transpose(0,2,1), beta= 1.5)[1].transpose(0,2,1) # 消除背景干扰

meanSample = np.mean(data_minus, axis= 0)
fft2d = np.fft.fft2(meanSample)
fft2d_shifted = np.fft.fftshift(fft2d)
magnitude = np.abs(fft2d_shifted)
peak_idx = np.argmax(np.abs(fft2d_shifted[1]))
ROI = 4 # range bin
start = max(0, peak_idx - ROI)
end = min(fft2d_shifted.shape[1], peak_idx + ROI)

num_chirps = data_minus.shape[0]
fft2dAll = np.zeros((num_chirps, 8, 8), dtype='complex')
for chirp in range(num_chirps):
    fft2dTmp = np.fft.fft2(data_minus[chirp,:,:])
    fft2dTmp = np.fft.fftshift(fft2dTmp)
    fft2dTmp = fft2dTmp[:, start:end]
    fft2dAll[chirp] = fft2dTmp

# Load validation data
validation_data = pd.read_excel('results_target1.xlsx')
validation_values = validation_data['target_column'].values  # Replace 'target_column' with the actual column name

# Preprocess data for CNN
X = np.abs(fft2dAll).reshape((num_chirps, 1, 8, 8))  # Reshape for CNN input
y = validation_values

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