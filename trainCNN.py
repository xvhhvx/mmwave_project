import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from simpleCNN import ChirpRegressionModel, SimpleCNN
from pathlib import Path
from toR_hat import toRhat
from getData import getData, mergeData, separateComplexData, getVali
from GroupDataset import GroupDataset

# ------------------- Setup -------------------
# Path setup
oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample/" # 文件夹路径
valiPath = oriFolderPath + "/HR.xlsx"
preProcessData = []
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # cuda for GPU, mps for Apple Silicon
model = ChirpRegressionModel().to(device)
criterion = nn.MSELoss()  # 回归任务使用MSE损失
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)  
epoch_num = 20 # 训练轮数
batch_size_set = 4 # 可以调高一些至8/16

# ------------------- Loading and Preprocessing Data -------------------
# Load and preprocess data

X = getData(oriFolderPath, 5)
X = mergeData(X)
X = separateComplexData(X)

# Load validation data
y = getVali(valiPath)


# ------------------- Input Dataloader -------------------
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_tensor.shape, y_tensor.shape)

# 对输入进行归一化
X_mean = X_tensor.mean(dim=0, keepdim=True)
X_std = X_tensor.std(dim=0, keepdim=True)
X_tensor = (X_tensor - X_mean) / (X_std + 1e-7)  # 添加小值避免除零
# 对目标进行归一化
y_mean = y_tensor.mean()
y_std = y_tensor.std()
y_tensor = (y_tensor - y_mean) / y_std

# 保存归一化参数
normalization_params = {
    'X_mean': X_mean.cpu().numpy(),
    'X_std': X_std.cpu().numpy(),
    'y_mean': y_mean.item(),
    'y_std': y_std.item()
}
with open('normalization_params.pkl', 'wb') as f:
    pickle.dump(normalization_params, f)

# Check for NaN and Inf values in PyTorch tensors
if torch.isnan(X_tensor).any():
    print("NaN values found in X_tensor")
if torch.isinf(X_tensor).any():
    print("Inf values found in X_tensor")

print(X_tensor.shape, y_tensor.shape)

# Create DataLoader
dataset = GroupDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size_set, shuffle=True)

# ------------------- Training Loop -------------------
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

# Save and load checkpoints
def save_checkpoint(epoch, model, optimizer, scheduler, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_dir / filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    checkpoint = torch.load(checkpoint_dir / filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


# ------------------- Training -------------------
# Modify training loop
start_epoch = 0
best_loss = float('inf')

# Check for existing checkpoint
latest_checkpoint = "latest_checkpoint.pt"
if os.path.exists(checkpoint_dir / latest_checkpoint):
    start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
    print(f"Resuming from epoch {start_epoch} with loss {best_loss}")

# Train the model
for epoch in range(start_epoch, epoch_num):
    train_loss = train_epoch(model, train_loader, device)
    scheduler.step(train_loss)
    
    # Save checkpoint
    save_checkpoint(epoch + 1, model, optimizer, scheduler, train_loss, latest_checkpoint)
    
    # Save best model
    if train_loss < best_loss:
        best_loss = train_loss
        save_checkpoint(epoch + 1, model, optimizer, scheduler, train_loss, "best_model.pt")
    
    print(f"Epoch {epoch+1}/{epoch_num}")
    print(f"Train Loss: {train_loss:.6f}")

# Evaluate the model
val_loss = evaluate(model, train_loader, device)
print(f"Validation MSE: {val_loss:.6f}")