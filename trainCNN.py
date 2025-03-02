import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from simpleCNN import ChirpRegressionModel, SimpleCNN
from pathlib import Path
from toR_hat import toRhat
from readDCA1000 import readDCA1000

# ------------------- Setup -------------------
# Path setup
oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample/" # 文件夹路径
preProcessData = []
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # cuda for GPU, mps for Apple Silicon
model = ChirpRegressionModel().to(device)
criterion = nn.MSELoss()  # 回归任务使用MSE损失
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)  
epoch_num = 20 # 训练轮数
batch_size_set = 4

# ------------------- Loading and Preprocessing Data -------------------
# Load and preprocess data

bin_files = [f for f in os.listdir(oriFolderPath) if f.endswith('.bin')]
for file_name in bin_files:
    file_path = os.path.join(oriFolderPath, file_name)
    
    try:
        # Process the file using toRhat function
        data = readDCA1000(file_path, 12, 200) # numChirps1200 * num_rx12 * numADCSamples200
        data = data[:,np.r_[0:4, 8:12], :] # pick TX1, TX3　only
        preProcessData.append(data)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Convert preProcessData to numpy array
preProcessData = np.array(preProcessData)
X = preProcessData

# Check for NaN and Inf values in numpy array
if np.isnan(X).any():
    print("NaN values found in X")
if np.isinf(X).any():
    print("Inf values found in X")

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


# ------------------- Input Dataloader -------------------
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(X_tensor.shape, y_tensor.shape)

# Check for NaN and Inf values in PyTorch tensors
if torch.isnan(X_tensor).any():
    print("NaN values found in X_tensor")
if torch.isinf(X_tensor).any():
    print("Inf values found in X_tensor")

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