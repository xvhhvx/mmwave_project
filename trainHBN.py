import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from twodCNN import twodCNNModel, HeartBeatNet
from getData import getData, getVali
from pathlib import Path
from toR_hat import toRhat
from sklearn.model_selection import train_test_split

# ------------------- Setup -------------------
# Path setup
oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/data"
bin_files = [f for f in os.listdir(oriFolderPath) if f.endswith('.csv') and not f.startswith('._')] #忽略._开头的隐藏文件
preProcessData = []
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # cuda for GPU, mps for Apple Silicon
model = HeartBeatNet().to(device)
criterion = nn.MSELoss()  # 回归任务使用MSE损失
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) # modified params from trainLSTM.py
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)                                                                       # modified params from trainLSTM.py
epoch_num = 20 # 训练轮数
batch_size = 4

# ------------------- Loading and Preprocessing Data -------------------
# Load and preprocess data
X = getData(oriFolderPath, bin_files)
y = getVali(oriFolderPath, bin_files)
X = np.array(X)
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
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = GroupDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size, shuffle=True)

# Split the data into training and validation sets
train_indices, val_indices = train_test_split(
    range(len(dataset)), 
    test_size=0.2, 
    random_state=42
)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------- Training Loop -------------------
# Evaluation metrics
def calculate_metrics(outputs, labels):
    """Calculate separate metrics for respiratory and heart rates"""
    rc1_mae = torch.mean(torch.abs(outputs[:, 0] - labels[:, 0]))
    hc1_mae = torch.mean(torch.abs(outputs[:, 1] - labels[:, 1]))

    return {
        'rc1_mae': rc1_mae.item(),
        'hc1_mae': hc1_mae.item()
    }

# Training loop
def train_epoch(model, dataloader, device):
    model.train()
    running_loss = 0.0
    metrics_sum = {'rc1_mae': 0, 'hc1_mae': 0}

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
        batch_metrics = calculate_metrics(outputs, labels)
        for key in metrics_sum:
            metrics_sum[key] += batch_metrics[key]
    
    num_batches = len(dataloader)
    epoch_loss = running_loss / num_batches
    epoch_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    return epoch_loss, epoch_metrics

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

# Evaluate the model
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = {'rc1_mae': 0, 'hc1_mae': 0}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            batch_metrics = calculate_metrics(outputs, labels)
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
    
    num_batches = len(dataloader)
    return total_loss / num_batches, {k: v / num_batches for k, v in metrics_sum.items()}

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Modify training loop
start_epoch = 0
best_loss = float('inf')

# Check for existing checkpoint
latest_checkpoint = "latest_checkpoint.pt"
if os.path.exists(checkpoint_dir / latest_checkpoint):
    start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
    print(f"Resuming from epoch {start_epoch} with loss {best_loss}")

# Training loop
best_val_loss = float('inf')
for epoch in range(start_epoch, epoch_num):
    # Training phase
    train_loss, train_metrics = train_epoch(model, train_loader, device)
    
    # Validation phase
    val_loss, val_metrics = evaluate(model, val_loader, device)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save checkpoint
    save_checkpoint(epoch + 1, model, optimizer, scheduler, val_loss, latest_checkpoint)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10)
  

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(epoch + 1, model, optimizer, scheduler, val_loss, "best_model.pt")
    
    # Print metrics
    print(f"\nEpoch {epoch+1}/{epoch_num}")
    print(f"Train Loss: {train_loss:.6f}")
    print("Train Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Val Loss: {val_loss:.6f}")
    print("Val Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")
