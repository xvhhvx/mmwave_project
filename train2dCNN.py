import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from twodCNN import twodCNNModel
from pathlib import Path
from toR_hat import toRhat
from sklearn.model_selection import train_test_split

# ------------------- Setup -------------------
# Path setup
oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/FMCW radar-based multi-person vital sign monitoring data/2_SymmetricalPosition/1_Radar_Raw_Data/" # 文件夹路径
preProcessData = []
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # cuda for GPU, mps for Apple Silicon
model = twodCNNModel().to(device)
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
        self.data = data      # [54, 1200, 2, 8, 8]
        self.labels = labels  # [54, 4]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # [1200, 2, 8, 8], [4]

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
    rc2_mae = torch.mean(torch.abs(outputs[:, 2] - labels[:, 2]))
    hc2_mae = torch.mean(torch.abs(outputs[:, 3] - labels[:, 3]))
    return {
        'rc1_mae': rc1_mae.item(),
        'hc1_mae': hc1_mae.item(),
        'rc2_mae': rc2_mae.item(),
        'hc2_mae': hc2_mae.item()
    }

# Training loop
def train_epoch(model, dataloader, device):
    model.train()
    running_loss = 0.0
    metrics_sum = {'rc1_mae': 0, 'hc1_mae': 0, 'rc2_mae': 0, 'hc2_mae': 0}

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
    metrics_sum = {'rc1_mae': 0, 'hc1_mae': 0, 'rc2_mae': 0, 'hc2_mae': 0}

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

