import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader
from simpleCNN import ChirpRegressionModel
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from getData import getData, getVali
from EarlyStopping import EarlyStopping
from GroupDataset import GroupDataset
from data_validation import (
    check_tensor_valid, 
    print_tensor_stats, 
    safe_normalize, 
    validate_data_pipeline,
    check_gradients,
    check_loss_value
)
from data_augmentation import augment_dataset,MmWaveAugmentation

# ------------------- Training Loop -------------------
# Training loop with validation checks
def train_epoch(model, dataloader, device):
    """
    Trains the model for one epoch with data validation checks.
    
    Args:
        model: The neural network model to train
        dataloader: DataLoader containing the training data
        device: Computing device (cuda, mps, or cpu)
        
    Returns:
        epoch_loss: Average loss for the epoch or NaN if no valid batches were processed
    """
    model.train()
    running_loss = 0.0
    valid_batches = 0
    
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # Validate batch data
        if not check_tensor_valid(inputs, f"Batch {batch_idx} inputs"):
            print(f"Skipping invalid batch {batch_idx}")
            continue
        if not check_tensor_valid(labels, f"Batch {batch_idx} labels"):
            print(f"Skipping invalid batch {batch_idx}")
            continue
            
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Check outputs for validity
        if not check_tensor_valid(outputs, f"Batch {batch_idx} outputs"):
            print(f"Invalid outputs in batch {batch_idx}, skipping")
            continue
            
        loss = criterion(outputs, labels)
        
        # Check if loss is valid
        if not check_loss_value(loss.item(), f"Batch {batch_idx}"):
            print(f"Invalid loss in batch {batch_idx}, skipping")
            continue
            
        loss.backward()
        
        # Check gradients
        if not check_gradients(model, threshold=5.0):
            print(f"Invalid gradients in batch {batch_idx}, clipping more aggressively")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # More aggressive clipping
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Standard clipping
            
        optimizer.step()
        
        running_loss += loss.item()
        valid_batches += 1
    
    # Avoid division by zero if all batches were invalid
    if valid_batches == 0:
        print("WARNING: No valid batches in epoch")
        return float('nan')
        
    epoch_loss = running_loss / valid_batches
    return epoch_loss

# Evaluate the model with validation checks
def evaluate(model, dataloader, device):
    """
    Evaluates the model on the provided dataset with data validation checks.
    
    Args:
        model: The neural network model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Computing device (cuda, mps, or cpu)
        
    Returns:
        total_loss / valid_batches: Average loss on the evaluation dataset or NaN if no valid batches
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Validate batch data
            if not check_tensor_valid(inputs, f"Val batch {batch_idx} inputs"):
                print(f"Skipping invalid validation batch {batch_idx}")
                continue
            if not check_tensor_valid(labels, f"Val batch {batch_idx} labels"):
                print(f"Skipping invalid validation batch {batch_idx}")
                continue
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            # Check outputs for validity
            if not check_tensor_valid(outputs, f"Val batch {batch_idx} outputs"):
                print(f"Invalid outputs in validation batch {batch_idx}, skipping")
                continue
                
            loss = criterion(outputs, labels)
            
            # Check if loss is valid
            if not check_loss_value(loss.item(), f"Val batch {batch_idx}"):
                print(f"Invalid loss in validation batch {batch_idx}, skipping")
                continue
                
            total_loss += loss.item()
            valid_batches += 1
    
    # Avoid division by zero if all batches were invalid
    if valid_batches == 0:
        print("WARNING: No valid batches in validation")
        return float('nan')
        
    return total_loss / valid_batches

# Save and load checkpoints
def save_checkpoint(epoch, model, optimizer, scheduler, loss, filename):
    """
    Saves model checkpoint to disk for future resumption of training.
    
    Args:
        epoch: Current epoch number
        model: The neural network model to save
        optimizer: The optimizer state to save
        scheduler: The learning rate scheduler state to save
        loss: Current loss value
        filename: Name of the checkpoint file
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_dir / filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Loads a previously saved model checkpoint to resume training.
    
    Args:
        model: The neural network model to load state into
        optimizer: The optimizer to load state into
        scheduler: The learning rate scheduler to load state into
        filename: Name of the checkpoint file to load
        
    Returns:
        tuple: (epoch, loss) from the loaded checkpoint
    """
    checkpoint = torch.load(checkpoint_dir / filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, 
                              criterion, device, epochs, patience=5):
    """
    Trains the model with early stopping mechanism and data validation.
    
    Implements a complete training pipeline with best model tracking, checkpoints,
    early stopping, data validation, and learning rate scheduling.
    
    Args:
        model: Neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for updating model weights
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Computing device (cuda, mps, or cpu)
        epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before early stopping
        
    Returns:
        tuple: (trained_model, train_losses, val_losses, best_epoch)
    """
    # 初始化早停
    early_stopping = EarlyStopping(patience=patience, checkpoint_dir=checkpoint_dir)
    train_losses = []
    val_losses = []
    
    start_epoch = 0
    best_loss = float('inf')
    best_epoch = 0
    
    # 检查是否存在检查点
    latest_checkpoint = "latest_checkpoint.pt"
    if os.path.exists(checkpoint_dir / latest_checkpoint):
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
        print(f"Resuming from epoch {start_epoch} with loss {best_loss}")
    
    # 训练循环
    for epoch in range(start_epoch, epochs):
        # 训练阶段
        train_loss = train_epoch(model, train_loader, device)
        
        # Check if train loss is valid
        if not check_loss_value(train_loss, f"Epoch {epoch+1} training"):
            print(f"WARNING: Invalid training loss in epoch {epoch+1}")
            # Try to recover by reducing learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"Reducing learning rate to {param_group['lr']}")
            # Skip this epoch
            continue
            
        train_losses.append(train_loss)
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, device)
        
        # Check if validation loss is valid
        if not check_loss_value(val_loss, f"Epoch {epoch+1} validation"):
            print(f"WARNING: Invalid validation loss in epoch {epoch+1}")
            # Try to recover by reducing learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"Reducing learning rate to {param_group['lr']}")
            # Skip this epoch
            continue
            
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)  # 使用验证损失来调整学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 保存检查点
        save_checkpoint(epoch + 1, model, optimizer, scheduler, train_loss, latest_checkpoint)
        
        # 打印训练进度
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Learning rate: {current_lr:.8f}")
        
        # 早停检查
        early_stopping(val_loss, model, optimizer, scheduler, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            best_epoch = epoch + 1 - early_stopping.patience  # 最佳模型的轮数
            break
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            save_checkpoint(epoch + 1, model, optimizer, scheduler, val_loss, "best_model.pt")
    
    # 如果没有触发早停，记录最佳轮数
    if not early_stopping.early_stop:
        best_epoch = val_losses.index(min(val_losses)) + 1
    
    # 加载最佳模型
    try:
        _, _ = load_checkpoint(model, optimizer, scheduler, "best_model.pt")
        print(f"Loaded best model from epoch {best_epoch}")
    except:
        print("Best model not found, using latest model")
    
    return model, train_losses, val_losses, best_epoch

# 根据心率范围创建分层拆分以确保训练和验证集中的分布相似
def create_stratified_split(X_tensor, y_tensor, y_mean, y_std, split_ratio=0.8):
    """
    Creates stratified train/validation split based on heart rate ranges.
    
    Ensures similar distributions of heart rates in both training and validation sets
    by using percentile-based binning for stratification. This helps prevent 
    distribution shift between training and evaluation.
    
    Args:
        X_tensor: Feature tensor data
        y_tensor: Target tensor data (normalized heart rates)
        y_mean: Mean value used for heart rate normalization
        y_std: Standard deviation used for heart rate normalization
        split_ratio: Fraction of data to use for training (default: 0.8)
        
    Returns:
        tuple: (train_dataset, val_dataset) containing GroupDataset objects
    """
    # Convert normalized y values back to original scale for binning
    original_y = y_tensor.numpy() * y_std + y_mean
    
    # Create data-driven heart rate bins using percentiles
    bins = np.percentile(original_y, [0, 20, 40, 60, 80, 100])
    # Ensure unique bin edges (required by np.digitize)
    bins = np.unique(bins)
    # Add small epsilon to the last bin edge to include maximum value
    if len(bins) > 1:
        bins[-1] += 0.001
    
    print(f"Data-driven bin edges: {bins}")

    bin_labels = np.digitize(original_y, bins[:-1])
    
    # Create stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-split_ratio, random_state=42)
    train_indices, val_indices = next(splitter.split(X_tensor, bin_labels))
    
    # Create train and validation datasets
    train_dataset = GroupDataset(X_tensor[train_indices], y_tensor[train_indices])
    val_dataset = GroupDataset(X_tensor[val_indices], y_tensor[val_indices])
    
    print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    print(f"Heart rate bin ranges: {list(zip(bins[:-1], bins[1:]))}")
    print(f"Training bin distribution: {np.bincount(bin_labels[train_indices])}")
    print(f"Validation bin distribution: {np.bincount(bin_labels[val_indices])}")
    
    return train_dataset, val_dataset

def filter_by_heart_rate(X_data, y_data, threshold=110, y_mean=0, y_std=1, is_normalized=True, device='cpu'):
    """
    Filters out data points with heart rates higher than the specified threshold.
    
    Parameters:
    -----------
    X_data : torch.Tensor or np.ndarray
        Input features
    y_data : torch.Tensor or np.ndarray
        Heart rate labels
    threshold : float
        Maximum allowed heart rate value
    y_mean : float
        Mean value used for normalization of y_data (if normalized)
    y_std : float
        Standard deviation used for normalization of y_data (if normalized)
    is_normalized : bool
        Whether y_data is normalized
    device : str
        Device to which the data should be moved (e.g., 'cpu', 'cuda')
            
    Returns:
    --------
    tuple: (filtered_X, filtered_y)
    """
    # Convert to numpy if tensors
    if isinstance(X_data, torch.Tensor):
        X_numpy = X_data.cpu().numpy()
    else:
        X_numpy = X_data
        
    if isinstance(y_data, torch.Tensor):
        y_numpy = y_data.cpu().numpy()
    else:
        y_numpy = y_data
    
    # Convert normalized values back to original scale if needed
    if is_normalized:
        original_y = y_numpy * y_std + y_mean
    else:
        original_y = y_numpy
    
    # Create mask for heart rates <= threshold
    mask = original_y <= threshold
    
    # Apply mask to filter both X and y
    filtered_X = X_numpy[mask]
    filtered_y = y_numpy[mask]
    
    # Convert back to tensors if inputs were tensors
    if isinstance(X_data, torch.Tensor):
        filtered_X = torch.tensor(filtered_X, dtype=torch.float32)
    
    if isinstance(y_data, torch.Tensor):
        filtered_y = torch.tensor(filtered_y, dtype=torch.float32)
    
    # Print stats about filtered data
    print(f"Original data points: {len(y_numpy)}")
    print(f"Filtered data points: {len(filtered_y)}")
    print(f"Removed {len(y_numpy) - len(filtered_y)} data points with heart rate > {threshold}")
    
    return filtered_X, filtered_y

if __name__ == "__main__":
    # ------------------- Setup -------------------
    # Path setup
    #oriFolderPath = r"/Sample" # 文件夹路径
    oriFolderPath = r"/Volumes/T7_Shield/mmwave_ip/Dataset/Sample3/" # 文件夹路径
    valiPath = oriFolderPath + "/HR.xlsx"
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
        factor=0.7,
        patience=10,
        min_lr=1e-7
    )  
    epoch_num = 300 # 训练轮数
    batch_size_set = 16 # 可以调高一些至8/16
    split_size_set = 0.8 # 训练集占比
    patience_set = 10 # 早停耐心值

    # ------------------- Loading and Preprocessing Data -------------------
    # Load and preprocess data
    X = getData(oriFolderPath, 5, loadFromFile = True, saveToFile = False) # 直接从文件中读取数据，5 为切割为 1 分钟，15 为切割为 20 秒

    # Load validation data
    y = getVali(valiPath) # 1（默认）为保持 1 分钟，3 为切割为 20 秒

    X, y = filter_by_heart_rate(X, y, threshold=95, is_normalized=False) # 过滤掉心率大于 95 的数据

    # ------------------- Input Dataloader -------------------
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    print(X_tensor.shape, y_tensor.shape)

    # Validate data before normalization
    print("Validating data before normalization...")
    if not validate_data_pipeline(X_tensor, y_tensor):
        print("WARNING: Data validation failed before normalization")
    
    # 使用安全归一化函数
    print("Performing safe normalization...")
    X_tensor, X_mean, X_std = safe_normalize(X_tensor, dim=0, eps=1e-7)
    y_tensor, y_mean, y_std = safe_normalize(y_tensor, eps=1e-7)

    # Validate data after normalization
    print("Validating data after normalization...")
    if not validate_data_pipeline(X_tensor, y_tensor):
        print("WARNING: Data validation failed after normalization")

    # 保存归一化参数
    normalization_params = {
        'X_mean': X_mean.cpu().numpy(),
        'X_std': X_std.cpu().numpy(),
        'y_mean': y_mean.item(),
        'y_std': y_std.item()
    }
    with open('normalization_params.pkl', 'wb') as f:
        pickle.dump(normalization_params, f)

    print(X_tensor.shape, y_tensor.shape)

    print("Performing data augmentation...")
    X_tensor, y_tensor = augment_dataset(
        X_tensor, 
        y_tensor, 
        augment_factor=3,
        primary_aug=MmWaveAugmentation.add_gaussian_noise,
        aug_probability=0.0,  # Set to 0 to disable random augmentations, set to 0.3 to enable random augmentations
        always_include_primary=True
    )  

    # Create DataLoader
    train_dataset, val_dataset = create_stratified_split(
        X_tensor, 
        y_tensor, 
        y_mean=y_mean.item(),  # Ensure scalar value
        y_std=y_std.item(),    # Ensure scalar value
        split_ratio=split_size_set
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size_set, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_set, shuffle=False)

    # ------------------- Training -------------------
    # 使用早停训练模型
    print("Starting training with early stopping and data validation...")
    model, train_losses, val_losses, best_epoch = train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,  
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=epoch_num,
        patience=patience_set
    )

    # 打印最终评估结果
    final_train_loss = evaluate(model, train_loader, device)
    final_val_loss = evaluate(model, val_loader, device)
    print(f"Final Training MSE: {final_train_loss:.6f}")
    print(f"Final Validation MSE: {final_val_loss:.6f}")
    print(f"Best model was from epoch {best_epoch}")

    # 可视化训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()
