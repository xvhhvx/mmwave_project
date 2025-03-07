import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from torch.utils.data import DataLoader
from simpleCNN import ChirpRegressionModel
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from getData import getData, getVali
from EarlyStopping import EarlyStopping
from GroupDataset import GroupDataset

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

def train_with_early_stopping(model, train_loader, val_loader, optimizer, scheduler, 
                              criterion, device, epochs, patience=5):
    """
    使用早停训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        device: 计算设备
        epochs: 最大训练轮数
        patience: 早停耐心值
        
    Returns:
        model: 训练好的模型
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        best_epoch: 最佳模型的轮数
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
        train_losses.append(train_loss)
        
        # 验证阶段
        val_loss = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)  # 使用验证损失来调整学习率
        
        # 保存检查点
        save_checkpoint(epoch + 1, model, optimizer, scheduler, train_loss, latest_checkpoint)
        
        # 打印训练进度
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
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


if __name__ == "__main__":
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
    criterion = nn.L1Loss()  # 回归任务使用L1损失
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )  
    epoch_num = 50 # 训练轮数
    batch_size_set = 16 # 可以调高一些至8/16
    split_size_set = 0.8 # 训练集占比
    patience_set = 5 # 早停耐心值

    # ------------------- Loading and Preprocessing Data -------------------
    # Load and preprocess data
    X = getData(oriFolderPath, 5, loadFromFile=True, saveToFile=False) # 直接从文件中读取数据

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

    train_size = int(split_size_set * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size_set, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_set, shuffle=False)

    # ------------------- Training -------------------
    # 使用早停训练模型
    print("Starting training with early stopping...")
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
