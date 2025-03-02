import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0, checkpoint_dir="checkpoints"):
        """
        Args:
            patience (int): 验证损失多少个epoch没有改善后停止训练
                            默认: 7
            verbose (bool): 是否打印早停信息
                            默认: False
            delta (float): 判定为改善的最小变化量
                          默认: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, val_loss, model, optimizer, scheduler, epoch, filename="early_stop_model.pt"):
        score = -val_loss  # 分数越高越好，所以取负

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch, filename)
        elif score < self.best_score + self.delta:
            # 验证损失没有显著改善
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 验证损失有所改善
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch, filename)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch, filename):
        '''当验证损失减少时，保存模型。'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
        self.val_loss_min = val_loss