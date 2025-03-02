from torch.utils.data import Dataset

class GroupDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # (75, 1200, 8, 200, 2)
        self.labels = labels  # (75, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回形状： (1200, 8, 200, 2), (1,)
        return self.data[idx], self.labels[idx]