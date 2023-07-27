import torch 
from torch.utils.data import Dataset

class Room2D(Dataset):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        x = torch.linspace(1, n, n)
        y = torch.linspace(1, m, m)
        points = torch.stack(torch.meshgrid(x,y, indexing="ij"))
        self.points = points.permute(*torch.arange(points.ndim - 1, -1, -1)).reshape((-1,2))
    
    def __len__(self):
        return self.n * self.m

    def __getitem__(self, idx):
        return self.points[idx]
