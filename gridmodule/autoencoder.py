import torch
from torch.nn import Module
from gridlayer import GridTissue

class OptModel(Module):
    def __init__(self, n):
        super().__init__()
        self.gridtissue =  GridTissue(n)
        self.lineardecoder = torch.nn.Linear(n, 2, bias=True)

    def forward(self, x):
        return self.lineardecoder(self.gridtissue(x))     



##############

