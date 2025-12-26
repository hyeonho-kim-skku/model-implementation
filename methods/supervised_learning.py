import torch.nn as nn
import torch.nn.functional as F

class SL(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        return loss