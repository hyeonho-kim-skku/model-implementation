import torch.nn as nn
from utils import nt_xent_loss

class SimCLR(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        (x1, x2), _ = batch # x1, x2: (B, C, H, W)
        
        # encoder + projection head
        _, z1 = self.model(x1) # z1: (B, out_dim)
        _, z2 = self.model(x2) # z2: (B, out_dim)

        loss = nt_xent_loss(z1, z2)
        return loss
