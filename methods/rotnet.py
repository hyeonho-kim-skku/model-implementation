import torch
import torch.nn as nn
import torch.nn.functional as F

class RotNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, batch): # batch: (x, y), x: (B,C,H,W), y: (B,)
        x, _ = batch

        # 0°, 90°, 180°, 270° 회전
        x0 = torch.rot90(x, k=0, dims=[2, 3])
        x1 = torch.rot90(x, k=1, dims=[2, 3])
        x2 = torch.rot90(x, k=2, dims=[2, 3])
        x3 = torch.rot90(x, k=3, dims=[2, 3])
        
        # 회전 label 생성
        batch_size = x.size(0)
        y0 = torch.full((batch_size,), 0, dtype=torch.long, device=x.device)
        y1 = torch.full((batch_size,), 1, dtype=torch.long, device=x.device)
        y2 = torch.full((batch_size,), 2, dtype=torch.long, device=x.device)
        y3 = torch.full((batch_size,), 3, dtype=torch.long, device=x.device)

        # concatenation
        x = torch.cat((x0, x1, x2, x3)) # (4*B, C, H, W)
        y = torch.cat((y0, y1, y2, y3)) # (4*B,)

        # loss
        loss = F.cross_entropy(self.model(x), y)
        return loss