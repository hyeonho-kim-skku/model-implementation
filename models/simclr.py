import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import *

class SimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        # ResNet20 backbone 가져오기.
        self.encoder = ResNet(BasicBlock, num_blocks=[3, 3, 3])
        feat_dim = 64*BasicBlock.expansion
        
        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim)
        )
    
    def forward(self, x):
        # encoder로 representation 추출
        h = self.encoder.forward_features(x) # (B, feat_dim)
        z = self.projection_head(h) # (B, out_dim)
        return h, z

class SimCLRClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet(BasicBlock, [3,3,3])
        feat_dim = 64*BasicBlock.expansion
        self.fc = nn.Linear(feat_dim, 10)

        state = torch.load('./checkpoint/simclr_ckpt.pth', map_location="cuda:0", weights_only=True)
        
        # encoder만 추출하며, "encoder." 제거.
        new_state_dict = {}
        for k, v in state['model'].items():
            if k.startswith('encoder.'):
                new_state_dict[k[8:]] = v # "encoder." 제거

        self.encoder.load_state_dict(new_state_dict)

        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        h = self.encoder.forward_features(x)
        logits = self.fc(h)
        return logits

