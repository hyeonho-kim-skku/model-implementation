import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def nt_xent_loss(zi, zj, temperature=0.5):
        batch_size = zi.size(0)

        # Concatenation, zi <-> zj 간 유사도 한 번에 구하기 위해.
        z = torch.cat([zi, zj], dim=0)
        # L2 Normalization.
        z = F.normalize(z, dim=1)
        # similarity matrix.
        similarity_matrix = torch.matmul(z, z.T) / temperature

        # positive pair
        mask_positive = torch.zeros(2*batch_size, 2*batch_size, dtype=torch.bool, device=z.device)
        for i in range(batch_size):
            mask_positive[i, i+batch_size] = True
            mask_positive[i+batch_size, i] = True    
        # anchor마다 positive만 추출.
        positive_similarities = similarity_matrix[mask_positive].view(2*batch_size, -1) # (2N, 1)

        # negative 구하기.
        mask_self = torch.eye(2*batch_size, dtype=torch.bool, device=z.device) # 자기 자신과의 유사도
        mask_negative = ~mask_self & ~mask_positive # 자신과 positive 제외.
        negative_similarities = similarity_matrix[mask_negative].view(2*batch_size, -1) # (2N, 2N-2)

        # positive를 맨앞에 두고, 뒤에 negative.
        logits = torch.cat([positive_similarities, negative_similarities], dim=1)
        # 정답 label은 항상 0(positive).
        labels = torch.zeros(2*batch_size, dtype=torch.long, device=z.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, batch):
        (x1, x2), _ = batch # x1, x2: (B, C, H, W)
        
        # encoder + projection head
        _, z1 = self.model(x1) # z1: (B, out_dim)
        _, z2 = self.model(x2) # z2: (B, out_dim)

        loss = nt_xent_loss(z1, z2)
        return loss
