import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, model, out_dim=128):
        super().__init__()

        # encoder
        self.encoder = model
        feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim)
        )

        # For knn evaluation
        self.model = self.encoder

    def _nt_xent_loss(self, zi, zj, temperature=0.5):
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
        (x1, x2), _ = batch
        
        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        loss = self._nt_xent_loss(z1, z2)

        return loss
