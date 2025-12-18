# ntxent.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        # positive pair 인덱스 마스크 (2B x 2B에서 diagonal 제외)
        self.mask = self._get_correlated_mask(batch_size).to(device)

        # 각 sample의 positive index (2B 길이)
        self.positives_mask = self._get_positives_mask(batch_size).to(device)

    def _get_correlated_mask(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(False)  # 자기 자신 제거
        return mask

    def _get_positives_mask(self, batch_size):
        # 2B개의 벡터: [z1_i, z2_i, z1_j, z2_j, ...]라고 할 때
        # i와 j는 서로의 positive
        N = 2 * batch_size
        positives = torch.zeros((N, N), dtype=torch.bool)
        for k in range(batch_size):
            i, j = k, k + batch_size
            positives[i, j] = True
            positives[j, i] = True
        return positives

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (B, D)
        """
        batch_size = z_i.size(0)
        assert batch_size == self.batch_size, "batch_size mismatch!"

        # 1) concat: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # 2) L2 normalize
        z = F.normalize(z, dim=1)

        # 3) similarity matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature  # cosine similarity (norm=1 이므로 dot=cos)

        # 4) positive, negative 분리
        # positives: (2B, 1)
        positives = sim[self.positives_mask].view(2 * batch_size, 1)
        # negatives: (2B, 2B-2)
        negatives = sim[self.mask].view(2 * batch_size, -1)

        # 5) logits: concat[positive, negatives]
        logits = torch.cat([positives, negatives], dim=1)

        # 6) label: 항상 0번이 positive
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(self.device)

        loss = F.cross_entropy(logits, labels, reduction='mean')
        return loss
