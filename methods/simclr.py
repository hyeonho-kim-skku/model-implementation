import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, model, out_dim=128):
        super().__init__()

        self.encoder = model
        feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
        )

        self.model = self.encoder

    def _nt_xent_loss(self, zi, zj, temperature=0.5):
        """
        Compute NT-Xent loss for a batch of paired projections zi and zj.
        """
        batch_size = zi.size(0)

        z = torch.cat([zi, zj], dim=0)
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T) / temperature
        similarity_matrix.fill_diagonal_(float('-inf'))
        labels = torch.cat([
                torch.arange(batch_size, 2*batch_size), # positive pairs for zi
                torch.arange(0, batch_size) # positive pairs for zj
            ], dim=0).to(z.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

    def forward(self, batch):
        (x1, x2), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        loss = self._nt_xent_loss(z1, z2)

        return loss
