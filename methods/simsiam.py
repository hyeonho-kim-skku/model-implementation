import torch.nn.functional as F
import torch.nn as nn

class SimSiam(nn.Module):
    def __init__(self, model, dim=2048, pred_dim=512):
        super().__init__()

        # Encoder는 backbone + projector로 구성.
        self.backbone = model
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 3-layer projector.
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim, bias=False),
            nn.BatchNorm1d(dim, affine=False)
        )

        # 2-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim,dim)
        )

        # For knn evaluation
        self.model = self.backbone

    def _calculate_loss(self, p, z):
        """
        Negative cosine similarity loss with stop-gradient
        """
        # Stop gradient
        z = z.detach()

        # L2-normalize
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        # Negative cosine similarity
        return -(p * z).sum(dim=1).mean()

    def forward(self, batch):
        (x1, x2), _ = batch

        # projections
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # predictions
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Symmetrized loss
        loss = self._calculate_loss(p1, z2) / 2 + self._calculate_loss(p2, z1) / 2

        return loss