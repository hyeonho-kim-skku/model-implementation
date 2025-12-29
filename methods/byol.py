import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BYOL(nn.Module):
    def __init__(self, model, dim=256, hidden_dim=4096, m=0.996):
        super().__init__()

        self.m = m

        # Online network
        # Encoder
        self.encoder_online = model
        feat_dim = self.encoder_online.fc.in_features
        self.encoder_online.fc = nn.Identity()

        # Projector
        self.projector_online = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)            
        )

        # Target network (copy of online network)
        self.encoder_target = copy.deepcopy(self.encoder_online)
        self.projector_target = copy.deepcopy(self.projector_online)

        # Freeze target network parameters
        for param in self.encoder_target.parameters():
            param.requires_grad = False
        for param in self.projector_target.parameters():
            param.requires_grad = False
        
        # model: Encoder for knn evaluation
        self.model = self.encoder_online

    @torch.no_grad()
    def _update_target_network(self):
        # Update encoder
        for param_online, param_target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            param_target.data = (self.m * param_target.data) + ((1. - self.m) * param_online.data)
        # Update projector
        for param_online, param_target in zip(self.projector_online.parameters(), self.projector_target.parameters()):
            param_target.data = (self.m * param_target.data) + ((1. - self.m) * param_online.data)

    def _calculate_loss(self, predictor, projector_target):
        # Normalize
        predictor = F.normalize(predictor, dim=-1)
        projector_target = F.normalize(projector_target, dim=-1)
        
        # Loss: 2 - 2 * cosine_similarity
        return 2 - 2*(predictor*projector_target).sum(dim=-1)

    def forward(self, batch):
        (view1, view2), _ = batch

        # Online network forward
        # View 1
        v1_encoder_online = self.encoder_online(view1)
        v1_projector_online = self.projector_online(v1_encoder_online)
        v1_predictor = self.predictor(v1_projector_online)

        # View 2
        v2_encoder_online = self.encoder_online(view2)
        v2_projector_online = self.projector_online(v2_encoder_online)
        v2_predictor = self.predictor(v2_projector_online)

        # Target network forward
        with torch.no_grad():
            # Update target network
            self._update_target_network()

            # View 1
            v1_encoder_target = self.encoder_target(view1)
            v1_projector_target = self.projector_target(v1_encoder_target)

            # View 2
            v2_encoder_target = self.encoder_target(view2)
            v2_projector_target = self.projector_target(v2_encoder_target)

        # Compute loss: v1_predictor predicts v2_projector_target and vice versa
        loss1 = self._calculate_loss(v1_predictor, v2_projector_target)
        loss2 = self._calculate_loss(v2_predictor, v1_projector_target)

        # Mean loss
        loss = (loss1 + loss2).mean()

        return loss