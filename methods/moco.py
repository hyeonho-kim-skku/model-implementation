import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MoCo(nn.Module):
    def __init__(self, model, dim=128, K=4096, m=0.99, T=0.1):
        """
            model: base encoder (e.g., ResNet with projection head)
            dim: feature dimension (default: 128)
            K: queue size; number of negative keys (default: 4096)
            m: momentum coefficient for updating key encoder (default: 0.99)
            T: temperature parameter for softmax (default: 0.1)
        """
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = model # query encoder (trained by gradient)
        feat_dim = self.encoder_q.fc.in_features
        # MoCo
        # self.encoder_q.fc = nn.linear(512*self.encoder_q.block.expansion, 128)
        # MoCo-v2
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        self.model = self.encoder_q # train에서 이름 맞추기 위해.

        self.encoder_k = copy.deepcopy(self.encoder_q) # key encoder (updated by momentum)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False

        # register_buffer: nn.Module의 함수로, 업데이트되나 학습되지는 않음. 모델 저장, GPU 이동시 함께.
        self.register_buffer("queue", torch.randn(dim, K)) 
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long)) # queue 현재 위치 포인터.
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # ptr부터 batch_size만큼 keys로 대체.
        self.queue[:, ptr:ptr+batch_size] = keys.T
        # move pointer.
        self.queue_ptr[0] = (ptr+batch_size) % self.K

    def contrastive_loss(self, q, k, queue):
        # positive similarity.
        logits_pos = torch.einsum('nc,nc->n',[q, k]).unsqueeze(-1) # (B, 1)
        # negative similarities.
        logits_neg = torch.einsum('nc,ck->nk',[q, queue.clone().detach()]) # (B, K)
        # 둘을 합침.
        logits = torch.cat([logits_pos, logits_neg], dim = 1) # (B, K+1)

        # temprature 적용.
        logits /= self.T

        # label(positive)는 모두 0.
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # loss 계산.
        loss = F.cross_entropy(logits, labels)

        return loss
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.m*param_k.data + (1. - self.m)*param_q.data

    def forward(self, batch):
        (im_q, im_k), _ = batch

        q = self.encoder_q(im_q)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)

        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        loss= self.contrastive_loss(q, k, self.queue)

        self._dequeue_and_enqueue(k)
        
        return loss