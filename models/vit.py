import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
positional embedding: standard learnable 1D
dropout: (after every dense layer) except for the the qkv-projections and directly (after adding positional- to patch embeddings)
MLP: The MLP contains two layers with a GELU non-linearity.
"""

class FeedForward(nn.Module): # MLP(Multi-Layer Perceptron)
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        dim: 입력과 출력 벡터 차원(Transformer 차원)
        hidden_dim: 중간 은닉층 차원(MLP 확장 차원, Tansformer에서 4배)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), # 확장
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), # 축소
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class Attention(nn.Module): # MSA(Multi-head Self Attention)
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim) # 헤드가 1개고, 차원이 같으면 변환할 필요 없음.

        self.heads = heads
        self.scale = dim_head ** -0.5 # attention 식의 분모(sqrt(d_k))

        self.norm = nn.LayerNorm(dim) # Pre-Norm

        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False) # Eq.5~7의 U_qkv 부분. 효율을 위해 w_q, w_k, w_v를 한번에 계산.

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x) # Eq.3

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 긴 벡터 inner_dim을 여러개의 헤드(h)와 작은 차원(d)으로 쪼갬.
        # head 차원(h)을 시퀀스(n) 앞으로 보냄. 그래야 헤드별 독립 연산 가능.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # [Batch, Heads, N, Head_dim]

        dots = torch.matmul(q, k.transpose(-1, -2)*self.scale) # [B, H, N, N]토큰들이 서로 얼마나 관계있는지 score map.

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # [B, H, N, D] 확률값(attn)에 실제 정보(v)를 곱해서 섞음.
        out = rearrange(out, 'b h n d -> b n (h d)') # head를 다시 합쳐 inner_dim으로 복구.
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0. ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
            Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
            FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x # skip connection
            x = feedforward(x) + x # skip connection
        
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim, depth, heads, dim_head, mlp_dim, emb_dropout=0., dropout=0., num_classes=10):
        """
        dim: transformer 내부 벡터 크기.
        """
        super(ViT, self).__init__()
        image_height, image_width = image_size # CIFAR10은 32, 32
        patch_height, patch_width = patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width # 패치 1개를 펼쳤을 때.

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), # (h w)는 시퀀스 길이.
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim), # 패치의 특징 추출.
            # nn.LayerNorm(dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.positional_embedding = nn.Parameter(torch.randn(num_patches + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, num_classes)
    
    def forward(self, img):
        batch_size = img.shape[0]
        
        x = self.to_patch_embedding(img) # [B, C, H, W] -> [B, num_patches, dim]

        cls_tokens = repeat(self.cls_token, '1 d -> b 1 d', b = batch_size) # [1, dim] -> [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, dim]

        x = x + self.positional_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = x[:,0] # CLS 토큰만 가져옴.
        x = self.mlp_head(x)
        return x