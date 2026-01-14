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

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        dim: 입력과 출력 벡터 차원(Transformer 차원)
        hidden_dim: 중간 은닉층 차원(MLP 확장 차원, Tansformer에서 4배)
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class MultiHeadSelfAttention(nn.Module): # MSA(Multi-head Self Attention)
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5 # attention 식의 분모(sqrt(d_k))

        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False) # Eq.5~7의 U_qkv 부분. 효율을 위해 w_q, w_k, w_v를 한번에 계산.

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 긴 벡터 inner_dim을 여러개의 헤드(h)와 작은 차원(d)으로 쪼갬.
        # head 차원(h)을 시퀀스(n) 앞으로 보냄. 그래야 헤드별 독립 연산 가능.
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # [Batch, Heads, N, Head_dim]

        dots = torch.matmul(q, k.transpose(-1, -2))*self.scale # [B, H, N, N]토큰들이 서로 얼마나 관계있는지 score map.

        attn = self.softmax(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # [B, H, N, D] 확률값(attn)에 실제 정보(v)를 곱해서 섞음.
        out = rearrange(out, 'b h n d -> b n (h d)') # head를 다시 합쳐 inner_dim으로 복구.
        out = self.to_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0. ):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSelfAttention(dim, heads, dim_head, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)
        
    def forward(self, x):
        out = x + self.mhsa(self.ln1(x))
        out = out + self.mlp(self.ln2(out))
        return out

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, channels, dim, depth, heads, dim_head, mlp_dim, emb_dropout=0., dropout=0., num_classes=10):
        """
        dim: transformer 내부 벡터 크기.
        """
        super(ViT, self).__init__()
        image_height, image_width = image_size # CIFAR10은 32, 32
        patch_height, patch_width = patch_size

        self.patch_size = patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width # 패치 1개를 펼쳤을 때.

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width), # (h w)는 시퀀스 길이.
            # nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim), # 패치의 특징 추출.
            # nn.LayerNorm(dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # (1, 1, D)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # (1, N+1, D)

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.ModuleList(
            TransformerBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)
        )

        self.fc = nn.Linear(dim, num_classes)
    
    def interpolate_positional_embedding(self, x, w, h):
        """
        입력 이미지의 크기(w, h)에 맞춰서, 기존에 학습된 Positional Embedding을
        Interpolation(보간)하여 반환하는 함수입니다.
        Multi-Crop 학습 시 다양한 크기의 이미지를 처리하기 위해 필수적입니다.
        """
        # 1. 현재 입력(x)의 패치 개수 확인 (CLS 토큰 제외)
        npatch = x.shape[1] - 1

        # 2. 기존에 학습된 패치 개수 확인 (CLS 토큰 제외)
        N = self.positional_embedding.shape[1] - 1
        
        # 3. 이미 학습된 패치 개수와 동일하면 원본 PE 반환 (정사각형 이미지 가정)
        if npatch == N:
            return self.positional_embedding
        
        # 4. CLS 토큰과 패치 토큰 분리
        # 현재 PE interpolation은 공간(spatial) 정보를 보간하는 것인데,
        # CLS 토큰의 PE는 공간(spatial)의 의미가 아닌 추상적인 대표성(Role)을 가지므로 제외해야 함.
        cls_pos_embedding = self.positional_embedding[:, 0] # (1, D)
        patch_pos_embedding = self.positional_embedding[:, 1:] # (1, N, D)

        dim = x.shape[-1]

        # 5. F.interpolate 위해 2차원으로 변환
        w0 = h0 = int(N ** 0.5)
        patch_pos_embedding = patch_pos_embedding.reshape(1, w0, h0, dim).permute(0, 3, 1, 2) # (1, D, w0, h0)

        # 6. 보간 수행
        w_new = h_new = int(npatch ** 0.5)
        patch_pos_embedding = F.interpolate(patch_pos_embedding, size=(w_new, h_new), mode='bicubic')

        # 7. 다시 1차원 변환
        patch_pos_embedding = patch_pos_embedding.permute(0, 2, 3, 1).reshape(1, w_new * h_new, dim) # (1, N_new, D)

        # 8. CLS 토큰 다시 결합
        new_pos_embedding = torch.cat((cls_pos_embedding.unsqueeze(1), patch_pos_embedding), dim=1) # (1, N_new + 1, D)
        return new_pos_embedding

    # CLS 토큰을 이용한 특징 추출 부분
    def forward_features(self, x):
        B, C, H, W = x.shape

        x = self.to_patch_embedding(x) # [B, C, H, W] -> [B, num_patches, dim]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B) # [1, 1, dim] -> [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1) # [B, num_patches + 1, dim]

        # 단순 덧셈 대신 보간 함수 사용
        # (B, N+1, D) + (1, N+1, D) -> Broadcasting
        x = x + self.interpolate_positional_embedding(x, W, H)
        
        x = self.dropout(x)

        for block in self.transformer:
            x = block(x)

        x = x[:,0] # CLS 토큰만 가져옴.
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

def vit_tiny():
    return ViT(
        image_size = (32, 32),
        patch_size = (4, 4),
        channels = 3,
        dim = 192,
        depth = 12,
        heads = 3,
        dim_head = 64,
        mlp_dim = 768,
        emb_dropout = 0.1,
        dropout = 0.1,
        num_classes = 10
    )