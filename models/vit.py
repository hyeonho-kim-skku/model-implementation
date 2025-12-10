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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # --- Attention Part ---
                nn.LayerNorm(dim),                               # 0: Attention Pre-Norm
                nn.Linear(dim, inner_dim * 3, bias=False),       # 1: to_qkv
                nn.Dropout(dropout),                             # 2: attn_dropout (softmax 후)
                nn.Sequential(                                   # 3: to_out (Linear + Dropout)
                    nn.Linear(inner_dim, dim),
                    nn.Dropout(dropout)
                ),
                
                # --- MLP Part ---
                nn.LayerNorm(dim),                               # 4: MLP Pre-Norm
                nn.Sequential(                                   # 5: MLP Network
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

        self.norm = nn.LayerNorm(dim) # Final Norm
        
    def forward(self, x):
        # layers 리스트에서 각 구성 요소를 꺼내어 절차적으로 연산
        for norm_attn, to_qkv, attn_dropout, to_out, norm_mlp, mlp_net in self.layers:
            
            # --- 1. Attention Logic ---
            residual = x
            x_norm = norm_attn(x) # Pre-Norm
            
            # QKV 계산 및 분할
            qkv = to_qkv(x_norm).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

            # Attention Score 계산
            dots = torch.matmul(q, k.transpose(-1, -2) * self.scale)
            attn = F.softmax(dots, dim=-1)
            attn = attn_dropout(attn)

            # 가중합 및 복원
            attn_out = torch.matmul(attn, v)
            attn_out = rearrange(attn_out, 'b h n d -> b n (h d)')
            attn_out = to_out(attn_out)
            
            x = residual + attn_out # Skip Connection

            # --- 2. MLP Logic ---
            residual = x
            x_norm = norm_mlp(x) # Pre-Norm
            mlp_out = mlp_net(x_norm)
            
            x = residual + mlp_out # Skip Connection

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