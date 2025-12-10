import torch.nn as nn
from einops import rearrange

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, in_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.fc2(out)
        return out

class MixerBlock(nn.Module):
    """
    - Token mixing MLP (spatial dimension 혼합)
    - Channel mixing MLP (feature dimension 혼합)
    """
    def __init__(self, dim, seq_len, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixing = MlpBlock(seq_len, tokens_mlp_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixing = MlpBlock(dim, channels_mlp_dim)

    def forward(self, x): # x: (B, S, C)
        # Token mixing: (B, S, C) -> (B, C, S) -> MLP -> (B, S, C)
        y = self.norm1(x)
        y = y.transpose(1, 2) # (B, S, C) -> (B, C, S)
        y = self.token_mixing(y)
        y = y.transpose(1, 2) # (B, C, S) -> (B, S, C)
        x = x + y # skip-connection

        # Channel-mixing: (B, S, C) -> MLP -> (B, S, C)
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y # skip-connection
        return x

class MlpMixer(nn.Module):
    def __init__(self,
                 num_classes,
                 num_blocks,
                 patch_size,
                 hidden_dim,  # C
                 tokens_mlp_dim, # D_S
                 channels_mlp_dim, # D_C
                 image_size):
        super().__init__()
        # Patch embedding layer
        self.stem = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)

        # sequence length
        seq_len = (image_size // patch_size) ** 2

        # Mixer blocks
        self.mixer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mixer_blocks.append(MixerBlock(hidden_dim, seq_len, tokens_mlp_dim, channels_mlp_dim))
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Patch embedding: (B, 3, H, W) -> (B, C, H', W')
        x = self.stem(x)

        # Patch -> Sequence: (B, C, H', W') -> (B, S, C)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Mixer blocks
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        # classification
        x = self.norm(x)
        x = x.mean(dim=1) # Global average pooling. 패치차원에 대한 평균. (B, S, C) -> (B, C)
        x = self.head(x)
        
        return x