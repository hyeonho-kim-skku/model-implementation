import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        # depthwise convolution with residual connection
        self.depthwise_block = Residual(
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )
        )

        # pointwise convolution (1x1 convolution)
        self.pointwise_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
    
    def forward(self, x):
        out = self.depthwise_block(x)
        out = self.pointwise_block(out)
        return out

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size, patch_size, num_classes):
        super().__init__()
        # patch embedding.
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

        # ConvBlock stack
        self.conv_blocks = nn.Sequential(*[ConvBlock(dim, kernel_size) for _ in range(depth)])

        # classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), # (B, dim, 1, 1)
            nn.Flatten(), # (B, dim)
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        out = self.patch_embed(x)
        out = self.conv_blocks(out)
        out = self.head(out)
        return out
