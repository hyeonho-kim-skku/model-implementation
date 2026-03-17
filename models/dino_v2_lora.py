import math
import timm
import torch
import torch.nn as nn
from .lora import LoRA

class DINOV2LoRA(nn.Module):
    def __init__(self, num_classes, rank=4):
        super().__init__()
        self.encoder = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, img_size=224, num_classes=0)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.w_a = nn.ModuleList()
        self.w_b = nn.ModuleList()

        for block in self.encoder.blocks:
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features

            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, rank)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, rank)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                qkv=w_qkv_linear,
                linear_a_q=w_a_linear_q,
                linear_b_q=w_b_linear_q,
                linear_a_v=w_a_linear_v,
                linear_b_v=w_b_linear_v,
            )
        
        self._reset_lora_weights()

        self.classifier = nn.Linear(self.encoder.embed_dim, num_classes)

    def _create_lora_layer(self, dim: int, rank: int):
        w_a = nn.Linear(dim, rank, bias=False)
        w_b = nn.Linear(rank, dim, bias=False)
        return w_a, w_b
    
    def _reset_lora_weights(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.encoder(x)
        logits = self.classifier(cls_token)
        return logits