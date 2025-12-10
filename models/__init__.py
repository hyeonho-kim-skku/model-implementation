import torch.nn as nn
import torch
from .resnet import *
from .pre_act_resnet import *
from .fractalnet import *
from .densenet import *
from .vit import *
from .mlp_mixer import *
from .conv_mixer import *


def load_model(model, **kwargs):
    if model == 'fractalnet':
        return FractalNet()
    elif model == 'pre_act_resnet':
        return PreActResNet(kwargs.get('block', PreActBottleneck), kwargs.get('num_blocks', [18, 18, 18]))
    elif model == 'densenet':
        return DenseNet(block=DenseBottleneck, num_blocks=[16, 16, 16], growth_rate=12, compression_factor=0.5)
    elif model == 'vit':
        return ViT(image_size=(32,32), patch_size=(4,4), channels=3, dim=256, depth=6, heads=8, dim_head=64, mlp_dim=512, emb_dropout=0.1, dropout=0.1, num_classes=10)
    elif model == 'mlp_mixer':
        return MlpMixer(num_classes=10, num_blocks=8, patch_size=4, hidden_dim=512, tokens_mlp_dim=256, channels_mlp_dim=2048, image_size=32) # Mixer-S/16과 비슷.
    elif model == 'conv_mixer':
        return ConvMixer(dim=256, depth=8, kernel_size=9, patch_size=1, num_classes=10)

# def load_model_bak(model_name, block, num_blocks):
#     if model_name == 'pre_act_resnet':
#         return PreActResNet(PreActBottleneck, num_blocks)

def load_resnet(block, num_blocks):
    if block == 'BasicBlock':
        return ResNet(BasicBlock, num_blocks)
    else:
        return ResNet(Bottleneck, num_blocks)

def load_criterion(criterion_name):
    if criterion_name == 'crossentropyloss':
        return nn.CrossEntropyLoss()

def load_optimizer(optimizer_name, model, lr, weight_decay, momentum=None):
    if optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def load_scheduler(scheduler_name, optimizer, num_epochs):
    if scheduler_name == 'MultiStepLR':
        milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]  # [50% epoch, 75% ecpoh]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
