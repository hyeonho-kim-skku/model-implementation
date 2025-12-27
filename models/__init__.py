import torch.nn as nn
import torch
from utils import *
from .resnet import *
from .pre_act_resnet import *
from .fractalnet import *
from .densenet import *
from .vit import *
from .mlp_mixer import *
from .conv_mixer import *
from .rotnet import *
from .simclr import *

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
    elif model == 'rotnet_pretrain':
        opt = {}
        opt['num_classes'] = 4
        opt['num_stages'] = 4
        return RotNet(opt)
    elif model == 'rotnet_classifier':
        opt = {}
        opt['num_classes'] = 10
        opt['num_stages'] = 4
        return RotNetConv2Classifier(opt)
    elif model == 'simclr':
        return SimCLR()
    elif model == 'simclr_classifier':
        return SimCLRClassifier()
