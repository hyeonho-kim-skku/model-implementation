import torch.nn as nn
from utils import *
from .resnet import *
from .pre_act_resnet import *
from .fractalnet import *
from .densenet import *
from .vit import *
from .mlp_mixer import *
from .conv_mixer import *
from .rotnet import *
from .timm_classifier import TIMMClassifier
from .timm_lora import TIMMLoRA
from .timm_pruned_linear_probe import TIMMPrunedLinearProbe
from .timm_pruned_lora import TIMMPrunedLoRA
from .timm_pruned_prompt import TIMMPrunedPromptRecovery, TIMMPrunedVPT

def load_model(**kwargs):
    model = kwargs.get('model')

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
    elif model == 'resnet18':
        return ResNet18()
    elif model == 'resnet50':
        return ResNet50()
    elif model == 'vit_tiny':
        return vit_tiny()
    elif model == 'timm_classifier':
        return TIMMClassifier(
            backbone_name=kwargs.get('backbone_name'),
            num_classes=kwargs.get('num_classes'),
            pretrained=kwargs.get('pretrained', False),
            img_size=kwargs.get('img_size'),
            freeze_encoder=kwargs.get('freeze_encoder', False),
            classifier_init=kwargs.get('classifier_init', 'random'),
        )
    elif model == 'timm_lora':
        return TIMMLoRA(
            backbone_name=kwargs.get('backbone_name'),
            num_classes=kwargs.get('num_classes'),
            rank=kwargs.get('lora_rank', 4),
            pretrained=kwargs.get('pretrained', True),
            img_size=kwargs.get('img_size'),
            lora_alpha=kwargs.get('lora_alpha'),
            qkv_lora_components=kwargs.get('qkv_lora_components'),
            lora_modules=kwargs.get('lora_modules'),
        )
    elif model == 'timm_pruned_lora':
        return TIMMPrunedLoRA(
            artifact_path=kwargs.get('artifact_path'),
            rank=kwargs.get('lora_rank', 4),
            lora_alpha=kwargs.get('lora_alpha'),
            qkv_lora_components=kwargs.get('qkv_lora_components'),
            lora_modules=kwargs.get('lora_modules'),
            reset_classifier=kwargs.get('reset_classifier', False),
            num_classes=kwargs.get('num_classes'),
        )
    elif model == 'timm_pruned_linear_probe':
        return TIMMPrunedLinearProbe(
            artifact_path=kwargs.get('artifact_path'),
            reset_classifier=kwargs.get('reset_classifier', True),
            num_classes=kwargs.get('num_classes'),
            freeze_encoder=kwargs.get('freeze_encoder', True),
        )
    elif model == 'timm_pruned_vpt':
        return TIMMPrunedVPT(
            artifact_path=kwargs.get('artifact_path'),
            prompt_mode=kwargs.get('prompt_mode', 'shallow'),
            num_prompt_tokens=kwargs.get('num_prompt_tokens', 1),
            reset_classifier=kwargs.get('reset_classifier', True),
            num_classes=kwargs.get('num_classes'),
            prompt_init_std=kwargs.get('prompt_init_std', 0.02),
            prompt_tokens_per_layer=kwargs.get('prompt_tokens_per_layer'),
            prompt_allocation_label=kwargs.get('prompt_allocation_label'),
        )
    elif model == 'timm_pruned_prompt':
        return TIMMPrunedPromptRecovery(
            artifact_path=kwargs.get('artifact_path'),
            prompt_components=kwargs.get('prompt_components', 'kv'),
            prompt_mode=kwargs.get('prompt_mode', 'deep'),
            num_prompt_tokens=kwargs.get('num_prompt_tokens', 1),
            reset_classifier=kwargs.get('reset_classifier', True),
            num_classes=kwargs.get('num_classes'),
            prompt_init_std=kwargs.get('prompt_init_std', 0.02),
            prompt_tokens_per_layer=kwargs.get('prompt_tokens_per_layer'),
            num_kv_prompt_tokens=kwargs.get('num_kv_prompt_tokens', 5),
            kv_prompt_tokens_per_layer=kwargs.get('kv_prompt_tokens_per_layer'),
            share_kv_prompt=kwargs.get('share_kv_prompt', True),
            prompt_allocation_label=kwargs.get('prompt_allocation_label'),
            lora_rank=kwargs.get('lora_rank'),
            lora_alpha=kwargs.get('lora_alpha'),
            lora_modules=kwargs.get('lora_modules'),
            qkv_lora_components=kwargs.get('qkv_lora_components'),
            initial_recovery_checkpoint=kwargs.get('initial_recovery_checkpoint'),
        )
