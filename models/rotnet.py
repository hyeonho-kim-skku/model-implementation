import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        padding = (kernel_size-1)//2
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

"""https://github.com/gidariss/FeatureLearningRotNet/blob/master/architectures/NetworkInNetwork.py"""
class RotNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        num_classes = opt['num_classes']
        num_stages = opt['num_stages'] if ('num_stages' in opt) else 3

        assert(num_stages >= 3)
        nChannels  = 192
        nChannels2 = 160
        nChannels3 = 96

        blocks = [nn.Sequential() for i in range(num_stages)]
        # 1st block
        blocks[0].add_module('Block1_ConvB1', BasicBlock(3, nChannels, 5))
        blocks[0].add_module('Block1_ConvB2', BasicBlock(nChannels,  nChannels2, 1))
        blocks[0].add_module('Block1_ConvB3', BasicBlock(nChannels2, nChannels3, 1))
        blocks[0].add_module('Block1_MaxPool', nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        # 2nd block
        blocks[1].add_module('Block2_ConvB1',  BasicBlock(nChannels3, nChannels, 5))
        blocks[1].add_module('Block2_ConvB2',  BasicBlock(nChannels,  nChannels, 1))
        blocks[1].add_module('Block2_ConvB3',  BasicBlock(nChannels,  nChannels, 1))
        blocks[1].add_module('Block2_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1)) # (B, 192, 8, 8)

        # 3rd block
        blocks[2].add_module('Block3_ConvB1',  BasicBlock(nChannels, nChannels, 3))
        blocks[2].add_module('Block3_ConvB2',  BasicBlock(nChannels, nChannels, 1))
        blocks[2].add_module('Block3_ConvB3',  BasicBlock(nChannels, nChannels, 1))

        for s in range(3, num_stages):
            blocks[s].add_module('Block'+str(s+1)+'_ConvB1',  BasicBlock(nChannels, nChannels, 3))
            blocks[s].add_module('Block'+str(s+1)+'_ConvB2',  BasicBlock(nChannels, nChannels, 1))
            blocks[s].add_module('Block'+str(s+1)+'_ConvB3',  BasicBlock(nChannels, nChannels, 1))
        
        # global average pooling and classifier
        blocks.append(nn.Sequential(
            GlobalAveragePooling(),
            nn.Linear(nChannels, num_classes)
        ))

        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier']
    
    def forward(self, x, out_feat_keys=None):
        """
        out_feat_keys=None      → 마지막 classifier 출력 반환 (pretraining)
        out_feat_keys=['conv2'] → conv2 블록의 feature map만 반환 (evaluation)
    	"""
        if out_feat_keys is None:
            out_feat_keys = ['classifier']
        
        max_out_feat = max([self.all_feat_names.index(k) for k in out_feat_keys]) # out-feat max만큼 forward 진행.
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat+1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat
        
        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
        return out_feats
    
class RotNetConv2Classifier(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = RotNet({'num_classes': 4, 'num_stages': 4}) # RotNet pretrained, backbone num_classes 4로 가져와도 forward에서 convB2까지만 진행됨.
        num_classes = opt['num_classes'] # 10
        
        # 가중치 불러오기
        state = torch.load('./checkpoint/rotnet_pretrain_ckpt.pth', map_location="cuda:0", weights_only=True)
        self.backbone.load_state_dict(state['model'])

        # feature extractor 부분은 freeze.
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        in_channels = 192 # ConvB2 기준.
        """https://github.com/gidariss/FeatureLearningRotNet/blob/master/architectures/NonLinearClassifier.py"""
        # convolution classifier 정의
        self.classifier = nn.Sequential(
            # conv2 feature: (B, 192, 8, 8) 기준
            BasicBlock(in_channels, 192, kernel_size=3),
            BasicBlock(192, 192, kernel_size=1),
            BasicBlock(192, 192, kernel_size=1),
            GlobalAveragePooling(), # (B, 192)
            nn.Linear(192, num_classes),  # (B, 10)
        )

        # (convolution classifier로 대체) conv2 feature shape: (B, 192, 8, 8)
        # in_dim = 192 * 8 * 8
        # self.classifier = nn.Linear(in_dim, 10)
    def forward(self, x):
        # conv2 feature만 뽑기.
        feat = self.backbone(x, out_feat_keys=['conv2']) # (B, 192, 8, 8)
        out = self.classifier(feat)
        return out