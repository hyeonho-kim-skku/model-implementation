import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
on CIFAR
SGD
batch size: 64
epochs: 300
initial learning rate: 0.1, divided by 50% and 75% total training epochs
weight decay: 1e-4
momentum: Nesterov momentum of 0.9 without dampening.
layer 구성: Three dense blocks that each has an equal number of layers.

"""
# BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)
# "In our experiments, we let each 1×1 convolution produce 4k feature-maps."
class DenseBottleneck(nn.Module): # H
    def __init__(self, in_channels, growth_rate): # growth_rate:k
        super(DenseBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = torch.cat([x, out], dim=1)
        return out

# BN-ReLu-Conv(1x1)-Avg.pool(2x2)
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        out = self.conv(out)
        out = self.pool(out)
        return out

"""
"Before entering the first dense block, a convolution with 16 (or twice the growth rate for DenseNet-BC) output channels is performed on the input images."
"On all datasets except ImageNet, the DenseNet used in our experiments has three dense blocks that each has an equal number of layers."
"At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached.
"""
class DenseNet(nn.Module):
    def __init__(self, block, num_blocks, growth_rate, compression_factor=0.5):
        super(DenseNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 2*growth_rate, kernel_size=3, padding=1, bias=False)

        in_channels = 2*growth_rate
        self.dense_block1 = self._make_dense_block(block, in_channels, growth_rate, num_blocks[0])
        in_channels += num_blocks[0]*growth_rate
        out_channels = int(math.floor(in_channels*compression_factor))
        self.trans1 = Transition(in_channels, out_channels) # Transition Layer로 채널 수 줄임.
        
        in_channels = out_channels
        self.dense_block2 = self._make_dense_block(block, in_channels, growth_rate, num_blocks[1])
        in_channels += num_blocks[1]*growth_rate
        out_channels = int(math.floor(in_channels*compression_factor))
        self.trans2 = Transition(in_channels, out_channels)

        in_channels = out_channels
        self.dense_block3 = self._make_dense_block(block, in_channels, growth_rate, num_blocks[2])
        final_channels = in_channels + num_blocks[2]*growth_rate
        
        self.gap = nn.AdaptiveAvgPool2d(1) # global average pooling. 채널별 평균을 구함.
        self.linear = nn.Linear(final_channels, 10)
    
    def _make_dense_block(self, block, in_channels, growth_rate, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_channels, growth_rate))
            in_channels += growth_rate # concatenation하면 featuer map이 k(growth_rate) 만큼 증가.
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense_block1(out)
        out = self.trans1(out)
        out = self.dense_block2(out)
        out = self.trans2(out)
        out = self.dense_block3(out)
        out = self.gap(out) # [B, C, 1, 1]
        out = out.view(out.size(0), -1) # [B, C]
        out = self.linear(out)
        return out