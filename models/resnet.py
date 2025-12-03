import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, first_stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=first_stride,
                            padding=1,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # projection shortcut 사용.
        self.shortcut = nn.Identity()
        if first_stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride, bias=False)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, middle_channels, first_stride):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3, stride=first_stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = nn.Conv2d(middle_channels, middle_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(middle_channels*self.expansion)

        self.shortcut = nn.Identity()
        if first_stride != 1 or in_channels != middle_channels*self.expansion:
            self.shortcut = nn.Conv2d(in_channels, middle_channels*self.expansion, kernel_size=1, stride=first_stride, bias=False)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layers1 = self._make_layers(block, num_blocks[0], 16, 16, 1)
        self.layers2 = self._make_layers(block, num_blocks[1], 16*block.expansion, 32, 2)
        self.layers3 = self._make_layers(block, num_blocks[2], 32*block.expansion, 64, 2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1) # global average pooling.
        self.fc = nn.Linear(in_features=64*block.expansion, out_features=10)
        
    def _make_layers(self, block, num_blocks, first_channel, out_channels, first_stride): # Bottleneck경우 out_channels는 middle_channels를 의미.
        strides = [first_stride] + [1]*(num_blocks-1)
        in_channels = [first_channel] + [out_channels*block.expansion]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            layers.append(block(in_channels[i], out_channels, strides[i]))
        return nn.Sequential(*layers) # unpacking.

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out