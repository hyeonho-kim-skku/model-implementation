import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if drop_rate > 0.0:
            self.dropout = nn.Dropout2d(p=drop_rate)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

"""
class FractalBlock():
    def __init__(self, depth):
        self.path1 = ConvBlock()
        self.path2 = nn.ModuleList([FractalBlock(depth-1), FractalBlock(depth-1)])

    def forward(self, x): # x -> [y1, y2, ..., yK], k = number of columns
        y1 = self.path1(x)
        [y2, ..., yK] = self.path2[0](x)
        [y2, ..., yK] = self.path2[1](join(y2, ..., yK))
        return [y1, ..., yK]
"""

class FractalBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, dropout_rate, local_drop_rate=0.15):
        super().__init__()
        self.depth = depth
        self.local_drop_rate = local_drop_rate

        if depth == 1:
            self.path1 = ConvBlock(in_channels, out_channels, dropout_rate)
        else:
            self.path1 = ConvBlock(in_channels, out_channels, dropout_rate)
            self.path2 = nn.ModuleList([FractalBlock(depth-1, in_channels, out_channels, dropout_rate),
                                        FractalBlock(depth-1, out_channels, out_channels, dropout_rate)])

    def join(self, y2_yK, global_chosen_columns): # y2_yK = [(B, C, H, W)_2, ..., (B, C, H, W)_K] -> element-wise mean (B, C, H, W)
        if not self.training:
            return torch.mean(torch.stack(y2_yK), dim=0)    
        
        B = y2_yK[0].size(0) # batch size.
        num_inputs = len(y2_yK)
        device = y2_yK[0].device

        # global drop path:
        if global_chosen_columns is not None:
            column_indices = torch.arange(num_inputs, 0, -1, device=device).repeat(B, 1) # (B, num_inputs), 뒤에서 앞으로 1, 2, ... 증가. (e.g. [[3,2,1], [3,2,1], ...])
            global_drop_mask = (column_indices == global_chosen_columns.view(B, 1)).float() # (B, num_inputs), 선택된 column에만 1. mask.
            global_drop_mask = global_drop_mask.view(B, num_inputs, 1, 1, 1)

            global_masked = torch.stack(y2_yK, dim=1) * global_drop_mask
            return global_masked.sum(dim=1) / global_drop_mask.sum(dim=1).clamp(min=1) # join 연산에 모두 참여하지 않을경우, clamp 사용하여 0 / 1로 처리.
        # local drop path:
        else:
            keep_prob = 1. - self.local_drop_rate
            local_drop_mask = torch.bernoulli(torch.full((B, num_inputs), keep_prob, device=device)) # [B, num_inputs]
            
            # 모두 dead인 경우 처리.
            dead_mask = (local_drop_mask.sum(dim=1) == 0)
            if dead_mask.any():
                dead_indices = torch.where(dead_mask)[0]
                random_choices = torch.randint(0, num_inputs, (dead_indices.size(0),), device=device)
                local_drop_mask[dead_indices, random_choices] = 1.

            # mask 적용.
            local_drop_mask = local_drop_mask.view(B, num_inputs, 1, 1, 1)
            local_masked = torch.stack(y2_yK, dim=1) * local_drop_mask

            return local_masked.sum(dim=1) / local_drop_mask.sum(dim=1)

    def forward(self, x, global_chosen_columns): # x -> [y1, y2, ..., yK], K = number of columns, x: (B, C_in, H, W)
        if self.depth == 1:
            out = [self.path1(x)]
        else:
            y1 = [self.path1(x)] # (B, C_in, H, W) -> [(B, C_out, H, W)]
            y2_yK = self.path2[0](x, global_chosen_columns)
            y2_yK = self.path2[1](self.join(y2_yK, global_chosen_columns), global_chosen_columns)
            out = y1 + y2_yK
        return out

class FractalNet(nn.Module):
    def __init__(self, num_columns=4, num_blocks=5):
        super().__init__()
        self.num_columns = num_columns
        in_channels = 3
        out_channels = [64, 128, 256, 512, 512] # num_filters
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4]

        self.layers = nn.ModuleList()
        for i in range(num_blocks):
            fractal_block = FractalBlock(num_columns, in_channels, out_channels[i], dropout_rates[i])
            self.layers.append(fractal_block)
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels[i]
        
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(512*1*1, 10))

    def forward(self, x):
        out = x

        global_chosen_columns = None
        B = x.size(0) # batch size
        if self.training:
            global_drop_ratio = 0.5
            if torch.rand(1).item() < global_drop_ratio: # 배치별로 global or local 선택.
                global_chosen_columns = torch.randint(1, self.num_columns+1, (B,), device=x.device) # 샘플 별 컬럼 1개 선택. 오른쪽에서 왼쪽으로 1, 2, ...

        for layer in self.layers:
            if isinstance(layer, FractalBlock):
                out = layer(out, global_chosen_columns)
                out = torch.mean(torch.stack(out), dim=0) # block의 마지막 join.
            else:
                out = layer(out)
        return out