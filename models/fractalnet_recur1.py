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

class FractalBlock(nn.Module):
    def __init__(self, C, in_channels, out_channels, dropout_rate, local_drop_rate=0.15): # C: number of columns
        super().__init__()
        self.C = C
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.local_drop_rate = local_drop_rate

        if C == 1:
            self.path1 = ConvBlock(in_channels, out_channels, dropout_rate)
        else:
            self.path1 = ConvBlock(in_channels, out_channels, dropout_rate)
            self.path2 = nn.ModuleList([FractalBlock(C-1, in_channels, out_channels, dropout_rate),
                                        FractalBlock(C-1, out_channels, out_channels, dropout_rate)])
    
    def join(self, x1, x2, global_chosen_columns):
        if not self.training:
            return (x1+x2) / 2
        
        B = x1.size(0) # B: batch size

        # 이번 배치가 global drop path인 경우.
        if global_chosen_columns is not None:
            global_mask = torch.zeros((B, 2), device=x1.device)
            
            global_mask[:, 0] = (global_chosen_columns == self.C).float()
            global_mask[:, 1] = (global_chosen_columns < self.C).float()

            global_mask = global_mask.view(B, 2, 1, 1)

            mask_x1 = global_mask[:, 0:1] # (B, 1, 1, 1)
            mask_x2 = global_mask[:, 1:2] # (B, 1, 1, 1)

            # global drop path는 (i) 둘 중 한 컬럼만 살아남거나, (ii) 둘 다 살아남지 못하거나.
            return x1*mask_x1 + x2*mask_x2
        # 이번 배치가 local drop path인 경우.
        else:
            keep_prob = 1. - self.local_drop_rate
            # local drop path 적용.
            local_drop_mask = torch.bernoulli(torch.full((B, 2), keep_prob, device=x1.device)) # [B, 2]
            
            dead_mask = (local_drop_mask.sum(dim=1) == 0)
            # 둘 다 dead인 경우 처리.
            if dead_mask.any():
                dead_indices = torch.where(dead_mask)[0]
                random_choices = torch.randint(0, 2, (dead_indices.size(0),), device=x1.device)
                local_drop_mask[dead_indices, random_choices] = 1.
            
            # [B, 2, 1, 1]로 확장.
            local_drop_mask = local_drop_mask.view(B, 2, 1, 1)

            # 각 경로 마스크.
            mask_x1 = local_drop_mask[:, 0:1] # [B, 1, 1, 1]
            mask_x2 = local_drop_mask[:, 1:2] # [B, 1, 1, 1]

            num_alive = mask_x1 + mask_x2

            return (x1*mask_x1 + x2*mask_x2) / num_alive

    def forward(self, x, global_chosen_columns):
        if self.C == 1:
            return self.path1(x)
        else:
            out1 = self.path1(x)
            out2_1 = self.path2[0](x, global_chosen_columns)
            out2_2 = self.path2[1](out2_1, global_chosen_columns)
            out = self.join(out1, out2_2, global_chosen_columns)
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
                global_chosen_columns = torch.randint(1, self.num_columns+1, (B,), device=x.device) # 샘플 별 컬럼 1개 선택.

        for layer in self.layers:
            if isinstance(layer, FractalBlock):
                out = layer(out, global_chosen_columns)
            else:
                out = layer(out)
        return out
