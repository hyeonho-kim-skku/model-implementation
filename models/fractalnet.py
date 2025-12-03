import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" implementation details
join:
    Here, a blob is the result of a conv layer:
    a tensor holding activations for a fixed number of channels over a spatial domain.
    The join layer merges all of its input feature blobs into a single output blob.
conv:
    As now standard, we employ batch normalization together with each conv layer
    (convolution, batch norm, then ReLU)
FractalNet
    5 blocks with 2 X 2 non-overlapping max-pooling and subsampling
    number of filter(CIFAR): channels within blocks 1 through 5 as (64, 128, 256, 512, 512) 
    epoch: We run for 400 epochs on CIFAR, 70 epochs on ImageNet
    learning rate: Our learning rate starts at 0.02,
                   drop the learning rate by a factor of 10 whenever the number of remaining epochs halves.
    optimizer: stochastic gradient descent, momentum 0.9
    batch size: 100(CIFAR), 32(ImageNet)
    initialization: Xavier initialization
    dropout: we fix drop rate per block at (0%, 10%, 20%, 30%, 40%)
    drop-path:
        local: a join drops each input with fixed probability(15%), but we make sure at least one survives.
"""

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
    def __init__(self, num_columns, in_channels, out_channels, drop_rate):
        super(FractalBlock, self).__init__()
        self.num_columns = num_columns
        self.grid_modules = nn.ModuleList([nn.ModuleList() for _ in range(num_columns)]) # [col_idx][depth]

        self.max_depth = 2 ** (num_columns-1)
        dist = self.max_depth # dist means distance between conv layers
        self.count = [0] * self.max_depth # 해당 깊이에서 join에 참여하는 열의 개수.
        # module 쌓기.
        for col_idx in range(len(self.grid_modules)):
            for depth in range(self.max_depth):
                # 각 컬럼에서 dist마다 conv 쌓기.
                if (depth+1) % dist == 0:
                    # 첫번째 conv 블록들은 입력채널 수가 in_channels,
                    if (depth+1) == dist:
                        module = ConvBlock(in_channels, out_channels, drop_rate)
                    # 나머지 conv 블록들은 입력채널 수가 out_channels
                    else:
                        module = ConvBlock(out_channels, out_channels, drop_rate)
                    
                    self.count[depth] += 1 # 해당 깊이의 join 참여 열 수 증가
                # dist가 아니면, 해당 위치에 layer 없음.
                else:
                    module =None

                self.grid_modules[col_idx].append(module)
            
            # 다음 열은 간격이 절반.
            dist //= 2
    
    # element-wise mean
    def join(self, input_feature_blobs, global_columns):
        output_blob = torch.stack(input_feature_blobs)

        num_blobs = len(input_feature_blobs)
        batch_size = len(input_feature_blobs[0])
        # training 중에만 drop-path 적용
        if self.training:
            # 배치의 앞 절반 샘플에는 global drop path 적용.
            num_global_drop = len(global_columns)
            # print(f"num_global_drop: {num_global_drop}")

            # global drop mask 생성
            global_drop_mask = np.zeros([num_blobs, num_global_drop], dtype=np.float32)
            start_col = self.num_columns - num_blobs
            # print(f"start_col: {start_col}") # debug
            # global_columns를 상대위치로 변환.
            relative_columns = global_columns - start_col
            # print(f"relative_columns: {relative_columns}") # debug
            valid_indices = np.where((relative_columns >= 0) & (relative_columns < num_blobs))[0] # global 선택된 column이 join에 참여하는 샘플의 인덱스들.
            global_drop_mask[relative_columns[valid_indices], valid_indices] = 1.
            # print(f"global_drop_mask: {global_drop_mask}") # debug

            # 배치의 뒤 절반 샘플에는 local drop path 적용.
            num_local_drop = batch_size - num_global_drop
            # [num_blobs][batch_size] 크기의 이진 마스크.
            # local_drop_mask = np.ones([num_blobs, num_local_drop], dtype=np.float32) # debug
            local_drop_mask = np.random.binomial(1, 1.0-0.15, [num_blobs, num_local_drop]).astype(np.float32)

            # 모든 열이 드롭된 샘플은 1개 열 살리기.
            alive_count = local_drop_mask.sum(axis=0) # 각 샘플별 살아있는 열 개수
            dead_indices = np.where(alive_count == 0.)[0] # 모든 열이 drop된 샘플 index
            
            if len(dead_indices) > 0:
                random_cols = np.random.randint(0, num_blobs, size=len(dead_indices)) # 살릴 열 고르기.
                local_drop_mask[random_cols, dead_indices] = 1.0

            # global + local drop mask 
            drop_mask = np.concatenate((global_drop_mask, local_drop_mask), axis=1)
            # 텐서로 변환.
            drop_mask = torch.from_numpy(drop_mask).to(output_blob.device)
            drop_mask = drop_mask.view(num_blobs, batch_size, 1, 1, 1) # 브로드캐스팅 위함.

            # drop path 적용.
            masked_output = output_blob * drop_mask # [num_blobs, batch_size, C, H, W]
            # mean 계산.
            n_alive = drop_mask.sum(dim=0) # 각 샘플별 살아있는 열 개수 [B, 1, 1, 1]
            # global drop path의 경우, input_feature_blobs가 모두 선택되지 않을 수 있음. 아래 나눌때 오류가 나지않게 임의변수(1.0) 주기.
            n_alive[n_alive==0.] = 1.
            # mean 계산
            output_blob = masked_output.sum(dim=0) / n_alive #
        else:
            output_blob = output_blob.mean(dim=0)

        return output_blob

    def forward(self, x, global_columns):
        outs = [x] * self.num_columns

        # 각 깊이를 순회.
        for depth in range(self.max_depth):
            start_col = self.num_columns - self.count[depth] # start column.

            cur_outs = [] # 현재 깊이의 출력.
            for col_idx in range(start_col, self.num_columns):
                cur_in = outs[col_idx]
                cur_module = self.grid_modules[col_idx][depth]
                cur_out = cur_module(cur_in)
                cur_outs.append(cur_out)
            
            # join
            # print(f"depth: {depth}") # debug
            joined = self.join(cur_outs, global_columns)

            # outs update.
            for col_idx in range(start_col, self.num_columns):
                outs[col_idx] = joined
        
        # 마지막 열의 출력 반환.
        return outs[-1]


class FractalNet(nn.Module):
    def __init__(self, num_columns, num_blocks=5):
        super(FractalNet, self).__init__()
        self.num_columns = num_columns

        layers = nn.ModuleList()
        out_channels = [64, 128, 256, 512, 512] # num_filters
        in_channels = 3
        drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
        for b in range(num_blocks):
            fractal_block = FractalBlock(num_columns, in_channels, out_channels[b], drop_rates[b])
            layers.append(fractal_block)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels[b]

        layers.append(nn.Flatten())
        layers.append(nn.Linear(512*1*1, 10))

        self.layers = layers

    def forward(self, x):
        out = x
        batch_size = x.size(0)

        if (self.training):
            global_drop_ratio = 0.5
            num_global_drop_samples = int(batch_size * global_drop_ratio) # global drop path 적용할 sample 수
            global_columns = np.random.randint(0, self.num_columns, size = num_global_drop_samples) # 살아남을 column 선택
        else:
            global_columns = None

        for layer in self.layers:
            # FractalBlock이면, global_columns 전달
            if isinstance(layer, FractalBlock):
                out = layer(out, global_columns)
            else:
                out = layer(out)
        return out