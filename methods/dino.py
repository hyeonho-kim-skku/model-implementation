import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        # weight_norm: weight W (out_dim, in_dim)를 g * v / ||v|| 로 재파라미터화
        # g: magnitude (out_dim, 1)  -> 각 output 노드 weight 벡터의 magnitude 결정.
        # v: direction (out_dim, in_dim) -> forward 시 ||v||로 정규화되어 방향만 사용.
        self.last_layer = nn.utils.parametrizations.weight_norm(self.last_layer)
        
        # magnitude를 1로 고정.
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2) # l2 normalize
        x = self.last_layer(x)
        return x

class DINO(nn.Module):
    def __init__(self, model, dim=65536, bottleneck_dim=256, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9, teacher_momentum=0.996):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.teacher_momentum = teacher_momentum

        # Prepare student network
        self.student = model
        feat_dim = self.student.fc.in_features
        self.student.fc = nn.Identity()
        self.student_head = DINOHead(feat_dim, dim, bottleneck_dim=bottleneck_dim)

        # Prepare teacher network (copy of student)
        self.teacher = copy.deepcopy(self.student)
        self.teacher_head = DINOHead(feat_dim, dim, bottleneck_dim=bottleneck_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Disable gradients for teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
        
        # Center buffer for teacher output centering
        self.register_buffer("center", torch.zeros(1, dim))

        # For knn evaluation in main.py
        self.model = self.student

    def forward(self, batch):
        """
        Multi-Crop Forward Logic
        batch: (crops, labels)
        crops: List of tensors [global1, global2, local1, local2, ... localM]
        """
        crops, _ = batch

        # 1. Input categorization
        # Global crops (32x32): 처음 2개
        global_crops = crops[:2]
        # Local crops (16x16): 나머지
        local_crops = crops[2:]
        
        # 1. Student forward pass
        # global, local 크기별로 따로 계산 후 합침
        global_inputs = torch.cat(global_crops, dim=0)  # [2*B, C, H, W]
        student_global_outputs = self.student_head(self.student(global_inputs))
        local_inputs = torch.cat(local_crops, dim=0)    # [M*B, C, H, W]
        student_local_outputs = self.student_head(self.student(local_inputs))

        student_output = torch.cat([student_global_outputs, student_local_outputs], dim=0)
        student_output = student_output.chunk(len(crops), dim=0)  # List of tensors

        # 2. Teacher forward pass
        # Teacher는 global crops만 사용
        with torch.no_grad():
            self._momentum_update_teacher()
            teacher_output = self.teacher_head(self.teacher(global_inputs))
            teacher_output = teacher_output.chunk(2, dim=0)  # List of tensors
        
        # 3. Compute DINO loss
        total_loss = 0.0
        n_loss_terms = 0

        # 모든 Teacher output (global)에 대해
        for t_idx, t_out in enumerate(teacher_output):
            # 모든 Student output에 대해 (global + local)
            for s_idx, s_out in enumerate(student_output):
                # 동일한 global view끼리는 비교 제외
                if t_idx == s_idx:
                    continue
                
                loss = self._dino_loss(s_out, t_out)
                total_loss += loss
                n_loss_terms += 1
        
        total_loss = total_loss / n_loss_terms

        # Update center
        self._update_center(torch.cat(teacher_output))

        return total_loss

    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = param_t.data * self.teacher_momentum + param_s.data * (1. - self.teacher_momentum)
        for param_s, param_t in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_t.data = param_t.data * self.teacher_momentum + param_s.data * (1. - self.teacher_momentum)   
    
    def _dino_loss(self, student_out, teacher_out):
        # Teacher centering and sharpening
        t = teacher_out - self.center
        t = F.softmax(t / self.teacher_temp, dim=-1)

        # Student sharpening
        s = F.log_softmax(student_out / self.student_temp, dim=-1)

        # Cross Entropy Loss: -sum(t * log(s))
        return -(t * s).sum(dim=-1).mean()
    
    def _update_center(self, teacher_out):
        batch_center = torch.mean(teacher_out, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)