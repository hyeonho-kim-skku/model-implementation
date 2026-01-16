# ntxent.py
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, LinearLR, SequentialLR

def load_optimizer(optimizer_name, model, lr, weight_decay, momentum=None, nesterov=False):
    params = (p for p in model.parameters() if p.requires_grad) # for문 돌면서, requires_grad=True인 것들만 추출.
    if optimizer_name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def load_scheduler(scheduler_name, optimizer, num_epochs, warmup_epochs=0):
    main_epochs = num_epochs - warmup_epochs
    
    if scheduler_name == 'MultiStepLR':
        milestones = [int(0.5 * num_epochs) - warmup_epochs, int(0.75 * num_epochs) - warmup_epochs]  # [50% epoch, 75% ecpoh]
        main_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        main_scheduler = CosineAnnealingLR(optimizer, main_epochs)

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        return scheduler
    else:
        return main_scheduler

@torch.no_grad()
def extract_features(model, dataloader, device):
    """모델에서 Feature와 Label을 추출하여 CPU 텐서로 반환합니다."""
    model.eval()
    feats, labels = [], []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        h = model.forward_features(x)
        feats.append(h.cpu())
        labels.append(y.cpu())

    feats = torch.cat(feats, dim=0) 
    labels = torch.cat(labels, dim=0)
    return feats, labels

@torch.no_grad()
def knn_monitor_k1(train_features, train_labels, test_features, test_labels, device, chunk_size):
    """
    1-NN 분류기로 Test Accuracy를 계산합니다.
    Memory Efficient Implementation (Chunking)
    """
    # Train feature를 GPU로 이동 및 정규화
    train_features = train_features.to(device)
    train_features = F.normalize(train_features, dim=1).t()  # [Dim, N_train]
    train_labels = train_labels.to(device)

    num_test_images = test_labels.shape[0]
    total_top1 = 0.0

    # Test feature를 chunk 단위로 처리
    for idx in range(0, num_test_images, chunk_size):
        # 현재 Chunk 데이터를 GPU로 이동
        chunk_test_features = test_features[idx:idx+chunk_size].to(device)  # [chunk, Dim]
        chunk_test_labels = test_labels[idx:idx+chunk_size].to(device)

        # 정규화 및 유사도 계산
        chunk_test_features = F.normalize(chunk_test_features, dim=1)
        sim = torch.matmul(chunk_test_features, train_features)  # [chunk, N_train]
        
        # 가장 유사한 train feature의 인덱스 추출 및 해당 label 예측
        pred_indices = sim.argmax(dim=1)  # [chunk]
        pred_labels = train_labels[pred_indices]  # [chunk]

        # 정확한 예측 개수 누적
        total_top1 += (pred_labels == chunk_test_labels).float().sum().item()

    return total_top1 / num_test_images * 100.0

def knn_eval(model, trainloader, testloader, device):
    train_feats, train_labels = extract_features(model, trainloader, device)
    test_feats, test_labels = extract_features(model, testloader, device)
    
    acc = knn_monitor_k1(train_feats, train_labels, test_feats, test_labels, device, chunk_size=1024)
    return acc

def move_to_device(batch, device):
    """배치를 device로 이동."""
    inputs, labels = batch

    # 1. Multi-Crop or Two-Crop: inputs가 list
    # Two-Crop batch: [[x1, x2], label]
    # Multi-Crop batch: [[x1, x2, x3, ...], label]
    # x, label: (batch_size, 3, 32, 32) tensor
    if isinstance(inputs, list):  
        inputs = [x.to(device) for x in inputs]
    # 2. Supervised learning: inputs가 tensor
    else:
        inputs = inputs.to(device)

    labels = labels.to(device)
    
    return inputs, labels

# def move_to_device(batch, device):
#     """배치를 device로 이동."""
#     # contrastive batch: [[x1, x2], label]. x1, x2, label: (batch_size, 3, 32, 32) tensor
#     if isinstance(batch[0], (list, tuple)):  
#         (x1, x2), label = batch
#         return ((x1.to(device), x2.to(device)), label.to(device))
#     else:  # supervised: (images, labels)
#         return tuple(b.to(device) for b in batch)