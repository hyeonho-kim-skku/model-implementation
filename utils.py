# ntxent.py
import torch
import torch.nn.functional as F

def load_optimizer(optimizer_name, model, lr, weight_decay, momentum=None, nesterov=False):
    params = (p for p in model.parameters() if p.requires_grad) # for문 돌면서, requires_grad=True인 것들만 추출.
    if optimizer_name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if optimizer_name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def load_scheduler(scheduler_name, optimizer, num_epochs):
    if scheduler_name == 'MultiStepLR':
        milestones = [int(0.5 * num_epochs), int(0.75 * num_epochs)]  # [50% epoch, 75% ecpoh]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

@torch.no_grad()
def extract_features(model, dataloader, device):
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
def knn_1nn_top1(train_feats, train_labels, test_feats, test_labels):
    # cosine similarity based knn
    train_feats = F.normalize(train_feats, dim=1)
    test_feats = F.normalize(test_feats, dim=1)

    sim = torch.matmul(test_feats, train_feats.transpose(0,1)) # [N_test, N_train]
    idx = sim.argmax(dim=1) # [N_test] 가장 유사도가 높은 train feature의 인덱스.
    pred = train_labels[idx] # [N_test] 예측한 label.
    acc = (pred == test_labels).float().mean().item() * 100.0
    return acc

def knn_eval(model, trainloader, testloader, device):
    train_feats, train_labels = extract_features(model, trainloader, device)
    test_feats, test_labels = extract_features(model, testloader, device)
    acc = knn_1nn_top1(train_feats, train_labels, test_feats, test_labels)
    return acc

def nt_xent_loss(zi, zj, temperature=0.5):
    batch_size = zi.size(0)

    # Concatenation, zi <-> zj 간 유사도 한 번에 구하기 위해.
    z = torch.cat([zi, zj], dim=0)
    # L2 Normalization.
    z = F.normalize(z, dim=1)
    # similarity matrix.
    similarity_matrix = torch.matmul(z, z.T) / temperature

    # positive pair
    mask_positive = torch.zeros(2*batch_size, 2*batch_size, dtype=torch.bool, device=z.device)
    for i in range(batch_size):
        mask_positive[i, i+batch_size] = True
        mask_positive[i+batch_size, i] = True    
    # anchor마다 positive만 추출.
    positive_similarities = similarity_matrix[mask_positive].view(2*batch_size, -1) # (2N, 1)

    # negative 구하기.
    mask_self = torch.eye(2*batch_size, dtype=torch.bool, device=z.device) # 자기 자신과의 유사도
    mask_negative = ~mask_self & ~mask_positive # 자신과 positive 제외.
    negative_similarities = similarity_matrix[mask_negative].view(2*batch_size, -1) # (2N, 2N-2)

    # positive를 맨앞에 두고, 뒤에 negative.
    logits = torch.cat([positive_similarities, negative_similarities], dim=1)
    # 정답 label은 항상 0(positive).
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=z.device)

    loss = F.cross_entropy(logits, labels)
    return loss