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

def move_to_device(batch, device):
    """배치를 device로 이동."""
    # contrastive batch: [[x1, x2], label]. x1, x2, label: (batch_size, 3, 32, 32) tensor
    if isinstance(batch[0], (list, tuple)):  
        (x1, x2), label = batch
        return ((x1.to(device), x2.to(device)), label.to(device))
    else:  # supervised: (images, labels)
        return tuple(b.to(device) for b in batch)