from datasets import load_dataset
from models import *
from methods import load_method
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

def train(args, method, optimizer, trainloader, writer, epoch):
    method.model.train()
    
    train_loss = 0.0
    for batch in trainloader:
        batch = tuple(b.to(device) for b in batch)

        optimizer.zero_grad()
        loss = method(batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    print(f'Epoch {epoch} - Train Loss: {train_loss/len(trainloader):.4f}')
    writer.add_scalar('train_loss',train_loss/len(trainloader),epoch)

def test(args, testloader, method, epoch, writer):
    method.model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device) # cuda

            outputs = method.model(images)
            loss = F.cross_entropy(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100. * correct / total
    global best_acc
    if acc > best_acc:
        best_acc = acc

        save_ckpt(args, method.model, acc, epoch)

    print(f'Epoch {epoch} - Test Loss: {test_loss/len(testloader):.4f}, Test Acc: {acc:.2f}%, Best Acc: {best_acc:.2f}')
    writer.add_scalar('test_loss',test_loss/len(testloader),epoch)
    writer.add_scalar('test_acc',acc,epoch)

def save_ckpt(args, model, acc, epoch):
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch
    }

    torch.save(state, './checkpoint/' + args.model + '_ckpt.pth')

def _main(args):
    model = load_model(args.model)
    model.to(device)
    
    trainloader, testloader = load_dataset(args.dataset, args.batch_size)

    method = load_method(args.method, model)
    # criterion = load_criterion(args.criterion, args.batch_size)
    optimizer = load_optimizer(args.optimizer, model, args.lr, args.weight_decay, args.momentum, args.nesterov)
    scheduler = load_scheduler(args.scheduler, optimizer, args.num_epochs)

    writer = SummaryWriter('./runs/' + args.model)

    for epoch in range(args.num_epochs): # resnet은 64k iteration => 64000*128/5000 = 약 163 epoch.
        train(args, method, optimizer, trainloader, writer, epoch)
        # pretrain시에 마지막 parameter일 경우 모델 저장.
        if args.pretrain:
            save_ckpt(args, model, -1, epoch)
        # SimCLR 같이 pretrain할 경우 test 진행 x.
        if testloader is not None:
            test(args, testloader, method, epoch, writer)
        scheduler.step()
    
    writer.close()

# 코드수정 없이 argument로만 조정하여 실행하는 것이 목표.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--criterion', type=str, default='crossentropyloss')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--nesterov', action='store_true') # --nesterov 적으면 True, 적지않으면 False 동작.
    parser.add_argument('--pretrain', action='store_true')
    args = parser.parse_args()
    _main(args)

""" command
CUDA_VISIBLE_DEVICES=7 python main.py --model=fractalnet --num_epochs=400 --batch_size=100 --lr=0.02 --scheduler=MultiStepLR
CUDA_VISIBLE_DEVICES=7 python main.py --model=densenet --num_epochs=300 --batch_size=64 --lr=0.01 --scheduler=MultiStepLR
CUDA_VISIBLE_DEVICES=7 python main.py --model=vit --num_epochs=400 --batch_size=128 --optimizer=AdamW --lr=0.0003 --weight_decay=0.02 --scheduler=CosineAnnealingLR
CUDA_VISIBLE_DEVICES=7 python main.py --model=mlp_mixer --num_epochs=400 --batch_size=128 --optimizer=AdamW --lr=0.001 --weight_decay=0.1 --scheduler=CosineAnnealingLR
CUDA_VISIBLE_DEVICES=7 python main.py --model=conv_mixer --num_epochs=200 --batch_size=128 --optimizer=AdamW --lr=0.001 --weight_decay=1e-3 --scheduler=CosineAnnealingLR
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_pretrain --dataset=CIFAR10_rotnet --num_epochs=200 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# rotnet pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_pretrain --dataset=CIFAR10_rotnet --num_epochs=200 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# rotnet pretrained + classifier
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_classifier --dataset=CIFAR10 --num_epochs=100 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# simclr pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=simclr --dataset=CIFAR10_SimCLR --num_epochs=300 --batch_size=512 --criterion=NTXent --optimizer=AdamW --lr=3e-4 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
"""
""" test command
CUDA_VISIBLE_DEVICES=7 python main.py --model=fractalnet --num_epochs=10 --batch_size=100 --lr=0.02 --scheduler=MultiStepLR
# rotnet pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_pretrain --dataset=CIFAR10_rotnet --num_epochs=10 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# rotnet pretrained + classifier
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_classifier --dataset=CIFAR10 --num_epochs=10 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# simclr pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=simclr --dataset=CIFAR10_SimCLR --num_epochs=10 --batch_size=512 --criterion=NTXent --optimizer=AdamW --lr=3e-4 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
# simclr pretrained + classifier
CUDA_VISIBLE_DEVICES=7 python main.py --model=simclr_classifier --dataset=CIFAR10 --num_epochs=100 --batch_size=512 --criterion=crossentropyloss --optimizer=AdamW --lr=3e-4 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
"""