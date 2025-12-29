from datasets import load_dataset
from models import load_model
from methods import load_method
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from utils import knn_eval, load_optimizer, load_scheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
best_knn_acc = 0

def train(args, method, optimizer, trainloader, writer, epoch):
    method.model.train()
    
    train_loss = 0.0
    for batch in trainloader:
        # 분기 필요
        # simclr, moco
        (x1, x2), label = batch
        batch = ((x1.to(device), x2.to(device)), label.to(device))
        # batch = tuple(b.to(device) for b in batch)

        optimizer.zero_grad()
        loss = method(batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    print(f'[Epoch {epoch}] - Train Loss: {train_loss/len(trainloader):.4f}')
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

        save_ckpt(args, method.model, acc, epoch, "_cls")

    print(f'[Epoch {epoch}] - Test Loss: {test_loss/len(testloader):.4f}, Test Acc: {acc:.2f}%, Best Acc: {best_acc:.2f}')
    writer.add_scalar('test_loss',test_loss/len(testloader),epoch)
    writer.add_scalar('test_acc',acc,epoch)

def save_ckpt(args, model, acc, epoch, suffix):
    state = {
        'model': model.state_dict(),
        'acc': acc,
        'epoch': epoch
    }

    torch.save(state, './checkpoint/' + args.method + '_' + args.model + suffix + '_ckpt.pth')

def _main(args):
    model = load_model(args.model)
    model.to(device)
    
    trainloader, testloader = load_dataset(args.dataset, args.batch_size)
    knn_trainloader = load_dataset("knn_train", args.batch_size) # 나중에 리팩토링

    method = load_method(args.method, model)
    method.to(device)
    
    optimizer = load_optimizer(args.optimizer, model, args.lr, args.weight_decay, args.momentum, args.nesterov)
    scheduler = load_scheduler(args.scheduler, optimizer, args.num_epochs)

    writer = SummaryWriter('./runs/' + args.method + '_' + args.model)

    for epoch in range(args.num_epochs):
        train(args, method, optimizer, trainloader, writer, epoch)

        # pretrain일때는 knn evaluation.
        if args.pretrain:
            if epoch%5 == 0: # 5 에폭마다 knn_eval 진행.
                knn_acc = knn_eval(model, knn_trainloader, testloader, device)

                global best_knn_acc
                if knn_acc > best_knn_acc:
                    best_knn_acc = knn_acc
                    save_ckpt(args, model, knn_acc, epoch, "_knn")

                print(f'[Epoch {epoch}] 1NN top-1: {knn_acc:.2f}% Best 1nn top-1: {best_knn_acc:.2f}%')
                writer.add_scalar('knn_acc', knn_acc, epoch)
        # supervised learning일 때는 classification.
        else:
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
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_pretrain --dataset=CIFAR10 --num_epochs=200 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# rotnet pretrained + classifier
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_classifier --dataset=CIFAR10 --num_epochs=100 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# simclr pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=simclr --method=simclr --dataset=CIFAR10_SimCLR --num_epochs=300 --batch_size=512 --optimizer=AdamW --lr=3e-4 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
"""

""" test command
CUDA_VISIBLE_DEVICES=7 python main.py --model=fractalnet --num_epochs=10 --batch_size=100 --lr=0.02 --scheduler=MultiStepLR
# rotnet pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_pretrain --dataset=CIFAR10 --num_epochs=10 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# rotnet pretrained + classifier
CUDA_VISIBLE_DEVICES=7 python main.py --model=rotnet_classifier --dataset=CIFAR10 --num_epochs=10 --batch_size=128 --criterion=crossentropyloss --optimizer=SGD --lr=0.1 --momentum=0.9 --weight_decay=5e-4 --scheduler=MultiStepLR --nesterov
# simclr pretrain
CUDA_VISIBLE_DEVICES=7 python main.py --model=simclr --method=simclr --dataset=CIFAR10_SimCLR --num_epochs=10 --batch_size=512 --optimizer=AdamW --lr=3e-4 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
# simclr pretrained + classifier
CUDA_VISIBLE_DEVICES=7 python main.py --model=simclr_classifier --dataset=CIFAR10 --num_epochs=100 --batch_size=512 --optimizer=AdamW --lr=3e-4 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
# moco pretrain (epoch: 200)
CUDA_VISIBLE_DEVICES=7 python main.py --model=resnet18 --method=moco --dataset=CIFAR10_MoCo --num_epochs=200 --batch_size=256 --optimizer=SGD --lr=0.03 --momentum=0.9 --weight_decay=5e-4 --scheduler=CosineAnnealingLR --pretrain
# BYOL (epochs: 300)
CUDA_VISIBLE_DEVICES=7 python main.py --model=resnet18 --method=byol --dataset=CIFAR10_SimCLR --num_epochs=16 --batch_size=128 --optimizer=SGD --lr=0.03 --momentum=0.9 --weight_decay=1e-4 --scheduler=CosineAnnealingLR --pretrain
"""