from datasets import load_dataset
from models import *
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

def train(criterion, optimizer, trainloader, model, writer, epoch):
    model.train()
    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) # cuda

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    print(f'Epoch {epoch} - Train Loss: {train_loss/len(trainloader):.4f}')
    writer.add_scalar('train_loss',train_loss/len(trainloader),epoch)


def test(args, testloader, model, criterion, epoch, writer):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device) # cuda

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.*correct/total
    global best_acc
    if acc > best_acc:
        best_acc = acc

        save_ckpt(args, model, acc, epoch)

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

    criterion = load_criterion(args.criterion)
    optimizer = load_optimizer(args.optimizer, model, args.lr, args.weight_decay, args.momentum)
    scheduler = load_scheduler(args.scheduler, optimizer, args.num_epochs)
    # milestones = [int(0.5 * args.num_epochs), int(0.75 * args.num_epochs)]  # [50% epoch, 75% ecpoh]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    writer = SummaryWriter('./runs/' + args.model)

    for epoch in range(args.num_epochs): # resnet은 64k iteration => 64000*128/5000 = 약 163 epoch.
        train(criterion, optimizer, trainloader, model, writer, epoch)
        test(args, testloader, model, criterion, epoch, writer)
        scheduler.step()
    
    writer.close()

# 코드수정 없이 argument로만 조정하여 실행하는 것이 목표.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--criterion', type=str, default='crossentropyloss')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str)
    args = parser.parse_args()
    _main(args)

""" command
CUDA_VISIBLE_DEVICES=6 python main.py --model=fractalnet --num_epochs=400 --batch_size=100 --lr=0.02 --scheduler=MultiStepLR
CUDA_VISIBLE_DEVICES=6 python main.py --model=densenet --num_epochs=300 --batch_size=64 --lr=0.01 --scheduler=MultiStepLR
CUDA_VISIBLE_DEVICES=6 python main.py --model=vit --num_epochs=400 --batch_size=128 --optimizer=AdamW --lr=0.0003 --weight_decay=0.02 --scheduler=CosineAnnealingLR
"""