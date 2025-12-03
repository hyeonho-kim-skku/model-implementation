from datasets import load_dataset
from models import *
import argparse
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_name = 'resnet'
model_name = 'pre_act_resnet'
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

        save(model, acc, epoch)

    print(f'Epoch {epoch} - Test Loss: {test_loss/len(testloader):.4f}, Test Acc: {acc:.2f}%')
    writer.add_scalar('test_loss',test_loss/len(testloader),epoch)
    writer.add_scalar('test_acc',acc,epoch)

def save(model, acc, epoch):
    if isinstance(model, torch.nn.DataParallel):
        state = {
            'model': model.module.state_dict(),  # DataParallel 
            'acc': acc,
            'epoch': epoch
        }
    else:
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch
        }

    torch.save(state, './checkpoint/' + model_name + '_ckpt.pth')

def _main(args):
    trainloader, testloader = load_dataset('CIFAR10')
    
    model = load_model(model_name, 'PreActBottleneck', num_blocks=[18, 18, 18]) # pre-act resent-164
    # model = load_resnet('BasicBlock', num_blocks=[3,3,3])
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = load_resnet_criterion()
    optimizer = load_resnet_optimizer(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[(32000*args.batch_size)//len(trainloader), (48000*args.batch_size)//len(trainloader)], gamma=0.1) # 50% 75%에서 drop.

    writer = SummaryWriter('./runs/' + model_name)

    for epoch in range(args.num_epochs): # resnet은 64k iteration => 64000*128/5000 = 약 163 epoch.
        train(criterion, optimizer, trainloader, model, writer, epoch)
        test(args, testloader, model, criterion, epoch, writer)
        scheduler.step()
    
    writer.close()

# 코드수정 없이 argument로만 조정하여 실행하는 것이 목표.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_epochs', type=int, default=163)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    _main(args)