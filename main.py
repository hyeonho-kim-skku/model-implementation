import datetime
import yaml
from datasets import get_loader
from models import load_model
from methods import load_method
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from utils import knn_eval, load_optimizer, load_scheduler, move_to_device

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
best_knn_acc = 0

def train(args, method, optimizer, trainloader, writer, epoch):
    method.model.train()
    
    train_loss = 0.0
    progress = trainloader
    if tqdm is not None:
        progress = tqdm(trainloader, desc=f"Epoch {epoch} [train]", leave=False)

    for step, batch in enumerate(progress, start=1):
        batch = move_to_device(batch, device)

        optimizer.zero_grad()
        loss = method(batch)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if tqdm is not None:
            progress.set_postfix(loss=f"{train_loss/step:.4f}")
    
    print(f'[Epoch {epoch}] - Train Loss: {train_loss/len(trainloader):.4f}')
    writer.add_scalar('train_loss',train_loss/len(trainloader),epoch)

def test(args, testloader, method, epoch, writer):
    method.model.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress = testloader
        if tqdm is not None:
            progress = tqdm(testloader, desc=f"Epoch {epoch} [test]", leave=False)

        for step, data in enumerate(progress, start=1):
            images, labels = data[0].to(device), data[1].to(device) # cuda

            outputs = method.model(images)
            loss = F.cross_entropy(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{test_loss/step:.4f}",
                    acc=f"{100. * correct / total:.2f}"
                )

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

    quant_suffix = "_quantized" if getattr(args, 'use_quantization', False) else ""

    filename = f'./checkpoint/{args.method}_{args.model}{quant_suffix}{suffix}_ckpt.pth'

    torch.save(state, filename)

def _main(args):
    model = load_model(**vars(args))
    model.to(device)
    
    # trainloader, testloader = load_dataset(args.dataset, args.batch_size)
    trainloader = get_loader(args.dataset, args.batch_size, args.mode, train=True, shuffle=True, drop_last=True)
    testloader = get_loader(args.dataset, args.batch_size, 'test', train=False, shuffle=False, drop_last=False)
    knn_trainloader = None
    if args.mode == 'two_crop' or args.mode == 'multi_crop':
        knn_trainloader = get_loader(args.dataset, args.batch_size, 'test', train=True, shuffle=False, drop_last=False)

    method = load_method(args.method, model)
    method.to(device)
    
    optimizer = load_optimizer(args.optimizer, method, args.lr, args.weight_decay, args.momentum, args.nesterov)
    scheduler = load_scheduler(args.scheduler, optimizer, args.num_epochs, args.warmup_epochs)

    timezone_kst = datetime.timezone(datetime.timedelta(hours=9))
    cur_time = datetime.datetime.now(tz=timezone_kst).strftime("%m%d-%H%M%S")

    exp_name = f"{args.model}_{args.dataset}_{args.method}"
    log_dir = f'./runs/{exp_name}/{cur_time}'
    
    writer = SummaryWriter(log_dir)

    for epoch in range(args.num_epochs):
        train(args, method, optimizer, trainloader, writer, epoch)

        # ssl 일때는 knn evaluation.
        if args.mode == 'two_crop' or args.mode == 'multi_crop':
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the yaml config file')

    parser.add_argument('--model', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--criterion', type=str, default='crossentropyloss')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--nesterov', action='store_true') # --nesterov 적으면 True, 적지않으면 False 동작.
    parser.add_argument('--mode', type=str, help='augmentation mode')
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--use_quantization', action='store_true', help='Use 4-bit quantization for the model')
    parser.add_argument('--lora_rank', '--rank', dest='lora_rank', type=int, help='Rank for LoRA adapters')
    parser.add_argument('--num_classes', type=int, help='Number of classes for classification')
    parser.add_argument('--pretrained_model', '--pretrained_model_name', dest='pretrained_model_name', type=str, help='Pre-trained model name')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        parser.set_defaults(**config_dict)
        args = parser.parse_args()

    _main(args)
