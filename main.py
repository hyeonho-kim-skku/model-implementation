import datetime
import yaml
from datasets import get_loader
from models import load_model
from methods import load_method
from trainer import Trainer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import load_optimizer, load_scheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    trainer = Trainer(
        args=args,
        method=method,
        optimizer=optimizer,
        scheduler=scheduler,
        trainloader=trainloader,
        testloader=testloader,
        writer=writer,
        device=device,
        knn_trainloader=knn_trainloader,
        checkpoint_dir=log_dir,
    )

    try:
        trainer.fit()
    finally:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the yaml config file')

    parser.add_argument('--model', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--num-epochs', dest='num_epochs', type=int)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    # parser.add_argument('--criterion', type=str, default='crossentropyloss')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--nesterov', action='store_true') # --nesterov 적으면 True, 적지않으면 False 동작.
    parser.add_argument('--mode', type=str, help='augmentation mode')
    parser.add_argument('--warmup-epochs', dest='warmup_epochs', type=int, default=0)
    parser.add_argument('--use-quantization', dest='use_quantization', action='store_true', help='Use 4-bit quantization for the model')
    parser.add_argument('--lora-rank', dest='lora_rank', type=int, help='Rank for LoRA adapters')
    parser.add_argument('--lora-alpha', dest='lora_alpha', type=float, help='Scaling factor for LoRA adapters')
    parser.add_argument('--lora-modules', dest='lora_modules', type=str, help='Comma-separated modules for LoRA: qkv,proj,mlp')
    parser.add_argument('--qkv-lora-components', dest='qkv_lora_components', type=str, help='Comma-separated qkv components for LoRA, e.g. q,v')
    parser.add_argument('--artifact-path', dest='artifact_path', type=str, help='Path to a pruned model artifact')
    parser.add_argument('--num-classes', dest='num_classes', type=int, help='Number of classes for classification')
    parser.add_argument('--backbone-name', dest='backbone_name', type=str, help='timm backbone name')
    parser.add_argument('--img-size', dest='img_size', type=int, help='Override timm model input size')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True, help='Load pretrained timm weights')
    parser.add_argument('--freeze-encoder', dest='freeze_encoder', action=argparse.BooleanOptionalAction, default=False, help='Train only the classifier head for timm_classifier')

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        parser.set_defaults(**config_dict)
        args = parser.parse_args()

    _main(args)
