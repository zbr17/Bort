# 用于mnist和cifar10的训练
import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
import random
import numpy as np

import timm
import torch
import torch.nn as nn
import torchvision.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.timm import utils
from timm.timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.timm.utils import ApexScaler, NativeScaler
from timm.timm.scheduler.cosine_lr import CosineLRScheduler
from src.bort.models import give_model
from src.bort.datasets import give_dataset, give_dataloader
from src.bort.utils import give_visualizer
from src.bort.optimizers import give_optim, give_scheduler
from src.bort.utils.misc import MetricHanlder, load_all, save_all, resume_config, give_pbar, ValueMeter
# new by rqh
from bort import BortS, BortA
# end
# import torch
import torch.distributed as dist

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
# /home/zbr/code/timm/vgg16_bort.yaml
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

#'/home/zbr/code/timm/vgg16_sgd.yaml'
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
# model parameters
group = parser.add_argument_group('Model parameters')
#model
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# 梯度裁剪的标准与模式
parser.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument("--local_rank", default=0, type=int)

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
parser.add_argument('--seed', default=42, type=int, metavar='N')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                     help='number of data loading workers (default: 8)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
parser.add_argument('--recon_ratio', default=0.95, type=float)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--device", type=int, nargs="+", default=[0])
parser.add_argument("--scheduler", type=str, default=None, help="cosine / none")
parser.add_argument("--print_interval", type=int, default=50)
parser.add_argument("--bs", type=int, default=256)
parser.add_argument("--dist_port", type=int, default=29500)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--output", type=str, default="/home/zbr/disk1/workspace/code/rqh-bort/ResultMnistNew/")
parser.add_argument('--checkpoint-hist', type=int, default=3, metavar='N',
                   help='number of checkpoints to keep (default: 3)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
parser.add_argument("--setting", type=str, default=None, 
                        help="It only applies to AllConv12 model")
parser.add_argument("--act_type", type=str, default="guided",
                        help="It only applies to AllConv12 model: guided / leaky")
# 每次跑实验前记得改实验名称
parser.add_argument('--exp', default='10.21ACNNDBortsL1v1', type=str, metavar='DIR', help='experiment name')
parser.add_argument('--model', default='simple', type=str, metavar='MODEL',help="lenet/simple")
parser.add_argument('--opt', default='borts', type=str, metavar='OPTIMIZER',
              help='Optimizer (default: "sgd", BortS, BortA)')
parser.add_argument('--gpu', type=int, default=1, 
                    help='gpu id')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
              help='learning rate, overrides lr-base if set (default: None)')
parser.add_argument('--weight-decay', type=float, default=0.005,
              help='weight decay (default: 2e-5)')
parser.add_argument('--dataset', metavar='NAME', default='mnist',
                    help='dataset name(chosen from mnist, cifar10)')
parser.add_argument('--amptitude', type=float, default=1,
                     help='amptitude, -1 means adaptive amptitude')
parser.add_argument('--mode', type=str, default='l1-fullrow',
                        help='mode for bort (default:l2-full, l1-fullrow, l1-fullcol, l1-row, l1-col)')
parser.add_argument('--gamma', type=float, default=0.05,
                        help='gamma for bort')
parser.add_argument('--dbort', action='store_true', default=True,
                   help='Use DBort')
parser.add_argument('--dbort-lambda', default=0.001, type=float, metavar='EPSILON',
                     help='DBort Epsilon (default: None, use opt default)')
parser.add_argument('--dbort-beta', default=0.1, type=float, metavar='BETA',
                        help='DBort Betas (default: None, use opt default)')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    args, args_text = _parse_args()
    path = args.output + args.exp
    if not os.path.exists(path):
        os.makedirs(path)
    log_path = os.path.join(path, 'log.txt')

    # 保存配置信息到yaml文件中
    yaml_path = os.path.join(path, 'args.yaml')
    with open(yaml_path, 'a') as f:
            f.write(args_text)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    model = give_model(args)

    for name, param in model.named_parameters():
        print(name)
        print(param.size())

    optimizer = give_optim(args, model)
    
    transform = None
    transform_list = []
    if args.dataset == 'mnist':
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)
    elif args.dataset == 'cifar10':
        if "resnet" in args.model or "vgg" in args.model or "allconv" in args.model:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        transform = transforms.Compose(transform_list)

    if args.dataset == 'mnist':
        dataset_train = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        dataset_eval = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        loader_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_eval = DataLoader(dataset_eval, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    elif args.dataset == 'cifar10':
        dataset_train = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        dataset_eval = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        loader_train = DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        loader_eval = DataLoader(dataset_eval, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    if args.scheduler == 'cosine':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=40,
            lr_min=args.min_lr,
            warmup_lr=args.warmup_lr,
            warmup_t=args.warmup_epochs,
        )
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()
    # 指定GPU
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    eval_metric = args.eval_metric
    decreasing_metric = eval_metric == 'loss'
    saver = None
    best_metric = None
    best_epoch = None
    output_dir = path

    saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            #model_ema=model_ema,
            #amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )

    def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if args.dbort:
                loss = loss - args.dbort_lambda * (outputs.pow(2).mean()).pow(args.dbort_beta)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if scheduler is not None:
                # scheduler.step_update(num_updates=, metric=loss.item())

            # 统计信息
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            if idx % args.print_interval == 0:
                print(f'[{idx}/{len(train_loader)}] Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

        print(f'Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%')
        with open(log_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}/{num_epochs}\n')
            f.write(f'Train Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%\n')

        return OrderedDict([('loss', total_loss/len(train_loader))])
    

# 6. 测试函数
    val_acc = {}
    def validate(model, test_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                pred = torch.argmax(outputs, dim = -1)
                total_loss += loss.item()
                # _, predicted = outputs.max(1)
                correct += torch.sum(pred == targets).item()
                total += len(targets)
                if idx % args.print_interval == 0:
                    print(f'[{idx}/{len(test_loader)}] Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')

        print(f'Test Loss: {total_loss/len(test_loader):.4f}, Accuracy: {100.*correct/total:.2f}%')
        with open(log_path, 'a') as f:
            f.write(f'Test Loss: {total_loss/len(test_loader):.4f}, Accuracy: {100.*correct/total:.2f}%\n')
        val_acc[epoch] = 100.*correct/total

        metrics = OrderedDict([('loss', total_loss/len(test_loader)), ('top1', 100.*correct/total)])
        return metrics

    # 7. 主训练循环
    num_epochs = args.epochs
    results = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        train_metrics = train_one_epoch(model, loader_train, criterion, optimizer, scheduler, device)
        val_metrics = validate(model, loader_eval, criterion, device)

        if val_metrics is not None:
            latest_metric = val_metrics[args.eval_metric]
        else:
            latest_metric = train_metrics[args.eval_metric]

        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

        results.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })
            
    best_acc = max(val_acc.values())
    print(f'Best Epoch: {max(val_acc, key=val_acc.get)}, Best Accuracy: {best_acc:.2f}%')
    with open(log_path, 'a') as f:
        f.write(f'Best Epoch: {max(val_acc, key=val_acc.get)}, Best Accuracy: {best_acc:.2f}%\n')
    print('Training complete!')

    results = {'all': results}
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    if best_metric is not None:
        results['best'] = results['all'][best_epoch - start_epoch]
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))
    print(f'--result\n{json.dumps(results, indent=4)}')


def setup_logging(log_path = None):
    # 获取根日志记录器
    log_file = log_path + '.txt'
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(
        '%(asctime)s-%(name)s- %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


if __name__ == '__main__':
    main()

