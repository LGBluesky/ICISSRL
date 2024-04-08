import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import argparse
from pathlib import Path
from data_aug import myDataset_linearProb
from torch.optim import SGD, Adam
from resnet_train import Resnet_train, Resnet_dist_train
import os
import torch.distributed as dist
import numpy as np
import torch.backends.cudnn as cudnn
from util import setup_for_distributed

parser = argparse.ArgumentParser(description='PyTorch Resnet50')
parser.add_argument('--data_train', metavar='DIR', default='',
                    help='path to linear probing train txt')
parser.add_argument('--data_val', metavar='DIR', default='',
                    help='path to linear probing val txt')
parser.add_argument('--pretrained_checkpoint', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--input_size', default=224, type=int,  help = 'model input size')
parser.add_argument('--workers', type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--optimizer', default='SGD', help='optimizer, Adam, SGD')
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (absolute lr)')
parser.add_argument('--blr', '--learning-rate',  type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--min_lr',  default=0.0, type=float, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')

parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--print_freq',default=20, type=int,
                    help = 'print frequency ')
parser.add_argument('--arch', metavar='ARCH', default='resnet50')


## output path
parser.add_argument('--output_dir', default='',
                    help='save checkpoint and tensorboard.log file')
parser.add_argument('--check_frequency', default=20, type = int,
                    help='save model weight frequency')
parser.add_argument('--check_min', default=60, type=int,
                    help='min epoch for  save model checkpoint')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')

## distributed train
parser.add_argument('--dist', action='store_true', help='DDP training')
parser.add_argument('--gpu', default='cuda:0', help='model device only one ')
parser.add_argument('--aug', default='', type=str, help='record data augmentation method')

## classifier head
parser.add_argument('--BN', action='store_true', help='whether FC or BN+FC ')
parser.add_argument('--cls_head', default='', type=str, help='record classifier head net')


def init_DDP(args):
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    setup_for_distributed(args.rank==0)
def main(args):

    if args.dist:
        init_DDP(args)
        seed = args.seed + dist.get_rank()
        print(args.dist)
    else:
        device = torch.device(args.gpu)
        args.device = device
        seed = args.seed

    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = True
    ## params write in args.txt
    print('--------------args------------------')
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
            print('%s: %s' % (key, value))
    print('--------------args------------------\n')

    model = torchvision.models.resnet50(weights=None)  # num_classes =2
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint))

    in_features = model.fc.in_features
    if args.BN:
        print(':::::::::Linear probing with BN + Linear::::::::::')
        model.fc = nn.Sequential(nn.BatchNorm1d(in_features, ), nn.Linear(in_features, 2))
    else:
        model.fc = nn.Linear(in_features, 2)

    ## freeze
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.fc.named_parameters():
        p.requires_grad = True

    ##
    # sampler + Dataloader
    transform_train = transforms.Compose([
        transforms.Resize(size=args.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    transform_val = transforms.Compose([
        transforms.Resize(size=args.input_size),
        transforms.ToTensor()])

    train_dataset = myDataset_linearProb(args.data_train, transform_train)
    if args.dist:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
    else:
        sampler_train = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)


    val_dataset = myDataset_linearProb(args.data_val, transform_val)
    if args.dist:
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
        )
    else:
        sampler_val = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.dist:
        args.lr = args.blr * args.batch_size * 2 / 256  # align with  MAE
    else:
        args.lr = args.blr * args.batch_size / 256


    ## optimizer and
    optimizer=None
    if args.optimizer=='SGD':
        optimizer = SGD(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer=="Adam":
        optimizer = Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.dist:
        Resnet = Resnet_dist_train(model=model, optimizer=optimizer, args=args)
        Resnet.train_and_val(train_loader, val_loader,sampler_train)
        dist.destroy_process_group()
    else:
        print('::::::::single gpu version failed::::::')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)