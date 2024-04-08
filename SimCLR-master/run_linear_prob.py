import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset_linearProb
from models.resnet_simclr import ResNetSimCLR_linearProbing
from simclr import SimCLR_linearProb, SimCLR_linearProb_one_GPU
from pathlib import Path
import os
import numpy as np
from util import setup_for_distributed
from torch.optim import Adam, SGD
from timm.models.layers import trunc_normal_


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data_train', metavar='DIR', default='',
                    help='path to linear probing train txt')
parser.add_argument('--data_val', metavar='DIR', default='',
                    help='path to linear probing val txt')
parser.add_argument('-dataset-name', default='custom',
                    help='dataset name', choices=['stl10', 'cifar10', 'custom'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

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


## output path
parser.add_argument('--output_dir', default='',
                    help='save checkpoint and tensorboard.log file')
parser.add_argument('--check_frequency', default=20, type = int,
                    help='save model weight frequency')
parser.add_argument('--check_min', default=60, type=int,
                    help='min epoch for  save model checkpoint')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=1, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')


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

    # sampler + Dataloader
    dataset = ContrastiveLearningDataset_linearProb(args.data_train, args)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, to_train=True)

    if args.dist:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
        )
    else:
        sampler_train = None
    train_loader = torch.utils.data.DataLoader(
            train_dataset, sampler=sampler_train, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, drop_last=True)


    dataset = ContrastiveLearningDataset_linearProb(args.data_val, args)
    val_dataset = dataset.get_dataset(args.dataset_name, args.n_views, to_train=False)

    if args.dist:
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
        )
    else:
        sampler_val = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # load pretrain model
    model = ResNetSimCLR_linearProbing(base_model=args.arch, out_dim=args.out_dim)
    checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.pretrained_checkpoint)

    checkpoint_model = checkpoint['state_dict']
    fc_layer = ['backbone.fc.0.weight', 'backbone.fc.0.bias', 'backbone.fc.2.weight', 'backbone.fc.2.bias']

    for k in fc_layer:
        if k in checkpoint_model:
            print(f'Remving key {k} from pretrained checkpoint ')
            del checkpoint_model[k]
    msg = model.load_state_dict(checkpoint_model, strict=False)

    # manually initialize fc layer: following MoCo v3    # add
    # trunc_normal_(model.backbone.fc.weight, std=0.01)
    if args.BN:
        print(':::::::::Linear probing with BN + Linear::::::::::')
        in_features = model.dim_mlp
        model.head = nn.Sequential(nn.BatchNorm1d(in_features, ), model.head)

    # freeze all but head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
    ## end model processing
    if args.dist:
        args.lr = args.blr * args.batch_size*2/256 # align with  MAE
    else:
        args.lr = args.blr * args.batch_size/ 256
    print('--------------args------------------')
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
            print('%s: %s' % (key, value))
    print('--------------args------------------\n')

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer=None
    if args.optimizer=='SGD':
        optimizer = SGD(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer=="Adam":
        optimizer = Adam(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
    #                                                        last_epoch=-1)

    if args.dist:
        simclr = SimCLR_linearProb(model=model, optimizer=optimizer, args=args)
        simclr.train_and_val(train_loader, val_loader, sampler_train)
        dist.destroy_process_group()
    else:
        simclr = SimCLR_linearProb_one_GPU(model=model, optimizer=optimizer, args=args)
        simclr.train_and_val(train_loader, val_loader)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)






