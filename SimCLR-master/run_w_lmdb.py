import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from pathlib import Path
import os
import numpy as np
from util import setup_for_distributed, LARS


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to txt')
parser.add_argument('--dataset-name', default='custom_lmdb',
                    help='dataset name', choices=['stl10', 'cifar10', 'custom', 'custom_lmdb'])
parser.add_argument('--lmdb_dir', default='', type=str)

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--input_size', default=224, type=int,  help = 'model input size')
parser.add_argument('--workers', type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


## output path
parser.add_argument('--output_dir', default='',
                    help='save checkpoint and tensorboard.log file')
parser.add_argument('--check_frequency', default=20, type = int,
                    help='save model weight frequency')
parser.add_argument('--check_min', default=150, type=int, help='save model min epoch')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
## optimizer parameter
parser.add_argument('--optimizer', default='', type=str, help='model optimizer')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', default=0.0, type=float)
parser.add_argument('--warmup_epochs', default=10, type=int)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
## data augmentation
parser.add_argument('--aug', default='', help='record data augmentation method')
parser.add_argument('--scale_min', default='', type=float, help= 'setting RandomCropResize scale minus')




def init_DDP(args):
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    setup_for_distributed(args.rank==0)

def main(args):
    init_DDP(args)
    print('--------------args------------------')
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
            print('%s: %s' % (key, value))
    print('--------------args------------------\n')


    seed = args.seed +dist.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False


    # sampler + Dataloader
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.input_size, args.scale_min, args.lmdb_dir)

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    if args.optimizer =='LARS':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader, sampler_train)
    dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
