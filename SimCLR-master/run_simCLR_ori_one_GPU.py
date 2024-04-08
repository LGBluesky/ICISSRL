import argparse
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR_ori
from pathlib import Path
import os
import numpy as np
from util import setup_for_distributed


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./data/sampler-80K-40K-9groups_Remote-lsj3090-nvme-224/Group1/MAE_pretrain/ratio_1-1.txt',
                    help='path to txt')
parser.add_argument('-dataset-name', default='custom',
                    help='dataset name', choices=['stl10', 'cifar10', 'custom'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--input_size', default=224, type=int,  help = 'model input size')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=192, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--optimizer', default='', type=str,
                    help='selection Adam or AdamW ')


## output path
parser.add_argument('--output_dir', default='../SimCLR_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090-224/Group1/SimCLR_pretrain/ratio_1-1-OneGPU',
                    help='save checkpoint and tensorboard.log file')
parser.add_argument('--check_frequency', default=20, type = int,
                    help='save model weight frequency')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=20, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')

parser.add_argument('--gpu', default='', type=str, help='model device')
parser.add_argument('--aug', default='RandomResizedCrop,RandomHorizontalFlip, RandomApply,RandomGrayscale,GaussianBlur',
                    type=str, help='record data augmentation')



def main(args):
    device = torch.device(args.gpu)
    # torch.cuda.set_device('cuda:0')
    args.device = device
    print('--------------args------------------')
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n'%(key, value))
            print('%s: %s' % (key, value))
    print('--------------args------------------\n')


    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

    # sampler + Dataloader
    dataset = ContrastiveLearningDataset(args.data)
    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.input_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = None
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    if args.optimizer=='AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.

    simclr = SimCLR_ori(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
