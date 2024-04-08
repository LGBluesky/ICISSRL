import argparse
import torch
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset_linearProb
from models.resnet_simclr import ResNetSimCLR_linearProbing
from pathlib import Path
import os
from util import misc, show_ROC, evaluate2

import json
import time
import datetime
import torch.nn as nn
from timm.utils import accuracy




model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR linear probing evalutate')
parser.add_argument('--data_val', metavar='DIR', default='',
                    help='path to linear probing val txt')
parser.add_argument('--data_test', metavar='DIR', default='',
                    help='path to linear probing test txt')
parser.add_argument('-dataset-name', default='custom',
                    help='dataset name', choices=['stl10', 'cifar10', 'custom'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')

parser.add_argument('--LP_checkpoint', default='', type=str, help='pretrained model checkpoint')

parser.add_argument('--input_size', default=224, type=int, help='model input size')
parser.add_argument('--workers', type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

## output path
parser.add_argument('--output_dir', default='',
                    help='save checkpoint and tensorboard.log file')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')

parser.add_argument('--n-views', default=1, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
##device
parser.add_argument('--device',default='cuda:0', type=str)
parser.add_argument('--BN', action='store_true', help='whether FC or BN+FC ')

@torch.no_grad()
def evaluate(model, test_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    soft = nn.Softmax(dim=1)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    pred_list = []
    target_list = []
    model.eval()

    for (images, target) in metric_logger.log_every(test_loader, 1, header):

        images = images[0]   # dataloader return a list
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            soft_output = soft(output)
            loss = criterion(output, target)
        acc1, acc2 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)  # default

        # ROC
        soft_out_np = soft_output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        pred_list.extend(soft_out_np)
        target_list.extend(target_np)

    evaluate_state, tpr, fpr, AUC_ROC = evaluate2(target_list, pred_list)
    metric_states = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metric_states.update(evaluate_state)

    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))

    return metric_states, tpr, fpr, AUC_ROC


def main(args):
    device = torch.device(args.device)
    print('--------------args------------------')
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
            print('%s: %s' % (key, value))
    print('--------------args------------------\n')


    # sampler + Dataloader

    dataset = ContrastiveLearningDataset_linearProb(args.data_val, args)
    val_dataset = dataset.get_dataset(args.dataset_name, args.n_views, to_train=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    dataset = ContrastiveLearningDataset_linearProb(args.data_test, args)
    test_dataset = dataset.get_dataset(args.dataset_name, args.n_views, to_train=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    ## model processing

    # load pretrain model
    model = ResNetSimCLR_linearProbing(base_model=args.arch, out_dim=args.out_dim)

    if args.BN:
        print(':::::::::Linear probing with BN + Linear::::::::::')
        in_features = model.dim_mlp
        model.head = nn.Sequential(nn.BatchNorm1d(in_features, ), model.head)

    checkpoint = torch.load(args.LP_checkpoint, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.LP_checkpoint)

    checkpoint_model = checkpoint['state_dict']
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    start_time = time.time()

    val_stats, tpr, fpr, AUC_ROC = evaluate(model, val_loader, device)
    show_ROC(tpr, fpr, AUC_ROC, key='val', root=args.output_dir)
    test_stats, tpr, fpr, AUC_ROC = evaluate(model, test_loader, device)
    show_ROC(tpr, fpr, AUC_ROC, key='test', root=args.output_dir)

    all_stats = {
        'val': val_stats,
        'test': test_stats}
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(all_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('evaluate  time {}'.format(total_time_str))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)






