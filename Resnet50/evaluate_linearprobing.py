# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from timm.utils import accuracy
import torch.nn as nn

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from data_aug import myDataset_linearProb



def get_args_parser():
    parser = argparse.ArgumentParser(description='ResNet50_evaluate')
    parser.add_argument('--GPU', default='', type=str)
    parser.add_argument('--data_test', metavar='DIR', default='',
                        help='path to linear probing train txt')
    parser.add_argument('--data_val', metavar='DIR', default='',
                        help='path to linear probing val txt')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--linearProbing_checkpoint', default='', type=str, help='resnet50 linearProbing checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='model input size')
    parser.add_argument('--workers', type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('--print_freq', default=20, type=int,
                        help='print frequency ')
    parser.add_argument('--BN', action='store_true', help='whether FC or BN+FC ')

    ## model save
    # output path
    parser.add_argument('--output_dir', default='',
                        help='save checkpoint and tensorboard.log file')
    # data augment
    return parser

def evaluate2(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.empty((y_scores.shape[0]))

    threshold_confusion = 0.5
    for i in range(y_scores.shape[0]):
        if y_scores[i, 0] >= threshold_confusion:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores[:, 1])
    AUC_ROC = roc_auc_score(y_true, y_scores[:, 1])

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 1])
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))

    # Confusion matrix

    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))
    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))

    evaluate_state = {'accuracy': accuracy,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'precision': precision,
                'F1_score': F1_score,
                'AUC_ROC': AUC_ROC}
    return evaluate_state, tpr, fpr, AUC_ROC


def show_ROC(tpr, fpr, AUC_ROC, key, root):
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    plt.plot(fpr, tpr, '-')
    plt.title('ROC curve, AUC = ' + str(AUC_ROC), fontsize=14)
    plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(root, key+'_'+'ROC'+'.png'), dpi=300)

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    soft = nn.Softmax(dim =1)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    pred_list = []
    target_list = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[-1]
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
    # misc.init_distributed_mode(args)
    args.device = torch.device(args.GPU)
    print('--------------args------------------')
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
            print('%s: %s' % (key, value))
    print('--------------args------------------\n')



    # linear probe: weak augmentation
    transform_val = transforms.Compose([
        transforms.Resize(size=args.input_size),
        transforms.ToTensor()])

    if args.data_val:
        val_dataset = myDataset_linearProb(args.data_val, transform_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
    
    test_dataset = myDataset_linearProb(args.data_test, transform_val)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)



    ## load model
    model = torchvision.models.resnet50(weights=None)  # num_classes =2
    in_features = model.fc.in_features
    if args.BN:
        print(':::::::::Linear probing with BN + Linear::::::::::')
        model.fc = nn.Sequential(nn.BatchNorm1d(in_features, ), nn.Linear(in_features, 2))
    else:
        model.fc = nn.Linear(in_features, 2)

    fc_checkpoint = torch.load(args.linearProbing_checkpoint, map_location='cpu')
    model.load_state_dict(fc_checkpoint['state_dict'])
    model.to(args.device)


    # add model revise module
    start_time = time.time()
    if args.data_val:
        val_stats, tpr, fpr, AUC_ROC = evaluate(val_loader, model, args.device)
        print(f"Accuracy of the network on the val_dataset {len(val_dataset)} test images: {val_stats['acc1']:.1f}%")
        show_ROC(tpr, fpr, AUC_ROC, key='val', root=args.output_dir)

    test_stats, tpr, fpr, AUC_ROC = evaluate(test_loader, model, args.device)
    print(f"Accuracy of the network on the test_dataset {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")
    show_ROC(tpr, fpr, AUC_ROC, key='test', root=args.output_dir)

    if args.data_val:
        all_stats = {
            'val': val_stats,
            'test': test_stats}
    else:
        all_stats = {
            'test': test_stats}
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(all_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('evaluate  time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
