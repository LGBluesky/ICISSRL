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
import torchvision.transforms as transforms

from timm.utils import accuracy
from util.datasets import myDataset,myDataset2
import torch.nn as nn

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.optim import SGD
import models_vit

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from util import Preprocess

def get_args_parser():
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)

    # Dataset parameters

    parser.add_argument('--data_path_test', default='./data/sampler-80K-40K-9groups_Remote-lsj3090-nvme/Group1/lineProb/linProb_Group1/lineProb_unified_test.txt',
                        type=str,
                        help='txt dataset path ')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir',
                        default='../mae_main_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/LineProb/linProb_Group2/ratio_1-1/ratio_100-100/topk',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--finetune', default='../mae_main_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/MAE_pretrain/ratio_1-1/checkpoint-199.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--resume', default='../mae_main_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/LineProb/linProb_Group2/ratio_1-1/ratio_100-100/w-cls-Blr-1.5e-3-Resize-toTensor-SGD-pretrain-epoch100/checkpoint-99.pth',
                        help='resume from checkpoint')    # supply

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

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
        print('OK')

    evaluate_state, tpr, fpr, AUC_ROC = evaluate2(target_list, pred_list)
    metric_states = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metric_states.update(evaluate_state)

    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
           .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))

    return metric_states, tpr, fpr, AUC_ROC


def main(args):
    # misc.init_distributed_mode(args)
    device = torch.device(args.device)
    # linear probe: weak augmentation


    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])


    dataset_test = myDataset2(os.path.join(args.data_path_test), transform=transform_val)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )


    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    else:
        print('::::::::::error:::::::')
        exit(0)

    # for linear prob only
    # hack: revise model's head with BN
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # freeze all but the head
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    # optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = SGD(model_without_ddp.head.parameters(), lr=0.001, weight_decay=0)
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)  # check

    start_time = time.time()

    test_stats, tpr, fpr, AUC_ROC = evaluate(data_loader_test, model, device)
    print(f"Accuracy of the network on the test_dataset {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
    show_ROC(tpr, fpr, AUC_ROC, key='test', root=args.output_dir)

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
