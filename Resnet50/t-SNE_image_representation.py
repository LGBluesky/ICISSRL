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
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import manifold
from data_aug import myDataset_linearProb
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler


def get_args_parser():
    parser = argparse.ArgumentParser(description='ResNet50_evaluate')
    parser.add_argument('--data_test', metavar='DIR', default='./data/sampler-80K-40K-9groups_Remote-lsj3090-nvme/Group1/UMAP_cluster_data_3090/Pos_1000-Neg_1000-seed_10.txt',
                        help='path to linear probing train txt')
    parser.add_argument('--rsnet_origin_W', action='store_true')
    parser.add_argument('--pretrained', default='', type=str, help='pretrained model checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='model input size')


    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=4,type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')


    ## model save
    # output path
    parser.add_argument('--output_dir', default='',
                        help='save checkpoint and tensorboard.log file')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save',default='tSNE_pretrain_ratio_1-1')
    parser.add_argument('--legend', action='store_true')

    # data augment
    return parser

@torch.no_grad()
def image_representation(data_loader, model, device):
    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 't-SNE'
    pred_list = []
    target_list = []

    # switch to evaluation mode
    model.eval()
    # for batch in metric_logger.log_every(data_loader, 1, header):
    for batch in data_loader:
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
        pred_list.extend(output.detach().cpu().numpy())
        target_list.extend(target.numpy())

    return np.array(pred_list), np.array(target_list)


class tSNE_representattion():
    def __init__(self, model_embedding, model_target, args):
        self.model_embedding = model_embedding
        self.model_target = model_target
        self.args = args

    def tSNE_reduce_dimension(self):
        reducer = manifold.TSNE(init='pca', n_iter=1000, random_state=32)
        # scaler = StandardScaler()
        # scaler= Normalizer()
        scaler= MinMaxScaler()
        features_normalized = scaler.fit_transform(self.model_embedding)
        self.tSNE_embedding = reducer.fit_transform(features_normalized)

        save_name = 'tSNE_embedding.npy'
        # save_path = os.path.join(self.args.output_dir, save_name)
        # np.save(save_path, self.tSNE_embedding)
        self.KL = reducer.kl_divergence_
    def show(self, figsize=(4, 4), save=None):

        fig, ax = plt.subplots(figsize=figsize)
        classes = list(np.unique(self.model_target))
        markers = 'oo' * len(classes)
        x_min, x_max = np.min(self.tSNE_embedding[:, 0]), np.max(self.tSNE_embedding[:, 1])
        y_min, y_max = np.min(self.tSNE_embedding[:, 1]), np.max(self.tSNE_embedding[:, 1])

        pos_color = '#fc8d62'
        neg_color = '#8da0cb'

        colors = [neg_color, pos_color]
        labels = ['neg', 'pos']

        for i, c in enumerate(classes):
            print(markers[i])
            ax.scatter(*self.tSNE_embedding[self.model_target == c].T, marker=markers[i], c=[colors[i]],
                       label=labels[i],
                       alpha=1, s=24)
        if args.legend:
            ax.legend()
        ax.axis("off")
        fig.set_facecolor('white')
        if args.legend:
            plt.legend(prop={'size': 20}, frameon=False, labelcolor=[colors[0], colors[1]])  # remove edge and revise
        if save:
            plt.tight_layout()
            plt.text(x_min, y_min, 'KL:%.2f' % self.KL, fontsize=18, color='b')
            # plt.title('t-SNE_ratio_1-1_n_iter-10000',color='w')
            # plt.savefig(os.path.join(self.args.output_dir, save), dpi=600, facecolor=fig.get_facecolor(),
            #             edgecolor='none')
            plt.show()
        else:
            plt.show()
    def infer(self):
        self.tSNE_reduce_dimension()
        self.show(save=args.save)





def main(args):

    device = torch.device(args.device)
    transform_test = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.ToTensor()])

    dataset_test = myDataset_linearProb(os.path.join(args.data_test), transform=transform_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    ## load model
    model = torchvision.models.resnet50(weights=None)  # num_classes =2
    model.fc = nn.Identity()
    if args.rsnet_origin_W:
        checkpoint_model = torch.load(args.pretrained, map_location='cpu')
        del_keys = ['fc.weight', 'fc.bias']
        for k in del_keys:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)
    else:
        checkpoint_model = torch.load(args.pretrained, map_location='cpu')
        fc_checkpoint=checkpoint_model['state_dict']
        del_keys = ['fc.weight', 'fc.bias']
        for k in del_keys:
            if k in fc_checkpoint:
                print(f"Removing key {k} from pretrained checkpoint")
                del fc_checkpoint[k]
        msg=model.load_state_dict(fc_checkpoint, strict=False)
        print(msg)
    # checkpoint_model = fc_checkpoint['state_dict']


    model.to(args.device)

    # add model revise module
    start_time = time.time()
    print(':::::::::::start processing t-SNE ::::::::::::')
    pred_list, target_list = image_representation(data_loader_test, model, device)
    target_Path = os.path.join(args.output_dir, 'target.npy')
    # np.save(target_Path, target_list)
    tSNE_reduce = tSNE_representattion(pred_list, target_list, args)
    tSNE_reduce.infer()

    total_time = time.time() - start_time
    print('evaluate  time:', total_time)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
