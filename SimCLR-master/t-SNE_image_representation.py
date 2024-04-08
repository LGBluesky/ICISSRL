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
import torch
from torchvision import models
import torchvision.transforms as transforms
from models.resnet_simclr import ResNetSimCLR_feature_representation as Resnet_feature
from pathlib import Path
import os
import numpy as np
from data_aug.custom_dataset import myDataset_linearProb
from sklearn import manifold
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('--data_test', default='./data/sampler-80K-40K-9groups_Remote-lsj3090-nvme/Group1/UMAP_cluster_data_3090/Pos_1000-Neg_1000-seed_10.txt',
                        help='path to test txt')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')

    parser.add_argument('--pretrained', default='', type=str, help='pretrained model checkpoint')
    parser.add_argument('--input_size', default=224, type=int, help='model input size')
    parser.add_argument('--workers', type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--num_workers', default=4, type= int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    ## output path
    parser.add_argument('--output_dir',default='../SimCLR_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090-224/Group1/t-sne', help='path to save t-sne ')

    ## device
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save',default='tSNE_pretrain_ratio_1-1')
    parser.add_argument('--legend',action='store_true')


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
        scaler = Normalizer()
        # scaler = MinMaxScaler()
        features_normalized = scaler.fit_transform(self.model_embedding)

        self.tSNE_embedding = reducer.fit_transform(features_normalized)

        self.KL = reducer.kl_divergence_
        save_name = 'tSNE_embedding.npy'
        save_path = os.path.join(self.args.output_dir, save_name)
        # np.save(save_path, self.tSNE_embedding)

    def show(self, figsize=(4, 4), save=None):

        fig, ax = plt.subplots(figsize=figsize)
        classes = list(np.unique(self.model_target))
        markers = 'oo' * len(classes)

        # pos_color = '#ffec3d'
        # neg_color = '#69b1ff'

        x_min, x_max = np.min(self.tSNE_embedding[:, 0]), np.max(self.tSNE_embedding[:, 1])
        y_min, y_max = np.min(self.tSNE_embedding[:, 1]), np.max(self.tSNE_embedding[:, 1])

        pos_color = '#fc8d62'
        neg_color='#8da0cb'

        colors = [neg_color, pos_color]
        labels = ['neg', 'pos']

        for i, c in enumerate(classes):
            print(markers[i])
            ax.scatter(*self.tSNE_embedding[self.model_target == c].T, marker=markers[i], c=[colors[i]], label=labels[i],
                       alpha=1, s=24)
        if args.legend:
            ax.legend()
        ax.axis("off")
        fig.set_facecolor('white')
        if args.legend:
            plt.legend(prop={'size': 20}, frameon=False, labelcolor=[colors[0], colors[1]]) # remove edge and revise
        if save:
            plt.tight_layout()
            # plt.title('t-SNE_ratio_1-1_n_iter-10000',color='w')
            plt.text(x_min, y_min, 'KL:%.2f' % self.KL, fontsize=18, color='b')
            # plt.savefig(os.path.join(self.args.output_dir, save),dpi=600, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.show()
        else:
            plt.show()
        # plt.show()
    def infer(self):
        self.tSNE_reduce_dimension()
        self.show(save=args.save)


def main(args):
    device = torch.device(args.device)
    # linear probe: weak augmentation
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])

    dataset_test = myDataset_linearProb(os.path.join(args.data_test), transform=transform_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = Resnet_feature(base_model=args.arch)

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.pretrained)
    checkpoint_model = checkpoint['state_dict']

    del_keys = ['backbone.fc.0.weight','backbone.fc.0.bias', 'backbone.fc.2.weight', 'backbone.fc.2.bias']
    for k in del_keys:
        if k in checkpoint_model:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    start_time = time.time()
    print(':::::::::::start processing t-SNE ::::::::::::')
    pred_list, target_list = image_representation(data_loader_test, model, device)
    target_Path = os.path.join(args.output_dir, 'target.npy')

    # np.save(target_Path, target_list)
    tSNE_reduce = tSNE_representattion(pred_list, target_list, args)
    tSNE_reduce.infer()

    total_time = time.time() - start_time
    print('evaluate  time:',total_time)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
