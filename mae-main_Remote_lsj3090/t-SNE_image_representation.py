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
from util.datasets import myDataset2
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
import models_vit
from sklearn import manifold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler


def get_args_parser():
    parser = argparse.ArgumentParser('tSNE', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--pretrained',
                        default='',
                        help='pretrained checkpoints')

    # Dataset parameters
    parser.add_argument('--data_path_test',
                        default='',
                        type=str, help='txt dataset path ')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    ## UMAP parameters

    ## Output parameters
    parser.add_argument('--output_dir',
                        default='../mae_main_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/Finetune/epoch_50/ratio_1-1/Finetune_Group1/ratio_10-100/Blr-5e-4-Resize-toTensor-SGD-finetune-epoch50/tsne',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save', default='tsne_pretrain_ratio_1-1_ratio_10-100_epoch50.png', help='umap reduce representation')
    parser.add_argument('--legend', action='store_true')
    return parser


@torch.no_grad()
def image_representation(data_loader, model, device):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 't-SNE'
    pred_list = []
    target_list = []

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 1, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.forward_features(images)
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
        # print(reducer.kl_divergence_)

        scaler = StandardScaler()
        # scaler = Normalizer()
        # scaler= MinMaxScaler()
        features_normalized = scaler.fit_transform(self.model_embedding)

        self.tSNE_embedding = reducer.fit_transform(features_normalized)

        print(reducer.kl_divergence_)
        save_name = 'tSNE_embedding.npy'
        save_path = os.path.join(self.args.output_dir, save_name)
        self.KL = reducer.kl_divergence_
        # np.save(save_path, self.tSNE_embedding)

    def show(self, figsize=(4, 4), save=None):

        fig, ax = plt.subplots(figsize=figsize)
        classes = list(np.unique(self.model_target))
        markers = 'oo' * len(classes)

        # pos_color = '#ffec3d'
        # neg_color = '#69b1ff'
        # pos_color = '#f9945e'  # candiate
        # neg_color = '#7990f7'

        pos_color = '#fc8d62'
        neg_color = '#8da0cb'
        x_min, x_max = np.min(self.tSNE_embedding[:, 0]), np.max(self.tSNE_embedding[:, 1])
        y_min, y_max = np.min(self.tSNE_embedding[:, 1]), np.max(self.tSNE_embedding[:, 1])

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
            plt.legend(loc=1,prop={'size': 20}, frameon=False, labelcolor=[colors[0], colors[1]])  # remove edge and revise
        if save:
            plt.tight_layout()
            # plt.title('t-SNE_ratio_1-1_n_iter-10000',color='w')
            plt.text(x_min, y_min,'KL:%.2f' %self.KL, fontsize=18, color='b')
            # plt.savefig(os.path.join(self.args.output_dir, save), dpi=600, facecolor=fig.get_facecolor(),
            #             edgecolor='none')
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

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.pretrained)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    # for key, val in checkpoint_model.items():
    #     print(key)
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # # interpolate position embedding
    # interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    # if args.global_pool:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    # else:
    #     assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
    #     for key, value in vars(args).items():
    #         f.write('%s:%s\n' % (key, value))

    start_time = time.time()
    print(':::::::::::start processing t-SNE::::::::::::')
    pred_list, target_list = image_representation(data_loader_test, model, device)
    target_Path = os.path.join(args.output_dir, 'target.npy')
    # np.save(target_Path, target_list)
    tSNE_reduce = tSNE_representattion(pred_list, target_list, args)
    tSNE_reduce.infer()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('evaluate  time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
