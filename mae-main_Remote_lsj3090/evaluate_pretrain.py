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

import pandas as pd
import torch

from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from timm.utils import accuracy
from util.datasets import myDataset3
import torch.nn as nn

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
import models_mae
import matplotlib.pyplot as plt
import shutil
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training test', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')  # default = false
    parser.set_defaults(norm_pix_loss=False)

    # model parameters
    parser.add_argument('--pretrained', default='../mae_main_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/MAE_pretrain/ratio_1-1/checkpoint-199.pth', type=str)
    parser.add_argument('--pretrained_ratio', default='ratio_1-1', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/sampler-80K-40K-9groups_Remote-lsj3090-nvme/Group1/pretrain_recon/pretrain_FN_recon.txt', type=str,
                        help='dataset txt path ')

    parser.add_argument('--output_dir', default='../mae_main_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/image_representation/MAE_pretrain_test1/FN',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=[0, 10, 20, 30, 40, 50, 60], type=list)


    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # data augment
    return parser


def show_image(image,dst, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip(image * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.savefig(dst, dpi=300, bbox_inches='tight', pad_inches=-0.1)
    return

def cal_psnr_ssim(pred, target):
    pred = torch.clip(pred*255, 0, 255).numpy()
    target = torch.clip(target*255, 0, 255).numpy()
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    print(pred.shape)
    print(target.shape)

    ssim_score = SSIM(pred, target, channel_axis=2)
    psnr_score = PSNR(target, pred)

    return ssim_score, psnr_score



@torch.no_grad()
def evaluate(data_loader, model, device, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'pretrain recon:'


    # switch to evaluation mode
    model.eval()
    for data_iter_step, (samples, _, imagePath) in enumerate(metric_logger.log_every(data_loader, 1, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        torch.manual_seed(args.seed[data_iter_step])
        print(imagePath[0])
        image_512_path = os.path.join(args.output_dir, 'image'+str(data_iter_step))
        if not os.path.exists(image_512_path):
            os.makedirs(image_512_path)
        shutil.copy(imagePath[0], image_512_path)

        samples = samples.to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)  # default)
        pred = model.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        samples = torch.einsum('nchw->nhwc', samples).detach().cpu()

        # masked image
        im_masked = samples * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = samples * (1 - mask) + pred * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [6, 6]
        # fig, ax = plt.subplots(1, 4)

        # plt.subplot(1, 4, 1)
        dst_Root = os.path.join(image_512_path, args.pretrained_ratio)
        if not os.path.exists(dst_Root):
            os.makedirs(dst_Root)
        fig1 = plt.figure(1)
        dst1 = os.path.join(dst_Root, 'original.png')
        show_image(samples[0], dst1)

        # plt.subplot(1, 4, 2)
        fig2 = plt.figure(2)
        dst2 = os.path.join(dst_Root, 'masked.png')
        show_image(im_masked[0], dst2)

        # plt.subplot(1, 4, 3)
        fig3 = plt.figure(3)
        dst3 = os.path.join(dst_Root, 'reconstruction.png')
        show_image(pred[0], dst3)


        # plt.subplot(1, 4, 4)
        fig4 = plt.figure(4)
        dst4 = os.path.join(dst_Root, 'reconstruction_visible.png')
        show_image(im_paste[0], dst4)

        excel_dst = os.path.join(dst_Root, 'SSIM_PSNR.xlsx')
        ssim_score, psnr_score = cal_psnr_ssim(im_paste[0], samples[0])
        eval_score = np.array([[ssim_score, psnr_score]])
        col_name = ['SSIM','PSNR']
        df = pd.DataFrame(eval_score, columns=col_name)
        df.to_excel(excel_dst, index=False, engine='openpyxl')

        # plt.show()
    return


def main(args):

    device = torch.device(args.device)

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])


    dataset_test = myDataset3(os.path.join(args.data_path), transform=transform_val)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)   # 实例化模型

    checkpoint = torch.load(args.pretrained, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.pretrained)
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model)
    print(msg)
    model.to(device)
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    start_time = time.time()

    evaluate(data_loader_test, model, device, args)


    # with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write(json.dumps(all_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('evaluate  time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
