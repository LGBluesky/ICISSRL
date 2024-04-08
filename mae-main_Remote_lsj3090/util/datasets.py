# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



class myDataset(Dataset):
    def __init__(self, txtPath,  transform=None,  is_train=True):
        super(myDataset, self).__init__()
        with open(txtPath, 'r') as f:
            self.imgList = f.readlines()
        self.transform = transform
        self.toTensor = transforms.ToTensor()
        self.cls1 = 'neg'
        self.cls2 = 'pos'
    def __len__(self):
         return len(self.imgList)

    def __getitem__(self, item):
        imgPath = self.imgList[item].strip('\n')
        key = os.path.split(imgPath)[0].split('/')[-1]
        img = cv2.imread(imgPath)

        # img = Image.open(imgPath).convert('RGB')
        if self.transform is not None:
            img = img[:, :, ::-1]  # toRGB
            img = self.transform(img)
            img = self.toTensor(img)
        if key == self.cls1:
            label = 0
        if key == self.cls2:
            label = 1

        return img, label


class myDataset2(Dataset):
    def __init__(self, txtPath,  transform=None,  is_train=True):
        super(myDataset2, self).__init__()
        with open(txtPath, 'r') as f:
            self.imgList = f.readlines()
        self.transform = transform
        self.cls1 = 'neg'
        self.cls2 = 'pos'
    def __len__(self):
         return len(self.imgList)

    def __getitem__(self, item):
        imgPath = self.imgList[item].strip('\n')
        key = os.path.split(imgPath)[0].split('/')[-1]
        img = Image.open(imgPath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if key == self.cls1:
            label = 0
        if key == self.cls2:
            label = 1
        return img, label


class myDataset3(Dataset):
    ## with names
    def __init__(self, txtPath,  transform=None,  is_train=True):
        super(myDataset3, self).__init__()
        with open(txtPath, 'r') as f:
            self.imgList = f.readlines()
        self.transform = transform
        self.cls1 = 'neg'
        self.cls2 = 'pos'
    def __len__(self):
         return len(self.imgList)

    def __getitem__(self, item):
        imgPath = self.imgList[item].strip('\n')
        key = os.path.split(imgPath)[0].split('/')[-1]
        img = Image.open(imgPath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if key == self.cls1:
            label = 0
        if key == self.cls2:
            label = 1
        return img, label, imgPath