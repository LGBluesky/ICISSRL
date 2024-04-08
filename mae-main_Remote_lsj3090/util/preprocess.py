# -*- coding: utf-8 -*-
"""
This file is a part of project "Aided-Diagnosis-System-for-Cervical-Cancer-Screening".
See https://github.com/ShenghuaCheng/Aided-Diagnosis-System-for-Cervical-Cancer-Screening for more information.

File name: preprocess
Description: preprocess data
"""

from .augmentations import Augmentations

__all__ = [
    "Preprocess"
]


class Preprocess:
    def __init__(
            self,
            dst_size,
            gamma=0.,
            sharp=0.5,
            blur=0.5,
            hsv_disturb=0.5,
            rgb_switch=0.5,
            rotate=0.5,
            h_flip=0.5,
            v_flip=0.5,
            norm=[0, 1],
            to_train=True
    ):
        self.dst_size = dst_size
        self.pre_funcs = [
            Augmentations.Resize(dst_size),
            Augmentations.RandomGamma(gamma),
            Augmentations.RandomSharp(sharp),
            Augmentations.RandomGaussainBlur(blur),
            Augmentations.RandomHSVDisturb(hsv_disturb),
            Augmentations.RandomRGBSwitch(rgb_switch),
            Augmentations.RandomRotate90(rotate),
            Augmentations.RandomHorizontalFlip(h_flip),
            Augmentations.RandomVerticalFlip(v_flip),
        ]
        if to_train:
            self.preprocess = Augmentations.Compose(*self.pre_funcs)
        else:
            self.preprocess = Augmentations.Resize(dst_size)
        self.normalize = Augmentations.Normalization(norm)

    def __call__(self, image):
        image = self.preprocess(image)
        # image = self.normalize(image)
        return image

if __name__ == '__main__':
    import cv2
    import numpy as np
    import random
    import torch
    from torchvision import transforms
    path = '/mnt/media_local/Data/YJX_data/class2/data7/neg/1160032_1254.tif.jpg'
    img = cv2.imread(path)
    cv2.imshow('4', img)
    img = img[:, :, ::-1]
    cv2.imshow('3', img)
    cv2.waitKey(0)

    toTensor = transforms.ToTensor()

    random.seed(10)
    for i in range(5):
        preProcess = Preprocess(dst_size=(224, 224))
        img_aug = preProcess(img)
        img_tensor = toTensor(img_aug)
        # print((torch.min(img_tensor), torch.max(img_tensor)))
        print((np.min(img_aug), np.max(img_aug)))
        cv2.imshow('1', img)
        cv2.imshow('2', img_aug)
        cv2.waitKey(50)
        cv2.destroyAllWindows()
    print('OK')
