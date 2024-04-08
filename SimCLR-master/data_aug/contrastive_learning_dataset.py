from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug import myDataset, myDataset_linearProb, my_lmdb_dataset, my_lmdb_linearProb


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, scale_size):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        s=1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size, scale=(scale_size, 1.0)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])


        # data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.RandomApply([color_jitter], p=0.8),
        #                                       transforms.RandomGrayscale(p=0.2),
        #                                       GaussianBlur(kernel_size=int(0.1 * size)),
        #                                       transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, aug_size: int, scale_size, lmdb_dir=None):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'custom': lambda: myDataset(self.root_folder,
                                                      transform=ContrastiveLearningViewGenerator(
                                                          self.get_simclr_pipeline_transform(aug_size, scale_size),
                                                          n_views)),
                          'custom_lmdb': lambda: my_lmdb_dataset(self.root_folder,
                                                      transform=ContrastiveLearningViewGenerator(
                                                          self.get_simclr_pipeline_transform(aug_size, scale_size),
                                                          n_views),lmdb_path=lmdb_dir)
                          }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


class ContrastiveLearningDataset_linearProb:
    def __init__(self, root_folder, args):
        self.root_folder = root_folder
        self.args = args

    @staticmethod
    def get_simclr_pipeline_transform(to_train, size):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""

        if to_train:
            # s = 1
            # # size = 224
            # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
            #                                       transforms.RandomHorizontalFlip(),
            #                                       transforms.RandomApply([color_jitter], p=0.8),
            #                                       transforms.RandomGrayscale(p=0.2),
            #                                       GaussianBlur(kernel_size=int(0.1 * size)),
            #                                       transforms.ToTensor()])

            # data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
            #                                       transforms.RandomApply([color_jitter], p=0.8),
            #                                       transforms.RandomGrayscale(p=0.2),
            #                                       GaussianBlur(kernel_size=int(0.1 * size)),
            #                                       transforms.ToTensor()])
            #
            #
            data_transforms = transforms.Compose([transforms.Resize(size=size),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor()])
        else:
            data_transforms = transforms.Compose([transforms.Resize(size=size),
                                                  transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views, to_train, lmdb_path=None):
        valid_datasets = {'custom': lambda: myDataset_linearProb(self.root_folder,
                                                                 transform=ContrastiveLearningViewGenerator(
                                                                     self.get_simclr_pipeline_transform(to_train, size=self.args.input_size),
                                                                     n_views)),
                          'custom_lmdb': lambda: my_lmdb_linearProb(self.root_folder,
                                                                 transform=ContrastiveLearningViewGenerator(
                                                                     self.get_simclr_pipeline_transform(to_train, size=self.args.input_size),
                                                                     n_views), lmdb_path=lmdb_path)
                          }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


if __name__ == '__main__':
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    dataPath = '../data/sampler-80K-40K-9groups_Remote-lsj3090/Group1/MAE_pretrain/ratio_1-1.txt'
    dataName = 'custom'
    dataset = ContrastiveLearningDataset(dataPath)
    train_dataset = dataset.get_dataset(dataName, 2, 224)
    for i in range(100):
        images, targets = train_dataset.__getitem__(i)
        print(images[0].shape)
        print('OK')

