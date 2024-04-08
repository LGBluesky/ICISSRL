import six

import torch
from torch.utils.data import Dataset

from PIL import Image
import cv2
import os
import lmdb
import numpy as np
import io
import pyarrow as pa




class myDataset(Dataset):
    def __init__(self, txtPath,  transform):
        super(myDataset, self).__init__()
        with open(txtPath, 'r') as f:
            self.imgList = f.readlines()
        self.transform = transform

    def __len__(self):
         return len(self.imgList)

    def __getitem__(self, item):
        imgPath = self.imgList[item].strip('\n')
        img = Image.open(imgPath).convert('RGB')
        img = self.transform(img.copy())
        return img, 0
class myDataset_linearProb(Dataset):
    def __init__(self, txtPath,  transform):
        super(myDataset_linearProb, self).__init__()
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
        img = self.transform(img.copy())
        if key == self.cls1:
            label = 0
        if key == self.cls2:
            label = 1
        return img, label

class my_lmdb_linearProb(Dataset):
    def __init__(self, txtPath,  transform, lmdb_path):
        super(my_lmdb_linearProb, self).__init__()
        with open(txtPath, 'r') as f:
            self.imgList = f.readlines()
        self.transform = transform
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, subdir= os.path.isdir(self.lmdb_path),
                             readonly=True, lock=False, readahead=False,meminit=False)
        self.cls1 = 'neg'
        self.cls2 = 'pos'


    def __len__(self):
         return len(self.imgList)

    def __getitem__(self, item):

        imgPath = self.imgList[item].strip('\n')
        key = os.path.split(imgPath)[0].split('/')[-1]
        env = self.env
        with env.begin(write=False) as txn:
            imgBin = txn.get(imgPath.encode())
        buf = io.BytesIO()
        buf.write(imgBin)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = self.transform(img.copy())
        if key == self.cls1:
            label = 0
        if key == self.cls2:
            label = 1

        return img, label



class my_lmdb_dataset(Dataset):
    def __init__(self, txtPath,  transform, lmdb_path):
        super(my_lmdb_dataset, self).__init__()
        with open(txtPath, 'r') as f:
            self.imgList = f.readlines()
        self.transform = transform
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, subdir= os.path.isdir(self.lmdb_path),
                             readonly=True, lock=False, readahead=False,meminit=False)


    def __len__(self):
         return len(self.imgList)

    def __getitem__(self, item):

        imgPath = self.imgList[item].strip('\n')
        env = self.env
        with env.begin(write=False) as txn:
            imgBin = txn.get(imgPath.encode())
            # imgBin = txn.get(self.keys[item])
        # unpacked = pa.deserialize(imgBin)
        # imgbuf =  unpacked[0]
        buf = io.BytesIO()
        buf.write(imgBin)
        buf.seek(0)

        # 二进制数据转换为PIL图像, 准还为十进制像素值
        # img = Image.open(io.BytesIO(imgBin))
        img = Image.open(buf).convert('RGB')
        img = self.transform(img.copy())

            ## opencv
            # imgBuf = np.frombuffer(imgBin, dtype=np.uint8)  #
            # img2 = cv2.imdecode(imgBuf, cv2.IMREAD_COLOR)
            # img2= cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            # img_tensor = torch.from_numpy(img2).permute(2, 0, 1)
        return img, 0



if __name__ == '__main__':
    root = '../data/sampler-80K-40K-9groups_Remote-lsj3090-nvme/Group1/MAE_pretrain/ratio_1-1.txt'
    transformer= None
    lmdb_dir = '/mnt/media_nvme/Data/YJX_data_lmdb/pretrain_120'
    myData = my_lmdb_dataset(root,transformer, lmdb_dir)
    # myData = myDataset(root, transformer)
    for i in range(1):
        img, target = myData.__getitem__(i)
        img = np.array(img)
        print(img.shape)
        # img.show()
        print('OK')
        # print(img.shape)

        # cv2.imshow('0', img[:,:,::-1].copy())
        # cv2.waitKey(10000)

    # cv2.destroyAllWindows()


    # src ='/mnt/media_nvme/Data/YJX_data/class2/data5/pos/190312601_45_125765_64641_417_417_ASC-US.tif.jpg'

