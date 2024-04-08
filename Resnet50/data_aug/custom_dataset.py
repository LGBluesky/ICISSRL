from torch.utils.data import Dataset
from PIL import Image
import cv2
import os

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


if __name__ == '__main__':
    root = '../data/sampler-80K-40K-9groups_Remote-lsj3090/Group1/MAE_pretrain/ratio_1-1.txt'
    myData = myDataset(root,)
    for i in range(1):
        img, target = myData.__getitem__(i)
        print(img.shape)
        print(target.shape)
        # cv2.imshow('0', img[:,:,::-1].copy())
        # cv2.imshow('1', target[:,:,::-1].copy())
        # cv2.waitKey(0)

        print('OK')
    cv2.destroyAllWindows()

