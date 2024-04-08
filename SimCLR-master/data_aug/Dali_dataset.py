import random
import torch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from nvidia import dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from PIL import Image
from torchvision import transforms
import  os

class DataSource(object):
    def __init__(self, img_path:str,shuffle=True,batch_size=64):
        self.batch_size = batch_size
        self.img_path = img_path
        self.__img_list()

        if shuffle:
            random.shuffle(self.imgList)
    def __img_list(self):

        with open(self.img_path, 'r') as f:
            self.imgList = f.readlines()


    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        imgs = []
        labels = []


        if self.i >= len(self.imgList):
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            # img_name, label = self.paths[self.i % len(self.paths)]
            imgPath = self.imgList[self.i].strip('\n')
            key = os.path.split(imgPath)[0].split('/')[-1]
            if key == self.cls1:
                label = 0
            if key == self.cls2:
                label = 1

            """
            dali定义的读取方式
            """
            img_file = open(imgPath, 'rb')
            imgs.append(np.frombuffer(img_file.read(), dtype=np.uint8))
            labels.append(np.array([label]))

            # img_seg_file = open(seg_img_path, 'rb')
            # seg_imgs.append(np.frombuffer(img_seg_file.read(), dtype=np.uint8))
            """
            自定义读取方式
            """
            # img = Image.open(image_path)
            # seg_img = Image.open(seg_img_path)
            # img = np.array(img)
            # seg_img = np.array(seg_img)
            # imgs.append(img)
            # seg_imgs.append(seg_img)
            # labels.append(np.array([label]))
            self.i += 1

        return (imgs, labels)

    def __len__(self):
        return len(self.imgList)

    next = __next__

class SourcePipeline(Pipeline):
    def __init__(self,  batch_size, num_threads, device_id, external_data,modeltype):
        super(SourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12,
                                                     exec_async=True,
                                                     exec_pipelined=True,
                                                     prefetch_queue_depth = 2
                                                     )
        self.input_data = ops.ExternalSource(num_outputs=3)
        self.external_data = external_data
        self.model_type = modeltype
        self.iterator = iter(self.external_data)
        self.res = ops.Resize(device="gpu", resize_x=256, resize_y=256)
        self.decode = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.cat = ops.Cat(device="gpu",axis=2)
        self.tran = ops.Transpose(device="gpu",perm=[2,0,1])
        self.crop = ops.RandomResizedCrop(device="gpu",size =256,random_area=[0.08, 1.25])
        self.resize = ops.Resize(device='gpu', resize_x=256, resize_y=256)
        self.no_mirrror_cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255, 0],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255, 1])
        self.mirrror_cmnp = ops.CropMirrorNormalize(device="gpu",
                                                   output_dtype=types.FLOAT,
                                                   output_layout=types.NCHW,
                                                   mirror = 1,
                                                   mean=[0.485 * 255, 0.456 * 255, 0.406 * 255, 0],
                                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255, 1])

        # self.flip = ops.random.CoinFlip(device="gpu", probability=0.5


    def define_graph(self):
        self.img,self.img_seg,self.labels = self.input_data()

        """
        读取图像数据(效果不是很好)
        """
        # images =self.img
        # img_seg = self.img_seg
        # img_seg = img_seg[:,:,dali.newaxis]

        """
        图像维度拼接，为统一预处理准备
        """
        # images = fn.cat(images,img_seg,axis = 2)
        # images = self.tran(images)


        """
        读取图像数据、seg_img变为一维通道
        """
        image = self.decode(self.img)
        img_seg = self.decode(self.img_seg)
        img_seg = img_seg[:,:,0:1]

        """
        cat维度拼接
        """
        fuse_img = self.cat(image, img_seg)

        if self.model_type == 'train':
            """
            对四通道图像Normalize处理、HWC-->CHW、裁剪、随机翻转
            """
            probability = random.random()
            if probability < 0.5:
                fuse_img = self.no_mirrror_cmnp(fuse_img)
            else:
                fuse_img = self.mirrror_cmnp(fuse_img)
            fuse_img = self.crop(fuse_img)
        if self.model_type == 'val':
            fuse_img = self.no_mirrror_cmnp(fuse_img)
            fuse_img = self.resize(fuse_img)

        """
        各数据源提取
        """
        image = fuse_img[0:3]
        img_seg = fuse_img[-1]
        img_seg = img_seg[dali.newaxis]

        """
        标签处理
        """
        label = self.labels[0]


        return (image,img_seg,label)

    def iter_setup(self):
        try:
            image,seg_img,labels = self.iterator.next()
            self.feed_input(self.img, image)
            self.feed_input(self.img_seg, seg_img)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class CustomDALIGenericIterator(DALIGenericIterator):
    def __init__(self, length,  pipelines,output_map, **argw):
        self._len = length # dataloader 的长度
        output_map = output_map
        super().__init__(pipelines, output_map, **argw)

    def __next__(self):
        batch = super().__next__()
        return self.parse_batch(batch)

    def __len__(self):
        return self._len

    def parse_batch(self, batch):
        img = batch[0]['imgs']
        label = batch[0]["labels"]  # bs * 1

        return {"image": img,"labels": label}
