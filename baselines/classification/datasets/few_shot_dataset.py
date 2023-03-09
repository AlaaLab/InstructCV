# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from abc import abstractmethod

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.dtd import DTD
from datasets.pets import Pets
from datasets.cars import Cars
from datasets.food import Food
from datasets.sun397 import SUN397
from datasets.voc2007 import VOC2007
from datasets.flowers import Flowers
from datasets.aircraft import Aircraft
from datasets.caltech101 import Caltech101

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from PIL import ImageEnhance

transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)

identity = lambda x:x


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def get_dataset(dset, root, split, transform):
    try:
        return dset(root, train=(split == 'train'), transform=transform, download=True)
    except:
        return dset(root, split=split, transform=transform, download=True)


class SetDataset:
    def __init__(self, dset, root, num_classes, batch_size, transform):
        self.sub_meta = {}
        self.cl_list = range(num_classes)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        # load the data from all splits in the datasetv
        if dset in [Aircraft, DTD, Flowers, VOC2007]:
            # if we have a predefined validation set
            train_dataset = get_dataset(dset, root, 'train', transform)
            valid_dataset = get_dataset(dset, root, 'val', transform)
            trainval_dataset = ConcatDataset([train_dataset, valid_dataset])
        elif dset in [datasets.CIFAR10, datasets.CIFAR100]:
            trainval_dataset = get_dataset(dset, root, 'train', identity)
        else:
            trainval_dataset = get_dataset(dset, root, 'train', transform)
        if dset in [datasets.CIFAR10, datasets.CIFAR100]:
            test_dataset = get_dataset(dset, root, 'test', identity)
        else:
            test_dataset = get_dataset(dset, root, 'test', transform)
        d = ConcatDataset([trainval_dataset, test_dataset])
        print(f'Total dataset size: {len(d)}')

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        for label in self.sub_meta:
            if len(self.sub_meta[label]) == 0:
                del self.sub_meta[label]
                del self.cl_list[label]

        print('Number of images per class')
        for key, item in self.sub_meta.items():
            print(len(self.sub_meta[key]))
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform if dset in [datasets.CIFAR10, datasets.CIFAR100] else identity)
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug=False, normalise=True):
        if aug:
            if normalise:
                transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
            else:
                transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor']
        else:
            if normalise:
                transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']
            else:
                transform_list = ['Scale','CenterCrop', 'ToTensor']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SetDataManager(DataManager):
    def __init__(self, dset, root, num_classes, image_size, n_way=5, n_support=5, n_query=16, n_episode=100, seed=0):        
        super(SetDataManager, self).__init__()
        self.dset = dset
        self.root = root
        self.num_classes = num_classes
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode

        self.trans_loader = TransformLoader(image_size)

        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_data_loader(self, aug, normalise): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug, normalise)
        dataset = SetDataset(self.dset, self.root, self.num_classes, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 12, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    pass
