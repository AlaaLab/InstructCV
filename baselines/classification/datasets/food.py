import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import numpy as np
from PIL import Image
from scipy.io import loadmat

import json


class Food(ImageFolder):
    """`Food-101 Dataset.
    Args:
        root (string): Root directory path to dataset.
        split (string): dataset split to load. E.g. ``train``
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    def __init__(self, root, split, transform=None, target_transform=None, download=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        f = open(os.path.join(root, 'food-101', 'meta', split + '.json'), 'r')
        class_to_image_ids = json.loads(f.read())
        samples = []
        with open(os.path.join(root, 'food-101', 'meta', 'classes.txt'), 'r') as f:
            classes = f.read().splitlines()
        for i, class_name in enumerate(classes):
            for image_id in class_to_image_ids[class_name]:
                samples.append((os.path.join(root, 'food-101', 'images', image_id + '.jpg'), i))
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
