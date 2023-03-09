import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader

from PIL import Image
import numpy as np


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(int(split_line[1].strip()))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'images', '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


class Pets(Dataset):
    """`Oxfod-IIT Pets <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_ Dataset.
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
        if split == 'train':
            split = 'trainval'
        image_ids, targets, classes, class_to_idx = find_classes(os.path.join(root, f'annotations/{split}.txt'))

        self.root = root
        self.loader = default_loader
        self.samples = make_dataset(self.root, image_ids, targets)
        self.len = len(self.samples)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

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
        return self.len
