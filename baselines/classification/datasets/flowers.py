import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader

from PIL import Image
import numpy as np
from scipy.io import loadmat


def make_dataset(root, split):
    # get indexes for split
    split = {'train': 'trnid', 'val': 'valid', 'test': 'tstid'}[split]
    split_idxs = loadmat(os.path.join(root, 'setid.mat'))[split].squeeze(0)
    # construct list of all image paths
    image_ids = []
    for element in split_idxs:
        image_ids.append(os.path.join(root, 'jpg', f'image_{element:05}.jpg'))

    # now we correct the indices to start from 0
    # they needed to start from 1 for the image paths
    split_idxs = split_idxs - 1

    # get all labels for the dataset
    all_labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'].squeeze(0)
    # get labels for this split
    labels = all_labels[split_idxs]
    # get classes
    classes = np.unique(labels)
    classes.sort()
    # make map from classes to indexes to use in training
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # correct labels to be the indexes we use in training
    labels = [class_to_idx[l] for l in labels]

    return image_ids, labels, classes, class_to_idx


class Flowers(Dataset):
    """`Oxfod-VGG Flowers <https://www.robots.ox.ac.uk/~vgg/data/flowers/>`_ Dataset.
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
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        image_ids, labels, classes, class_to_idx = make_dataset(root, split)
        self.samples = list(zip(image_ids, labels))

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
