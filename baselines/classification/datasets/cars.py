import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

import numpy as np
from PIL import Image
from scipy.io import loadmat


def make_dataset(root, split, annotations_path):
    annotations = loadmat(annotations_path)['annotations'][0]
    image_ids = []
    labels = []
    for element in annotations:
        image_ids.append(os.path.join(root, 'cars_' + split, str(element[-1][0])))
        labels.append(int(element[-2]))
    classes = np.unique(labels)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    # correct labels to be the indexes we use in training
    labels = [class_to_idx[l] for l in labels]
    return image_ids, labels, classes, class_to_idx


class Cars(ImageFolder):
    """`Standford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html/>`_ Dataset.
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

        #file_names = glob(os.path.join(root, 'cars_' + split))
        if split == 'train':
            annot_file = 'cars_train_annos.mat'
        elif split == 'test':
            annot_file = 'cars_test_annos_withlabels.mat'
        image_ids, labels, classes, class_to_idx = make_dataset(root, split, os.path.join(root, 'devkit', annot_file))
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
