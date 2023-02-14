import os
from glob import glob
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader

from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET


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


def make_dataset(dir, image_ids, targets, task='segmentation'):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        img_path = os.path.join(dir, 'images', '%s.jpg' % image_ids[i])
        seg_path = os.path.join(dir, 'annotations/trimaps', '%s.png' % image_ids[i])
        bbox = None
        xml_file = os.path.join(dir, 'annotations/xmls', '%s.xml' % image_ids[i])
        if os.path.exists(xml_file) == True:
            tree = ET.parse(xml_file)
            obj = tree.find('object')
            bndbox = obj.find('bndbox')
            bbox = [int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text)]
        elif task=='detection':
            #print('[Warning]: lack file of {}'.format(xml_file))
            #pass
            continue
        else:
            pass
        item = (img_path, seg_path, targets[i], bbox)
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
    
    def __init__(self, root, split, transform=None, target_transform=None, download=None, loader=default_loader, task='segmentation', **kwargs):
        # if split == 'train':
        #     split = 'trainval'
        image_ids, targets, classes, class_to_idx = find_classes(os.path.join(root, f'annotations/{split}.txt'))
        
        self.root = root
        self.loader = default_loader
        self.samples = make_dataset(self.root, image_ids, targets, task)
        self.len = len(self.samples)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.task = task

        print('sample:{}'.format(len(self.samples)))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        img_path, seg_path, target, bbox = self.samples[index]
        sample = self.loader(img_path)
        seg = self.loader(seg_path)
        target_img= self.get_seg_img(seg)
        if self.task == 'detection':
            target_img = self.get_bbox_img(sample, bbox)
        if self.transform is not None:
            sample = self.transform(sample)
            target_img = self.transform(target_img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target_img, target
    
    def __len__(self):
        return self.len

    def get_seg_img(self, seg):
        seg = np.array(seg)
        seg -= 2
        seg = Image.fromarray(seg)
        return seg
    
    def get_bbox_img(self, img, bbox):
        if bbox is None:
            return None

        #img.save('new_test.jpg')
        box_img = img.copy()
        a = ImageDraw.ImageDraw(box_img)
        a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='red', width=2)
        #img.save('new_test_2.jpg')
        return box_img
