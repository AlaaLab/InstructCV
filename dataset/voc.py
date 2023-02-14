import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def voc_label_indices(colormap):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def read_file_list(root, is_train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        filenames = f.read().split()
    images = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in filenames]
    labels = [os.path.join(root, 'SegmentationClass', i + '.png') for i in filenames]
    return images, labels  # file list


def voc_rand_crop(image, label, height, width):
    """
    Random crop image (PIL image) and label (PIL image).
    """
    i, j, h, w = transforms.RandomCrop.get_params(
        image, output_size=(height, width))

    image = transforms.functional.crop(image, i, j, h, w)
    label = transforms.functional.crop(label, i, j, h, w)

    return image, label


class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train , root, transform, **kwargs):
        """
        crop_size: (h, w)
        """
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.transform = transform

        images, labels = read_file_list(root=root, is_train=is_train)
        self.images = self.filter(images)  # images list
        self.labels = self.filter(labels)  # labels list
        print('Read ' + str(len(self.images)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
                Image.open(img).size[1] >= 0 and
                Image.open(img).size[0] >= 0)]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image).convert('RGB')
        label = Image.open(label).convert('RGB')

        # image, label = voc_rand_crop(image, label,
        #                              *self.crop_size)
        image = self.transform(image)
        # label = voc_label_indices(label)
        label = self.transform(label)

        return image, label  # float32 tensor, uint8 tensor

    def __len__(self):
        return len(self.images)
