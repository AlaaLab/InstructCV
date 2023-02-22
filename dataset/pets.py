import os
from glob import glob
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import datasets, transforms

from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET

import jieba

def word_token2idx(language, word_token):
    
    seg_list = list(jieba.cut(language))
    tokens = []
    for w in seg_list:
        if w == ' ':
            continue
        if len(tokens) >= 20:
            continue
        idx = word_token[w]
        tokens.append(idx)

    for _ in range(len(tokens), 20):
        tokens.append(0)

    return tokens


def build_token_dict(word_token_file, samples=[]):
    '''
    Return: word token
    '''

    word_token = {}
    if os.path.exists(word_token_file):
        for line in open(word_token_file):
            words = line.strip().split()
            w = words[0]
            idx = int(words[1])
            word_token[w] = idx
    for img_path, task, target in samples:
        if task == 'segmentation':
            language = 'segment cat'
        elif task == 'detection':
            language = 'detect cat'
        else:
            language = 'is this a photo of cat?'
        seg_list = list(jieba.cut(language))

        for w in seg_list:
            if w == ' ':
                continue
            if w not in word_token:
                word_token[w] = len(word_token) + 1
    
    fp = open(word_token_file, 'w')
    for w in word_token:
        fp.write(w + ' ' + str(word_token[w]) + '\n')

    fp.close()
    return word_token


def find_classes(classes_file):
    '''
    Read classes file, separating out image IDs and class names
    E.g., image_ids: Abyssinian_100; classes/targets: 1; 
    class_to_idx: dict to store class & idx mapping
    '''

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
    
    '''
    Args:
        task: list if multi-task
    Return:
        Images: append the list of Classificaiton, Segmentation, Detection
                    Classificaiton: ((img_path, tk, target)...,)
                    Segmentation: ((img_path, tk, seg_path)...,)
                    Detection: ((img_path, tk, bbox)...,) ;test -- ((img_path, tk, target)...,) 
    '''
    
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)

    if type(task) != list:
        task = [task]

    for tk in task:
        for i in range(len(image_ids)):
            if tk == 'classification' and targets[i] not in [1, 2]: # 1 cat, 2 dog
                continue
            if tk == 'classification':
                img_path = os.path.join(dir, 'images', '%s.jpg' % image_ids[i])
                item = (img_path, tk, targets[i])
                images.append(item)
    
            elif tk  == 'segmentation':
                img_path = os.path.join(dir, 'images', '%s.jpg' % image_ids[i])
                seg_path = os.path.join(dir, 'annotations/trimaps', '%s.png' % image_ids[i])
                item = (img_path, tk, seg_path)
                images.append(item)
    
            elif tk == 'detection':
                img_path = os.path.join(dir, 'images', '%s.jpg' % image_ids[i])
                bbox = None
                xml_file = os.path.join(dir, 'annotations/xmls', '%s.xml' % image_ids[i])
                if os.path.exists(xml_file) == False:
                    continue
    
                tree = ET.parse(xml_file)
                obj = tree.find('object')
                bndbox = obj.find('bndbox')
                bbox = [int(bndbox.find('xmin').text),
                        int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text),
                        int(bndbox.find('ymax').text)]
                item = (img_path, tk, bbox)
                images.append(item)
            else: # for test
                img_path = os.path.join(dir, 'images', '%s.jpg' % image_ids[i])
                item = (img_path, tk, targets[i])
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
    
    def __init__(self, root, split, transform=None, target_transform=None, download=None, loader=default_loader, task='segmentation', g_task_id = False, word_token_file='language_word.txt', **kwargs):
        '''
        g_task_id: identify if this is for the language dataloader
        word_token_file: predefined file
        task: list (tasks)
        '''
        image_ids, targets, classes, class_to_idx = find_classes(os.path.join(root, f'annotations/{split}.txt'))
        
        self.root = root
        self.task = task
        self.g_task_id = g_task_id
        self.loader = default_loader
        self.samples = make_dataset(self.root, image_ids, targets, task) # task:list
        self.len = len(self.samples)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform

        print('sample:{}'.format(len(self.samples)))
        print('g_task_id:{}'.format(g_task_id ))

        self.word_token = build_token_dict(word_token_file, self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        img_path, task, target = self.samples[index]
        sample = self.loader(img_path)

        if task == 'segmentation':
            seg = self.loader(target)
            target = self.get_seg_img(seg)
            language = 'segment cat'
        
        elif task == 'detection':
            target = self.get_bbox_img(sample, target)
            language = 'detect cat'

        elif task == 'classification':
            target = self.get_class_img(sample, target)
            language = 'is this a photo of cat?'
        else:
            target = self.loader(img_path)

        if self.g_task_id == True:
            lan_token = torch.tensor(word_token2idx(language, self.word_token))

            #print('language:{}'.format(language))
            #print('lan_token:{}'.format(lan_token))

            return lan_token

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if type(self.task) == list:
            #sample = self.merge_imgs(sample, target)
            sample = torch.cat([sample, target], dim=1)

        return sample, target

    def __len__(self):
        return self.len

    def get_seg_img(self, seg):
        seg = np.array(seg)
        seg -= 2
        seg_img = Image.fromarray(seg)
        return seg_img
    
    def get_bbox_img(self, img, bbox):
        if bbox is None:
            return None

        #img.save('new_test.jpg')
        box_img = img.copy()
        a = ImageDraw.ImageDraw(box_img)
        a.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), fill=None, outline='red', width=6)
        #img.save('new_test_2.jpg')
        return box_img

    def get_class_img(self, sample, target):
        if target == 1:
            img = Image.new('RGB', sample.size, (255, 255, 255))
        elif target == 2:
            img = Image.new('RGB', sample.size, (0, 0, 0))
        else:
            img = sample.copy()

        return img

    def merge_imgs(self, sample, target):
        dst = Image.new('RGB', (sample.width, sample.height+target.height))
        dst.paste(sample, (0, 0))
        dst.paste(target, (0, sample.height))
        return dst


