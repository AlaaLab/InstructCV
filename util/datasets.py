# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import pdb
import os
import PIL
import torch
from dataset import Pets
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from universal_mae_helpers import imagenet_mean, imagenet_std
import config

normalise_dict = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
Datasets = {"oxford-pets": Pets}

def make_signal_task_dataset(dataset_name, task, batch_size, split):

    image_size = config.dataset_params[dataset_name]['IMG_SIZE']
    dataset_path = config.dataset_params[dataset_name]['data_path']

    if task == 'segmentation':
        transform_aug = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            ])

        dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":transform_aug}
        dataset = Datasets[dataset_name](**dataset_params, task=task, split=split)
    
        if split != 'test':
            dataset_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        else:
            dataset_loader = DataLoader(dataset, batch_size, shuffle=False)

    elif task == 'detection':
        transform_aug = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            ])

        dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":transform_aug}
        dataset = Datasets[dataset_name](**dataset_params, task=task, split=split)
    
        if split != 'test':
            dataset_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        else:
            dataset_loader = DataLoader(dataset, batch_size, shuffle=False)
    elif task == 'classification':
        transform_aug = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            ])

        dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":transform_aug}
        dataset = Datasets[dataset_name](**dataset_params, task=task, split=split)
    
        if split != 'test':
            dataset_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        else:
            dataset_loader = DataLoader(dataset, batch_size, shuffle=False)
    else: # for test
        transform_aug = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            ])

        dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":transform_aug}
        dataset = Datasets[dataset_name](**dataset_params, task=task, split=split)

        if split != 'test':
            dataset_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        else:
            dataset_loader = DataLoader(dataset, batch_size, shuffle=False)

    return dataset_loader

def make_multi_task_dataset(dataset_name, tasks, batch_size, split):
    '''
    Create two dataloder: dataset_loader, language_dataset_loader
    Args:
        tasks: ["classification", "segmentation", "detection"]
    '''
    
    image_size = config.dataset_params[dataset_name]['IMG_SIZE']
    dataset_path = config.dataset_params[dataset_name]['data_path']

    transform_aug = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        ])

    #NOTE: for test
    if tasks == ['multi-task'] and split == 'test':
        dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":None}
        dataset = Datasets[dataset_name](**dataset_params, task=tasks, split=split)
        dataset_loader = DataLoader(dataset, batch_size)
        return dataset_loader

    #image dataset
    dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":transform_aug}
    dataset = Datasets[dataset_name](**dataset_params, task=tasks, split=split)

    #language dataset
    dataset_params = {"root":dataset_path, "transform":transform_aug, "target_transform":None}
    language_dataset = Datasets[dataset_name](**dataset_params, task=tasks, split=split, g_task_id=True)

    g=torch.Generator()
    g.manual_seed(0)

    if split != 'test':
        dataset_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, generator=g)
        language_dataset_loader = DataLoader(language_dataset, batch_size, shuffle=True, drop_last=True, generator=g)
    else:
        dataset_loader = DataLoader(dataset, batch_size, shuffle=False)
        language_dataset_loader = DataLoader(language_dataset, batch_size, shuffle=False)


    for lan in language_dataset_loader:
        print('lan:{}'.format(lan))
        break

    return (dataset_loader, language_dataset_loader)


def build_dataset(dataset_name, batch_size, task, split):
    '''
    Enable multi-task datasets
    '''
 
    if type(task) == list:
        train_loader = make_multi_task_dataset(dataset_name, task, batch_size, split)
        val_loader = make_multi_task_dataset(dataset_name, task, batch_size, split='test')
    else:
        train_loader = make_signal_task_dataset(dataset_name, task, batch_size, split)
        val_loader = make_signal_task_dataset(dataset_name, task, batch_size, split='test')
    
    return train_loader, val_loader


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

make_dataset = {"single_task": make_signal_task_dataset, "multi-task": make_multi_task_dataset}
