#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import subprocess

import torch
import torch.nn as nn
from torchvision import models


MODELS = {
    'supervised': ['', ''],
    'moco-v1': ['https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar', 'pth'],
    'moco-v2': ['https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar', 'pth'],
    'byol': ['https://storage.googleapis.com/deepmind-byol/checkpoints/pretrain_res50x1.pkl', 'pkl'],
    'swav': ['https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar', 'pth'],
    'deepcluster-v2': ['https://dl.fbaipublicfiles.com/deepcluster/deepclusterv2_800ep_pretrain.pth.tar', 'pth'],
    'sela-v2': ['https://dl.fbaipublicfiles.com/deepcluster/selav2_400ep_pretrain.pth.tar', 'pth'],
    'infomin': ['https://www.dropbox.com/sh/87d24jqsl6ra7t2/AAAzMTynP3Qc8mIE4XWkgILUa/InfoMin_800.pth', 'pth'],
    'insdis': ['https://www.dropbox.com/sh/87d24jqsl6ra7t2/AACcsSIt1_Njv7GsmsuzZ6Sta/InsDis.pth', 'pth'],
    'pirl': ['https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth', 'pth'],
    'pcl-v1': ['https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v1_epoch200.pth.tar', 'pth'],
    'pcl-v2': ['https://storage.googleapis.com/sfr-pcl-data-research/PCL_checkpoint/PCL_v2_epoch200.pth.tar', 'pth']
}


def convert_byol(tf_path, pth_path):
    byol = pickle.load(open(tf_path, 'rb'))

    tf_params_dict = dict(byol['experiment_state'].online_params)
    tf_state_dict = dict(byol['experiment_state'].online_state)
    conv_layers_tf = [value for (key, value) in tf_params_dict.items() if 'conv' in key]
    bn_layers_tf = [value for (key, value) in tf_params_dict.items() if 'batchnorm' in key]
    bn_ema_tf = [value for (key, value) in tf_state_dict.items() if 'batchnorm' in key]

    model = models.resnet50()
    conv_layers_pt = [m for m in list(model.modules()) if type(m) is nn.Conv2d]
    bn_layers_pt = [m for m in list(model.modules()) if type(m) is nn.BatchNorm2d]

    def load_conv_layer(idx, bias=False):
        m = conv_layers_pt[idx]
        # assert the weight of conv has the same shape
        d = conv_layers_tf[idx]
        assert torch.from_numpy(d['w']).permute(3, 2, 0, 1).shape == m.weight.data.shape
        m.weight.data = torch.from_numpy(d['w']).permute(3, 2, 0, 1)

    def load_bn_layer(idx, bias=False):
        m = bn_layers_pt[idx]
        d = bn_layers_tf[idx]
        bn_mean = bn_ema_tf[2 * idx]
        bn_var = bn_ema_tf[2 * idx + 1]
        assert torch.from_numpy(d['scale']).shape[-1] == m.weight.data.shape[-1]
        assert torch.from_numpy(d['offset']).shape[-1] == m.bias.data.shape[-1]
        assert torch.from_numpy(bn_mean['average']).shape[-1] == m.running_mean.shape[-1]
        assert torch.from_numpy(bn_var['average']).shape[-1] == m.running_var.shape[-1]
        m.weight.data = torch.from_numpy(d['scale']).reshape(m.weight.data.shape)
        m.bias.data = torch.from_numpy(d['offset']).reshape(m.bias.data.shape)
        m.running_mean = torch.from_numpy(bn_mean['average']).reshape(m.running_mean.shape)
        m.running_var = torch.from_numpy(bn_var['average']).reshape(m.running_var.shape)

    # first match initial layers
    conv_layers_tf.insert(0, conv_layers_tf.pop(-1))
    bn_layers_tf.insert(0, bn_layers_tf.pop(-1))
    bn_ema_tf.insert(0, bn_ema_tf.pop(-1))
    bn_ema_tf.insert(0, bn_ema_tf.pop(-1))

    # the rest of the layers
    for i in range(len(conv_layers_pt)):
        load_conv_layer(i)
        load_bn_layer(i)

    # save the PyTorch weights.
    torch.save({'state_dict': model.state_dict()}, pth_path)


class LoadedResNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

        if model_name == 'supervised':
            self.model = models.resnet50(pretrained=True)
            del self.model.fc
        else:
            self.model = models.resnet50(pretrained=False)
            del self.model.fc

            path = os.path.join('models', f'{self.model_name}.pth')
            print(path)
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'resnet' in state_dict:
                state_dict = state_dict['resnet']

            if model_name == 'simclr-v2':
                state_dict_keys = [k for k in state_dict.keys() if 'tracked' not in k and 'fc' not in k]
                simclr_dict = pickle.load(open('./simclr_keys.pkl', 'rb'))
                state_dict = {key: state_dict[tf_key] for key, tf_key in simclr_dict.items()}
            else:
                state_dict = self.rename(state_dict)
                state_dict = self.remove_keys(state_dict)

            self.model.load_state_dict(state_dict)

        self.model.train()
        print("num parameters:", sum(p.numel() for p in self.model.parameters()))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def rename(self, d):
        unwanted_prefixes = {
            'supervised': '',
            'simclr-v1': '',
            'simclr-v2': '',
            'byol': '',
            'swav': 'module.',
            'deepcluster-v2': 'module.',
            'sela-v2': 'module.',
            'moco-v1': 'module.encoder_q.',
            'moco-v2': 'module.encoder_q.',
            'cmc': 'module.encoder1.',
            'infomin': 'module.encoder.',
            'insdis': 'module.encoder.',
            'pirl': 'module.encoder.',
            'pcl-v1': 'module.encoder_q.',
            'pcl-v2': 'module.encoder_q.',
        }
        prefix = unwanted_prefixes[self.model_name]
        l = len(prefix)
        new_d = {}
        for key in d.keys():
            if prefix in key:
                new_d[key[l:]] = d[key]
            else:
                new_d[key] = d[key]
        return new_d

    def remove_keys(self, d):
        for key in list(d.keys()):
            if 'module.jigsaw' in key or 'module.head_jig' in key:
                print('warning, jigsaw stream in model')
                d.pop(key)
            elif 'projection' in key or 'prototypes' in key or 'fc' in key or 'linear' in key or 'head' in key:
                print(f'removed {key}')
                d.pop(key)
        return d

def download_pretrained_models(pretrained_models_path='models'):
    for model_name, (url, file_format) in MODELS.items():
        if url and file_format:
            dst = os.path.join(pretrained_models_path, f'{model_name}.{file_format}')
            if not os.path.isfile(dst):
                print(f'Downloading {model_name} from {url} to {dst}')
                subprocess.run(['wget', url, '-O', dst])
            else:
                print(f'Found {model_name} at {dst}')


if __name__ == '__main__':
    os.makedirs("./models", exist_ok=True)
    print('Warning, SimCLR-v1 and SimCLR-v2 models need to be downloaded manually and converted into PyTorch format. See readme.md for details.')
    # download models from URLs
    # download_pretrained_models()
    # if os.path.isfile('models/byol.pkl'):
    #     # BYOL is provided in a pickle format, so we need to handle that first
    #     convert_byol('models/byol.pkl', 'models/byol.pth')
    #     os.remove('models/byol.pkl')
    # resave each model in the same format
    # for model_name in MODELS:
    model_name = 'supervised'
    print(model_name)
    model = LoadedResNet(model_name)
    state_dict = model.state_dict()
    state_dict = {key.replace('model.', ''): val for key, val in state_dict.items()}
    torch.save(state_dict, os.path.join('models', f'{model_name}.pth'))