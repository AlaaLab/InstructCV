import sys
import os
import requests
import config

import torch
import numpy as np

import models_mae
from universal_mae_helpers import *
import types


def prompted_masking(self, x, mask_ratio):

    """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
    """

    N, L, D     = x.shape  # batch, length, dim
    len_keep    = int(L * (1 - mask_ratio))
    
    noise       = torch.arange(1, L+1).type(torch.int64).view(1, -1).to(config.device) #torch.rand(N, L, device=x.device)  # noise in [0, 1]

    all_ids     = set(list(np.arange(0, L))) 
    ids_remove  = set(list(np.arange(0, L).reshape(config.NUM_PATCHES, config.NUM_PATCHES)[int(config.NUM_PATCHES/2):, :int(config.NUM_PATCHES/2)].reshape(-1,)))
    ids_to_keep = all_ids - ids_remove

    ids_shuffle = torch.Tensor(np.array(list(ids_to_keep) + list(ids_remove))).type(torch.int64).view(1, -1).to(config.device)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_restore = ids_restore.repeat(N, 1)

    len_keep    = len(list(ids_to_keep))
    ids_keep    = torch.Tensor(np.array(list(ids_to_keep))).type(torch.int64).view(1, -1).to(config.device)

    ids_keep    = ids_keep.repeat(N, 1)

    x_masked    = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))


    # generate the binary mask: 0 is keep, 1 is remove
    mask               = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask               = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def prompt_reconstruction(self, x, mask_ratio):

    """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
    """

    N, L, D     = x.shape  # batch, length, dim
    len_keep    = int(L * (1 - mask_ratio))
    
    noise       = torch.arange(1, L+1).type(torch.int64).view(1, -1).to(config.device) #torch.rand(N, L, device=x.device)  # noise in [0, 1]

    all_ids     = set(list(np.arange(0, L))) 
    ids_remove  = set(list(np.arange(0, L).reshape(config.NUM_PATCHES, config.NUM_PATCHES)[int(config.NUM_PATCHES/2):, :int(config.NUM_PATCHES/2)].reshape(-1,)))
    ids_to_keep = all_ids - ids_remove

    ids_shuffle = torch.Tensor(np.array(list(ids_to_keep) + list(ids_remove))).type(torch.int64).view(1, -1).to(config.device)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_restore = ids_restore.repeat(N, 1)

    len_keep    = len(list(ids_to_keep))
    ids_keep    = torch.Tensor(np.array(list(ids_to_keep))).type(torch.int64).view(1, -1).to(config.device)

    ids_keep    = ids_keep.repeat(N, 1)

    x_masked    = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))


    # generate the binary mask: 0 is keep, 1 is remove
    mask               = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask               = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def promptify(model):
    
    model.random_masking = types.MethodType(prompted_masking, model)
    
    return model
