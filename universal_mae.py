# Copyright (c) 2022, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Script for training/finetuning universal MAEs
# ---------------------------------------------------------------------------------
# References:
# ----------
# Original MAE repo: https://github.com/facebookresearch/mae
# timm backbone models: https://github.com/rwightman/pytorch-image-models
# ---------------------------------------------------------------------------------

# ** import all necessary modules ** 

from __future__ import absolute_import, division, print_function

# system imports

import sys
import os
import requests
import argparse
import warnings

warnings.filterwarnings("ignore")

if not sys.warnoptions:
    
    warnings.simplefilter("ignore")

# torch and timm imports

import torch
from torch.optim import *

# If diff timm version is installed: os.system("pip3 install timm==0.4.5")

import timm
assert timm.__version__ == "0.4.5"

# local imports and utils

import models_mae
from universal_mae_helpers import load_model, get_cuda_devices, get_out_dir, initialize_visual_cue, get_optimizer
from universal_mae_trainer import train
from util.datasets import * 

# plotting and other general imports

from matplotlib import pyplot as plt
import numpy as np
import pickle 
from distutils.util import strtobool
from PIL import Image
from colorama import Fore, Style

# logging tools

from comet_ml import Experiment

# Define global constants
imagenet_mean = np.array(timm.data.constants.IMAGENET_DEFAULT_MEAN)
imagenet_std  = np.array(timm.data.constants.IMAGENET_DEFAULT_STD)



# ** Get command line arguments **

def get_args_parser():
    
    # TODO: add regular logger facility
    
    parser = argparse.ArgumentParser('Universal MAE: Fine-tuning pre-trained MAE to solve new tasks', add_help=False)
    
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--finetune', type=strtobool, default="False", help='Fully finetune the MAE? False means only optimize the visual prompt') 
    parser.add_argument('--seed', default=0, type=int)

    # Model parameters
    parser.add_argument('--model', default='base', type=str, help='ViT model to train (base/large/huge)')
    parser.add_argument('--image_size', default=224, type=int, help='Dimensions of the inpu image')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate (absolute lr)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')
    
    # Dataset, task and prompt parameters  nargs='*'
    parser.add_argument('--dataset_name', default='maze', type=str, help='Name of the source dataset')
    parser.add_argument('--multi_task', type=strtobool, default="False", help='Finetune for multiple tasks') 
    parser.add_argument('--task_list', default=['path_finding_5x5', 'path_finding_6x6'], type=str, nargs='*', help='Description of the vision tasks (only read if multi-task flag is True)')
    parser.add_argument('--task', default='path_finding_5x5', type=str, help='Description of the vision task (discarded if multi-task)')
    parser.add_argument('--prompt_path', default='./prompts/', type=str, help='Path for saving visual prompts')
    parser.add_argument('--model_path', default='./models/', type=str, help='Path for saving finetuned models')
    
    # comet logger parameters
    parser.add_argument('--enable_comet_logger', type=strtobool, default="False", help='Keep track of training progress via online comet logger') 
    parser.add_argument('--log_dir', default='./logs', help='path for logger outputd')
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    return parser


# ** main script **

def main(args):
    
    # fix the seed for reproducibility
    # ........................................

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    # set up cuda device and collect all opt,
    # model, logging and training parameters
    # ........................................
    # TODO: Create paths for outputs
    
    device                   = get_cuda_devices()
    fine_tune, multi_task    = args.finetune==1, "multiple_tasks" if args.multi_task==1 else "single_task"
    num_tasks                = len(args.task_list) if args.multi_task==1 else 1
    experiment               = get_comet_experiment() if args.enable_comet_logger==1 else None 
    experiment_name          = args.dataset_name + "_" + args.task + "_ViT-" + args.model 
    model_pth, prompt_pth    = get_out_dir(args.dataset_name, args.task, args.model, fine_tune)
    selected_tasks           = args.task_list if args.multi_task==1 else args.task

    if experiment is not None:
        
        experiment.set_name(experiment_name)


    model_params             = dict({"model_type": args.model})
    optimizer_params         = dict({"learning_rate": args.lr, "weight_decay": args.weight_decay, "layer_decay": args.layer_decay})
    training_params          = dict({"finetune": fine_tune, "device": device, "batch_size": args.batch_size, "num_epochs": args.epochs})
    logger_params            = dict({"experiment": experiment, "print_freq": 10})    
    out_dir_params           = dict({"model_path": model_pth, "prompt_path": prompt_pth})
    dataset_params           = dict({"dataset_name": args.dataset_name, "task": selected_tasks, "batch_size": args.batch_size})
    
    # build the training & validation datasets
    # ........................................
    # TODO: merge two calls inside build_dataset
    
    #train_loader, val_loader = tuple([make_dataset[multi_task](**dataset_params, split=split) for split in ["train", "val"]]) 
    #val_loader               = make_dataset[multi_task](dataset_name=args.dataset_name, task=selected_tasks, batch_size=1, split="val")

    # data_test              = iter(make_dataset["single_task"](dataset_name="oxford-pets", task="detection", batch_size=16, split="test"))

    # data_batch             = next(data_test)

  
    print('dataset_params:{}'.format(dataset_params))
    train_loader, val_loader = build_dataset(**dataset_params, split="trainval")

    print("Number of training samples: ", len(train_loader.dataset))
    print("Number of validation samples: ", len(val_loader.dataset))
    
    # make_dataset
    
    # load the MAE model pre-trained on ImageNet-1K
    # .............................................
    # NOTE: here we "promptify" the MAE model to
    #       modify the masking pattern of i/p image
    # .............................................
    # + Initialize viisual cue using xavier init
    # .............................................
    # TODO: save model in models/pretrained/
    
    print("Loading pre-trained MAE-ViT model and initializing the visual cue...")
    
    model, patch_size        = load_model(ViT_mode=args.model, prompt=True)
    visual_cue               = initialize_visual_cue(num_tasks, args.image_size)

    # Resume training
    visula_cue_path = out_dir_params['prompt_path']+'/'+model_params['model_type']+'.p'
    if os.path.exists(visula_cue_path):
        file                   = open(visula_cue_path, "rb")
        visual_cue             = pickle.load(file)
        print('load visual_cue file success!, path:{}'.format(visula_cue_path))

    chkpt_dir = out_dir_params['model_path']+'/'+model_params['model_type']+'.pth'
    print('chkpt_dir:{}'.format(chkpt_dir))
    if os.path.exists(chkpt_dir):
        state = torch.load(chkpt_dir)
        model.load_state_dict(state)
        print('load checkpoint success! ckpt:{}'.format(chkpt_dir))
    print('visual_cue:{}'.format(visual_cue.shape))
    
    
    model.to(device)
    visual_cue.to(device)
    
    model_params.update(dict({"patch_size": patch_size}))
    
    # construct the optimizer based on i/p args
    # .........................................
    # TODO: fix lard optimizer 
    # .........................................
    
    optimizer                = get_optimizer(model=model, visual_cue=visual_cue, **training_params, **optimizer_params)
    
    # Finetune the MAE model
    # ......................
    
    model, visual_cue, loss_train, loss_val = train(model, visual_cue, optimizer, train_loader, val_loader,
                                                    **training_params, **logger_params, **model_params, **out_dir_params)
    


if __name__ == '__main__':
    
    args = get_args_parser()
    args = args.parse_args()

    main(args)

    

