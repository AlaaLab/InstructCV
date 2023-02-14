from __future__ import absolute_import, division, print_function

# system imports

import sys
import os
import requests
import argparse
import warnings

#warnings.filterwarnings("ignore")

#if not sys.warnoptions:
    
#    warnings.simplefilter("ignore")

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
import config

# plotting and other general imports

from matplotlib import pyplot as plt
import numpy as np
import pickle 
from distutils.util import strtobool
from PIL import Image
from colorama import Fore, Style

# logging tools

from comet_ml import Experiment



model, _               = load_model(ViT_mode='large', prompt=True)
