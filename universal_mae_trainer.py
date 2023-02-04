# Copyright (c) 2022, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)
# ---------------------------------------------------------------------------------
# ** Description ** Training functions and tools for universal MAE
# ---------------------------------------------------------------------------------

# ** import all necessary modules ** 

from __future__ import absolute_import, division, print_function

# system imports

import sys
import os
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
from universal_mae_helpers import *
from util.datasets import * 
import config

# plotting and other general imports

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


# helper functions
"""
def stitch_visual_collage(train_batch, visual_cue, batch_size):
    
    if type(train_batch) is list:
        
        train_batch = [torch.cat([train_batch[0], train_batch[1]], dim=2)]
    
                   
    if visual_cue.shape[0]==1:
    
        train = torch.cat([torch.einsum("nchw->nhwc", train_batch[0]), visual_cue[0, :, :, :].unsqueeze(0).repeat(batch_size, 1, 1, 1)], dim=2) 
    
    else:

        data_batch, label_batch = train_batch
        train                   = torch.cat([torch.einsum("nchw->nhwc", data_batch[0]), visual_cue[label_batch, :, :, :]], dim=2) 
    
    train = torch.einsum("nhwc->nchw", (train - torch.tensor(imagenet_mean)) / torch.tensor(imagenet_std)).float()
    
    return train
"""        

    
def print_training_progress(epoch, num_epochs):    
    
    print(Fore.GREEN + Style.BRIGHT + "-----------" + ''.join(map(str, ["-"] * num_epochs)) + Style.RESET_ALL)
    print(Fore.YELLOW + Style.BRIGHT + "Epoch: {} ".format(epoch), ''.join(map(str, ["|"] * epoch)), ''.join(map(str, [" "] * (num_epochs - epoch))), "* " + "(" + str(int(epoch/num_epochs * 100))+"%)")
    print(Fore.GREEN + Style.BRIGHT + "-----------" + ''.join(map(str, ["-"] * num_epochs)) + Style.RESET_ALL)    
    
    

# epoch training function
    
    
def train_epoch(model, visual_cue, optimizer, data_loader, device, print_freq=10, experiment=None, epoch_num=None):
    
    """
    
    A function for training/finetuning the MAE-ViT model for one epoch
    ..................................................................
    
    :param model: The pre-trained MAE-ViT model (an instantiation of the MaskedAutoencoderViT class
    
    :param visual_cue: A parameter matrix of size (image_size x image_size/2)
    
    :param optimizer: Typically an instance of torch's Adam optimizer
    
    :param data_loader: Training data loader
    
    :param print_freq: The frequency of printing training progress
    
    :param experiment: Comet experiment instance class
    
    :param epoch_num: The epoch training number
        
    """
    
    n_iterations       = data_loader.__len__() if visual_cue.shape[0]==1 else data_loader[0].__len__()
    batch_size         = data_loader.batch_size if visual_cue.shape[0]==1 else data_loader[0].batch_size
    training_loader    = data_loader if visual_cue.shape[0]==1 else zip(data_loader[0], data_loader[1])
    image_size         = visual_cue.shape[1]
    
    iter_counter       = 0
    loss_list          = []
    
    
    for train_batch in training_loader: # replace with enumerate and remove iter_counter

        # create a visual collage and apply forward pass

        visual_collage = stitch_visual_collage(train_batch, visual_cue, batch_size).to(device)   
        loss, y, mask  = model(visual_collage)
        
            
        # Clear gradients
    
        optimizer.zero_grad()

        # Calculating gradients
        
        if torch.cuda.device_count() > 1:
 
            loss.sum().backward()
            loss_list.append(np.float(loss.sum().cpu().detach().numpy()) / torch.cuda.device_count())
            
        else:
            
            loss.backward()
            loss_list.append(np.float(loss.cpu().detach().numpy()))
        
        # Update parameters
        optimizer.step()  
        
        iter_counter += 1
        
        if experiment is not None:
        
            experiment.log_metric("Training loss / iteration", loss_list[-1], step=epoch_num * n_iterations + iter_counter)
        
        if (iter_counter % print_freq) == 0:
            
            print("Iteration: {} \t--- Training Loss: {:.4f}".format(iter_counter, loss_list[-1]))

    epoch_loss = np.mean(np.array(loss_list))    
            
    return epoch_loss  





def get_validatation_loss(model, visual_cue, optimizer, data_loader, model_type, patch_size, num_epochs, epoch_loss, epoch, 
                          val_loss, best_val_loss, best_v_cue, model_path, prompt_path, print_freq=10, finetune=False, experiment=None):
    
    
    valid_loader  = data_loader if visual_cue.shape[0]==1 else zip(data_loader[0], data_loader[1])
    
    with torch.no_grad():
        
        model.eval()
        
        #curr_val_loss = mae_forward_pass(model, valid_loader, visual_cue=visual_cue, IMG_SIZE=visual_cue.shape[1], 
        #                                 PATCH_SIZE=patch_size, NUM_PATCHES=config.NUM_PATCHES, segment=False)
        
        val_losses         = []
        
        for val_batch in valid_loader:
            
            visual_collage = stitch_visual_collage(val_batch, visual_cue, valid_loader.batch_size).to(device)  
            loss, _, __    = model(visual_collage)
            
            if torch.cuda.device_count() > 1:
                
                val_losses.append(np.float(loss.sum().cpu().detach().numpy()) / torch.cuda.device_count())
            
            else:
                
                val_losses.append(np.float(loss.cpu().detach().numpy()))
            
            
        val_loss.append(np.mean(val_losses))

        """
        
        if experiment is not None:
            
            experiment.log_metric("Validation loss", curr_val_loss, step=epoch)
            
            
        val_batches   = []
        val_labels    = []
        
        
                
        valid_loader  = data_loader if visual_cue.shape[0]==1 else zip(data_loader[0], data_loader[1])
        
        
        for _data in valid_loader:
            
            if visual_cue.shape[0]==1:
                val_batches.append(_data[0])
            
            else:
                val_batches.append(_data[0][0])
                val_labels.append(_data[1][0].item())
        
        val_data      = torch.einsum("nchw->nhwc", torch.cat(val_batches))
        val_labels    = np.array(val_labels)
            
        n_visuals     = 9  
        viz_images    = forward_collage(model, 
                                        val_data[:n_visuals, :, :, :], 
                                        visual_cue=visual_cue[val_labels[:n_visuals], :, :, :], 
                                        IMG_SIZE=visual_cue.shape[1], 
                                        PATCH_SIZE=patch_size, 
                                        NUM_PATCHES=config.NUM_PATCHES)
            
        if experiment is not None:
            
            experiment.log_image(visual_cue[0, :, :, :], name=prompt_path + "_visual_prompt")
            
        for _ in range(n_visuals):
                
            if experiment is not None:
            
                experiment.log_image(viz_images[_], name="visualize_" + str(_))
                
                
        fig, axs   = plt.subplots(1, n_visuals, figsize=(72, 12))
        viz_images = [tensor_2_pil(torch.einsum("hwc->chw", viz_images[k])) for k in range(len(viz_images))]

        for i, image in enumerate(viz_images):
                
            axs[i].imshow(image)
            axs[i].set_axis_off()
            
        if experiment is not None:
                
            experiment.log_figure(figure=fig, figure_name="image_visualization")
            
        """
        
    if (epoch > 0):  
        
        if (val_loss[-1] < best_val_loss):
            
            best_v_cue    = visual_cue.clone() 
            best_val_loss = val_loss[-1]
                
            pickle.dump(best_v_cue, open(prompt_path + "/" + model_type + ".p", "wb" ))
 
            if finetune:
    
                torch.save(model.state_dict(), model_path + "/" + model_type + ".pth")
        
        else:
            
            best_v_cue = None
        
    else:
        
        best_v_cue = visual_cue
    
    
    print(Fore.CYAN + Style.BRIGHT + "........................................." + Style.RESET_ALL)
    print(Fore.CYAN + Style.BRIGHT + "Epoch: {} \t--- Training Loss: {:.4f} \t--- Validation Loss: {:.4f}".format(epoch, epoch_loss[-1], val_loss[-1]))
        
    return val_loss, best_v_cue, best_val_loss
    

    
def train(model, visual_cue, optimizer, train_data_loader, val_data_loader, device, model_type,  
          patch_size, num_epochs, model_path, prompt_path, print_freq=10, finetune=False, experiment=None, **kwargs):
    
    epoch_loss      = [] 
    val_loss        = [] 
     
    best_val_loss   = 1e6 
    n_iterations    = train_data_loader.__len__() if visual_cue.shape[0]==1 else train_data_loader[0].__len__()
    batch_size      = train_data_loader.batch_size if visual_cue.shape[0]==1 else train_data_loader[0].batch_size
    image_size      = visual_cue.shape[1]
    best_v_cue      = visual_cue.detach().clone()

    for epoch in range(num_epochs): 

        print_training_progress(epoch, num_epochs)
        
        epoch_loss_ = train_epoch(model, visual_cue, optimizer, train_data_loader, device, print_freq=print_freq, experiment=experiment, epoch_num=epoch)
        
        epoch_loss.append(epoch_loss_) 
        
        
        if experiment is not None:
            
            experiment.log_metric("Training loss", epoch_loss_, step=epoch)
    
    
        val_loss, best_v_cue, best_val_loss    = get_validatation_loss(model, visual_cue, optimizer, 
                                                                       data_loader=val_data_loader, model_type=model_type, 
                                                                       patch_size=patch_size, num_epochs=num_epochs, 
                                                                       epoch_loss=epoch_loss, epoch=epoch, 
                                                                       val_loss=val_loss, best_val_loss=best_val_loss, 
                                                                       best_v_cue=best_v_cue, 
                                                                       model_path=model_path, prompt_path=prompt_path, 
                                                                       print_freq=10, finetune=finetune, experiment=experiment)

    torch.cuda.empty_cache()
    
    if finetune:
        
        torch.save(model.state_dict(), model_path + "/" + model_type + ".pth")
    
    else:

        visual_cue   = best_v_cue 
    
    
    pickle.dump(visual_cue, open(prompt_path + "/" + model_type + ".p", "wb" ))
    
    return model, visual_cue, epoch_loss, val_loss














